import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings

from transform import image2world, world2image

@torch.jit.script
def project_bev_graph(
                    node_coords: torch.Tensor,
                    node_depths: torch.Tensor,
                    K_left: torch.Tensor,
                    R_left2world: torch.Tensor,
                    t_left2world: torch.Tensor,
                    K_sat: torch.Tensor,
                    R_sat2world: torch.Tensor,
                    t_sat2world: torch.Tensor) -> torch.Tensor:
    """
    Projects a batch of graph nodes from the rover's image plane to the satellite's image plane.
    """
    # Project from rover image plane to 3D world coordinates
    x_world = image2world(node_coords, K_left, R_left2world, t_left2world, node_depths)
    
    # Project from 3D world coordinates to satellite image plane
    bev_graph_coords = world2image(x_world, K_sat, R_sat2world, t_sat2world)
    
    return bev_graph_coords

class GraphMatcher(nn.Module):

    def __init__(self, in_dim, hid_dim):
        super().__init__()

        self.node_in_proj = nn.Linear(in_dim, hid_dim)
        self.sat_in_proj = nn.Conv2d(in_dim, hid_dim, kernel_size=1, padding=0)

        self.graph_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @torch.jit.script
    def project_bev_graph(self,
                          node_coords: torch.Tensor,
                          node_depths: torch.Tensor,
                          K_left: torch.Tensor,
                          R_left2world: torch.Tensor,
                          t_left2world: torch.Tensor,
                          K_sat: torch.Tensor,
                          R_sat2world: torch.Tensor,
                          t_sat2world: torch.Tensor) -> torch.Tensor:
        """
        Projects a batch of graph nodes from the rover's image plane to the satellite's image plane.
        """
        # Project from rover image plane to 3D world coordinates
        x_world = image2world(node_coords, K_left, R_left2world, t_left2world, node_depths)
        
        # Project from 3D world coordinates to satellite image plane
        bev_graph_coords = world2image(x_world, K_sat, R_sat2world, t_sat2world)
        
        return bev_graph_coords

    def forward(self,
                node_coords: torch.Tensor,
                node_scores: torch.Tensor,
                node_features: torch.Tensor,
                node_depths: torch.Tensor,
                K_left: torch.Tensor,
                R_left2world: torch.Tensor,
                t_left2world_init: torch.Tensor,
                sat_featmap: torch.Tensor,
                K_sat: torch.Tensor,
                R_sat2world: torch.Tensor,
                t_sat2world: torch.Tensor,
                search_radius: torch.Tensor,
                search_steps: int):
        
        B, N, _ = node_features.shape
        device = node_features.device

        node_features = self.node_in_proj(node_features)  # Shape: (B, N, C)
        sat_featmap = self.sat_in_proj(sat_featmap)      # Shape: (B, C, H, W)

        _, C, H, W = sat_featmap.shape

        norm_node_features = F.normalize(node_features, p=2, dim=-1)
        norm_sat_featmap = F.normalize(sat_featmap, p=2, dim=1)

        sim_map = torch.einsum('bnc,bchw->bnhw', norm_node_features, norm_sat_featmap)

        step_range = torch.linspace(-1, 1, steps=search_steps, device=device)
        offset_y, offset_x = torch.meshgrid(step_range, step_range, indexing='ij')
        # Shape: (S, 2), where S = search_steps * search_steps
        normalized_offsets = torch.stack([offset_x, offset_y], dim=-1).view(-1, 2)
        S = normalized_offsets.shape[0]

        # Scale offsets by the per-batch search radius (in meters)
        # search_radius shape: (B,) -> (B, 1, 1) for broadcasting
        world_offsets_2d = normalized_offsets.unsqueeze(0) * search_radius.view(B, 1, 1) # Shape: (B, S, 2)
        
        # Create 3D world offsets (assuming movement is on the XY plane)
        world_offsets_3d = F.pad(world_offsets_2d, (0, 1), "constant", 0.0) # Shape: (B, S, 3)

        candidate_t_left2world = t_left2world_init.unsqueeze(1) + world_offsets_3d # Shape: (B, S, 3)

        # --- Project all candidate graphs in a single batched operation ---
        # Reshape and expand tensors to run projection for all B*S candidates at once
        all_candidate_coords = project_bev_graph(
            node_coords=node_coords.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, N, 2),
            node_depths=node_depths.unsqueeze(1).expand(-1, S, -1).reshape(B * S, N),
            K_left=K_left.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, 3, 3),
            R_left2world=R_left2world.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, 3, 3),
            t_left2world=candidate_t_left2world.reshape(B * S, 3),
            K_sat=K_sat.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, 3, 3),
            R_sat2world=R_sat2world.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, 3, 3),
            t_sat2world=t_sat2world.unsqueeze(1).expand(-1, S, -1).reshape(B * S, 3)
        )
        # Reshape back to (B, S, N, 2)
        candidate_graphs_coords = all_candidate_coords.view(B, S, N, 2)

        # --- Sample similarity for each node in each candidate graph ---
        # Prepare coordinates for grid_sample: shape (B*N, S, 1, 2)
        # Normalize coordinates to [-1, 1] range for grid_sample
        norm_coords = candidate_graphs_coords.permute(0, 2, 1, 3).reshape(B * N, S, 1, 2)
        norm_coords = norm_coords / torch.tensor([W - 1, H - 1], device=device) * 2 - 1
        
        # Sample from the similarity map
        sim_per_node = F.grid_sample(
            sim_map.reshape(B * N, 1, H, W),
            norm_coords,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        ).view(B, N, S).permute(0, 2, 1) # Shape: (B, S, N)

        # --- Score each candidate graph ---
        # Weight node similarities by their scores and sum to get graph-level logits
        candidate_graph_logits = torch.sum(node_scores.unsqueeze(1) * sim_per_node, dim=2)  # Shape: (B, S)
        
        # Apply scaling and compute softmax scores
        candidate_graph_scores = F.softmax(candidate_graph_logits * self.graph_logit_scale.exp(), dim=1)  # Shape: (B, S)
        
        # Find the best candidate graph based on scores
        best_indices = torch.argmax(candidate_graph_scores, dim=1)  # Shape: (B,)
        
        # Select the coordinates of the best matching graph
        match_sat_coords = candidate_graphs_coords[torch.arange(B), best_indices] # Shape: (B, N, 2)

        # Create a validity mask for the best matching graph
        valid_mask = (
            (match_sat_coords[..., 0] >= 0) & (match_sat_coords[..., 0] < W) &
            (match_sat_coords[..., 1] >= 0) & (match_sat_coords[..., 1] < H)
        ) # Shape: (B, N)

        return match_sat_coords, valid_mask, candidate_t_left2world, candidate_graph_scores

    def loss(self, 
            sat_featmap_size: tuple[int, int],
            node_coords: torch.Tensor,
            node_depths: torch.Tensor,
            node_scores: torch.Tensor,
            K_left: torch.Tensor,
            R_left2world: torch.Tensor,
            t_left2world: torch.Tensor,
            K_sat: torch.Tensor,
            R_sat2world: torch.Tensor,
            t_sat2world: torch.Tensor,
            candidate_t_left2world: torch.Tensor,
            candidate_graph_scores: torch.Tensor):

        bev_graph_coords = project_bev_graph(
            node_coords, node_depths, 
            K_left, R_left2world, t_left2world, 
            K_sat, R_sat2world, t_sat2world
        ) # Shape: (B, N, 2)

        H, W = sat_featmap_size
        bev_valid_mask = (
            (bev_graph_coords[..., 0] >= 0) & (bev_graph_coords[..., 0] < W) &
            (bev_graph_coords[..., 1] >= 0) & (bev_graph_coords[..., 1] < H)
        )

        if torch.all(bev_valid_mask):
            invalid_loss = torch.zeros([], device=node_scores.device)
        else:
            invalid_loss = torch.sum(node_scores[~bev_valid_mask])

        dist_per_candidate = torch.norm(candidate_t_left2world - t_left2world.unsqueeze(1), dim=-1) # Shape: (B, S)

        # The target label is the index of the candidate pose with the minimum distance
        target_label = torch.argmin(dist_per_candidate, dim=1)  # Shape: (B,)
        min_dist = torch.gather(dist_per_candidate, dim=1, index=target_label[:,None]).squeeze(1)
        error = torch.max(min_dist).item()
        if error > 0.5:
            warnings.warn(f'Distance between target candidate and ground truth {error} exceeds limit.')
        log_probs = torch.log(candidate_graph_scores + 1e-9)
        nll_loss = -log_probs[torch.arange(log_probs.shape[0]), target_label].mean()  # Shape: (B,)

        return nll_loss, invalid_loss