import os, sys
code_dir = os.path.dirname(os.path.realpath(__file__))
if code_dir not in sys.path:
    sys.path.append(code_dir)
sys.path.append(os.path.dirname(code_dir))

import torch
import torch.nn as nn
from collections import OrderedDict

from DINOv3.dinov3_encoder import DINOv3
from dpt import DPTHead
from node_sampler import NodeSampler
from node_fusion import NodeFusion
from graph_matcher import GraphMatcher
from pose_estimator import PoseEstimator


class GRAVITY(nn.Module):
    """
        GRAVITY: GRaph-based Alignment of Viewpoint Invariant TopologY
    """

    def __init__(self, num_nodes, hid_dim, num_fusion_layers, num_fusion_heads, device):
        super().__init__()
        self.dino = DINOv3(device, sat=False)
        self.scale = self.dino.scale

        self.dpt = DPTHead(in_channels=self.dino.embed_dim, features=hid_dim, final_out_channels=hid_dim,
                           out_channels=[hid_dim//2, hid_dim, hid_dim*2, hid_dim*2])

        self.left_in_proj = nn.Linear(self.dino.embed_dim, hid_dim)
        self.sat_in_proj = nn.Linear(self.dino.embed_dim, hid_dim)

        self.node_sampler = NodeSampler(num_nodes=num_nodes,
                                        hid_dim=hid_dim,
                                        scale=self.scale)

        self.node_fusion = NodeFusion(hid_dim=hid_dim,
                                      num_layers=num_fusion_layers,
                                      num_heads=num_fusion_heads)

        
        self.graph_matcher = GraphMatcher(in_dim=hid_dim, hid_dim=hid_dim)
        self.pose_estimator = PoseEstimator()
        self.device = device
        
        self.to(device)

    def train_params(self):

        params = [{'params': [p for p in self.parameters() if p.requires_grad], 'lr_scale': 1}]
        return params
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Overrides the default state_dict() method to exclude the parameters
        of the frozen 'dino' submodule.
        """
        original_state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        filtered_state_dict = OrderedDict()

        for key, value in original_state_dict.items():
            if not key.startswith(prefix + 'dino.'):
                filtered_state_dict[key] = value

        return filtered_state_dict

    def load_state_dict(self, state_dict, strict=True):
        """
        Overrides the default load_state_dict() method to gracefully handle
        the missing 'dino' parameters by always using strict=False internally.
        """
        super().load_state_dict(state_dict, strict=False)

    def forward(self, data):

        left_image = data['left_image'].to(self.device)
        sat_image = data['sat_image'].to(self.device)
        left_depth_map = data['left_depth'].to(self.device)
        K_left = data['K_left'].to(self.device)
        K_sat = data['K_sat'].to(self.device)
        R_left2world = data['R_left2world'].to(self.device)
        t_left2world_init = data['t_left2world_init'].to(self.device)
        R_sat2world = data['R_sat2world'].to(self.device)
        t_sat2world = data['t_sat2world'].to(self.device)
        max_init_offset = data['max_init_offset'].to(self.device)

        B, C, H1, W1 = left_image.shape
        B, C, H2, W2 = sat_image.shape

        left_features = self.left_in_proj(self.dino(left_image))
        left_featmap = left_features.permute(0,2,1).unflatten(-1, (H1//self.scale, W1//self.scale))
        
        sat_intermediate_features = self.dino.get_intermediate_layers(sat_image)
        sat_features = self.sat_in_proj(sat_intermediate_features[-1][0])

        norm_node_coords, node_weights, node_features = self.node_sampler(left_featmap)
        node_scores, node_features = self.node_fusion(node_weights=node_weights,
                                                      node_features=node_features,
                                                      left_features=left_features,
                                                      sat_features=sat_features)

        node_coords = torch.stack([(norm_node_coords[..., 0] + 1) / 2 * (W1-1),
                                   (norm_node_coords[..., 1] + 1) / 2 * (H1-1)], dim=-1)

        node_depths = torch.nn.functional.grid_sample(left_depth_map.unsqueeze(1), 
                                                      norm_node_coords.unsqueeze(1), 
                                                      mode='bilinear', 
                                                      align_corners=False).view(node_scores.shape)

        sat_featmap = self.dpt(sat_intermediate_features,
                               patch_size=(H2//self.scale, W2//self.scale),
                               out_size=(H2,W2))

        meter_per_pixel = 167.82 / H2    
        search_radius = max_init_offset
        search_steps = int(torch.max(max_init_offset).item() * 2 / meter_per_pixel) + 1
        match_result = self.graph_matcher(node_coords,
                                          node_scores,
                                          node_features,
                                          node_depths,
                                          K_left,
                                          R_left2world,
                                          t_left2world_init,
                                          sat_featmap,
                                          K_sat,
                                          R_sat2world,
                                          t_sat2world,
                                          search_radius=search_radius,
                                          search_steps=search_steps)
        match_sat_coords, valid, candidate_t_left2world, candidate_graph_scores = match_result
        match_weights = node_scores * valid.float()

        t_left2world = self.pose_estimator(x_left=node_coords,
                                           depth_left=node_depths,
                                           K_left=K_left,
                                           R_left2world=R_left2world,
                                           x_sat=match_sat_coords,
                                           K_sat=K_sat,
                                           R_sat2world=R_sat2world,
                                           t_sat2world=t_sat2world,
                                           weights=match_weights)
        
        pred = {'t_left2world': t_left2world,
                'node_coords': node_coords,
                'node_depths': node_depths,
                'node_scores': node_scores,
                'sat_featmap_size': sat_featmap.shape[-2:],
                'K_left': K_left,
                'R_left2world': R_left2world,
                'K_sat': K_sat,
                'R_sat2world': R_sat2world,
                't_sat2world': t_sat2world,
                'candidate_t_left2world': candidate_t_left2world,
                'candidate_graph_scores': candidate_graph_scores,
                'match_sat_coords': match_sat_coords,}

        return pred
    
    def loss(self, pred, label):

        t_left2world_label = label['t_left2world'].to(self.device)


        match_loss, invalid_loss = self.graph_matcher.loss(sat_featmap_size=pred['sat_featmap_size'],
                                                           node_coords=pred['node_coords'],
                                                           node_depths=pred['node_depths'],
                                                           node_scores=pred['node_scores'],
                                                           K_sat=pred['K_sat'],
                                                           R_sat2world=pred['R_sat2world'],
                                                           t_sat2world=pred['t_sat2world'],
                                                           K_left=pred['K_left'],
                                                           R_left2world=pred['R_left2world'],
                                                            t_left2world=t_left2world_label,
                                                            candidate_t_left2world=pred['candidate_t_left2world'],
                                                            candidate_graph_scores=pred['candidate_graph_scores'])
                                       

        t_error = torch.norm(t_left2world_label - pred['t_left2world'], dim=-1).mean()

        loss = match_loss
        loss_dict = {'loss': loss.item(), 'm': match_loss.item(), 'i': invalid_loss.item(), 't': t_error.item()}
        return loss, loss_dict