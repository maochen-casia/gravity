import torch
import torch.nn as nn

from transform import image2world, world2cam

class PoseEstimator(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,
                x_left: torch.Tensor,
                depth_left: torch.Tensor,
                K_left: torch.Tensor,
                R_left2world: torch.Tensor,
                x_sat: torch.Tensor,
                K_sat: torch.Tensor,
                R_sat2world: torch.Tensor,
                t_sat2world: torch.Tensor,
                weights: torch.Tensor,) -> torch.Tensor:
        
        
        B, N, _ = x_left.shape

        X_world_no_t = image2world(x_left, K_left, R_left2world, torch.zeros([B, 3], device=x_left.device), depth_left)

        X_sat_cam_no_t = world2cam(X_world_no_t, R_sat2world, t_sat2world)
        
        C = (K_sat @ X_sat_cam_no_t.transpose(-2, -1)).transpose(-2, -1)
        C_x, C_y, C_z = C.unbind(dim=-1)

        u, v = x_sat.unbind(dim=-1) # Each is (B, K)

        R_world2sat = R_sat2world.transpose(-2, -1)
        M = K_sat @ R_world2sat
        M0 = M[:, 0, :].unsqueeze(1) # (B, 1, 3)
        M1 = M[:, 1, :].unsqueeze(1) # (B, 1, 3)
        M2 = M[:, 2, :].unsqueeze(1) # (B, 1, 3)

        A_row1 = M0 - u.unsqueeze(-1) * M2 # (B, K, 3)
        A_row2 = M1 - v.unsqueeze(-1) * M2 # (B, K, 3)

        b_val1 = u * C_z - C_x # (B, K)
        b_val2 = v * C_z - C_y # (B, K)

        A = torch.stack([A_row1, A_row2], dim=2).view(B, 2 * N, 3)
        b = torch.stack([b_val1, b_val2], dim=2).view(B, 2 * N)

        weights = weights.clamp(min=0.0)
        w_sqrt = torch.sqrt(weights + 1e-8)
        w_expanded = w_sqrt.repeat_interleave(2, dim=1) # Shape: (B, 2*K)

        A_w = w_expanded.unsqueeze(-1) * A # Shape: (B, 2*K, 3)
        b_w = w_expanded * b               # Shape: (B, 2*K)

        try:
            solution = torch.linalg.lstsq(A_w, b_w)
            t_solved = solution.solution # Shape: (B, 3)
        except torch.linalg.LinAlgError:
            # If solving fails, return the initial guess as a fallback
            print("Warning: Least-squares solver failed. Returning initial translation.")
            return None

        return t_solved