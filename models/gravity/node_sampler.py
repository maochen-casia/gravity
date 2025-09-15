import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NodeSampler(nn.Module):

    def __init__(self, num_nodes, hid_dim, scale=16):
        super().__init__()

        self.num_nodes = num_nodes
        self.scale = scale

        self.weight_proj = nn.Sequential(
            nn.Conv2d(hid_dim, hid_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hid_dim, scale*scale, kernel_size=1, padding=0)
        )
        self.feat_proj = nn.Conv2d(hid_dim, hid_dim, kernel_size=1, padding=0)

    def simple_nms(self, scores, nms_radius, mask_val='-inf'):
        """ Fast Non-maximum suppression to remove nearby points """

        assert(nms_radius >= 0)
        assert mask_val in {'zero', '-inf'}

        def max_pool(x):
            return torch.nn.functional.max_pool2d(
                x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

        mask_val = {'zero': 0.0, '-inf': float('-inf')}[mask_val]
        mask_val = torch.ones_like(scores) * mask_val
        max_mask = scores == max_pool(scores)
        for _ in range(2):
            supp_mask = max_pool(max_mask.float()) > 0
            supp_scores = torch.where(supp_mask, mask_val, scores)
            new_max_mask = supp_scores == max_pool(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, scores, mask_val)

    def mask_borders(self, scores, border, mask_val='-inf'):
        """ Masks out scores that are too close to the border """
        b, h, w = scores.shape
        mask = torch.zeros((b, h, w), device=scores.device)
        mask[:, border:h-border, border:w-border] = 1
        mask_val = {'zero': 0.0, '-inf': float('-inf')}[mask_val]
        mask_val = torch.ones_like(scores) * mask_val
        scores = torch.where(mask > 0, scores, mask_val)
        return scores

    def forward(self, featmap):

        B, C, H, W = featmap.shape
        H_new, W_new = H * self.scale, W * self.scale

        weight_map = self.weight_proj(featmap).permute(0,2,3,1).reshape(B,H,W,self.scale,self.scale)
        weight_map = weight_map.permute(0, 1, 3, 2, 4).reshape(B, H_new, W_new)

        weight_map = self.mask_borders(weight_map, border=H_new//256, mask_val='-inf')
        weight_map = self.simple_nms(weight_map, nms_radius=H_new//256, mask_val='-inf')

        weights, indices = torch.topk(weight_map.flatten(1), self.num_nodes, dim=1)
        y, x = indices // W_new, indices % W_new

        featmap = self.feat_proj(featmap)
        norm_y, norm_x = y / (H_new-1) * 2 - 1, x / (W_new-1) * 2 - 1
        norm_xy = torch.stack([norm_x, norm_y], dim=-1).view(B, 1, self.num_nodes, 2)
        features = torch.nn.functional.grid_sample(featmap, norm_xy, mode='bilinear', align_corners=False)
        features = features.reshape(B, -1, self.num_nodes).permute(0, 2, 1)

        return norm_xy.squeeze(1), weights, features
