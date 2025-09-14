import torch

from .gravity import GRAVITY

def build_gravity(config):
    device = torch.device(config.device)
    model = GRAVITY(num_nodes=config.num_nodes,
                    hid_dim=config.hid_dim,
                    num_fusion_layers=config.num_fusion_layers,
                    num_fusion_heads=config.num_fusion_heads,
                    device=device)
    return model