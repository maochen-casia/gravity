import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FusionLayer(nn.Module):

    def __init__(self, hid_dim, num_heads):

        super().__init__()

        self.self_attn = nn.MultiheadAttention(hid_dim, num_heads, batch_first=True)
        self.cross_attn1 = nn.MultiheadAttention(hid_dim, num_heads, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(hid_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 4),
            nn.GELU(),
            nn.Linear(hid_dim * 4, hid_dim)
        )

        self.self_ln = nn.LayerNorm(hid_dim)
        self.cross_query_ln1 = nn.LayerNorm(hid_dim)
        self.cross_key_ln1 = nn.LayerNorm(hid_dim)
        self.cross_query_ln2 = nn.LayerNorm(hid_dim)
        self.cross_key_ln2 = nn.LayerNorm(hid_dim)
        self.ffn_ln = nn.LayerNorm(hid_dim)
    
    def forward(self, nodes, left_features, sat_features):

        ln_query = self.self_ln(nodes)
        self_out = self.self_attn(ln_query, ln_query, ln_query)[0]
        nodes = nodes + self_out

        ln_query = self.cross_query_ln1(nodes)
        ln_key = self.cross_key_ln1(sat_features)
        cross_out1 = self.cross_attn1(ln_query, ln_key, ln_key)[0]
        nodes = nodes + cross_out1

        ln_query = self.cross_query_ln2(nodes)
        ln_key = self.cross_key_ln2(left_features)
        cross_out2 = self.cross_attn2(ln_query, ln_key, ln_key)[0]
        nodes = nodes + cross_out2

        ln_ffn = self.ffn_ln(nodes)
        ffn_out = self.ffn(ln_ffn)
        nodes = nodes + ffn_out

        return nodes
        

class NodeFusion(nn.Module):

    def __init__(self, hid_dim, num_layers, num_heads, ):
        super().__init__()

        self.hid_dim = hid_dim

        self.node_in_proj = nn.Linear(hid_dim, hid_dim)
        self.node_pre_ln = nn.LayerNorm(hid_dim)

        self.left_in_proj = nn.Linear(hid_dim, hid_dim)
        self.left_pre_ln = nn.LayerNorm(hid_dim)

        self.sat_in_proj = nn.Linear(hid_dim, hid_dim)
        self.sat_pre_ln = nn.LayerNorm(hid_dim)

        self.layers = nn.ModuleList([FusionLayer(hid_dim, num_heads) for _ in range(num_layers)])

        self.node_post_ln = nn.LayerNorm(hid_dim)
        self.node_feature_proj = nn.Linear(hid_dim, hid_dim)
        self.node_weight_proj = nn.Linear(hid_dim, 1)

    
    def forward(self, node_weights, node_features, left_features, sat_features):

        node_features = self.node_pre_ln(self.node_in_proj(node_features))
        left_features = self.left_pre_ln(self.left_in_proj(left_features))
        sat_features = self.sat_pre_ln(self.sat_in_proj(sat_features))

        for layer in self.layers:
            node_features = layer(node_features, left_features, sat_features)
        
        node_features = self.node_post_ln(node_features)
        
        fuse_node_weights = self.node_weight_proj(node_features).squeeze(-1)
        node_weights = node_weights + fuse_node_weights
        node_scores = F.softmax(node_weights / 8, dim=-1)

        node_features = self.node_feature_proj(node_features)

        return node_scores, node_features