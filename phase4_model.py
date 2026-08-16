"""
Phase 4: Model Architecture (Original / Iteration 1 settings restored)
This is the exact configuration that produced val MSE 2.6232 and
test RMSE 2.2974 the first time. No LayerNorm, no extra dropout tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GATConv

MOL_NODE_FEATURES = 5
POCKET_NODE_FEATURES = 20

HIDDEN_DIM = 128
NUM_MOL_LAYERS = 3
NUM_POCKET_LAYERS = 3
ATTENTION_HEADS = 4
DROPOUT = 0.2


class MoleculeEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super(MoleculeEncoder, self).__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList()
        layer_index = 0
        while layer_index < num_layers:
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.layers.append(GINConv(mlp))
            layer_index = layer_index + 1

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        x = F.relu(x)

        layer_index = 0
        while layer_index < len(self.layers):
            residual = x
            x = self.layers[layer_index](x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + residual
            layer_index = layer_index + 1

        return x


class PocketEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, heads):
        super(PocketEncoder, self).__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList()
        layer_index = 0
        while layer_index < num_layers:
            self.layers.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=DROPOUT))
            layer_index = layer_index + 1

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        x = F.relu(x)

        layer_index = 0
        while layer_index < len(self.layers):
            residual = x
            x = self.layers[layer_index](x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + residual
            layer_index = layer_index + 1

        return x


class CrossAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, heads):
        super(CrossAttentionFusion, self).__init__()
        self.mol_to_pocket_attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.pocket_to_mol_attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)

    def forward(self, mol_x, pocket_x):
        mol_seq = mol_x.unsqueeze(0)
        pocket_seq = pocket_x.unsqueeze(0)

        mol_attended, _ = self.mol_to_pocket_attn(mol_seq, pocket_seq, pocket_seq)
        pocket_attended, _ = self.pocket_to_mol_attn(pocket_seq, mol_seq, mol_seq)

        mol_fused = mol_x + mol_attended.squeeze(0)
        pocket_fused = pocket_x + pocket_attended.squeeze(0)

        return mol_fused, pocket_fused


class RegressionHead(nn.Module):
    def __init__(self, hidden_dim):
        super(RegressionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, mol_pooled, pocket_pooled):
        combined = torch.cat([mol_pooled, pocket_pooled], dim=-1)
        output = self.mlp(combined)
        return output


class BindingAffinityModel(nn.Module):
    def __init__(self):
        super(BindingAffinityModel, self).__init__()
        self.mol_encoder = MoleculeEncoder(MOL_NODE_FEATURES, HIDDEN_DIM, NUM_MOL_LAYERS)
        self.pocket_encoder = PocketEncoder(POCKET_NODE_FEATURES, HIDDEN_DIM, NUM_POCKET_LAYERS, ATTENTION_HEADS)
        self.fusion = CrossAttentionFusion(HIDDEN_DIM, ATTENTION_HEADS)
        self.head = RegressionHead(HIDDEN_DIM)

    def forward(self, mol_x, mol_edge_index, pocket_x, pocket_edge_index):
        mol_embed = self.mol_encoder(mol_x, mol_edge_index)
        pocket_embed = self.pocket_encoder(pocket_x, pocket_edge_index)

        mol_fused, pocket_fused = self.fusion(mol_embed, pocket_embed)

        mol_pooled = mol_fused.mean(dim=0, keepdim=True)
        pocket_pooled = pocket_fused.mean(dim=0, keepdim=True)

        prediction = self.head(mol_pooled, pocket_pooled)
        return prediction.squeeze()


def test_forward_pass():
    import os
    graphs_dir = "graphs"
    files = os.listdir(graphs_dir)
    first_file = files[0]
    data = torch.load(os.path.join(graphs_dir, first_file))

    model = BindingAffinityModel()
    model.eval()

    with torch.no_grad():
        prediction = model(
            data["mol_x"], data["mol_edge_index"],
            data["pocket_x"], data["pocket_edge_index"]
        )

    print("Tested on complex:", data["pdb_id"])
    print("True label (pKd):", data["label"])
    print("Model prediction (untrained, random weights):", prediction.item())
    print("\nForward pass successful, no shape errors.")


if __name__ == "__main__":
    test_forward_pass()
