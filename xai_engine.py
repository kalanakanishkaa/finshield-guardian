import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import SAGEConv, HeteroConv

# ==============================================================================
# FINSHIELD GUARDIAN: EXPLAINABLE AI (XAI) ENGINE - STANDALONE & OPTIMIZED
# ==============================================================================

# Embedded model architecture to prevent circular import issues with main.py
class FinShieldGraphSAGEOpt(nn.Module):
    def __init__(self, num_nodes, in_edge_channels, hidden_channels):
        super(FinShieldGraphSAGEOpt, self).__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_channels)
        self.conv1 = HeteroConv({('account', 'transacts', 'account'): SAGEConv((-1, -1), hidden_channels)}, aggr='mean')
        self.conv2 = HeteroConv({('account', 'transacts', 'account'): SAGEConv((-1, -1), hidden_channels)}, aggr='mean')
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2 + in_edge_channels, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, edge_index_dict, edge_attr_dict, edge_label_index, x_dict):
        src_nodes = edge_label_index[0]
        dst_nodes = edge_label_index[1]
        h_src = x_dict['account'][src_nodes]
        h_dst = x_dict['account'][dst_nodes]
        edge_features = edge_attr_dict[('account', 'transacts', 'account')]
        edge_representations = torch.cat([h_src, h_dst, edge_features], dim=-1)
        return self.edge_mlp(edge_representations)

class XAIWrapperModel(torch.nn.Module):
    def __init__(self, base_model):
        super(XAIWrapperModel, self).__init__()
        self.base_model = base_model

    def forward(self, x, edge_index, edge_attr_dict, edge_label_index):
        out = self.base_model.conv1(x, edge_index)
        out = {k: F.relu(v) for k, v in out.items()}
        out = self.base_model.conv2(out, edge_index)
        out = {k: F.relu(v) for k, v in out.items()}
        return self.base_model(edge_index, edge_attr_dict, edge_label_index, out)

class FraudExplainer:
    def __init__(self, model_path="finshield_gnn_best.pth", graph_path="finshield_hetero_graph.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Initializing XAI Engine on {self.device}...")

        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"[ERROR] Graph dataset '{graph_path}' not found.")
        self.graph_data = torch.load(graph_path, weights_only=False).to(self.device)

        num_nodes = self.graph_data['account'].num_nodes
        in_edge_channels = self.graph_data['account', 'transacts', 'account'].edge_attr.size(1)
        
        self.base_model = FinShieldGraphSAGEOpt(
            num_nodes=num_nodes, 
            in_edge_channels=in_edge_channels, 
            hidden_channels=32
        ).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[ERROR] Model weights '{model_path}' not found.")
        self.base_model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model = XAIWrapperModel(self.base_model).to(self.device)
        self.model.eval()

        self.explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=150),
            explanation_type='model',
            edge_mask_type='object',
            node_mask_type='attributes',
            model_config=dict(
                mode='multiclass_classification',
                task_level='edge',
                return_type='raw', 
            ),
        )
        print("[SUCCESS] FinShield XAI Engine successfully initialized.")

    # --- අලුත් වෙනස: මෙතන original_edge_index සහ original_edge_attr කියන 2 ම ගන්නවා ---
    def explain_transaction(self, original_edge_index, original_edge_attr):
        edge_type = ('account', 'transacts', 'account')
        full_edge_index = self.graph_data[edge_type].edge_index
        
        nodes_of_interest = original_edge_index.view(-1).unique()
        
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=nodes_of_interest,
            num_hops=2,
            edge_index=full_edge_index,
            relabel_nodes=True,
            flow='source_to_target'
        )
        
        sub_x_dict = {'account': self.base_model.node_emb(subset).detach()}
        sub_edge_index_dict = {edge_type: sub_edge_index}
        
        # --- අලුත් වෙනස: target edge එකේ features විතරක් යවනවා ---
        sub_edge_attr_dict = {edge_type: original_edge_attr}
        
        target_edge = mapping.view(2, 1)

        try:
            explanation = self.explainer(
                x=sub_x_dict,
                edge_index=sub_edge_index_dict,
                edge_attr_dict=sub_edge_attr_dict,
                edge_label_index=target_edge
            )
            return explanation, subset, sub_edge_index
        except Exception as e:
            print(f"[ERROR] Failed to generate explanation: {e}")
            return None, None, None