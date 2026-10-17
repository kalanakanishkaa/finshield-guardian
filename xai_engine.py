import torch
import torch.nn.functional as F
import os
import sys
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

# ==============================================================================
# FINSHIELD GUARDIAN: EXPLAINABLE AI (XAI) ENGINE - OPTIMIZED
# ==============================================================================

try:
    from main import FinShieldGraphSAGEOpt
except ImportError:
    print("[ERROR] Failed to import 'FinShieldGraphSAGEOpt' from main.py.")
    sys.exit(1)

# 1. XAI Wrapper Model 
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

    def explain_transaction(self, original_edge_index):
        print(f"\n[INFO] Extracting 2-hop Subgraph to prevent Memory Crash...")
        edge_type = ('account', 'transacts', 'account')
        
        full_edge_index = self.graph_data[edge_type].edge_index
        full_edge_attr = self.graph_data[edge_type].edge_attr
        
        nodes_of_interest = original_edge_index.view(-1).unique()
        
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=nodes_of_interest,
            num_hops=2,
            edge_index=full_edge_index,
            relabel_nodes=True,
            flow='source_to_target'
        )
        
        sub_edge_attr = full_edge_attr[edge_mask]
        print(f"[INFO] Subgraph extracted: {len(subset)} Nodes, {sub_edge_index.size(1)} Edges.")
        
        # FIX: Added .detach() to prevent PyTorch from computing gradients for the model embeddings
        sub_x_dict = {'account': self.base_model.node_emb(subset).detach()}
        sub_edge_index_dict = {edge_type: sub_edge_index}
        sub_edge_attr_dict = {edge_type: sub_edge_attr}
        
        target_edge = mapping.view(2, 1)

        print("[INFO] Running GNNExplainer on the optimized Subgraph...")
        try:
            # FIX: Removed the 'target' parameter to resolve the UserWarning
            explanation = self.explainer(
                x=sub_x_dict,
                edge_index=sub_edge_index_dict,
                edge_attr_dict=sub_edge_attr_dict,
                edge_label_index=target_edge
            )
            print("[SUCCESS] Explanation generated successfully.")
            return explanation, subset, sub_edge_index
            
        except Exception as e:
            print(f"[ERROR] Failed to generate explanation: {e}")
            return None, None, None

if __name__ == "__main__":
    try:
        xai = FraudExplainer()
        
        edge_type = ('account', 'transacts', 'account')
        target_edge = xai.graph_data[edge_type].edge_index[:, 0:1]
        
        exp_result, sub_nodes, sub_edges = xai.explain_transaction(target_edge)
        
        if exp_result is not None:
            print("\n==========================================")
            print("       XAI EXPLANATION RESULTS            ")
            print("==========================================")
            mask_dict = exp_result.edge_mask_dict if hasattr(exp_result, 'edge_mask_dict') else exp_result.edge_mask
            print(f"Total Edges Analyzed : {mask_dict[edge_type].size(0)}")
            print(f"Top Risk Importance  : {mask_dict[edge_type].max().item():.4f}")
            print("==========================================\n")
            
    except Exception as ex:
        print(f"\n[FATAL ERROR] {ex}")