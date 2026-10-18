# ==============================================================================
# FINSHIELD GUARDIAN: FASTAPI BACKEND MICROSERVICE
# ==============================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from torch_geometric.nn import SAGEConv, HeteroConv

# --- NEW: Import our newly created Engines ---
from xai_engine import FraudExplainer
from recommendation_engine import PreventionRecommender

model = None
graph_data = None
xai_engine = None
recommender = None
OPTIMIZED_THRESHOLD = 0.9900
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, graph_data, xai_engine, recommender
    print("[INFO] Loading FinShield model and graph artifacts...")
    
    GRAPH_PATH = 'finshield_hetero_graph.pt'
    MODEL_PATH = 'finshield_gnn_best.pth'
    
    if not os.path.exists(GRAPH_PATH) or not os.path.exists(MODEL_PATH):
        raise RuntimeError("[ERROR] Required model or graph checkpoint missing in folder!")
        
    graph_data = torch.load(GRAPH_PATH, weights_only=False).to(device)
    
    NUM_NODES = graph_data['account'].num_nodes
    IN_EDGE_CHANNELS = graph_data['account', 'transacts', 'account'].edge_attr.size(1)
    
    model = FinShieldGraphSAGEOpt(num_nodes=NUM_NODES, in_edge_channels=IN_EDGE_CHANNELS, hidden_channels=32).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # --- NEW: Initialize XAI and Recommendation Modules ---
    print("[INFO] Initializing XAI and Recommendation Modules...")
    xai_engine = FraudExplainer()
    recommender = PreventionRecommender()
    
    print("[SUCCESS] FinShield Guardian API ready to serve predictions!")
    
    yield
    print("[INFO] Shutting down FinShield Guardian API...")

app = FastAPI(
    title="FinShield Guardian API",
    description="Real-Time Heterogeneous Graph Neural Network Fraud Detection Engine with XAI",
    version="1.1.0",
    lifespan=lifespan
)

class TransactionRequest(BaseModel):
    source_account_id: int
    dest_account_id: int
    edge_features: list[float]

# --- NEW: Request Model for XAI Analysis ---
class XAITransactionRequest(BaseModel):
    transaction_edge_index: int  # The specific transaction ID/Index from the dataset

@app.post("/predict-fraud")
def predict_fraud(payload: TransactionRequest):
    try:
        edge_index = torch.tensor([[payload.source_account_id], [payload.dest_account_id]], dtype=torch.long, device=device)
        edge_attr = torch.tensor([payload.edge_features], dtype=torch.float, device=device)
        
        edge_index_dict = graph_data.edge_index_dict
        
        with torch.no_grad():
            num_nodes = model.node_emb.num_embeddings
            account_ids = torch.arange(num_nodes, device=device)
            x_dict = {'account': model.node_emb(account_ids)}
            x_dict = model.conv1(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = model.conv2(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            
            temp_attr_dict = {('account', 'transacts', 'account'): edge_attr}
            out = model(edge_index_dict, temp_attr_dict, edge_index, x_dict)
            probs = F.softmax(out, dim=-1)[0]
            fraud_prob = probs[1].item()
            
        is_fraud = fraud_prob >= OPTIMIZED_THRESHOLD
        
        return {
            "status": "success",
            "fraud_probability": round(fraud_prob * 100, 4),
            "is_fraud_flagged": bool(is_fraud),
            "decision_threshold": OPTIMIZED_THRESHOLD,
            "risk_level": "CRITICAL" if fraud_prob > 0.99 else ("MODERATE" if fraud_prob > 0.5 else "LOW")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW: API Endpoint for Mobile App XAI & Recommendations ---
@app.post("/api/v1/analyze_transaction")
async def analyze_transaction(req: XAITransactionRequest):
    try:
        edge_type = ('account', 'transacts', 'account')
        
        # FIX: අදාළ Index එක පාවිච්චි කරලා ග්‍රාෆ් එකෙන් හරියටම Edge එකයි ඒකෙ Features ටිකයි වෙන් කරගැනීම
        target_edge = xai_engine.graph_data[edge_type].edge_index[:, req.transaction_edge_index : req.transaction_edge_index + 1]
        target_edge_attr = xai_engine.graph_data[edge_type].edge_attr[req.transaction_edge_index : req.transaction_edge_index + 1]
        
        # FIX: Run XAI Explainer passing BOTH target_edge and target_edge_attr
        exp_result, sub_nodes, sub_edges = xai_engine.explain_transaction(target_edge, target_edge_attr)
        
        if exp_result is None:
            raise HTTPException(status_code=500, detail="Failed to generate XAI explanation.")
            
        # Extract Risk Score and Top Factors
        mask_dict = exp_result.edge_mask_dict if hasattr(exp_result, 'edge_mask_dict') else exp_result.edge_mask
        
        # Using the XAI importance max as a proxy risk score for the recommendation engine
        risk_score = mask_dict[edge_type].max().item()
        
        # Simulate detected factors based on score
        detected_factors = []
        if risk_score > 0.75:
            detected_factors.append("fraud_ring_proximity")
            detected_factors.append("new_device")
            
        # Generate Personalized Recommendations
        final_response = recommender.generate_recommendations(risk_score, detected_factors)
        
        return final_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))