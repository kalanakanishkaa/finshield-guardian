import json

# ==============================================================================
# FINSHIELD GUARDIAN: PREVENTION RECOMMENDATION ENGINE
# ==============================================================================

class PreventionRecommender:
    """
    Rule-based Recommendation Engine for FinShield Guardian.
    Maps XAI-derived risk factors to personalized prevention guidance.
    Designed according to Research Proposal Specifications (Table 2).
    """
    def __init__(self):
        # Define urgency levels based on probability thresholds
        self.risk_threshold_high = 0.75
        self.risk_threshold_medium = 0.40

    def generate_recommendations(self, risk_score, detected_factors=None):
        """
        Generates a list of actionable recommendations based on the risk score
        and specific factors identified by the GNNExplainer.
        """
        recommendations = []
        urgency = "LOW"

        if detected_factors is None:
            detected_factors = []

        # 1. Determine Urgency Level
        if risk_score >= self.risk_threshold_high:
            urgency = "HIGH"
        elif risk_score >= self.risk_threshold_medium:
            urgency = "MEDIUM"

        # 2. Apply Rule-based Mappings (From Proposal Table 2)
        if "new_device" in detected_factors or urgency == "HIGH":
            recommendations.append({
                "issue": "New/unrecognised device or High Risk",
                "action": "Enable multi-factor authentication (2FA) immediately."
            })
        
        if "fraud_ring_proximity" in detected_factors or risk_score > 0.8:
            recommendations.append({
                "issue": "Fraud-ring proximity detected",
                "action": "Change credentials and enable real-time alerts."
            })

        if "suspicious_merchant" in detected_factors or urgency == "MEDIUM":
            recommendations.append({
                "issue": "Suspicious merchant link",
                "action": "Verify merchant details before proceeding with transactions."
            })

        if "multi_location" in detected_factors:
            recommendations.append({
                "issue": "Multi-location anomaly",
                "action": "Review recent account activity for unauthorized access."
            })

        # 3. Default Safe Action for Elevated Risk
        if urgency == "MEDIUM" and not recommendations:
            recommendations.append({
                "issue": "Elevated risk level",
                "action": "Monitor account closely and review recent transactions."
            })
            
        if urgency == "LOW":
            recommendations.append({
                "issue": "No significant anomalies",
                "action": "Transaction is safe. No immediate action required."
            })

        return {
            "risk_score": round(risk_score, 4),
            "urgency_level": urgency,
            "recommendations": recommendations
        }

if __name__ == "__main__":
    # Initialize the Recommendation Engine
    recommender = PreventionRecommender()
    
    print("\n==========================================")
    print("   TESTING RECOMMENDATION ENGINE        ")
    print("==========================================")

    # TEST 1: Simulating our actual XAI output from earlier (Score: 0.1093)
    print("\n[TEST 1] Low Risk Scenario (Our previous XAI result):")
    sample_risk_score = 0.1093
    sample_factors = [] 
    result_low = recommender.generate_recommendations(sample_risk_score, sample_factors)
    print(json.dumps(result_low, indent=4))
    
    # TEST 2: Simulating a High Risk (Fraud) scenario with XAI factors
    print("\n[TEST 2] High Risk Scenario (Simulated):")
    high_risk_score = 0.8521
    high_risk_factors = ["fraud_ring_proximity", "new_device"]
    result_high = recommender.generate_recommendations(high_risk_score, high_risk_factors)
    print(json.dumps(result_high, indent=4))
    print("\n==========================================\n")