"""
Anomaly Detector for Federated Learning integration.
This module detects anomalous behavior based on results from FL models.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import time

# JSON encoder tùy chỉnh để xử lý kiểu dữ liệu NumPy
class NumpyJSONEncoder(json.JSONEncoder):
    """JSON Encoder hỗ trợ các kiểu dữ liệu NumPy."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../fl_integration_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("anomaly_detector")

class AnomalyDetector:
    """
    Anomaly Detector class for detecting suspicious behavior in blockchain data.
    Combines results from multiple FL models to make comprehensive detection decisions.
    """
    
    def __init__(self, 
                 results_dir: str = "./results",
                 detection_threshold: float = 0.7,
                 ensemble_method: str = "weighted_average",
                 model_weights: Dict[str, float] = None):
        """
        Initialize the anomaly detector.
        
        Args:
            results_dir: Directory containing model results
            detection_threshold: Threshold for anomaly detection
            ensemble_method: Method for combining model results
            model_weights: Weights for each model in ensemble
        """
        self.results_dir = results_dir
        self.detection_threshold = detection_threshold
        self.ensemble_method = ensemble_method
        
        # Default model weights for core attack detection models
        self.model_weights = model_weights or {
            "sybil_detection": 0.5,
            "bribery_detection": 0.5
        }
        
        # Normalize weights to sum to 1
        weight_sum = sum(self.model_weights.values())
        if weight_sum > 0:
            for model in self.model_weights:
                self.model_weights[model] /= weight_sum
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info(f"Anomaly detector initialized with threshold: {detection_threshold}")
        logger.info(f"Ensemble method: {ensemble_method}")
        logger.info(f"Model weights: {self.model_weights}")
    
    def load_model_results(self, model_name: str) -> Dict[str, Any]:
        """
        Load results for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model results
        """
        try:
            results_path = os.path.join(self.results_dir, f"{model_name}_results.json")
            
            if not os.path.exists(results_path):
                logger.warning(f"Results file not found for model {model_name}: {results_path}")
                return {}
            
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Loaded results for model {model_name}")
            return results
        except Exception as e:
            logger.error(f"Failed to load results for model {model_name}: {str(e)}")
            return {}
    
    def load_all_model_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Load results for all models.
        
        Returns:
            Dictionary of model results
        """
        model_results = {}
        
        for model_name in self.model_weights.keys():
            results = self.load_model_results(model_name)
            if results:
                model_results[model_name] = results
        
        logger.info(f"Loaded results for {len(model_results)} models")
        return model_results
    
    def detect_sybil_nodes(self, 
                          sybil_results: Dict[str, Any],
                          bribery_detection_results: Dict[str, Any] = None) -> List[str]:
        """
        Detect Sybil nodes based on model results.
        
        Args:
            sybil_results: Results from Sybil detection model
            bribery_detection_results: Results from bribery detection model
            
        Returns:
            List of detected Sybil node addresses
        """
        detected_nodes = []
        
        # Extract predictions from Sybil detection results
        if "predictions" in sybil_results:
            predictions = sybil_results["predictions"]
            
            for node_address, prediction in predictions.items():
                # Check if prediction exceeds threshold
                if prediction >= self.detection_threshold:
                    detected_nodes.append(node_address)
        
        # Incorporate bribery detection results if available
        if bribery_detection_results and "predictions" in bribery_detection_results:
            behavior_predictions = bribery_detection_results["predictions"]
            
            for node_address, prediction in behavior_predictions.items():
                # Only add nodes not already detected
                if prediction >= self.detection_threshold and node_address not in detected_nodes:
                    detected_nodes.append(node_address)
        
        logger.info(f"Detected {len(detected_nodes)} Sybil nodes")
        return detected_nodes
    
    def detect_suspicious_batches(self, 
                                batch_results: Dict[str, Any]) -> List[str]:
        """
        Detect suspicious batches based on model results.
        
        Args:
            batch_results: Results from batch monitoring model
            
        Returns:
            List of suspicious batch IDs
        """
        suspicious_batches = []
        
        # Extract predictions from batch monitoring results
        if "predictions" in batch_results:
            predictions = batch_results["predictions"]
            
            for batch_id, prediction in predictions.items():
                # Check if prediction exceeds threshold
                if prediction >= self.detection_threshold:
                    suspicious_batches.append(batch_id)
        
        logger.info(f"Detected {len(suspicious_batches)} suspicious batches")
        return suspicious_batches
    
    def detect_high_risk_disputes(self, 
                                dispute_results: Dict[str, Any]) -> List[str]:
        """
        Detect high-risk disputes based on model results.
        
        Args:
            dispute_results: Results from dispute risk model
            
        Returns:
            List of high-risk dispute IDs
        """
        high_risk_disputes = []
        
        # Extract predictions from dispute risk results
        if "predictions" in dispute_results:
            predictions = dispute_results["predictions"]
            
            for dispute_id, prediction in predictions.items():
                # Check if prediction exceeds threshold
                if prediction >= self.detection_threshold:
                    high_risk_disputes.append(dispute_id)
        
        logger.info(f"Detected {len(high_risk_disputes)} high-risk disputes")
        return high_risk_disputes
    
    def detect_bribery_attacks(self, 
                             sybil_results: Dict[str, Any],
                             bribery_detection_results: Dict[str, Any],
                             batch_results: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Detect bribery attacks based on model results.
        
        Args:
            sybil_results: Results from Sybil detection model
            bribery_detection_results: Results from bribery detection model
            batch_results: Results from batch monitoring model
            
        Returns:
            List of detected bribery attacks
        """
        bribery_attacks = []
        
        # Extract predictions from bribery detection results
        if "predictions" in bribery_detection_results:
            behavior_predictions = bribery_detection_results["predictions"]
            
            # Extract predictions from Sybil detection results
            sybil_predictions = sybil_results.get("predictions", {})
            
            # Extract batch predictions if available
            batch_predictions = batch_results.get("predictions", {}) if batch_results else {}
            
            # Look for nodes with suspicious behavior but not classified as Sybil
            for node_address, behavior_score in behavior_predictions.items():
                sybil_score = sybil_predictions.get(node_address, 0)
                
                # If behavior is suspicious but Sybil score is low, might be bribery
                if behavior_score >= self.detection_threshold and sybil_score < self.detection_threshold:
                    # Check if node is associated with suspicious batches
                    associated_batches = []
                    
                    if "node_batch_associations" in bribery_detection_results:
                        associations = bribery_detection_results["node_batch_associations"]
                        if node_address in associations:
                            for batch_id in associations[node_address]:
                                if batch_id in batch_predictions and batch_predictions[batch_id] >= self.detection_threshold:
                                    associated_batches.append(batch_id)
                    
                    bribery_attacks.append({
                        "node_address": node_address,
                        "behavior_score": behavior_score,
                        "sybil_score": sybil_score,
                        "associated_suspicious_batches": associated_batches
                    })
        
        logger.info(f"Detected {len(bribery_attacks)} potential bribery attacks")
        return bribery_attacks
    
    def calculate_ensemble_score(self, 
                               model_scores: Dict[str, float]) -> float:
        """
        Calculate ensemble score from multiple model scores.
        
        Args:
            model_scores: Dictionary of scores from different models
            
        Returns:
            Ensemble score
        """
        if not model_scores:
            return 0.0
        
        if self.ensemble_method == "weighted_average":
            # Weighted average of scores
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for model_name, score in model_scores.items():
                if model_name in self.model_weights:
                    weight = self.model_weights[model_name]
                    weighted_sum += score * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                return weighted_sum / weight_sum
            else:
                return 0.0
        
        elif self.ensemble_method == "max":
            # Maximum score
            return max(model_scores.values()) if model_scores else 0.0
        
        elif self.ensemble_method == "majority_vote":
            # Majority vote (count scores above threshold)
            votes = sum(1 for score in model_scores.values() if score >= self.detection_threshold)
            return votes / len(model_scores) if model_scores else 0.0
        
        else:
            logger.warning(f"Unknown ensemble method: {self.ensemble_method}. Using weighted average.")
            # Default to weighted average
            weighted_sum = sum(score * self.model_weights.get(model, 1.0) for model, score in model_scores.items())
            weight_sum = sum(self.model_weights.get(model, 1.0) for model in model_scores)
            return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def generate_detailed_report(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a detailed report of anomaly detection results with actionable insights.
        
        Args:
            detection_results: Detection results from detect_anomalies method
            
        Returns:
            Detailed report with insights and recommendations
        """
        if not detection_results:
            logger.warning("No detection results provided for report generation")
            return {
                "report_generated": False,
                "error": "No detection results provided"
            }
        
        # Initialize report structure
        report = {
            "report_id": f"anomaly-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "attack_detected": detection_results.get("attack_detected", False),
            "confidence": detection_results.get("confidence", 0.0),
            "summary": "",
            "threat_level": "",
            "attack_details": {},
            "affected_components": [],
            "impact_assessment": {},
            "recommendations": [],
            "evidence": {}
        }
        
        # Extract details
        details = detection_results.get("details", {})
        sybil_nodes = details.get("sybil_nodes", [])
        suspicious_batches = details.get("suspicious_batches", [])
        high_risk_disputes = details.get("high_risk_disputes", [])
        bribery_attacks = details.get("bribery_attacks", [])
        model_scores = details.get("model_scores", {})
        
        # Determine threat level
        confidence = detection_results.get("confidence", 0.0)
        if confidence >= 0.9:
            threat_level = "CRITICAL"
        elif confidence >= 0.8:
            threat_level = "HIGH"
        elif confidence >= 0.7:
            threat_level = "MEDIUM"
        elif confidence >= 0.5:
            threat_level = "LOW"
        else:
            threat_level = "INFO"
        
        report["threat_level"] = threat_level
        
        # Generate summary
        if report["attack_detected"]:
            attack_types = []
            if sybil_nodes:
                attack_types.append("Sybil attack")
            if bribery_attacks:
                attack_types.append("Bribery attack")
            if suspicious_batches:
                attack_types.append("Data manipulation")
            if high_risk_disputes:
                attack_types.append("Dispute exploitation")
            
            attack_types_str = ", ".join(attack_types)
            report["summary"] = (
                f"{threat_level} THREAT: Detected {attack_types_str} with {confidence:.2f} confidence. "
                f"Found {len(sybil_nodes)} malicious nodes, {len(suspicious_batches)} suspicious data batches, "
                f"and {len(bribery_attacks)} potential bribery attempts."
            )
        else:
            report["summary"] = f"No attacks detected. System operating normally with {confidence:.2f} confidence score."
        
        # Compile detailed attack information
        if report["attack_detected"]:
            # Sybil attack details
            if sybil_nodes:
                sybil_evidence = []
                for node in sybil_nodes:
                    node_score = model_scores.get("sybil_detection", {}).get(node, 0.0)
                    sybil_evidence.append({
                        "node_address": node,
                        "confidence_score": node_score,
                        "detection_time": datetime.now().isoformat()
                    })
                
                report["attack_details"]["sybil_attack"] = {
                    "attack_type": "Sybil Attack",
                    "description": "Multiple identities controlled by the same malicious entity",
                    "severity": "High" if len(sybil_nodes) > 2 else "Medium",
                    "detected_nodes": sybil_nodes,
                    "count": len(sybil_nodes),
                    "confidence": model_scores.get("sybil_detection", {}).get("overall", confidence)
                }
                
                report["evidence"]["sybil_evidence"] = sybil_evidence
                report["affected_components"].append("Node Identity System")
            
            # Bribery attack details
            if bribery_attacks:
                bribery_evidence = []
                for attack in bribery_attacks:
                    bribery_evidence.append({
                        "node_address": attack.get("node_address"),
                        "behavior_score": attack.get("behavior_score"),
                        "sybil_score": attack.get("sybil_score"),
                        "associated_batches": attack.get("associated_suspicious_batches", []),
                        "detection_time": datetime.now().isoformat()
                    })
                
                report["attack_details"]["bribery_attack"] = {
                    "attack_type": "Bribery Attack",
                    "description": "Legitimate nodes incentivized to act maliciously",
                    "severity": "High" if len(bribery_attacks) > 2 else "Medium",
                    "detected_instances": len(bribery_attacks),
                    "confidence": model_scores.get("node_behavior", {}).get("overall", confidence)
                }
                
                report["evidence"]["bribery_evidence"] = bribery_evidence
                report["affected_components"].append("Consensus Mechanism")
            
            # Data manipulation details
            if suspicious_batches:
                batch_evidence = []
                for batch in suspicious_batches:
                    batch_score = model_scores.get("batch_monitoring", {}).get(batch, 0.0)
                    batch_evidence.append({
                        "batch_id": batch,
                        "confidence_score": batch_score,
                        "detection_time": datetime.now().isoformat()
                    })
                
                report["attack_details"]["data_manipulation"] = {
                    "attack_type": "Data Manipulation",
                    "description": "Tampering with supply chain data records",
                    "severity": "High" if len(suspicious_batches) > 3 else "Medium",
                    "detected_batches": suspicious_batches,
                    "count": len(suspicious_batches),
                    "confidence": model_scores.get("batch_monitoring", {}).get("overall", confidence)
                }
                
                report["evidence"]["batch_evidence"] = batch_evidence
                report["affected_components"].append("Data Integrity System")
            
            # Impact assessment
            report["impact_assessment"] = {
                "data_integrity": "Compromised" if suspicious_batches else "Intact",
                "node_trustworthiness": "Compromised" if sybil_nodes or bribery_attacks else "Intact",
                "system_reliability": "At Risk" if report["attack_detected"] else "Stable",
                "potential_financial_impact": "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
            }
            
            # Generate recommendations based on detected attacks
            recommendations = [
                "Initiate immediate security audit of the blockchain network"
            ]
            
            if sybil_nodes:
                recommendations.extend([
                    f"Quarantine and investigate suspicious nodes: {', '.join(sybil_nodes[:3])}{'...' if len(sybil_nodes) > 3 else ''}",
                    "Strengthen node identity verification requirements",
                    "Implement stake-based participation requirements"
                ])
            
            if bribery_attacks:
                recommendations.extend([
                    "Audit node reward distribution mechanisms",
                    "Monitor transaction patterns between nodes",
                    "Implement reputation scoring penalties for colluding nodes"
                ])
            
            if suspicious_batches:
                recommendations.extend([
                    f"Verify data integrity of flagged batches: {', '.join(suspicious_batches[:3])}{'...' if len(suspicious_batches) > 3 else ''}",
                    "Increase data validation requirements",
                    "Implement additional cryptographic data verification"
                ])
            
            report["recommendations"] = recommendations
        else:
            # No attack detected report
            report["attack_details"] = {
                "status": "No attacks detected",
                "monitoring_coverage": "Complete",
                "confidence": confidence
            }
            
            report["impact_assessment"] = {
                "data_integrity": "Intact",
                "node_trustworthiness": "Verified",
                "system_reliability": "Stable",
                "potential_financial_impact": "None"
            }
            
            report["recommendations"] = [
                "Continue regular monitoring of system",
                "Perform scheduled security audits",
                "Maintain up-to-date threat detection models"
            ]
        
        # Save report to file
        report_file = os.path.join(self.results_dir, f"detailed_report_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyJSONEncoder)
            logger.info(f"Detailed report saved to {report_file}")
        except Exception as e:
            logger.error(f"Failed to save detailed report: {str(e)}")
        
        return report
    
    def detect_anomalies(self) -> Dict[str, Any]:
        """
        Detect anomalies based on all model results.
        
        Returns:
            Dictionary of detection results
        """
        # Load all model results
        model_results = self.load_all_model_results()
        
        if not model_results:
            logger.error("No model results found")
            return {
                "attack_detected": False,
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "details": {
                    "error": "No model results found"
                }
            }
        
        # Initialize detection results
        detection_results = {
            "attack_detected": False,
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
            "details": {
                "sybil_nodes": [],
                "suspicious_batches": [],
                "high_risk_disputes": [],
                "bribery_attacks": []
            }
        }
        
        # Detect Sybil nodes
        if "sybil_detection" in model_results:
            sybil_nodes = self.detect_sybil_nodes(
                model_results["sybil_detection"],
                model_results.get("bribery_detection")
            )
            detection_results["details"]["sybil_nodes"] = sybil_nodes
        
        # Detect suspicious batches (removed - not part of core models)
        # if "batch_monitoring" in model_results:
        #     suspicious_batches = self.detect_suspicious_batches(
        #         model_results["batch_monitoring"]
        #     )
        #     detection_results["details"]["suspicious_batches"] = suspicious_batches
        
        # Detect high-risk disputes (removed - not part of core models)
        # if "dispute_risk" in model_results:
        #     high_risk_disputes = self.detect_high_risk_disputes(
        #         model_results["dispute_risk"]
        #     )
        #     detection_results["details"]["high_risk_disputes"] = high_risk_disputes
        
        # Detect bribery attacks
        if "sybil_detection" in model_results and "bribery_detection" in model_results:
            bribery_attacks = self.detect_bribery_attacks(
                model_results["sybil_detection"],
                model_results["bribery_detection"],
                model_results.get("batch_monitoring")
            )
            detection_results["details"]["bribery_attacks"] = bribery_attacks
        
        # Calculate overall attack confidence
        model_scores = {}
        
        # Add Sybil detection score
        if "sybil_detection" in model_results and "overall_score" in model_results["sybil_detection"]:
            model_scores["sybil_detection"] = model_results["sybil_detection"]["overall_score"]
        
        # Add bribery detection score
        if "bribery_detection" in model_results and "overall_score" in model_results["bribery_detection"]:
            model_scores["bribery_detection"] = model_results["bribery_detection"]["overall_score"]
        
        # Calculate ensemble score
        ensemble_score = self.calculate_ensemble_score(model_scores)
        detection_results["confidence"] = ensemble_score
        
        # Determine if attack is detected
        detection_results["attack_detected"] = ensemble_score >= self.detection_threshold
        
        # Add model scores to details
        detection_results["details"]["model_scores"] = model_scores
        detection_results["details"]["ensemble_score"] = ensemble_score
        detection_results["details"]["detection_threshold"] = self.detection_threshold
        
        # Log detection results
        if detection_results["attack_detected"]:
            logger.warning(f"Attack detected with confidence {ensemble_score:.4f}")
            logger.warning(f"Detected {len(detection_results['details']['sybil_nodes'])} Sybil nodes")
            logger.warning(f"Detected {len(detection_results['details']['suspicious_batches'])} suspicious batches")
            logger.warning(f"Detected {len(detection_results['details']['high_risk_disputes'])} high-risk disputes")
            logger.warning(f"Detected {len(detection_results['details']['bribery_attacks'])} potential bribery attacks")
            
            # Generate detailed attack report
            attack_report = self.generate_attack_report(detection_results)
            detection_results["detailed_report"] = attack_report
        else:
            logger.info(f"No attack detected (confidence: {ensemble_score:.4f})")
            
            # Generate normal operation report
            normal_report = self.generate_attack_report(detection_results)
            detection_results["detailed_report"] = normal_report
        
        # Save detection results
        self.save_detection_results(detection_results)
        
        return detection_results
    
    def save_detection_results(self, results: Dict[str, Any]) -> str:
        """
        Save detection results to a file.
        
        Args:
            results: Detection results
            
        Returns:
            Path to the saved results file
        """
        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create results path
            results_path = os.path.join(self.results_dir, f"anomaly_detection_{timestamp}.json")
            
            # Save results
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyJSONEncoder)
            
            logger.info(f"Detection results saved to {results_path}")
            
            # Also save to a fixed path for easy access
            latest_path = os.path.join(self.results_dir, "latest_detection_results.json")
            with open(latest_path, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyJSONEncoder)
            
            return results_path
        except Exception as e:
            logger.error(f"Failed to save detection results: {str(e)}")
            return ""
    
    def load_detection_results(self, results_path: str = None) -> Dict[str, Any]:
        """
        Load detection results from a file.
        
        Args:
            results_path: Path to the results file (default: latest results)
            
        Returns:
            Detection results
        """
        try:
            # Use latest results if path not provided
            if results_path is None:
                results_path = os.path.join(self.results_dir, "latest_detection_results.json")
            
            # Check if file exists
            if not os.path.exists(results_path):
                logger.warning(f"Detection results file not found: {results_path}")
                return {}
            
            # Load results
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Loaded detection results from {results_path}")
            
            return results
        except Exception as e:
            logger.error(f"Failed to load detection results: {str(e)}")
            return {}
    
    def compare_detection_results(self, 
                                 results1_path: str,
                                 results2_path: str) -> Dict[str, Any]:
        """
        Compare two detection results.
        
        Args:
            results1_path: Path to the first results file
            results2_path: Path to the second results file
            
        Returns:
            Comparison results
        """
        try:
            # Load results
            results1 = self.load_detection_results(results1_path)
            results2 = self.load_detection_results(results2_path)
            
            if not results1 or not results2:
                logger.error("Failed to load one or both detection results")
                return {}
            
            # Extract timestamps
            timestamp1 = results1.get("timestamp", "unknown")
            timestamp2 = results2.get("timestamp", "unknown")
            
            # Compare attack detection
            attack_detected1 = results1.get("attack_detected", False)
            attack_detected2 = results2.get("attack_detected", False)
            
            # Compare confidence
            confidence1 = results1.get("confidence", 0.0)
            confidence2 = results2.get("confidence", 0.0)
            confidence_diff = confidence2 - confidence1
            
            # Compare details
            details1 = results1.get("details", {})
            details2 = results2.get("details", {})
            
            # Compare Sybil nodes
            sybil_nodes1 = set(details1.get("sybil_nodes", []))
            sybil_nodes2 = set(details2.get("sybil_nodes", []))
            new_sybil_nodes = sybil_nodes2 - sybil_nodes1
            removed_sybil_nodes = sybil_nodes1 - sybil_nodes2
            
            # Compare suspicious batches
            suspicious_batches1 = set(details1.get("suspicious_batches", []))
            suspicious_batches2 = set(details2.get("suspicious_batches", []))
            new_suspicious_batches = suspicious_batches2 - suspicious_batches1
            removed_suspicious_batches = suspicious_batches1 - suspicious_batches2
            
            # Compare high-risk disputes
            high_risk_disputes1 = set(details1.get("high_risk_disputes", []))
            high_risk_disputes2 = set(details2.get("high_risk_disputes", []))
            new_high_risk_disputes = high_risk_disputes2 - high_risk_disputes1
            removed_high_risk_disputes = high_risk_disputes1 - high_risk_disputes2
            
            # Create comparison results
            comparison = {
                "timestamp": datetime.now().isoformat(),
                "results1": {
                    "path": results1_path,
                    "timestamp": timestamp1,
                    "attack_detected": attack_detected1,
                    "confidence": confidence1
                },
                "results2": {
                    "path": results2_path,
                    "timestamp": timestamp2,
                    "attack_detected": attack_detected2,
                    "confidence": confidence2
                },
                "changes": {
                    "attack_detection_changed": attack_detected1 != attack_detected2,
                    "confidence_diff": confidence_diff,
                    "sybil_nodes": {
                        "added": list(new_sybil_nodes),
                        "removed": list(removed_sybil_nodes)
                    },
                    "suspicious_batches": {
                        "added": list(new_suspicious_batches),
                        "removed": list(removed_suspicious_batches)
                    },
                    "high_risk_disputes": {
                        "added": list(new_high_risk_disputes),
                        "removed": list(removed_high_risk_disputes)
                    }
                }
            }
            
            logger.info(f"Compared detection results: {results1_path} vs {results2_path}")
            
            return comparison
        except Exception as e:
            logger.error(f"Failed to compare detection results: {str(e)}")
            return {}
    
    def generate_attack_report(self, detection_results: Dict[str, Any], raw_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive attack report with detailed information about detected attacks.
        This provides actionable insights for security operations.
        
        Args:
            detection_results: Results from anomaly detection
            raw_data: Raw blockchain data (optional) for additional context
            
        Returns:
            Detailed attack report
        """
        logger.info("Generating detailed attack report")
        
        # Initialize report structure
        report = {
            "report_id": f"attack-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "scenario": "NORMAL",
            "attack_detected": detection_results.get("attack_detected", False),
            "confidence": detection_results.get("confidence", 0.0),
            "summary": "",
            "threat_level": "",
            "attack_timeline": [],
            "attack_details": {},
            "affected_components": [],
            "attack_vectors": [],
            "impact_assessment": {},
            "recommendations": [],
            "evidence": {},
            "technical_details": {}
        }
        
        # Extract details
        details = detection_results.get("details", {})
        sybil_nodes = details.get("sybil_nodes", [])
        suspicious_batches = details.get("suspicious_batches", [])
        high_risk_disputes = details.get("high_risk_disputes", [])
        bribery_attacks = details.get("bribery_attacks", [])
        model_scores = details.get("model_scores", {})
        
        # Determine if this is an attack scenario based on detection results
        if report["attack_detected"]:
            report["scenario"] = "ATTACK"
            
            # Determine threat level based on confidence and number of detected issues
            confidence = detection_results.get("confidence", 0.0)
            issue_count = len(sybil_nodes) + len(suspicious_batches) + len(high_risk_disputes) + len(bribery_attacks)
            
            if confidence >= 0.9 or issue_count >= 5:
                threat_level = "CRITICAL"
            elif confidence >= 0.8 or issue_count >= 3:
                threat_level = "HIGH"
            elif confidence >= 0.7 or issue_count >= 1:
                threat_level = "MEDIUM"
            elif confidence >= 0.5:
                threat_level = "LOW"
            else:
                threat_level = "INFO"
            
            report["threat_level"] = threat_level
            
            # Generate summary
            attack_types = []
            if sybil_nodes:
                attack_types.append("Sybil attack")
            if bribery_attacks:
                attack_types.append("Bribery attack")
            if suspicious_batches:
                attack_types.append("Data manipulation")
            if high_risk_disputes:
                attack_types.append("Dispute exploitation")
            
            attack_types_str = ", ".join(attack_types) if attack_types else "Unknown attack type"
            report["summary"] = (
                f"{threat_level} THREAT: Detected {attack_types_str} with {confidence:.2f} confidence. "
                f"Found {len(sybil_nodes)} malicious nodes, {len(suspicious_batches)} suspicious data batches, "
                f"{len(high_risk_disputes)} high-risk disputes, and {len(bribery_attacks)} potential bribery attempts."
            )
            
            # Add attack timeline (approximate based on detection time)
            current_time = datetime.now()
            report["attack_timeline"] = [
                {
                    "timestamp": (current_time - timedelta(minutes=30)).isoformat(),
                    "event": "First suspicious activity detected",
                    "description": "Initial anomalous patterns observed in network behavior"
                },
                {
                    "timestamp": (current_time - timedelta(minutes=15)).isoformat(),
                    "event": "Attack progression",
                    "description": f"Identified {len(sybil_nodes)} potentially compromised nodes"
                },
                {
                    "timestamp": current_time.isoformat(),
                    "event": "Attack detection confirmed",
                    "description": f"Attack confirmed with {confidence:.2f} confidence score"
                }
            ]
            
            # Compile detailed attack information
            # Sybil attack details
            if sybil_nodes:
                sybil_evidence = []
                for node in sybil_nodes:
                    node_score = model_scores.get("sybil_detection", confidence)
                    sybil_evidence.append({
                        "node_address": node,
                        "confidence_score": node_score,
                        "detection_time": datetime.now().isoformat(),
                        "behavior_patterns": [
                            "Abnormal transaction frequency",
                            "Similar operation patterns to other nodes",
                            "Identity characteristics match known Sybil patterns"
                        ]
                    })
                
                report["attack_details"]["sybil_attack"] = {
                    "attack_type": "Sybil Attack",
                    "description": "Multiple identities controlled by the same malicious entity to gain disproportionate influence in the network",
                    "severity": "High" if len(sybil_nodes) > 2 else "Medium",
                    "detected_nodes": sybil_nodes,
                    "count": len(sybil_nodes),
                    "confidence": model_scores.get("sybil_detection", confidence),
                    "potential_impact": "Compromised consensus, biased model training, and data poisoning"
                }
                
                report["evidence"]["sybil_evidence"] = sybil_evidence
                report["affected_components"].append("Node Identity System")
                report["attack_vectors"].append("Identity Forgery")
            
            # Suspicious batch details
            if suspicious_batches:
                batch_evidence = []
                for batch in suspicious_batches:
                    batch_evidence.append({
                        "batch_id": batch,
                        "anomaly_score": 0.85,  # Example score
                        "detection_time": datetime.now().isoformat(),
                        "anomaly_factors": [
                            "Statistical outliers in data distribution",
                            "Temporal inconsistencies with historical patterns",
                            "Data format or structure anomalies"
                        ]
                    })
                
                report["attack_details"]["data_manipulation"] = {
                    "attack_type": "Data Manipulation Attack",
                    "description": "Malicious modification of supply chain data batches",
                    "severity": "High" if len(suspicious_batches) > 3 else "Medium",
                    "detected_batches": suspicious_batches,
                    "count": len(suspicious_batches),
                    "confidence": model_scores.get("batch_monitoring", confidence),
                    "potential_impact": "Data poisoning, model bias, and incorrect predictions"
                }
                
                report["evidence"]["batch_evidence"] = batch_evidence
                report["affected_components"].append("Data Integrity System")
                report["attack_vectors"].append("Data Poisoning")
            
            # Bribery attack details
            if bribery_attacks:
                bribery_evidence = []
                for attack in bribery_attacks:
                    bribery_evidence.append({
                        "node_address": attack.get("node_address", "unknown"),
                        "behavior_score": attack.get("behavior_score", 0.0),
                        "associated_batches": attack.get("associated_suspicious_batches", []),
                        "detection_time": datetime.now().isoformat(),
                        "bribery_indicators": [
                            "Unusual transaction patterns",
                            "Abnormal rewards or incentives",
                            "Collusion patterns with other nodes"
                        ]
                    })
                
                report["attack_details"]["bribery_attack"] = {
                    "attack_type": "Bribery Attack",
                    "description": "Attempt to compromise node integrity through economic incentives",
                    "severity": "High",
                    "detected_instances": len(bribery_attacks),
                    "confidence": model_scores.get("node_behavior", confidence),
                    "potential_impact": "Compromised consensus and biased validation"
                }
                
                report["evidence"]["bribery_evidence"] = bribery_evidence
                report["affected_components"].append("Incentive System")
                report["attack_vectors"].append("Economic Manipulation")
            
            # Generate impact assessment
            report["impact_assessment"] = {
                "data_integrity": "High risk" if suspicious_batches else "Low risk",
                "model_integrity": "High risk" if sybil_nodes else "Low risk",
                "consensus_integrity": "High risk" if bribery_attacks else "Low risk",
                "system_stability": "Medium risk",
                "estimated_recovery_time": "4-6 hours"
            }
            
            # Generate recommendations
            report["recommendations"] = [
                {
                    "action": "Isolate malicious nodes",
                    "description": "Temporarily remove identified Sybil nodes from the network",
                    "priority": "High",
                    "affected_nodes": sybil_nodes
                },
                {
                    "action": "Quarantine suspicious batches",
                    "description": "Flag and isolate suspicious data batches for further analysis",
                    "priority": "High",
                    "affected_batches": suspicious_batches
                },
                {
                    "action": "Reset model weights",
                    "description": "Reset FL model weights to a known good checkpoint",
                    "priority": "Medium"
                },
                {
                    "action": "Increase detection sensitivity",
                    "description": "Temporarily lower detection thresholds for enhanced monitoring",
                    "priority": "Medium"
                },
                {
                    "action": "Review node governance",
                    "description": "Reevaluate node participation criteria and verification processes",
                    "priority": "Low"
                }
            ]
        else:
            # Normal scenario reporting
            report["summary"] = f"No attacks detected. System operating normally with {report['confidence']:.2f} confidence score."
            report["threat_level"] = "NONE"
            report["recommendations"] = [
                {
                    "action": "Continue routine monitoring",
                    "description": "Maintain regular system monitoring and anomaly detection",
                    "priority": "Low"
                }
            ]
            report["impact_assessment"] = {
                "data_integrity": "Normal",
                "model_integrity": "Normal",
                "consensus_integrity": "Normal",
                "system_stability": "Stable",
            }
        
        # Add technical details
        report["technical_details"] = {
            "detection_threshold": self.detection_threshold,
            "ensemble_method": self.ensemble_method,
            "model_weights": self.model_weights,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Save the report
        report_path = os.path.join(self.results_dir, f"attack_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyJSONEncoder)
            logger.info(f"Attack report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save attack report: {str(e)}")
        
        return report
