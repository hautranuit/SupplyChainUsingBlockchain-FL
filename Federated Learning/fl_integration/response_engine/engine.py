"""
Response Engine for Federated Learning integration.
This module provides automated responses to detected anomalies.
"""

import os
import json
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import time
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
logger = logging.getLogger("response_engine")

class ResponseEngine:
    """
    Response Engine class for providing automated responses to detected anomalies.
    """
    
    def __init__(self, 
                 config_path: str = None,
                 suspicious_batch_threshold: int = 1,
                 high_risk_dispute_threshold: int = 1,
                 bribery_attack_threshold: int = 1,
                 arbitrator_bias_threshold: int = 1,
                 overall_confidence_threshold: float = 0.7,
                 notification_channels: List[str] = ["log"]):
        """
        Initialize the Response Engine.
        
        Args:
            config_path: Path to configuration file
            suspicious_batch_threshold: Threshold for number of suspicious batches
            high_risk_dispute_threshold: Threshold for number of high-risk disputes
            bribery_attack_threshold: Threshold for number of detected bribery attacks
            arbitrator_bias_threshold: Threshold for number of arbitrator bias incidents
            overall_confidence_threshold: Threshold for overall confidence
            notification_channels: List of notification channels
        """
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Extract response engine configuration
                if 'response_engine' in config:
                    re_config = config['response_engine']
                    suspicious_batch_threshold = re_config.get('suspicious_batch_threshold', suspicious_batch_threshold)
                    high_risk_dispute_threshold = re_config.get('high_risk_dispute_threshold', high_risk_dispute_threshold)
                    bribery_attack_threshold = re_config.get('bribery_attack_threshold', bribery_attack_threshold)
                    arbitrator_bias_threshold = re_config.get('arbitrator_bias_threshold', arbitrator_bias_threshold)
                    overall_confidence_threshold = re_config.get('overall_confidence_threshold', overall_confidence_threshold)
                
                # Extract notification channels
                if 'notification_channels' in config:
                    notification_channels = config['notification_channels']
            except Exception as e:
                logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
        
        self.suspicious_batch_threshold = suspicious_batch_threshold
        self.high_risk_dispute_threshold = high_risk_dispute_threshold
        self.bribery_attack_threshold = bribery_attack_threshold
        self.arbitrator_bias_threshold = arbitrator_bias_threshold
        self.overall_confidence_threshold = overall_confidence_threshold
        self.notification_channels = notification_channels
        
        # Actions configuration
        self.actions = {
            "log_alert": True,
            "notify_admin": False,
            "flag_node_on_chain": False,
            "flag_batch_on_chain": False
        }
        
        # Load actions from config if available
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                if 'response_engine' in config and 'actions' in config['response_engine']:
                    self.actions.update(config['response_engine']['actions'])
            except Exception as e:
                logger.error(f"Failed to load actions configuration: {str(e)}")
        
        logger.info(f"Response Engine initialized with thresholds: "
                   f"suspicious_batch={self.suspicious_batch_threshold}, "
                   f"high_risk_dispute={self.high_risk_dispute_threshold}, "
                   f"bribery_attack={self.bribery_attack_threshold}, "
                   f"arbitrator_bias={self.arbitrator_bias_threshold}, "
                   f"overall_confidence={self.overall_confidence_threshold}")
        logger.info(f"Notification channels: {self.notification_channels}")
        logger.info(f"Actions: {self.actions}")
    
    def evaluate_detection_results(self, 
                                  detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate detection results and determine if action is needed.
        
        Args:
            detection_results: Dictionary of detection results
            
        Returns:
            Dictionary with evaluation results
        """
        # Extract detection counts
        suspicious_batches_count = detection_results.get('suspicious_batches_count', 0)
        high_risk_disputes_count = detection_results.get('high_risk_disputes_count', 0)
        bribery_attacks_count = detection_results.get('bribery_attacks_count', 0)
        arbitrator_bias_count = detection_results.get('arbitrator_bias_count', 0)
        
        # Extract confidence scores
        batch_confidence = detection_results.get('batch_confidence', 0.0)
        dispute_confidence = detection_results.get('dispute_confidence', 0.0)
        bribery_confidence = detection_results.get('bribery_confidence', 0.0)
        arbitrator_confidence = detection_results.get('arbitrator_confidence', 0.0)
        
        # Calculate overall confidence
        confidence_values = []
        if batch_confidence > 0:
            confidence_values.append(batch_confidence)
        if dispute_confidence > 0:
            confidence_values.append(dispute_confidence)
        if bribery_confidence > 0:
            confidence_values.append(bribery_confidence)
        if arbitrator_confidence > 0:
            confidence_values.append(arbitrator_confidence)
        
        overall_confidence = np.mean(confidence_values) if confidence_values else 0.0
        
        # Determine if thresholds are exceeded
        batch_threshold_exceeded = suspicious_batches_count >= self.suspicious_batch_threshold
        dispute_threshold_exceeded = high_risk_disputes_count >= self.high_risk_dispute_threshold
        bribery_threshold_exceeded = bribery_attacks_count >= self.bribery_attack_threshold
        arbitrator_threshold_exceeded = arbitrator_bias_count >= self.arbitrator_bias_threshold
        confidence_threshold_exceeded = overall_confidence >= self.overall_confidence_threshold
        
        # Determine if action is needed
        action_needed = (
            (batch_threshold_exceeded or 
             dispute_threshold_exceeded or 
             bribery_threshold_exceeded or
             arbitrator_threshold_exceeded) and 
            confidence_threshold_exceeded
        )
        
        # Determine attack type
        attack_types = []
        if bribery_threshold_exceeded:
            attack_types.append("bribery")
        if batch_threshold_exceeded:
            attack_types.append("suspicious_batch")
        if dispute_threshold_exceeded:
            attack_types.append("high_risk_dispute")
        if arbitrator_threshold_exceeded:
            attack_types.append("arbitrator_bias")
        
        # Create evaluation results
        evaluation_results = {
            "action_needed": action_needed,
            "overall_confidence": float(overall_confidence),
            "confidence_threshold_exceeded": confidence_threshold_exceeded,
            "attack_detected": len(attack_types) > 0,
            "attack_types": attack_types,
            "thresholds_exceeded": {
                "batch": batch_threshold_exceeded,
                "dispute": dispute_threshold_exceeded,
                "bribery": bribery_threshold_exceeded,
                "arbitrator": arbitrator_threshold_exceeded
            },
            "counts": {
                "suspicious_batches": suspicious_batches_count,
                "high_risk_disputes": high_risk_disputes_count,
                "bribery_attacks": bribery_attacks_count,
                "arbitrator_bias": arbitrator_bias_count
            },
            "confidence_scores": {
                "batch": float(batch_confidence),
                "dispute": float(dispute_confidence),
                "bribery": float(bribery_confidence),
                "arbitrator": float(arbitrator_confidence)
            }
        }
        
        return evaluation_results
    
    def generate_alert_message(self, 
                              evaluation_results: Dict[str, Any]) -> str:
        """
        Generate alert message based on evaluation results.
        
        Args:
            evaluation_results: Dictionary with evaluation results
            
        Returns:
            Alert message
        """
        # Extract information from evaluation results
        action_needed = evaluation_results.get('action_needed', False)
        overall_confidence = evaluation_results.get('overall_confidence', 0.0)
        attack_detected = evaluation_results.get('attack_detected', False)
        attack_types = evaluation_results.get('attack_types', [])
        counts = evaluation_results.get('counts', {})
        confidence_scores = evaluation_results.get('confidence_scores', {})
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate alert message
        if action_needed and attack_detected:
            # Critical alert - action needed
            alert_level = "CRITICAL"
            attack_type_str = ", ".join(attack_types)
            
            message = f"[{alert_level}] ATTACK DETECTED - {timestamp}\n\n"
            message += f"Attack Types: {attack_type_str}\n"
            message += f"Overall Confidence: {overall_confidence:.4f}\n\n"
            
            message += "Detection Details:\n"
            
            if "bribery" in attack_types:
                message += f"- Bribery Attacks: {counts.get('bribery_attacks', 0)} detected "
                message += f"(Confidence: {confidence_scores.get('bribery', 0.0):.4f})\n"
            
            if "suspicious_batch" in attack_types:
                message += f"- Suspicious Batches: {counts.get('suspicious_batches', 0)} detected "
                message += f"(Confidence: {confidence_scores.get('batch', 0.0):.4f})\n"
            
            if "high_risk_dispute" in attack_types:
                message += f"- High-Risk Disputes: {counts.get('high_risk_disputes', 0)} detected "
                message += f"(Confidence: {confidence_scores.get('dispute', 0.0):.4f})\n"
                
            if "arbitrator_bias" in attack_types:
                message += f"- Arbitrator Bias: {counts.get('arbitrator_bias', 0)} detected "
                message += f"(Confidence: {confidence_scores.get('arbitrator', 0.0):.4f})\n"
            
            message += "\nACTION REQUIRED: Please investigate and respond to this security incident."
        
        elif attack_detected:
            # Warning alert - attack detected but action not needed
            alert_level = "WARNING"
            attack_type_str = ", ".join(attack_types)
            
            message = f"[{alert_level}] POTENTIAL ATTACK DETECTED - {timestamp}\n\n"
            message += f"Attack Types: {attack_type_str}\n"
            message += f"Overall Confidence: {overall_confidence:.4f}\n\n"
            
            message += "Detection Details:\n"
            
            if "bribery" in attack_types:
                message += f"- Bribery Attacks: {counts.get('bribery_attacks', 0)} detected "
                message += f"(Confidence: {confidence_scores.get('bribery', 0.0):.4f})\n"
            
            if "suspicious_batch" in attack_types:
                message += f"- Suspicious Batches: {counts.get('suspicious_batches', 0)} detected "
                message += f"(Confidence: {confidence_scores.get('batch', 0.0):.4f})\n"
            
            if "high_risk_dispute" in attack_types:
                message += f"- High-Risk Disputes: {counts.get('high_risk_disputes', 0)} detected "
                message += f"(Confidence: {confidence_scores.get('dispute', 0.0):.4f})\n"
                
            if "arbitrator_bias" in attack_types:
                message += f"- Arbitrator Bias: {counts.get('arbitrator_bias', 0)} detected "
                message += f"(Confidence: {confidence_scores.get('arbitrator', 0.0):.4f})\n"
            
            message += "\nNOTE: Confidence level below threshold for automatic response."
        
        else:
            # Informational alert - no attack detected
            alert_level = "INFO"
            
            message = f"[{alert_level}] SECURITY SCAN COMPLETED - {timestamp}\n\n"
            message += f"No attacks detected\n"
            message += f"Overall Confidence: {overall_confidence:.4f}\n\n"
            
            message += "Scan Details:\n"
            message += f"- Bribery Attacks: {counts.get('bribery_attacks', 0)} detected "
            message += f"(Confidence: {confidence_scores.get('bribery', 0.0):.4f})\n"
            message += f"- Suspicious Batches: {counts.get('suspicious_batches', 0)} detected "
            message += f"(Confidence: {confidence_scores.get('batch', 0.0):.4f})\n"
            message += f"- High-Risk Disputes: {counts.get('high_risk_disputes', 0)} detected "
            message += f"(Confidence: {confidence_scores.get('dispute', 0.0):.4f})\n"
            message += f"- Arbitrator Bias: {counts.get('arbitrator_bias', 0)} detected "
            message += f"(Confidence: {confidence_scores.get('arbitrator', 0.0):.4f})\n"
        
        return message
    
    def send_notification(self, 
                         message: str, 
                         channels: List[str] = None) -> Dict[str, bool]:
        """
        Send notification through specified channels.
        
        Args:
            message: Notification message
            channels: List of notification channels
            
        Returns:
            Dictionary with notification results
        """
        if channels is None:
            channels = self.notification_channels
        
        results = {}
        
        for channel in channels:
            if channel == "log":
                # Log to file
                logger.warning(message)
                results["log"] = True
            
            elif channel == "email":
                # Send email notification
                try:
                    # Email configuration (should be loaded from config in production)
                    smtp_server = "smtp.example.com"
                    smtp_port = 587
                    smtp_username = "alerts@example.com"
                    smtp_password = "password"
                    from_email = "alerts@example.com"
                    to_email = "admin@example.com"
                    subject = "Supply Chain Security Alert"
                    
                    # Create message
                    email_message = MIMEMultipart()
                    email_message["From"] = from_email
                    email_message["To"] = to_email
                    email_message["Subject"] = subject
                    
                    # Attach message
                    email_message.attach(MIMEText(message, "plain"))
                    
                    # Send email
                    with smtplib.SMTP(smtp_server, smtp_port) as server:
                        server.starttls()
                        server.login(smtp_username, smtp_password)
                        server.send_message(email_message)
                    
                    logger.info(f"Email notification sent to {to_email}")
                    results["email"] = True
                except Exception as e:
                    logger.error(f"Failed to send email notification: {str(e)}")
                    results["email"] = False
            
            elif channel == "webhook":
                # Send webhook notification
                try:
                    # Webhook configuration (should be loaded from config in production)
                    webhook_url = "https://example.com/webhook"
                    
                    # Prepare payload
                    payload = {
                        "message": message,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Send webhook
                    response = requests.post(webhook_url, json=payload)
                    
                    if response.status_code == 200:
                        logger.info(f"Webhook notification sent to {webhook_url}")
                        results["webhook"] = True
                    else:
                        logger.error(f"Failed to send webhook notification: {response.status_code}")
                        results["webhook"] = False
                except Exception as e:
                    logger.error(f"Failed to send webhook notification: {str(e)}")
                    results["webhook"] = False
            
            else:
                logger.warning(f"Unknown notification channel: {channel}")
                results[channel] = False
        
        return results
    
    def flag_node_on_blockchain(self, 
                               node_id: str, 
                               reason: str,
                               confidence: float,
                               blockchain_connector: Any = None) -> bool:
        """
        Flag a node on the blockchain.
        
        Args:
            node_id: ID of the node to flag
            reason: Reason for flagging
            confidence: Confidence score
            blockchain_connector: Blockchain connector instance
            
        Returns:
            True if successful, False otherwise
        """
        if not self.actions.get("flag_node_on_chain", False):
            logger.info(f"Node flagging on blockchain is disabled")
            return False
        
        if blockchain_connector is None:
            logger.error(f"Blockchain connector not provided")
            return False
        
        try:
            # Flag node on blockchain
            # This is a placeholder - actual implementation would depend on the blockchain connector
            # blockchain_connector.flag_node(node_id, reason, confidence)
            
            logger.info(f"Node {node_id} flagged on blockchain: {reason} (Confidence: {confidence:.4f})")
            return True
        except Exception as e:
            logger.error(f"Failed to flag node {node_id} on blockchain: {str(e)}")
            return False
    
    def flag_batch_on_blockchain(self, 
                                batch_id: str, 
                                reason: str,
                                confidence: float,
                                blockchain_connector: Any = None) -> bool:
        """
        Flag a batch on the blockchain.
        
        Args:
            batch_id: ID of the batch to flag
            reason: Reason for flagging
            confidence: Confidence score
            blockchain_connector: Blockchain connector instance
            
        Returns:
            True if successful, False otherwise
        """
        if not self.actions.get("flag_batch_on_chain", False):
            logger.info(f"Batch flagging on blockchain is disabled")
            return False
        
        if blockchain_connector is None:
            logger.error(f"Blockchain connector not provided")
            return False
        
        try:
            # Flag batch on blockchain
            # This is a placeholder - actual implementation would depend on the blockchain connector
            # blockchain_connector.flag_batch(batch_id, reason, confidence)
            
            logger.info(f"Batch {batch_id} flagged on blockchain: {reason} (Confidence: {confidence:.4f})")
            return True
        except Exception as e:
            logger.error(f"Failed to flag batch {batch_id} on blockchain: {str(e)}")
            return False
    
    def respond_to_detection(self, 
                            detection_results: Dict[str, Any],
                            blockchain_connector: Any = None) -> Dict[str, Any]:
        """
        Respond to detection results.
        
        Args:
            detection_results: Dictionary of detection results
            blockchain_connector: Blockchain connector instance
            
        Returns:
            Dictionary with response results
        """
        # Evaluate detection results
        evaluation_results = self.evaluate_detection_results(detection_results)
        
        # Generate alert message
        alert_message = self.generate_alert_message(evaluation_results)
        
        # Initialize response results
        response_results = {
            "evaluation": evaluation_results,
            "alert_message": alert_message,
            "notifications": {},
            "blockchain_actions": {
                "nodes_flagged": [],
                "batches_flagged": []
            }
        }
        
        # Send notifications if action needed or attack detected
        if evaluation_results.get('action_needed', False) or evaluation_results.get('attack_detected', False):
            # Determine notification channels based on severity
            if evaluation_results.get('action_needed', False):
                # Critical - use all channels
                channels = self.notification_channels
            else:
                # Warning - use log channel only
                channels = ["log"]
            
            # Send notifications
            notification_results = self.send_notification(alert_message, channels)
            response_results["notifications"] = notification_results
        
        # Take blockchain actions if action needed
        if evaluation_results.get('action_needed', False) and blockchain_connector is not None:
            # Flag suspicious batches
            if evaluation_results.get('thresholds_exceeded', {}).get('batch', False):
                suspicious_batches = detection_results.get('suspicious_batches', [])
                batch_confidence = evaluation_results.get('confidence_scores', {}).get('batch', 0.0)
                
                for batch in suspicious_batches:
                    batch_id = batch.get('id', '')
                    if batch_id:
                        success = self.flag_batch_on_blockchain(
                            batch_id=batch_id,
                            reason="Detected as suspicious batch",
                            confidence=batch_confidence,
                            blockchain_connector=blockchain_connector
                        )
                        
                        if success:
                            response_results["blockchain_actions"]["batches_flagged"].append(batch_id)
        
        # Log response summary
        logger.info(f"Response completed: "
                   f"action_needed={evaluation_results.get('action_needed', False)}, "
                   f"attack_detected={evaluation_results.get('attack_detected', False)}, "
                   f"notifications={list(response_results['notifications'].keys())}, "
                   f"nodes_flagged={len(response_results['blockchain_actions']['nodes_flagged'])}, "
                   f"batches_flagged={len(response_results['blockchain_actions']['batches_flagged'])}")
        
        return response_results
    
    def save_response_results(self, 
                             response_results: Dict[str, Any],
                             output_dir: str = "./results") -> str:
        """
        Save response results to file.
        
        Args:
            response_results: Dictionary with response results
            output_dir: Output directory
            
        Returns:
            Path to the saved file
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"response_results_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(response_results, f, indent=2, cls=NumpyJSONEncoder)
            
            logger.info(f"Response results saved to {filepath}")
            
            return filepath
        except Exception as e:
            logger.error(f"Failed to save response results: {str(e)}")
            return ""
