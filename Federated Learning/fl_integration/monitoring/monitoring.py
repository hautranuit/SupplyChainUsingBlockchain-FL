"""
Monitoring and Logging system for Federated Learning integration.
This module provides comprehensive logging, metrics collection, and monitoring
capabilities for the FL system.
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import threading
import traceback
import numpy as np

# Custom JSON encoder to handle NumPy data types
class NumpyJSONEncoder(json.JSONEncoder):
    """JSON Encoder supporting NumPy data types."""
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
logger = logging.getLogger("monitoring")

class MonitoringSystem:
    """
    Monitoring System class for FL integration.
    Provides logging, metrics collection, and monitoring capabilities.
    """
    
    def __init__(self, 
                 log_dir: str = "./logs",
                 metrics_dir: str = "./metrics",
                 log_level: str = "INFO",
                 enable_performance_monitoring: bool = True,
                 enable_health_checks: bool = True,
                 check_interval: int = 60):  # seconds
        """
        Initialize the monitoring system.
        
        Args:
            log_dir: Directory for log files
            metrics_dir: Directory for metrics files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_performance_monitoring: Whether to enable performance monitoring
            enable_health_checks: Whether to enable periodic health checks
            check_interval: Interval for health checks in seconds
        """
        self.log_dir = log_dir
        self.metrics_dir = metrics_dir
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_health_checks = enable_health_checks
        self.check_interval = check_interval
        self.is_running = True  # Add is_running attribute
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Configure logger
        self.configure_logger()
        
        # Initialize metrics
        self.metrics = {
            "system": {
                "start_time": datetime.now().isoformat(),
                "component_status": {},
                "errors": []
            },
            "performance": {
                "blockchain_connector": {
                    "requests": 0,
                    "errors": 0,
                    "avg_response_time": 0
                },
                "fl_orchestrator": {
                    "training_rounds": 0,
                    "clients_participated": 0,
                    "avg_training_time": 0
                },
                "anomaly_detector": {
                    "detections": 0,
                    "false_positives": 0,
                    "false_negatives": 0
                }
            },
            "events": []
        }
        
        # Health check thread
        self.health_check_thread = None
        self.stop_health_checks_event = threading.Event()  # Renamed to avoid conflict
        
        if self.enable_health_checks:
            self.start_health_checks()
        
        logger.info(f"Monitoring system initialized with log directory: {log_dir}")
        logger.info(f"Metrics directory: {metrics_dir}")
        logger.info(f"Performance monitoring enabled: {enable_performance_monitoring}")
        logger.info(f"Health checks enabled: {enable_health_checks}")
    
    def configure_logger(self):
        """Configure the logger with appropriate settings."""
        # Create a file handler for the main log file
        main_log_file = os.path.join(self.log_dir, "fl_system.log")
        file_handler = logging.FileHandler(main_log_file)
        file_handler.setLevel(self.log_level)
        
        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(file_handler)
        
        # Set the logger level
        logger.setLevel(self.log_level)
    
    def log_event(self, 
                 component: str, 
                 event_type: str, 
                 message: str, 
                 details: Dict[str, Any] = None,
                 level: str = "INFO"):
        """
        Log an event.
        
        Args:
            component: Component that generated the event
            event_type: Type of event
            message: Event message
            details: Additional event details
            level: Logging level
        """
        # Get the logging level
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # Log the message
        logger.log(log_level, f"[{component}] {message}")
        
        # Add to events list
        event = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "type": event_type,
            "message": message,
            "level": level,
            "details": details or {}
        }
        
        self.metrics["events"].append(event)
        
        # If it's an error, add to errors list
        if level.upper() in ["ERROR", "CRITICAL"]:
            self.metrics["system"]["errors"].append(event)
        
        # Save metrics if performance monitoring is enabled
        if self.enable_performance_monitoring:
            self.save_metrics()
    
    def update_component_status(self, 
                              component: str, 
                              status: str, 
                              details: Dict[str, Any] = None):
        """
        Update the status of a component.
        
        Args:
            component: Component name
            status: Status string (e.g., "healthy", "degraded", "failed")
            details: Additional status details
        """
        self.metrics["system"]["component_status"][component] = {
            "status": status,
            "last_updated": datetime.now().isoformat(),
            "details": details or {}
        }
        
        logger.info(f"Component {component} status updated to {status}")
        
        # Save metrics if performance monitoring is enabled
        if self.enable_performance_monitoring:
            self.save_metrics()
    
    def record_performance_metric(self, 
                                component: str, 
                                metric: str, 
                                value: Any):
        """
        Record a performance metric.
        
        Args:
            component: Component name
            metric: Metric name
            value: Metric value
        """
        if not self.enable_performance_monitoring:
            return
        
        # Ensure component exists in performance metrics
        if component not in self.metrics["performance"]:
            self.metrics["performance"][component] = {}
        
        # Update metric
        self.metrics["performance"][component][metric] = value
        
        logger.debug(f"Performance metric recorded: {component}.{metric} = {value}")
        
        # Save metrics
        self.save_metrics()
    
    def start_timer(self, component: str, operation: str) -> int:
        """
        Start a timer for performance measurement.
        
        Args:
            component: Component name
            operation: Operation name
            
        Returns:
            Timer ID
        """
        if not self.enable_performance_monitoring:
            return -1
        
        timer_id = int(time.time() * 1000)  # Use timestamp as ID
        
        # Create timer entry
        timer_entry = {
            "component": component,
            "operation": operation,
            "start_time": time.time(),
            "end_time": None,
            "duration": None
        }
        
        # Store timer entry
        if "timers" not in self.metrics:
            self.metrics["timers"] = {}
        
        self.metrics["timers"][timer_id] = timer_entry
        
        return timer_id
    
    def stop_timer(self, timer_id: int, success: bool = True) -> float:
        """
        Stop a timer and record the duration.
        
        Args:
            timer_id: Timer ID from start_timer
            success: Whether the operation was successful (optional)
            
        Returns:
            Duration in seconds
        """
        if not self.enable_performance_monitoring or timer_id == -1:
            return 0.0
        
        if "timers" not in self.metrics or timer_id not in self.metrics["timers"]:
            logger.warning(f"Timer {timer_id} not found")
            return 0.0
        
        # Get timer entry
        timer_entry = self.metrics["timers"][timer_id]
        
        # Calculate duration
        end_time = time.time()
        duration = end_time - timer_entry["start_time"]
        
        # Update timer entry
        timer_entry["end_time"] = end_time
        timer_entry["duration"] = duration
        timer_entry["success"] = success  # Store success status
        
        # Record as performance metric
        component = timer_entry["component"]
        operation = timer_entry["operation"]
        
        # Create metric name
        metric_name = f"{operation}_time"
        
        # Record metric
        self.record_performance_metric(component, metric_name, duration)
        
        # Record success/failure metrics
        success_metric_name = f"{operation}_success_count" if success else f"{operation}_failure_count"
        
        if component in self.metrics["performance"]:
            component_metrics = self.metrics["performance"][component]
            
            if success_metric_name not in component_metrics:
                component_metrics[success_metric_name] = 0
            
            component_metrics[success_metric_name] += 1
        
        # Also update average time if applicable
        avg_metric_name = f"avg_{operation}_time"
        
        if component in self.metrics["performance"]:
            component_metrics = self.metrics["performance"][component]
            
            if avg_metric_name in component_metrics:
                # Update running average
                count_metric_name = f"{operation}_count"
                
                if count_metric_name not in component_metrics:
                    component_metrics[count_metric_name] = 0
                
                count = component_metrics[count_metric_name] + 1
                component_metrics[count_metric_name] = count
                
                # Calculate new average
                old_avg = component_metrics[avg_metric_name]
                new_avg = ((old_avg * (count - 1)) + duration) / count
                
                # Update average
                self.record_performance_metric(component, avg_metric_name, new_avg)
            else:
                # First time, just set the value
                self.record_performance_metric(component, avg_metric_name, duration)
                self.record_performance_metric(component, f"{operation}_count", 1)
        
        return duration
    
    def save_metrics(self):
        """Save metrics to a file."""
        try:
            # Create metrics file path
            metrics_file = os.path.join(self.metrics_dir, "fl_metrics.json")
            
            # Save metrics
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2, cls=NumpyJSONEncoder)
            
            # Also save a timestamped version periodically
            current_hour = datetime.now().strftime("%Y%m%d_%H")
            timestamped_file = os.path.join(self.metrics_dir, f"fl_metrics_{current_hour}.json")
            
            # Only save if file doesn't exist yet for this hour
            if not os.path.exists(timestamped_file):
                with open(timestamped_file, 'w') as f:
                    json.dump(self.metrics, f, indent=2, cls=NumpyJSONEncoder)
        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")
    
    def load_metrics(self) -> Dict[str, Any]:
        """
        Load metrics from file.
        
        Returns:
            Metrics dictionary
        """
        try:
            # Create metrics file path
            metrics_file = os.path.join(self.metrics_dir, "fl_metrics.json")
            
            # Check if file exists
            if not os.path.exists(metrics_file):
                logger.warning(f"Metrics file not found: {metrics_file}")
                return {}
            
            # Load metrics
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            return metrics
        except Exception as e:
            logger.error(f"Failed to load metrics: {str(e)}")
            return {}
    
    def health_check(self):
        """Perform a health check on all components."""
        logger.info("Performing health check")
        
        try:
            # Check each component
            components = [
                "blockchain_connector",
                "data_processor",
                "fl_orchestrator",
                "model_repository",
                "anomaly_detector",
                "response_engine"
            ]
            
            for component in components:
                # Placeholder for actual health check logic
                # In a real implementation, this would check the actual status of each component
                
                # For now, just mark as healthy if it exists in component_status
                if component in self.metrics["system"]["component_status"]:
                    status = self.metrics["system"]["component_status"][component]["status"]
                    
                    # If status is not "failed", keep it as is
                    if status != "failed":
                        continue
                
                # Default to healthy
                self.update_component_status(component, "healthy", {
                    "last_check": datetime.now().isoformat()
                })
            
            # Check for errors
            error_count = len(self.metrics["system"]["errors"])
            if error_count > 0:
                logger.warning(f"System has {error_count} errors")
            
            # Update system uptime
            start_time = datetime.fromisoformat(self.metrics["system"]["start_time"])
            uptime_seconds = (datetime.now() - start_time).total_seconds()
            
            self.metrics["system"]["uptime_seconds"] = uptime_seconds
            
            # Save metrics
            self.save_metrics()
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            logger.error(traceback.format_exc())
    
    def health_check_loop(self):
        """Run health checks in a loop."""
        while not self.stop_health_checks_event.is_set():
            try:
                self.health_check()
            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
            
            # Wait for next check
            self.stop_health_checks_event.wait(self.check_interval)
    
    def start_health_checks(self):
        """Start the health check thread."""
        if not self.enable_health_checks:
            return
        
        if self.health_check_thread is not None and self.health_check_thread.is_alive():
            logger.warning("Health check thread already running")
            return
        
        # Reset stop event
        self.stop_health_checks_event.clear()
        
        # Create and start thread
        self.health_check_thread = threading.Thread(target=self.health_check_loop)
        self.health_check_thread.daemon = True
        self.health_check_thread.start()
        
        logger.info(f"Health check thread started with interval {self.check_interval} seconds")
    
    def stop_health_checks(self):
        """Stop the health check thread."""
        if not self.enable_health_checks or self.health_check_thread is None:
            return
        
        # Set stop event
        self.stop_health_checks_event.set()
        
        # Wait for thread to stop
        if self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=5)
        
        logger.info("Health check thread stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current system status.
        
        Returns:
            System status dictionary
        """
        # Perform a health check
        self.health_check()
        
        # Create status summary
        status = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": self.metrics["system"].get("uptime_seconds", 0),
            "components": self.metrics["system"]["component_status"],
            "error_count": len(self.metrics["system"]["errors"]),
            "recent_errors": self.metrics["system"]["errors"][-5:] if self.metrics["system"]["errors"] else []
        }
        
        # Determine overall status
        component_statuses = [c["status"] for c in status["components"].values()]
        
        if "failed" in component_statuses:
            status["overall_status"] = "failed"
        elif "degraded" in component_statuses:
            status["overall_status"] = "degraded"
        else:
            status["overall_status"] = "healthy"
        
        return status
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Performance metrics dictionary
        """
        return self.metrics["performance"]
    
    def get_recent_events(self, 
                         count: int = 10, 
                         level: str = None, 
                         component: str = None) -> List[Dict[str, Any]]:
        """
        Get recent events.
        
        Args:
            count: Number of events to return
            level: Filter by log level
            component: Filter by component
            
        Returns:
            List of event dictionaries
        """
        events = self.metrics["events"]
        
        # Apply filters
        if level:
            events = [e for e in events if e["level"].upper() == level.upper()]
        
        if component:
            events = [e for e in events if e["component"] == component]
        
        # Sort by timestamp (newest first)
        events.sort(key=lambda e: e["timestamp"], reverse=True)
        
        # Return requested number of events
        return events[:count]
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive system report.
        
        Returns:
            Report dictionary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_status": self.get_system_status(),
            "performance_metrics": self.get_performance_metrics(),
            "recent_events": self.get_recent_events(count=20),
            "recent_errors": self.get_recent_events(count=10, level="ERROR")
        }
        
        # Save report
        report_file = os.path.join(self.log_dir, f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyJSONEncoder)
            
            logger.info(f"System report generated and saved to {report_file}")
        except Exception as e:
            logger.error(f"Failed to save system report: {str(e)}")
        
        return report
    
    def cleanup_old_logs(self, max_age_days: int = 30):
        """
        Clean up old log files.
        
        Args:
            max_age_days: Maximum age of log files in days
        """
        try:
            # Get current time
            now = time.time()
            
            # Calculate cutoff time
            cutoff_time = now - (max_age_days * 24 * 60 * 60)
            
            # Check log directory
            for filename in os.listdir(self.log_dir):
                file_path = os.path.join(self.log_dir, filename)
                
                # Skip if not a file
                if not os.path.isfile(file_path):
                    continue
                
                # Check file modification time
                mod_time = os.path.getmtime(file_path)
                
                if mod_time < cutoff_time:
                    # Delete old file
                    os.remove(file_path)
                    logger.info(f"Deleted old log file: {file_path}")
            
            # Check metrics directory
            for filename in os.listdir(self.metrics_dir):
                file_path = os.path.join(self.metrics_dir, filename)
                
                # Skip if not a file or if it's the current metrics file
                if not os.path.isfile(file_path) or filename == "fl_metrics.json":
                    continue
                
                # Check file modification time
                mod_time = os.path.getmtime(file_path)
                
                if mod_time < cutoff_time:
                    # Delete old file
                    os.remove(file_path)
                    logger.info(f"Deleted old metrics file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to clean up old logs: {str(e)}")
    
    def shutdown(self):
        """Shutdown the monitoring system."""
        logger.info("Shutting down monitoring system")
        
        # Set running status to False
        self.is_running = False
        
        # Stop health checks
        self.stop_health_checks()
        
        # Save final metrics
        self.save_metrics()
        
        # Generate final report
        self.generate_report()
        
        logger.info("Monitoring system shutdown complete")
