"""
MLflow integration for experiment tracking and model versioning.
Tracks experiments, parameters, metrics, and artifacts for MLOps.
"""
import os
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

from ..utils.config import BASE_DIR
from ..utils.logger import setup_logger

logger = setup_logger("mlflow_tracker")


class MLflowTracker:
    """
    MLflow tracker for experiment tracking, model versioning, and metrics logging.
    
    This class provides a clean interface for:
    - Experiment tracking (parameters, metrics, artifacts)
    - Model versioning and registry
    - Reproducibility (logging all configurations)
    
    Example:
        >>> tracker = MLflowTracker(experiment_name="cv_ranking_v1")
        >>> tracker.start_run()
        >>> tracker.log_params({"semantic_weight": 0.3, "llm_weight": 0.7})
        >>> tracker.log_metrics({"accuracy": 0.85, "precision": 0.82})
        >>> tracker.log_model(vectorstore, "vectorstore")
        >>> tracker.end_run()
    """
    
    def __init__(
        self,
        experiment_name: str = "cv_ranking",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[Path] = None
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI. If None, uses local file store.
            artifact_location: Location for artifacts. If None, uses default.
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI (default: local file store)
        if tracking_uri is None:
            mlflow_dir = BASE_DIR / "mlruns"
            tracking_uri = f"file://{mlflow_dir.absolute()}"
        
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=str(artifact_location) if artifact_location else None
                )
                logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            else:
                logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
        except Exception as e:
            logger.warning(f"Could not set up experiment: {e}. Using default.")
        
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        self.current_run = None
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for this run. If None, uses timestamp.
            tags: Optional dictionary of tags for this run.
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_run = mlflow.start_run(run_name=run_name)
        
        if tags:
            mlflow.set_tags(tags)
        
        logger.info(f"Started MLflow run: {run_name} (ID: {self.current_run.info.run_id})")
    
    def end_run(self):
        """End the current MLflow run."""
        if self.current_run:
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.current_run.info.run_id}")
            self.current_run = None
        else:
            logger.warning("No active run to end")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters (hyperparameters, configuration) to MLflow.
        
        Args:
            params: Dictionary of parameter names and values
            
        Example:
            >>> tracker.log_params({
            ...     "semantic_weight": 0.3,
            ...     "llm_weight": 0.7,
            ...     "embedding_model": "all-MiniLM-L6-v2"
            ... })
        """
        # Convert all values to strings (MLflow requirement)
        params_str = {k: str(v) for k, v in params.items()}
        mlflow.log_params(params_str)
        logger.debug(f"Logged {len(params)} parameters to MLflow")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for time-series metrics
            
        Example:
            >>> tracker.log_metrics({
            ...     "ranking_accuracy": 0.85,
            ...     "avg_llm_score": 7.2,
            ...     "avg_semantic_similarity": 0.75
            ... })
        """
        if step is not None:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value, step=step)
        else:
            mlflow.log_metrics(metrics)
        logger.debug(f"Logged {len(metrics)} metrics to MLflow")
    
    def log_artifact(self, local_path: Path, artifact_path: Optional[str] = None):
        """
        Log an artifact (file or directory) to MLflow.
        
        Args:
            local_path: Path to file or directory to log
            artifact_path: Optional path within artifact directory
            
        Example:
            >>> tracker.log_artifact(Path("config.yaml"), "config")
        """
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
        logger.debug(f"Logged artifact: {local_path}")
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None
    ):
        """
        Log a model to MLflow.
        
        Args:
            model: Model object to log
            artifact_path: Path within artifact directory
            registered_model_name: Optional name for model registry
            
        Note:
            This is a placeholder. Actual model logging depends on model type.
            For vectorstores, we log the path. For sklearn models, use mlflow.sklearn.log_model().
        """
        # For now, log as artifact if it's a path
        if isinstance(model, (str, Path)):
            self.log_artifact(Path(model), artifact_path)
        else:
            logger.warning(f"Model logging for type {type(model)} not fully implemented")
            # In production, you'd use appropriate MLflow model logging:
            # mlflow.sklearn.log_model(model, artifact_path)
            # mlflow.langchain.log_model(model, artifact_path)
    
    def log_config(self, config_dict: Dict[str, Any]):
        """
        Log configuration dictionary as both params and artifact.
        
        Args:
            config_dict: Configuration dictionary to log
        """
        import json
        
        # Log as parameters
        self.log_params(config_dict)
        
        # Also save as JSON artifact
        config_file = BASE_DIR / "temp_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.log_artifact(config_file, "config")
        config_file.unlink()  # Clean up temp file
    
    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()


def create_experiment_tracker(
    experiment_name: str = "cv_ranking",
    run_name: Optional[str] = None
) -> MLflowTracker:
    """
    Convenience function to create and start an MLflow tracker.
    
    Args:
        experiment_name: Name of the experiment
        run_name: Optional name for the run
        
    Returns:
        MLflowTracker instance with active run
    """
    tracker = MLflowTracker(experiment_name=experiment_name)
    tracker.start_run(run_name=run_name)
    return tracker

