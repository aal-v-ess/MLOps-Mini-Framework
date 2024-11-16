import json
# import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import pickle
from dataclasses import dataclass
# import hashlib
# import yaml
from enum import Enum

class ModelStage(Enum):
    """Enum for model lifecycle stages"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

@dataclass
class ExperimentMetadata:
    """Metadata for an experiment run"""
    experiment_id: str
    name: str
    description: Optional[str]
    tags: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"

class ExperimentTracker:
    """Core class for tracking ML experiments"""
    
    def __init__(self, base_dir: Union[str, Path] = "experiments"):
        """
        Initialize the experiment tracker.
        
        Args:
            base_dir: Base directory for storing experiments
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._current_experiment: Optional[str] = None
        
    def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Create a new experiment.
        
        Args:
            name: Name of the experiment
            description: Optional description
            tags: Optional list of tags for categorization
            
        Returns:
            experiment_id: Unique identifier for the experiment
        """
        # Generate unique experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{name.lower().replace(' ', '_')}_{timestamp}"
        
        # Create experiment directory
        experiment_dir = self.base_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            name=name,
            description=description,
            tags=tags or [],
            start_time=datetime.now()
        )
        
        # Save metadata
        self._save_metadata(experiment_id, metadata)
        self._current_experiment = experiment_id
        
        return experiment_id
    
    def log_params(self, params: Dict[str, Any], experiment_id: Optional[str] = None) -> None:
        """
        Log parameters for an experiment.
        
        Args:
            params: Dictionary of parameters to log
            experiment_id: Optional experiment ID. Uses current experiment if not specified.
        """
        experiment_id = self._validate_experiment_id(experiment_id)
        params_file = self._get_experiment_dir(experiment_id) / "params.json"
        
        # Load existing params if any
        if params_file.exists():
            existing_params = self._load_json(params_file)
            # Update with new params
            existing_params.update(params)
            params = existing_params
        
        self._save_json(params_file, params)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        experiment_id: Optional[str] = None
    ) -> None:
        """
        Log metrics for an experiment.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
            experiment_id: Optional experiment ID. Uses current experiment if not specified.
        """
        experiment_id = self._validate_experiment_id(experiment_id)
        metrics_dir = self._get_experiment_dir(experiment_id) / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        
        # Add timestamp and step
        metrics_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            **metrics
        }
        
        # Append to metrics file
        metrics_file = metrics_dir / "metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics_entry) + "\n")
    
    def save_model(
        self,
        model: Any,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None,
        stage: ModelStage = ModelStage.DEVELOPMENT
    ) -> str:
        """
        Save a model with its metadata.
        
        Args:
            model: The model object to save
            name: Name of the model
            metadata: Optional model metadata
            experiment_id: Optional experiment ID. Uses current experiment if not specified.
            stage: Model lifecycle stage
            
        Returns:
            model_id: Unique identifier for the saved model
        """
        experiment_id = self._validate_experiment_id(experiment_id)
        models_dir = self._get_experiment_dir(experiment_id) / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Generate model ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{name}_{timestamp}"
        
        # Save model
        model_path = models_dir / f"{model_id}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Save metadata
        model_metadata = {
            "model_id": model_id,
            "name": name,
            "timestamp": timestamp,
            "stage": stage.value,
            "experiment_id": experiment_id,
            **(metadata or {})
        }
        
        metadata_path = models_dir / f"{model_id}_metadata.json"
        self._save_json(metadata_path, model_metadata)
        
        return model_id
    
    def load_model(self, model_id: str, experiment_id: Optional[str] = None) -> Any:
        """
        Load a saved model.
        
        Args:
            model_id: ID of the model to load
            experiment_id: Optional experiment ID. Uses current experiment if not specified.
            
        Returns:
            model: The loaded model object
        """
        experiment_id = self._validate_experiment_id(experiment_id)
        models_dir = self._get_experiment_dir(experiment_id) / "models"
        model_path = models_dir / f"{model_id}.pkl"
        
        if not model_path.exists():
            raise ValueError(f"Model {model_id} not found in experiment {experiment_id}")
        
        with open(model_path, "rb") as f:
            return pickle.load(f)
    
    def get_experiment_metrics(
        self,
        experiment_id: Optional[str] = None,
        metric_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get metrics for an experiment as a DataFrame.
        
        Args:
            experiment_id: Optional experiment ID. Uses current experiment if not specified.
            metric_names: Optional list of metric names to filter
            
        Returns:
            DataFrame containing metrics
        """
        experiment_id = self._validate_experiment_id(experiment_id)
        metrics_file = self._get_experiment_dir(experiment_id) / "metrics" / "metrics.jsonl"
        
        if not metrics_file.exists():
            return pd.DataFrame()
        
        # Read metrics
        metrics = []
        with open(metrics_file, "r") as f:
            for line in f:
                metrics.append(json.loads(line))
        
        df = pd.DataFrame(metrics)
        
        # Filter metrics if specified
        if metric_names:
            columns = ["timestamp", "step"] + metric_names
            df = df[columns]
        
        return df
    
    def _validate_experiment_id(self, experiment_id: Optional[str] = None) -> str:
        """Validate and return experiment ID"""
        if experiment_id is None:
            if self._current_experiment is None:
                raise ValueError("No experiment ID provided and no current experiment set")
            experiment_id = self._current_experiment
        return experiment_id
    
    def _get_experiment_dir(self, experiment_id: str) -> Path:
        """Get experiment directory path"""
        return self.base_dir / experiment_id
    
    def _save_metadata(self, experiment_id: str, metadata: ExperimentMetadata) -> None:
        """Save experiment metadata"""
        metadata_file = self._get_experiment_dir(experiment_id) / "metadata.json"
        metadata_dict = {
            k: v if not isinstance(v, datetime) else v.isoformat()
            for k, v in metadata.__dict__.items()
        }
        self._save_json(metadata_file, metadata_dict)
    
    def _save_json(self, path: Path, data: Dict) -> None:
        """Save data as JSON"""
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load_json(self, path: Path) -> Dict:
        """Load JSON data"""
        with open(path, "r") as f:
            return json.load(f)

# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = ExperimentTracker("experiments")
    
    # Create experiment
    experiment_id = tracker.create_experiment(
        name="test_experiment",
        description="Test experiment",
        tags=["test", "example"]
    )
    
    # Log parameters
    tracker.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    })
    
    # Log metrics
    for step in range(5):
        tracker.log_metrics({
            "loss": 0.5 - step * 0.1,
            "accuracy": 0.8 + step * 0.04
        }, step=step)
    
    # Save dummy model
    class DummyModel:
        def __init__(self):
            self.name = "dummy"
    
    model = DummyModel()
    model_id = tracker.save_model(
        model,
        name="model_name",
        metadata={"framework": "pytorch", "type": "classification"}
    )
    
    # Get metrics
    metrics_df = tracker.get_experiment_metrics(metric_names=["loss", "accuracy"])
    print("\nMetrics:")
    print(metrics_df)