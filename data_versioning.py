import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import pickle
from dataclasses import dataclass
import hashlib
import yaml
from enum import Enum
import shutil
from urllib.parse import urlparse
import logging
from contextlib import contextmanager
from experiments import ExperimentTracker

# ... [Previous code remains the same until after ModelStage class] ...

@dataclass
class DatasetMetadata:
    """Metadata for a versioned dataset"""
    dataset_id: str
    name: str
    version: str
    description: Optional[str]
    format: str
    size_bytes: int
    n_rows: Optional[int]
    n_columns: Optional[int]
    column_types: Optional[Dict[str, str]]
    tags: List[str]
    created_at: datetime
    hash: str
    source_url: Optional[str] = None
    parent_version: Optional[str] = None
    transformation_info: Optional[Dict[str, Any]] = None

class DataVersion:
    """Class for handling dataset versioning"""
    
    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir) / "datasets"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _compute_hash(self, data: Union[pd.DataFrame, np.ndarray, Path]) -> str:
        """
        Compute hash of dataset for versioning.
        
        Args:
            data: Dataset to hash (DataFrame, numpy array, or file path)
            
        Returns:
            sha256 hash of the dataset
        """
        hasher = hashlib.sha256()
        
        if isinstance(data, pd.DataFrame):
            # Hash DataFrame content
            for column in data.columns:
                hasher.update(column.encode())
                hasher.update(data[column].values.tobytes())
        elif isinstance(data, np.ndarray):
            # Hash numpy array
            hasher.update(data.tobytes())
        elif isinstance(data, Path):
            # Hash file content in chunks
            with open(data, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
            
        return hasher.hexdigest()
    
    def _get_dataset_path(self, dataset_id: str, version: str) -> Path:
        """Get path for dataset storage"""
        return self.base_dir / dataset_id / version
    
    def _get_data_format(self, data: Union[pd.DataFrame, np.ndarray, Path]) -> str:
        """Determine format of the dataset"""
        if isinstance(data, pd.DataFrame):
            return 'pandas.DataFrame'
        elif isinstance(data, np.ndarray):
            return 'numpy.ndarray'
        elif isinstance(data, Path):
            return data.suffix[1:]  # Remove the dot from extension
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _save_data(self, data: Union[pd.DataFrame, np.ndarray, Path], save_path: Path) -> None:
        """Save dataset to disk"""
        if isinstance(data, pd.DataFrame):
            data.to_parquet(save_path / 'data.parquet')
        elif isinstance(data, np.ndarray):
            np.save(save_path / 'data.npy', data)
        elif isinstance(data, Path):
            shutil.copy2(data, save_path / data.name)
    
    def _load_data(self, load_path: Path, format: str) -> Any:
        """Load dataset from disk"""
        if format == 'pandas.DataFrame':
            return pd.read_parquet(load_path / 'data.parquet')
        elif format == 'numpy.ndarray':
            return np.load(load_path / 'data.npy')
        else:
            # For other file formats, return the path
            data_files = list(load_path.glob(f'*.{format}'))
            if not data_files:
                raise FileNotFoundError(f"No files with format {format} found")
            return data_files[0]
    
    def _get_column_types(self, data: Union[pd.DataFrame, np.ndarray, Path]) -> Optional[Dict[str, str]]:
        """Get column types for tabular data"""
        if isinstance(data, pd.DataFrame):
            return {col: str(dtype) for col, dtype in data.dtypes.items()}
        return None
    
    def register_dataset(
        self,
        data: Union[pd.DataFrame, np.ndarray, Path],
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source_url: Optional[str] = None,
        parent_version: Optional[str] = None,
        transformation_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Register a new dataset version.
        
        Args:
            data: Dataset to register
            name: Name of the dataset
            description: Optional description
            tags: Optional list of tags
            source_url: Optional source URL
            parent_version: Optional previous version ID
            transformation_info: Optional information about transformations applied
            
        Returns:
            Tuple of (dataset_id, version)
        """
        # Generate dataset ID if new dataset
        dataset_id = name.lower().replace(' ', '_')
        
        # Compute version hash
        version = self._compute_hash(data)[:12]
        
        # Create dataset directory
        dataset_path = self._get_dataset_path(dataset_id, version)
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Get data format and size
        data_format = self._get_data_format(data)
        if isinstance(data, Path):
            size_bytes = data.stat().st_size
        else:
            # Approximate size for DataFrame/ndarray
            size_bytes = data.memory_usage(deep=True).sum() if isinstance(data, pd.DataFrame) else data.nbytes
        
        # Create metadata
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            name=name,
            version=version,
            description=description,
            format=data_format,
            size_bytes=size_bytes,
            n_rows=len(data) if hasattr(data, '__len__') else None,
            n_columns=len(data.columns) if isinstance(data, pd.DataFrame) else None,
            column_types=self._get_column_types(data),
            tags=tags or [],
            created_at=datetime.now(),
            hash=version,
            source_url=source_url,
            parent_version=parent_version,
            transformation_info=transformation_info
        )
        
        # Save data and metadata
        self._save_data(data, dataset_path)
        with open(dataset_path / 'metadata.json', 'w') as f:
            json.dump(metadata.__dict__, f, default=str, indent=2)
        
        self.logger.info(f"Registered dataset '{name}' with version {version}")
        return dataset_id, version
    
    def load_dataset(self, dataset_id: str, version: str) -> Tuple[Any, DatasetMetadata]:
        """
        Load a specific version of a dataset.
        
        Args:
            dataset_id: ID of the dataset
            version: Version hash
            
        Returns:
            Tuple of (dataset, metadata)
        """
        dataset_path = self._get_dataset_path(dataset_id, version)
        if not dataset_path.exists():
            raise ValueError(f"Dataset {dataset_id} version {version} not found")
        
        # Load metadata
        with open(dataset_path / 'metadata.json', 'r') as f:
            metadata_dict = json.load(f)
            metadata = DatasetMetadata(**{
                k: datetime.fromisoformat(v) if k == 'created_at' else v
                for k, v in metadata_dict.items()
            })
        
        # Load data
        data = self._load_data(dataset_path, metadata.format)
        
        return data, metadata
    
    def get_dataset_versions(self, dataset_id: str) -> List[DatasetMetadata]:
        """
        Get all versions of a dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            List of metadata for all versions
        """
        dataset_path = self.base_dir / dataset_id
        if not dataset_path.exists():
            return []
        
        versions = []
        for version_path in dataset_path.iterdir():
            if version_path.is_dir():
                with open(version_path / 'metadata.json', 'r') as f:
                    metadata_dict = json.load(f)
                    metadata = DatasetMetadata(**{
                        k: datetime.fromisoformat(v) if k == 'created_at' else v
                        for k, v in metadata_dict.items()
                    })
                    versions.append(metadata)
        
        return sorted(versions, key=lambda x: x.created_at, reverse=True)
    
    def get_dataset_lineage(self, dataset_id: str, version: str) -> List[DatasetMetadata]:
        """
        Get the lineage (version history) of a dataset.
        
        Args:
            dataset_id: ID of the dataset
            version: Starting version hash
            
        Returns:
            List of metadata objects representing the dataset's lineage
        """
        lineage = []
        current_version = version
        
        while current_version:
            try:
                _, metadata = self.load_dataset(dataset_id, current_version)
                lineage.append(metadata)
                current_version = metadata.parent_version
            except ValueError:
                break
        
        return lineage

@contextmanager
def track_dataset_usage(tracker: ExperimentTracker, dataset_id: str, version: str):
    """
    Context manager to track dataset usage in experiments.
    
    Args:
        tracker: ExperimentTracker instance
        dataset_id: ID of the dataset
        version: Version hash
    """
    # Log dataset usage at start
    tracker.log_params({
        f"dataset_{dataset_id}": {
            "version": version,
            "used_at": datetime.now().isoformat()
        }
    })
    
    try:
        yield
    finally:
        # You could add additional logging here if needed
        pass

# # Example usage
# if __name__ == "__main__":
#     # Initialize data versioning
#     data_version = DataVersion("experiments")
    
#     # Create sample dataset
#     df = pd.DataFrame({
#         'A': np.random.rand(100),
#         'B': np.random.randint(0, 10, 100),
#         'C': ['cat', 'dog'] * 50
#     })
    
#     # Register dataset
#     dataset_id, version = data_version.register_dataset(
#         data=df,
#         name="sample_dataset",
#         description="Sample dataset for testing",
#         tags=["test", "sample"],
#         transformation_info={
#             "preprocessing": "none"
#         }
#     )
    
#     # Create modified version
#     df_modified = df.copy()
#     df_modified['D'] = df_modified['A'] * 2
    
#     # Register new version
#     dataset_id, new_version = data_version.register_dataset(
#         data=df_modified,
#         name="sample_dataset",
#         description="Modified sample dataset",
#         tags=["test", "sample", "modified"],
#         parent_version=version,
#         transformation_info={
#             "preprocessing": "added column D = A * 2"
#         }
#     )
    
#     # Get all versions
#     versions = data_version.get_dataset_versions(dataset_id)
#     print("\nDataset versions:")
#     for v in versions:
#         print(f"Version: {v.version}, Created: {v.created_at}")
    
#     # Get lineage
#     lineage = data_version.get_dataset_lineage(dataset_id, new_version)
#     print("\nDataset lineage:")
#     for l in lineage:
#         print(f"Version: {l.version}, Transformation: {l.transformation_info}")
    
#     # Use with experiment tracker
#     tracker = ExperimentTracker("experiments")
#     experiment_id = tracker.create_experiment(
#         name="dataset_test",
#         description="Testing dataset versioning"
#     )
    
#     with track_dataset_usage(tracker, dataset_id, new_version):
#         # Load and use dataset
#         data, metadata = data_version.load_dataset(dataset_id, new_version)
#         print("\nLoaded dataset shape:", data.shape)