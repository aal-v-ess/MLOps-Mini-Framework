from experiments import ExperimentTracker
from data_versioning import DataVersion, track_dataset_usage
import pandas as pd
import numpy as np

# # Example usage
# if __name__ == "__main__":
#     # Initialize tracker
#     experiment_path = "experiments"
#     tracker = ExperimentTracker(experiment_path)
    
#     # Create experiment
#     experiment_id = tracker.create_experiment(
#         name="test_experiment",
#         description="Test experiment",
#         tags=["test", "example"]
#     )
    
#     # Log parameters
#     tracker.log_params({
#         "learning_rate": 0.001,
#         "batch_size": 32,
#         "epochs": 10
#     })
    
#     # Log metrics
#     for step in range(5):
#         tracker.log_metrics({
#             "loss": 0.5 - step * 0.1,
#             "accuracy": 0.8 + step * 0.04
#         }, step=step)
    
#     # Save dummy model (temporary)
#     class DummyModel:
#         def __init__(self):
#             self.name = "dummy"
    
#     model = DummyModel()
#     model_id = tracker.save_model(
#         model,
#         name="model_name",
#         metadata={"framework": "pytorch", "type": "classification"}
#     )
    
#     # Get metrics
#     metrics_df = tracker.get_experiment_metrics(metric_names=["loss", "accuracy"])
#     print("\nMetrics:")
#     print(metrics_df)



    
# Example usage
if __name__ == "__main__":
    # Initialize data versioning
    data_version = DataVersion("experiments")
    
    # Create sample dataset
    df = pd.DataFrame({
        'A': np.random.rand(100),
        'B': np.random.randint(0, 10, 100),
        'C': ['cat', 'dog'] * 50
    })
    
    # Register dataset
    dataset_id, version = data_version.register_dataset(
        data=df,
        name="sample_dataset",
        description="Sample dataset for testing",
        tags=["test", "sample"],
        transformation_info={
            "preprocessing": "none"
        }
    )
    
    # Create modified version
    df_modified = df.copy()
    df_modified['D'] = df_modified['A'] * 2
    
    # Register new version
    dataset_id, new_version = data_version.register_dataset(
        data=df_modified,
        name="sample_dataset",
        description="Modified sample dataset",
        tags=["test", "sample", "modified"],
        parent_version=version,
        transformation_info={
            "preprocessing": "added column D = A * 2"
        }
    )
    
    # Get all versions
    versions = data_version.get_dataset_versions(dataset_id)
    print("\nDataset versions:")
    for v in versions:
        print(f"Version: {v.version}, Created: {v.created_at}")
    
    # Get lineage
    lineage = data_version.get_dataset_lineage(dataset_id, new_version)
    print("\nDataset lineage:")
    for lin in lineage:
        print(f"Version: {lin.version}, Transformation: {lin.transformation_info}")
    
    # Use with experiment tracker to track dataset
    tracker = ExperimentTracker("experiments")
    experiment_id = tracker.create_experiment(
        name="dataset_test",
        description="Testing dataset versioning"
    )
    
    with track_dataset_usage(tracker, dataset_id, new_version):
        # Load and use dataset
        data, metadata = data_version.load_dataset(dataset_id, new_version)
        print("\nLoaded dataset shape:", data.shape)

    # Initialize tracker
    experiment_path = "experiments"
    tracker = ExperimentTracker(experiment_path)
    
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
    
    # Save dummy model (temporary)
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
