# MLOps-Mini-Framework

# Experiments

File organization:

## Models
experiments/
├── experiment_id_1/
│   ├── metadata.json
│   ├── params.json
│   ├── metrics/
│   │   └── metrics.jsonl
│   └── models/
│       ├── model_id_1.pkl
│       └── model_id_1_metadata.json
└── experiment_id_2/
    └── ...


## Data
experiments/
├── datasets/
│   └── dataset_id/
│       ├── version1/
│       │   ├── data.parquet/data.npy/original_file
│       │   └── metadata.json
│       └── version2/
│           ├── data.parquet/data.npy/original_file
│           └── metadata.json


## Data Quality

Basic data quality checks:
- Schema validation
- Missing value detection
- Value range validation
- Data type validation
- Categorical value set validation


Drift detection:
- Data drift using Kolmogorov-Smirnov test
- Concept drift monitoring through distribution shifts
- Configurable thresholds for drift detection


Comprehensive reporting:
- Quality check results
- Failed validations
- Drift analysis
- Timestamp tracking



# TODO

Next steps could include:
- Implementing artifact tracking
- Adding experiment comparison utilities
- Creating visualization tools
- Adding support for distributed training
- Implementing experiment search/filtering
- Adding support for different storage backends
