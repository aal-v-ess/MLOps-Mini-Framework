import great_expectations as ge
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union
from scipy.stats import ks_2samp

class DataQualityChecker:
    """
    A class to perform data quality checks and drift detection using Great Expectations.
    """
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None):
        """
        Initialize the DataQualityChecker.
        
        Args:
            reference_data: Optional reference dataset for drift detection
        """
        self.reference_data = reference_data
        if reference_data is not None:
            self.reference_stats = self._compute_statistics(reference_data)
    
    def _compute_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Compute summary statistics for numerical columns.
        """
        stats = {}
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'q1': df[col].quantile(0.25),
                'q3': df[col].quantile(0.75)
            }
        return stats

    def create_basic_suite(self, df: pd.DataFrame) -> ge.core.ExpectationSuite:
        """
        Create a basic expectation suite based on the data.
        """
        context = ge.data_context.DataContext()
        suite = context.create_expectation_suite(
            expectation_suite_name="basic_suite",
            overwrite_existing=True
        )
        
        ge_df = ge.from_pandas(df)
        
        # Add basic expectations for all columns
        for column in df.columns:
            # Check for missing values
            ge_df.expect_column_values_to_not_be_null(column)
            
            # Add type-specific expectations
            if df[column].dtype in ['int64', 'float64']:
                ge_df.expect_column_values_to_be_in_type_list(
                    column,
                    ['INTEGER', 'FLOAT', 'DOUBLE', 'DECIMAL']
                )
                
                # Add range expectations based on current data
                min_val = df[column].min()
                max_val = df[column].max()
                ge_df.expect_column_values_to_be_between(
                    column,
                    min_value=min_val,
                    max_value=max_val
                )
            
            # TODO: expect_column_values_to_be_in_type_list throws ValueError: No recognized numpy/python type in list: ['STRING', 'OBJECT']
            # elif df[column].dtype == 'object':
            #     print(df[column].iloc[0])
            #     ge_df.expect_column_values_to_be_in_type_list(
            #         column,
            #         ['STRING', 'OBJECT']
            #     )
                
            #     # If column has few unique values, create value set expectation
            #     if df[column].nunique() / len(df) < 0.1:  # Less than 10% unique values
            #         value_set = list(df[column].unique())
            #         ge_df.expect_column_values_to_be_in_set(
            #             column,
            #             value_set
            #         )
        
        return suite

    def check_data_quality(self, df: pd.DataFrame, suite: ge.core.ExpectationSuite) -> Dict:
        """
        Run data quality checks using the provided expectation suite.
        """
        ge_df = ge.from_pandas(df)
        validation_result = ge_df.validate(
            expectation_suite=suite,
            result_format="COMPLETE"
        )
        
        return {
            'success': validation_result.success,
            'results': validation_result.results,
            'statistics': validation_result.statistics
        }

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.05
    ) -> Dict[str, Dict]:
        """
        Detect both data drift and potential concept drift.
        
        Args:
            current_data: New data to compare against reference
            threshold: p-value threshold for drift detection
        
        Returns:
            Dictionary containing drift detection results
        """
        if self.reference_data is None:
            raise ValueError("Reference data must be set to detect drift")
        
        drift_results = {
            'data_drift': {},
            'concept_drift': {}
        }
        
        # Check data drift for numerical columns
        numerical_cols = current_data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            # Perform Kolmogorov-Smirnov test
            ks_statistic, p_value = ks_2samp(
                self.reference_data[col].dropna(),
                current_data[col].dropna()
            )
            
            drift_results['data_drift'][col] = {
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'drift_detected': p_value < threshold
            }
            
            # Check for concept drift indicators
            current_stats = self._compute_statistics(current_data)
            
            # Calculate distribution shifts
            mean_shift = abs(current_stats[col]['mean'] - self.reference_stats[col]['mean'])
            std_shift = abs(current_stats[col]['std'] - self.reference_stats[col]['std'])
            
            drift_results['concept_drift'][col] = {
                'mean_shift': mean_shift,
                'std_shift': std_shift,
                'mean_shift_percentage': mean_shift / abs(self.reference_stats[col]['mean']) * 100,
                'std_shift_percentage': std_shift / abs(self.reference_stats[col]['std']) * 100
            }
        
        return drift_results

    def generate_report(
        self,
        quality_results: Dict,
        drift_results: Optional[Dict] = None
    ) -> Dict:
        """
        Generate a comprehensive data quality and drift report.
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_quality': {
                'overall_success': quality_results['success'],
                'failed_checks': [
                    result for result in quality_results['results']
                    if not result['success']
                ]
            }
        }
        
        if drift_results:
            report['drift_analysis'] = {
                'data_drift': {
                    'columns_with_drift': [
                        col for col, result in drift_results['data_drift'].items()
                        if result['drift_detected']
                    ]
                },
                'concept_drift': {
                    'significant_shifts': [
                        col for col, result in drift_results['concept_drift'].items()
                        if result['mean_shift_percentage'] > 10 or result['std_shift_percentage'] > 10
                    ]
                }
            }
        
        return report