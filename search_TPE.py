"""
search_TPE.py

This script performs hyperparameter optimization using TPE (Tree-structured Parzen Estimator) search for each dataset.

Usage:
    python search_TPE.py [start:end:increment]
    python search_TPE.py [dataset_index]
    python search_TPE.py [dataset_name]
"""

import sys
import os
import dataset_names
import main_module
from feature_selection import rfe_feature_selection

from set_seed import set_seed

# Set the seed for reproducibility
set_seed(88)

if __name__ == "__main__":
    # Retrieve the list of datasets to work on
    dataset_list = dataset_names.get_database_list_from_arguments(sys.argv)

    print("Datasets to work on:")
    print(dataset_list)

    # Define the search type
    search_type = "HyperoptSearchTPE"

    # Ensure the "Results" directory exists
    if not os.path.exists('Results'):
        os.makedirs('Results')

    # Perform hyperparameter optimization using TPE search for each dataset
    for dataset_name in dataset_list:
        initial_features = "min,max,mean,cos"

        # Perform recursive feature elimination to find the best features
        best_features, _ = rfe_feature_selection(
            dataset_name=dataset_name,
            initial_features=initial_features,
            bag_size=20,  # Initial value for bag_size
            n_prototypes=5,  # Initial value for n_prototypes
            lambda_fused_lasso=1e-5,  # Initial value for lambda_fused_lasso
            n_iter=10,
            search_max_epoch=100
        )
        features_to_use_str = ','.join(best_features)

        # Find the best hyperparameters using TPE search
        main_module.find_result_for_one(
            search_type=search_type,
            dataset_name=dataset_name,
            n_iter=100,
            search_max_epoch=1000,
            features_to_use_str=features_to_use_str
        )
