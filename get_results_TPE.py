"""
get_results_TPE.py

This script retrieves and evaluates the best hyperparameters found using TPE search for each dataset.

Usage:
    python get_results_TPE.py [start:end:increment]
    python get_results_TPE.py [dataset_index]
    python get_results_TPE.py [dataset_name]
"""

import sys
import main_module as helper_experiments
import dataset_names

if __name__ == "__main__":
    # Retrieve the list of datasets to work on
    dataset_list = dataset_names.get_database_list_from_arguments(sys.argv)

    # Iterate over each dataset and evaluate the best hyperparameters found using TPE search
    for dataset_name in dataset_list:
        helper_experiments.get_best_result_for_one_dataset(
            search_type="HyperoptSearchTPE",
            dataset_name=dataset_name,
            n_iter=100,
            search_max_epoch=1000,
            best_result_max_epoch=5000
        )
