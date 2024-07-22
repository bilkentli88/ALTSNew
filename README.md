# ALTSNew: Advanced Learning-based Time Series Classification

## Overview

ALTSNew is a shapelet-based neural network model implemented using the PyTorch framework for time series classification. The model uses four features between each time series instance and each shapelet candidate:

1. Minimum distance
2. Maximum distance
3. Arithmetic mean distance
4. Cosine similarity

ALTSNew is designed to be both accurate and interpretable, leveraging the UCR Benchmark repository as the baseline for performance comparison. The hyperparameters tuned in this model are:

1. `bag_size`: Length of the candidate shapelet, proportional to the time series length.
2. `n_prototypes`: Number of shapelets created.
3. `lambda_fused_lasso`: A regularization parameter for Fused Lasso Regularization.

## Requirements

- Python 3.7+
- PyTorch
- Skorch
- Hyperopt
- NumPy
- Pandas
- Scikit-learn

## Installation

1. Clone this repository:

    ```sh
    git clone https://github.com/yourusername/ALTSNew.git
    cd ALTSNew
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Project Structure

- `main_module.py`: Contains the main functions for hyperparameter search, model training, and evaluation.
- `shapelet_regularization.py`: Defines a custom Skorch NeuralNetClassifier with regularization.
- `shapelet_generation.py`: Defines the ShapeletGeneration3LN neural network model with shapelet-based learning.
- `helper_datasets.py`: Functions for loading and preprocessing datasets, as well as converting data into shapelet bags.
- `dataset_names.py`: Provides functions to retrieve lists of dataset names.
- `feature_selection.py`: Functions for evaluating model performance and performing recursive feature elimination (RFE).
- `search_TPE.py`: Script for performing hyperparameter optimization using TPE search.
- `get_results_TPE.py`: Script for retrieving and evaluating the best hyperparameters found using TPE search.
- `set_seed.py`: Utility for setting the random seed for reproducibility.
- `Datasets/`: Directory containing the UCR Benchmark datasets divided into training and testing files.
- `Results/`: Directory where the results of the hyperparameter search and evaluations are saved.

## Usage

### Running the Hyperparameter Search

To run the hyperparameter optimization using TPE search:

```sh
python search_TPE.py [start:end:increment]
python search_TPE.py [dataset_index]
python search_TPE.py [dataset_name]


### Evaluating the Best Hyperparameters

To evaluate the best hyperparameters found using TPE search:

```sh

python get_results_TPE.py [start:end:increment]
python get_results_TPE.py [dataset_index]
python get_results_TPE.py [dataset_name]
