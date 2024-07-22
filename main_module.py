"""
main_module.py

This module contains functions for hyperparameter optimization, model evaluation, and result retrieval.
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from skorch.callbacks import EarlyStopping
from shapelet_generation import ShapeletGeneration as ShapeletNN
from shapelet_regularization import ShapeletRegularizedNet
from helper_datasets import load_dataset
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

STRIDE_RATIO = 0.10
CV_COUNT = 5  # Increased number of folds for more rigorous validation
LIST_N_PROTOTYPES = list(range(1, 101))
LIST_LEARNING_RATES = [0.0001]  # Decreased learning rate

def get_list_bag_sizes(time_series_length):
    """
    Generates a list of bag sizes based on the length of the time series.

    Parameters:
    - time_series_length (int): The length of the time series.

    Returns:
    - list: A list of bag sizes.
    """
    BAG_SIZE_START = 10
    BAG_SIZE_END = 40
    BAG_SIZE_COUNT = 50

    bag_size_start = int(time_series_length * (BAG_SIZE_START / 100))
    bag_size_end = int(time_series_length * (BAG_SIZE_END / 100))

    list_bag_sizes = np.linspace(bag_size_start, bag_size_end, BAG_SIZE_COUNT, endpoint=True, dtype=int)
    return np.unique(list_bag_sizes).tolist()

def hyperopt_objective(search_space):
    """
    Objective function for hyperparameter optimization using Hyperopt.

    Parameters:
    - search_space (dict): The search space for hyperparameters.

    Returns:
    - dict: A dictionary with the loss and status of the optimization.
    """
    dataset_name = search_space["dataset_name"]
    train, y_train, _, _ = load_dataset(dataset_name)
    y_train = torch.from_numpy(y_train).float()
    y_train_labels = np.argmax(y_train, axis=1)
    n_classes = y_train.shape[1]

    nn_shapelet_generator = ShapeletNN(
        n_prototypes=int(search_space["n_prototypes"]),  # Ensure n_prototypes is an integer
        bag_size=search_space["bag_size"],
        n_classes=n_classes,
        stride_ratio=STRIDE_RATIO,
        features_to_use_str=search_space["features_to_use_str"],
    )

    net = get_skorch_regularized_classifier(nn_shapelet_generator, search_space["max_epoch"],
                                            use_early_stopping=True)
    skf = StratifiedKFold(n_splits=CV_COUNT)
    accuracies = []

    for train_index, val_index in skf.split(train, y_train_labels):
        X_train, X_val = train[train_index], train[val_index]
        y_train_fold, y_val_fold = y_train_labels[train_index], y_train_labels[val_index]

        net.fit(X_train, y_train_fold)
        y_pred = net.predict(X_val)
        y_pred_labels = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
        accuracies.append(accuracy_score(y_val_fold, y_pred_labels))

    avg_accuracy = np.mean(accuracies)
    return {'loss': -avg_accuracy, 'status': STATUS_OK}

def find_best_hyper_params_hyperopt_search_tpe(dataset_name, n_iter=100, search_max_epoch=1000, features_to_use_str=None):
    """
    Finds the best hyperparameters using Hyperopt TPE search.

    Parameters:
    - dataset_name (str): The name of the dataset.
    - n_iter (int): The number of iterations for the search.
    - search_max_epoch (int): The maximum number of epochs for the search.
    - features_to_use_str (str): The features to use for the model.

    Returns:
    - dict: A dictionary with the best hyperparameters and related information.
    """
    return find_best_hyper_params_hyperopt_search(
        dataset_name, tpe.suggest, n_iter, search_max_epoch, features_to_use_str
    )

def find_best_hyper_params_hyperopt_search(dataset_name, search_algorithm, n_iter=100, search_max_epoch=1000, features_to_use_str=None):
    """
    Finds the best hyperparameters using Hyperopt search.

    Parameters:
    - dataset_name (str): The name of the dataset.
    - search_algorithm: The search algorithm to use (e.g., tpe.suggest).
    - n_iter (int): The number of iterations for the search.
    - search_max_epoch (int): The maximum number of epochs for the search.
    - features_to_use_str (str): The features to use for the model.

    Returns:
    - dict: A dictionary with the best hyperparameters and related information.
    """
    train, y_train, _, _ = load_dataset(dataset_name)
    y_train = torch.from_numpy(y_train).float()
    y_train_labels = np.argmax(y_train, axis=1)
    n_classes = y_train.shape[1]

    list_bag_sizes = get_list_bag_sizes(train.shape[1])

    search_space = {
        'n_prototypes': hp.quniform('n_prototypes', 2, 100, 1),  # Changed range for n_prototypes
        'cv_count': hp.choice('cv_count', [CV_COUNT]),
        'dataset_name': hp.choice('dataset_name', [dataset_name]),
        'max_epoch': hp.choice('max_epoch', [search_max_epoch]),
        'features_to_use_str': hp.choice('features_to_use_str', [features_to_use_str]),
        'bag_size': hp.choice('bag_size', list_bag_sizes),
        'lambda_fused_lasso': hp.loguniform('lambda_fused_lasso', np.log(1e-8), np.log(1)),  # Adjusted range
        'lr': hp.choice('lr', LIST_LEARNING_RATES)
    }

    trials = Trials()
    best_loss = float('inf')
    no_improvement_count = 0
    early_stop_threshold = 10  # Number of iterations without improvement before stopping

    def early_stop_fn(trials, early_stop_threshold):
        nonlocal best_loss, no_improvement_count
        losses = [trial['result']['loss'] for trial in trials]
        if len(losses) > early_stop_threshold:
            if min(losses[-early_stop_threshold:]) >= best_loss:
                return True, {}  # Early stop
            best_loss = min(best_loss, min(losses))
        return False, {}

    best = fmin(fn=hyperopt_objective, space=search_space, algo=search_algorithm, max_evals=n_iter, trials=trials,
                return_argmin=False, verbose=True, rstate=np.random.default_rng(seed=88),
                early_stop_fn=lambda trials: early_stop_fn(trials.trials, early_stop_threshold))

    return {
        "dataset_name": dataset_name,
        "search_type": "HyperoptSearch",
        "features_to_use": features_to_use_str,
        "bag_size": best["bag_size"],
        "n_classes": n_classes,
        "n_prototypes": int(best["n_prototypes"]),  # Ensure n_prototypes is an integer
        "stride_ratio": STRIDE_RATIO,
        "lambda_fused_lasso": best["lambda_fused_lasso"],
        "n_iter": n_iter,
        "cv_count": CV_COUNT,
        "train_accuracy": -1 * min(trials.losses())
    }

def evaluate_model_performance(dataset_name, bag_size, n_prototypes, max_epoch, phase, features_to_use_str):
    """
    Evaluates the model performance on the test set.

    Parameters:
    - dataset_name (str): The name of the dataset.
    - bag_size (int): The bag size.
    - n_prototypes (int): The number of prototypes.
    - max_epoch (int): The maximum number of epochs.
    - phase (str): The phase of training (training or test).
    - features_to_use_str (str): The features to use for the model.

    Returns:
    - float: The accuracy score on the test set.
    """
    train, y_train, test, y_test = load_dataset(dataset_name)
    y_train = torch.from_numpy(y_train).float()
    y_train_labels = np.argmax(y_train, axis=1)
    n_classes = y_train.shape[1]

    nn_shapelet_generator = ShapeletNN(
        n_prototypes=n_prototypes,
        bag_size=bag_size,
        n_classes=n_classes,
        stride_ratio=STRIDE_RATIO,
        features_to_use_str=features_to_use_str
    )

    use_early_stopping = (phase == "training")
    net = get_skorch_regularized_classifier(nn_shapelet_generator, max_epoch,
                                            use_early_stopping=use_early_stopping)

    net.fit(train, y_train_labels)
    y_test_labels = np.argmax(y_test, axis=1)
    y_predict = net.predict(test)
    y_predict_labels = np.argmax(y_predict, axis=1) if y_predict.ndim > 1 else y_predict
    return accuracy_score(y_test_labels, y_predict_labels)

def get_skorch_regularized_classifier(nn_shapelet_generator, max_epochs, use_early_stopping):
    """
    Returns a Skorch regularized classifier.

    Parameters:
    - nn_shapelet_generator (ShapeletNN): The shapelet neural network generator.
    - max_epochs (int): The maximum number of epochs.
    - use_early_stopping (bool): Whether to use early stopping.

    Returns:
    - ShapeletRegularizedNet: The Skorch regularized classifier.
    """
    callbacks = []
    if use_early_stopping:
        early_stopping = EarlyStopping(monitor='valid_loss',
                                       patience=10,
                                       threshold=0.001,
                                       threshold_mode='rel',
                                       lower_is_better=True)
        callbacks.append(early_stopping)

    return ShapeletRegularizedNet(
        module=nn_shapelet_generator,
        max_epochs=max_epochs,
        lr=0.0001,  # Decreased learning rate
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        callbacks=callbacks
    )

def get_best_result_for_one_dataset(search_type, dataset_name, n_iter, search_max_epoch, best_result_max_epoch):
    """
    Retrieves the best result for one dataset.

    Parameters:
    - search_type (str): The type of search (e.g., HyperoptSearch).
    - dataset_name (str): The name of the dataset.
    - n_iter (int): The number of iterations for the search.
    - search_max_epoch (int): The maximum number of epochs for the search.
    - best_result_max_epoch (int): The maximum number of epochs for the best result.

    Returns:
    - None
    """
    search_filename = get_filename_output_for_search(search_type, dataset_name, n_iter, search_max_epoch)
    if not os.path.exists(search_filename):
        print(f"{search_filename} does not exist, SKIPPING")
        return

    output_best_results_filename = get_filename_output_for_best_results(search_type, dataset_name, n_iter, search_max_epoch, best_result_max_epoch)

    df_search = pd.read_csv(search_filename)

    if df_search.empty or len(df_search) < 1:
        print(f"{search_filename} is empty or does not have enough data, SKIPPING")
        return

    train_accuracy = df_search['train_accuracy'].iloc[0] if 'train_accuracy' in df_search.columns else None

    if os.path.exists(output_best_results_filename):
        df_results = pd.read_csv(output_best_results_filename)

        if not df_results.empty and dataset_name in df_results["dataset_name"].values:
            dataset_index = df_results[df_results["dataset_name"] == dataset_name].index[0]
            if train_accuracy is not None:
                df_results.at[dataset_index, "train_accuracy"] = train_accuracy
            df_results["train_accuracy"] = df_results["train_accuracy"].astype(float).map('{:.5f}'.format)
            df_results["test_accuracy"] = df_results["test_accuracy"].astype(float).map('{:.5f}'.format)

            df_results.to_csv(output_best_results_filename, index=False)
            print(f"Updated existing results for {dataset_name} in {output_best_results_filename}")
            return

    df = pd.read_csv(search_filename)
    print(f"RUNNING for {dataset_name}")
    bag_size = int(df["bag_size"][0])
    n_prototypes = df["n_prototypes"][0]
    lambda_fused_lasso = df["lambda_fused_lasso"][0]
    features_to_use_str = df["features_to_use"].iloc[0]

    test_accuracy = evaluate_model_performance(
        dataset_name=dataset_name,
        bag_size=bag_size,
        n_prototypes=n_prototypes,
        max_epoch=best_result_max_epoch,
        phase="test",
        features_to_use_str=features_to_use_str
    )

    print(dataset_name, train_accuracy, test_accuracy)

    d = {
        "dataset_name": dataset_name,
        "bag_size": bag_size,
        "n_prototypes": n_prototypes,
        "stride_ratio": STRIDE_RATIO,
        "max_epoch": best_result_max_epoch,
        "features_to_use_str": features_to_use_str,
        "lambda_fused_lasso": lambda_fused_lasso,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    }

    # Ensure the directory exists
    if not os.path.exists('Results'):
        os.makedirs('Results')

    if os.path.exists(output_best_results_filename):
        df_results = pd.read_csv(output_best_results_filename)
    else:
        df_results = pd.DataFrame()

    df_results = pd.concat([df_results, pd.DataFrame([d])], ignore_index=True)
    df_results["train_accuracy"] = df_results["train_accuracy"].astype(float).map('{:.5f}'.format)
    df_results["test_accuracy"] = df_results["test_accuracy"].astype(float).map('{:.5f}'.format)

    df_results.to_csv(output_best_results_filename, index=False)

# Helper functions to get filenames
def get_filename_output_for_search(search_type, dataset_name, n_iter, max_epoch):
    """
    Generates the filename for search results.

    Parameters:
    - search_type (str): The type of search (e.g., HyperoptSearch).
    - dataset_name (str): The name of the dataset.
    - n_iter (int): The number of iterations for the search.
    - max_epoch (int): The maximum number of epochs for the search.

    Returns:
    - str: The filename for search results.
    """
    return f"Results/{search_type}_params_{dataset_name}_n_iter_{n_iter}_max_epoch_{max_epoch}.csv"

def get_filename_output_for_best_results(search_type, dataset_name, n_iter, search_max_epoch, best_result_max_epoch):
    """
    Generates the filename for the best results.

    Parameters:
    - search_type (str): The type of search (e.g., HyperoptSearch).
    - dataset_name (str): The name of the dataset.
    - n_iter (int): The number of iterations for the search.
    - search_max_epoch (int): The maximum number of epochs for the search.
    - best_result_max_epoch (int): The maximum number of epochs for the best result.

    Returns:
    - str: The filename for the best results.
    """
    return f"Results/Test_results_according_to_{search_type}_n_iter_{n_iter}_search_max_epoch_{search_max_epoch}_result_max_epoch_{best_result_max_epoch}.csv"

dispatcher = {"HyperoptSearchTPE": find_best_hyper_params_hyperopt_search_tpe}

def find_result_for_one(search_type, dataset_name, n_iter=100, search_max_epoch=1000, features_to_use_str=None):
    """
    Finds the result for one dataset.

    Parameters:
    - search_type (str): The type of search (e.g., HyperoptSearch).
    - dataset_name (str): The name of the dataset.
    - n_iter (int): The number of iterations for the search.
    - search_max_epoch (int): The maximum number of epochs for the search.
    - features_to_use_str (str): The features to use for the model.

    Returns:
    - None
    """
    output_filename = get_filename_output_for_search(search_type, dataset_name, n_iter, search_max_epoch)

    if os.path.isfile(output_filename):
        print(f"dataset: {dataset_name} exists in {output_filename}")
    else:
        print(f"RUNNING for dataset: {dataset_name}")
        result_dict = dispatcher[search_type](dataset_name, n_iter=n_iter, search_max_epoch=search_max_epoch, features_to_use_str=features_to_use_str)
        result_dict["search_type"] = search_type

        df = pd.DataFrame(result_dict, index=[0])
        df.to_csv(output_filename, index=False)
        print(f"Saved dataset: {dataset_name} in {output_filename}")
