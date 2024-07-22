"""
feature_selection.py

This module contains functions for evaluating model performance and performing recursive feature elimination (RFE).
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from helper_datasets import load_dataset
from shapelet_regularization import ShapeletRegularizedNet
from shapelet_generation import ShapeletGeneration as ShapeletNN

def evaluate_model_performance(dataset_name, bag_size, n_prototypes, max_epoch, features_to_use_str, lambda_fused_lasso):
    """
    Evaluates the model performance on the test set.

    Parameters:
    - dataset_name (str): The name of the dataset.
    - bag_size (int): The bag size.
    - n_prototypes (int): The number of prototypes.
    - max_epoch (int): The maximum number of epochs.
    - features_to_use_str (str): The features to use for the model.
    - lambda_fused_lasso (float): The regularization parameter for fused lasso.

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
        stride_ratio=0.10,
        features_to_use_str=features_to_use_str,
        lambda_fused_lasso=lambda_fused_lasso
    )

    net = ShapeletRegularizedNet(
        module=nn_shapelet_generator,
        max_epochs=max_epoch,
        lr=0.0001,  # Decreased learning rate
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        callbacks=[]
    )

    net.fit(train, y_train_labels)
    y_test_labels = np.argmax(y_test, axis=1)
    y_predict = net.predict(test)
    y_predict_labels = np.argmax(y_predict, axis=1) if y_predict.ndim > 1 else y_predict
    return accuracy_score(y_test_labels, y_predict_labels)

def rfe_feature_selection(dataset_name, initial_features, bag_size, n_prototypes, lambda_fused_lasso, n_iter=10, search_max_epoch=100):
    """
    Performs recursive feature elimination (RFE) for feature selection.

    Parameters:
    - dataset_name (str): The name of the dataset.
    - initial_features (str): The initial set of features to use.
    - bag_size (int): The bag size.
    - n_prototypes (int): The number of prototypes.
    - lambda_fused_lasso (float): The regularization parameter for fused lasso.
    - n_iter (int): The number of iterations for the feature selection.
    - search_max_epoch (int): The maximum number of epochs for the search.

    Returns:
    - tuple: The best set of features and the best accuracy score.
    """
    features = initial_features.split(',')
    best_features = features[:]
    best_accuracy = 0

    for feature in features:
        features_to_use = ','.join([f for f in best_features if f != feature])
        val_accuracy = evaluate_model_performance(
            dataset_name=dataset_name,
            bag_size=bag_size,
            n_prototypes=n_prototypes,
            max_epoch=search_max_epoch,
            features_to_use_str=features_to_use,
            lambda_fused_lasso=lambda_fused_lasso
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_features = [f for f in best_features if f != feature]

    return best_features, best_accuracy
