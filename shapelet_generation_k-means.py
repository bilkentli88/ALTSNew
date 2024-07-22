# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 01:09:11 2024

@author: tayip
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from helper_datasets import convert_to_bags, load_dataset

class ShapeletGeneration3LN(nn.Module):
    def __init__(self,
                 n_prototypes,
                 bag_size,
                 n_classes,
                 stride_ratio,
                 features_to_use_str,
                 lambda_prototypes=None
                 lambda_linear_params=None,
                 lambda_fused_lasso=None,
                 dropout_rate=0.50,
                 dataset_name=None):
        super(ShapeletGeneration3LN, self).__init__()

        self.n_prototypes = n_prototypes
        self.bag_size = bag_size
        self.n_classes = n_classes
        self.stride_ratio = stride_ratio
        self.features_to_use = features_to_use_str.split(",")
        self.lambda_prototypes = lambda_prototypes
        self.lambda_fused_lasso = lambda_fused_lasso
        self.lambda_linear_params = lambda_linear_params
        self.dropout_rate = dropout_rate

        self.prototypes = self.initialize_prototypes_kmeans(dataset_name).requires_grad_()
        #self.prototypes = nn.Parameter(
            #self.initialize_prototypes_kmeans(n_prototypes, bag_size, stride_ratio, dataset_name))
        input_size = len(self.features_to_use) * n_prototypes
        hidden_size = int(input_size * 2.5)

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.selu1 = nn.SELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.selu2 = nn.SELU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_size, n_classes)

    def initialize_prototypes_kmeans(self, dataset_name):
        if not isinstance(dataset_name, str):
            raise ValueError("dataset_name must be a string")

        X, _, _, _ = load_dataset(dataset_name)
        X_bags = convert_to_bags(X, self.bag_size, self.stride_ratio)
        X_bags_flattened = X_bags.reshape(X_bags.shape[0] * X_bags.shape[1], -1)

        kmeans = KMeans(n_clusters=self.n_prototypes, random_state=79, n_init=10)
        kmeans.fit(X_bags_flattened)

        centroids = kmeans.cluster_centers_
        prototypes = centroids.reshape(1, self.n_prototypes, self.bag_size)

        return torch.tensor(prototypes, dtype=torch.float32)

    def pairwise_distances(self, x, y):
        x_norm = x.norm(dim=2)[:, :, None].float()
        y_t = y.permute(0, 2, 1).contiguous()
        y_norm = y.norm(dim=2)[:, None]
        y_t = torch.cat([y_t] * x.shape[0], dim=0)
        dist = x_norm + y_norm - 2.0 * torch.bmm(x.float(), y_t)
        return torch.clamp(dist, 0.0, np.inf)

    def layer_norm(self, feature):
        mean = feature.mean(keepdim=True, dim=-1)
        std = feature.std(keepdim=True, dim=-1)
        return (feature - mean) / (std + 1e-5)

    def get_output_from_prototypes(self, batch_inp):
        dist = self.pairwise_distances(batch_inp, self.prototypes)
        features = []

        if "min" in self.features_to_use:
            features.append(self.layer_norm(dist.min(dim=1)[0]))
        if "max" in self.features_to_use:
            features.append(self.layer_norm(dist.max(dim=1)[0]))
        if "mean" in self.features_to_use:
            features.append(self.layer_norm(dist.mean(dim=1)))
        if "std" in self.features_to_use:
            features.append(self.layer_norm(dist.std(dim=1)))

        return torch.cat(features, dim=1)

    def forward(self, x):
        x_bags = convert_to_bags(x, self.bag_size, self.stride_ratio)
        features = self.get_output_from_prototypes(x_bags)

        out = self.linear1(features)
        out = self.selu1(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.selu2(out)
        out = self.dropout2(out)
        out = self.output_layer(out)

        return out
