"""
shapelet_generation.py

This module defines the ShapeletGeneration3LN neural network model with shapelet-based learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from helper_datasets import convert_to_bags

class ShapeletGeneration(nn.Module):
    def __init__(self,
                 n_prototypes,
                 bag_size,
                 n_classes,
                 stride_ratio,
                 features_to_use_str,
                 lambda_prototypes=None,
                 lambda_linear_params=None,
                 lambda_fused_lasso=None,
                 dropout_rate=0.50,
                 dataset_name=None):
        super(ShapeletGeneration, self).__init__()
        features_to_use = features_to_use_str.split(",")
        self.prototypes = nn.Parameter((torch.randn((1, n_prototypes, bag_size)) * 0.5))
        self.n_p = n_prototypes
        self.bag_size = bag_size
        self.N = n_classes
        self.stride_ratio = stride_ratio
        self.features_to_use = features_to_use
        self.lambda_prototypes = lambda_prototypes
        self.lambda_fused_lasso = lambda_fused_lasso
        self.lambda_linear_params = lambda_linear_params
        self.dropout_rate = dropout_rate

        input_size = len(self.features_to_use) * n_prototypes
        hidden_size = int(input_size * 1.5)

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.selu1 = nn.SELU()
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)  # Add Batch Normalization
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.selu2 = nn.SELU()
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)  # Add Batch Normalization
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_size, n_classes)

    def pairwise_distances(self, x, y):
        """
        Computes pairwise distances between x and y.

        Parameters:
        - x (torch.Tensor): The first set of samples.
        - y (torch.Tensor): The second set of samples.

        Returns:
        - torch.Tensor: The pairwise distances.
        """
        x_norm = (x.norm(dim=2)[:, :, None]).float()
        y_t = y.permute(0, 2, 1).contiguous()
        y_norm = (y.norm(dim=2)[:, None])
        y_t = torch.cat([y_t] * x.shape[0], dim=0)
        dist = x_norm + y_norm - 2.0 * torch.bmm(x.float(), y_t)
        return torch.clamp(dist, 0.0, np.inf)

    def cosine_similarity(self, x, y):
        """
        Computes cosine similarity between x and y.

        Parameters:
        - x (torch.Tensor): The first set of samples.
        - y (torch.Tensor): The second set of samples.

        Returns:
        - torch.Tensor: The cosine similarity.
        """
        x_normalized = F.normalize(x, p=2, dim=2)
        y_normalized = F.normalize(y, p=2, dim=2)
        y_t = y_normalized.permute(0, 2, 1).contiguous()
        y_t = torch.cat([y_t] * x.shape[0], dim=0)
        cos_sim = torch.bmm(x_normalized, y_t)
        return cos_sim

    def layer_norm(self, feature):
        """
        Applies layer normalization to the input feature.

        Parameters:
        - feature (torch.Tensor): The input feature.

        Returns:
        - torch.Tensor: The normalized feature.
        """
        mean = feature.mean(keepdim=True, dim=-1)
        var = ((feature - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + 1e-5).sqrt()
        y = (feature - mean) / std
        return y

    def get_output_from_prototypes(self, batch_inp):
        """
        Extracts features from prototypes based on input batch.

        Parameters:
        - batch_inp (torch.Tensor): The input batch.

        Returns:
        - torch.Tensor: The extracted features.
        """
        dist = self.pairwise_distances(batch_inp, self.prototypes)
        cos_sim = self.cosine_similarity(batch_inp, self.prototypes)
        l_features = []
        if "min" in self.features_to_use:
            min_dist = self.layer_norm(dist.min(dim=1)[0])
            l_features.append(min_dist)
        if "max" in self.features_to_use:
            max_dist = self.layer_norm(dist.max(dim=1)[0])
            l_features.append(max_dist)
        if "mean" in self.features_to_use:
            mean_dist = self.layer_norm(dist.mean(dim=1))
            l_features.append(mean_dist)
        if "cos" in self.features_to_use:
            cos_sim_feature = self.layer_norm(cos_sim.max(dim=1)[0])
            l_features.append(cos_sim_feature)
        all_features = torch.cat(l_features, dim=1)
        return all_features

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        - x (torch.Tensor): The input data.

        Returns:
        - torch.Tensor: The model output.
        """
        x_bags = convert_to_bags(x, self.bag_size, self.stride_ratio)
        all_features = self.get_output_from_prototypes(x_bags)
        out = self.linear1(all_features)
        out = self.selu1(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.selu2(out)
        out = self.dropout2(out)
        out = self.output_layer(out)
        return out
