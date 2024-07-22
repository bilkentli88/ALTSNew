"""
shapelet_regularization.py

This module defines a custom Skorch NeuralNetClassifier with regularization.
"""

import torch
from skorch import NeuralNetClassifier

class ShapeletRegularizedNet(NeuralNetClassifier):
    def __init__(self, *args,
                 lambda_prototypes=0.1,
                 lambda_linear_params=0.05,
                 lambda_fused_lasso=0.01,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_prototypes = lambda_prototypes
        self.lambda_linear_params = lambda_linear_params
        self.lambda_fused_lasso = lambda_fused_lasso

    def get_loss(self, y_pred, y_true, X=None, training=False):
        """
        Calculates the loss function including the regularization terms.

        Parameters:
        - y_pred (torch.Tensor): The predicted output.
        - y_true (torch.Tensor): The true labels.
        - X (torch.Tensor): The input data (default: None).
        - training (bool): Whether the model is in training mode (default: False).

        Returns:
        - torch.Tensor: The calculated loss.
        """
        # Calculate the primary loss using the parent class method
        loss_softmax = super().get_loss(y_pred, y_true, X=X, training=training)

        # L2 regularization on prototypes
        loss_prototypes = torch.norm(self.module_.prototypes, p=2)

        # L2 regularization on linear layer parameters
        loss_weight_reg = torch.tensor(0.0)  # Initialize as a tensor to ensure compatibility with computation graph
        for param in self.module_.linear1.parameters():
            loss_weight_reg += param.norm(p=2).sum()

        # Fused Lasso regularization (total variation regularization)
        fused_lasso_reg = torch.sum(torch.abs(self.module_.prototypes[:, 1:] - self.module_.prototypes[:, :-1]))

        # Combine all loss components
        total_loss = (loss_softmax +
                      self.lambda_prototypes * loss_prototypes +
                      self.lambda_linear_params * loss_weight_reg +
                      self.lambda_fused_lasso * fused_lasso_reg)

        return total_loss
