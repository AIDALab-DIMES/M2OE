from math import ceil

from tensorflow.python.keras.layers import ReLU, BatchNormalization

from .AETabularMM_SC import TabularMM_SC
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Add, Multiply, Activation, Concatenate


import numpy as np


class TabularMM_Pref(TabularMM_SC):
    """
    This module provides an implementation of the neural masking model for a single tabular outlier. It learns feature *choice* and *mask* to transform an outlier into a counterfactual close to normals. This variant, tailored to outliers evolving over time, takes into account information from past explanation steps.

    Parameters
    ----------
    preferences : np.ndarray or Sequence[float]
        Per-feature preference weights `(D,)` multiplied with the sparsity term;
        values `< 1` *lower* the penalty (making features more likely to be re-selected).
    normal_data : np.ndarray
        Normal data `(N, D)` to compute `normal_dist` `(D,)`.
    loss_weights : Sequence[float], optional, default=[1.0, 1.2, 0.3]
        Three non-negative weights `[alpha1, alpha2, alpha3]` controlling, respectively,
        (i) subspace contrast, (ii) patch proximity, and (iii) sparsity of the choice.
    lr : float, optional, default=0.001
        Learning rate.
    epochs : int, optional, default=30
        Training epochs used by explainers.
    bs : int, optional, default=16
        Batch size used by explainers.
        
    Attributes
    ----------
    MASK: keras.Module
        Neural module advocated to single out the features to include in the explanation.
    CHOOSE: keras.Module
        Neural module advocated to find the modification to apply to sample's features.
    MASKAPPLY: keras.Module
        Sub-network that composes input, mask and choice.
    normal_dist: ndarray of shape (n_features,)
        Difference statistic vector storing the average feature-wise difference of provided normal samples.

    """

    def __init__(self, preferences, normal_data, loss_weights=[1.0, 1.2, 0.3], lr=0.001, epochs=30, bs=16):
        super(TabularMM_Pref, self).__init__(normal_data, loss_weights, lr, epochs, bs)

        self.preferences = preferences
        

    def compute_loss(self, data, patches, mask, choose):
        """
        Compute the composite loss for the masking/choice model.

        This loss combines three per-sample terms:
        1) a proximity term that pulls the patched outlier toward its reference in the selected subspace, 
        2) a contrastive term that favors subspaces where the outlier deviates from normals (using `self.normal_dist`)
        3) a sparsity term on the choice vector to prefer compact explanations, a preference weight promotes feature re-selection across different snapshots.

        Parameters
        ----------
        data : Sequence[tf.Tensor] or Tuple[tf.Tensor, tf.Tensor]
            Pair `[data_o, data_i]` where:
              - `data_o` : tf.Tensor, shape `(B, D)`
                    Batch of outlier samples **O**.
              - `data_i` : tf.Tensor, shape `(B, D)`
                    Batch of reference/normal samples **R** (e.g., kNN of **O**).
        patches : tf.Tensor
            Patched/counterfactual samples **O'**, shape `(B, D)`, produced by
            the mask applier.
        mask : tf.Tensor
            Real-valued mask magnitudes, shape `(B, D)`.
        choose : tf.Tensor
            Real-valued per-feature selectors in `[0, 1]`, shape `(B, D)`. Acts as a
            soft/binary **choice** of dimensions composing the explanation subspace.

        Returns
        -------
        tf.Tensor
            Scalar loss (0-D tensor): `mean( α1 * margin_n + α2 * sample_distance + α3 * ndim_loss )`.

        Notes
        -----
        Let `Δ = R − O` (element-wise), `D` the feature count, and `α = self.loss_weights`.

        Per-sample components (all shape `(B,)`):
        - **Sparsity / dimensionality**:
            `ndim_loss = ||choose * w||₂ = sqrt( sum_j choose[:, j]^2*self.preferences[j] )`
        - **Proximity in the chosen subspace** (weighted L2, normalized by `sqrt(D)`):
            `margin_n = sqrt( sum_j ((O' − R)[:, j]^2 * choose[:, j]) ) / sqrt(D)`
        - **Contrast vs. normals** (favoring subspaces with larger normal dispersion):
            `differences_red = sum_j (Δ[:, j]^2 * choose[:, j]^2)`
            `normal_dist = sqrt( sum_j (self.normal_dist[j] * choose[:, j]) )`
            `sample_distance = normal_dist / (differences_red + 1e-4)`

        Final scalar loss:
        `loss = mean( α[0] * margin_n + α[1] * sample_distance + α[2] * ndim_loss )`

        References
        ----------
        - Angiulli, F., Fassetti, F., Nisticò, S., & Palopoli, L. (2025). Explaining evolving outliers for uncovering key aspects of the green comparative advantage. Array, 100518.
        """
        
        data_o = data[0]
        data_i = data[1]
        
        ndim_loss = tf.sqrt(tf.reduce_sum(self.preferences * (choose ** 2), axis=1))
        margin_n = tf.sqrt(tf.reduce_sum(((patches - data_i) ** 2) * choose, axis=1)) / np.sqrt(data_o.shape[1]) 
        differences = (- data_o + data_i)
        differences_red = tf.reduce_sum((differences ** 2) * (choose ** 2), axis=1)
        normal_dist = tf.sqrt(tf.reduce_sum(self.normal_dist * (choose), axis=1))
        sample_distance = normal_dist / (differences_red + 1e-4)
        
        loss = tf.reduce_mean(self.loss_weights[0] * margin_n +
                              self.loss_weights[1] * sample_distance +
                              self.loss_weights[2] * ndim_loss)

        return loss
    
   
    
