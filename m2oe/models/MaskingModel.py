from math import ceil

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.neighbors import NearestNeighbors

import numpy as np


class MaskingModel(keras.Model):
    """
    This Keras Model subclass provides an abstract implementation of the neural masking model that learn feature *choice* and *mask* to transform an outlier into a counterfactual close to normals.

    References:
    - Angiulli, F., Fassetti, F., Nisticò, S., & Palopoli, L. (2025). Explaining evolving outliers for uncovering key aspects of the green comparative advantage. Array, 100518.
    - Angiulli, F., Fassetti, F., Nisticò, S., & Palopoli, L. (2024). Explaining outliers and anomalous groups via subspace density contrastive loss. Machine Learning, 113(10), 7565-7589.
    - Angiulli, F., Fassetti, F., Nisticó, S., & Palopoli, L. (2023, October). Counterfactuals explanations for outliers via subspaces density contrastive loss. In International Conference on Discovery Science (pp. 159-173). Cham: Springer Nature Switzerland.
    
    Parameters
    ----------
    loss_weights : Sequence[float], default=[1.0, 1.2, 0.3]
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
    """

    def __init__(self, loss_weights, lr, epochs, bs):
        super(MaskingModel, self).__init__()
        self.MASK = None
        self.CHOOSE = None
        self.MASKAPPLY = None
        
        
        self.loss_weights = loss_weights
        self.lr = lr
        self.epochs = epochs
        self.bs = bs

    def defineMaskGen(self, in_shape):
        """
        Build the **mask** and **choice** generator sub-networks.
        Sets the `self.MASK` and `self.CHOOSE` neural sub-modules (Keras `Model`s)
            with signatures roughly:
                - `MASK([O, R]) -> mask` of shape `(B, D)`
                - `CHOOSE([O, R]) -> choice` of shape `(B, D)` with values in `[0, 1]`.

        Parameters
        ----------
        in_shape : int or Tuple[int]
            Input feature dimensionality `D` (an integer) or a shape tuple that can be
            resolved to `D`.

        """
        raise NotImplementedError('subclasses must override defineMaskGen!')

    def defineMaskApply(self, in_shape):
        """
        Build the **mask applier** sub-network that composes input, mask and choice.
        Sets attribute `self.MASKAPPLY` (Keras `Model`) with signature:
                - `MASKAPPLY([O, mask, choice]) -> O'` where all tensors are `(B, D)`.
                  

        Parameters
        ----------
        in_shape : int or Tuple[int]
            Input feature dimensionality `D`.

        """

        raise NotImplementedError('subclasses must override defineMaskApply!')
        

    def test(self, id, classes, train_images, train_labels, drawplot=True):
        raise NotImplementedError('subclasses must override test!')
        
         
        
        

