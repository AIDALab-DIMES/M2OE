from math import ceil

from tensorflow.python.keras.layers import ReLU, BatchNormalization

from .MaskingModel import MaskingModel
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Add, Multiply, Activation, Concatenate


import numpy as np


class TabularMM_SC(MaskingModel):
    """
    This module provides an implementation of the neural masking model for a single tabular outlier. It learns an unique feature *choice* and *mask* to transform an outlier into a counterfactual close to normals.
    The choice network produces a **batch-shared** per-feature vector which is broadcast to `(B, D)`.

    Parameters
    ----------
    normal_data : np.ndarray
        Normal/reference data `(N, D)` to compute `normal_dist` `(D,)`.
    loss_weights : Sequence[float], optional
        Loss weights `[alpha1, alpha2, alpha3]`.
    lr : float, optional
        Learning rate.
    epochs : int, optional
        Training epochs used by explainers.
    bs : int, optional
        Batch size used by explainers.
    """

    def __init__(self, normal_data, loss_weights=[1.0, 1.2, 0.3], lr=0.001, epochs=30, bs=16):
        super(TabularMM_SC, self).__init__(loss_weights, lr, epochs, bs)
        self.in_shape = normal_data.shape[1]
        
        differences = (- normal_data[:, np.newaxis, :] + normal_data[np.newaxis, :, :])**2
        self.normal_dist = differences.sum(axis=1) / (differences.shape[1]-1)
        self.normal_dist = self.normal_dist.mean(axis=0)
        
        # Build the network
        self.defineMaskGen(self.in_shape)
        self.defineMaskApply(self.in_shape)
            

    def call(self, inputs, training=None, mask=None):
        """
        Forward pass producing counterfactual patches, raw masks, and choices. Overrides the Keras `Model`s `call` function.

        Parameters
        ----------
        inputs : Sequence[tf.Tensor]
            `[O, R]` with shapes `(B, D)` each (outliers, references).

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            `(patches, masks, choices)` each `(B, D)`.
        """
        ones_input = np.ones_like(inputs[0])
        masks = self.MASK(inputs)
        choose = self.CHOOSE(ones_input)
        patches = self.MASKAPPLY([inputs[0], masks, choose])
        return patches, mask, choose


    def defineMaskGen(self, in_shape):
        """
        Build the **mask** and **choice** generator sub-networks.
        Sets the `self.MASK` and `self.CHOOSE` neural sub-modules (Keras `Model`s) with signatures roughly:
        - `MASK([O, R]) -> mask` of shape `(B, D)`
        - `CHOOSE([O, R]) -> choice` of shape `(B, D)` with values in `[0, 1]`. An unique choice is produced for all the samples,  which is broadcast to `(B, D)`.

        Parameters
        ----------
        in_shape : int or Tuple[int]
            Input feature dimensionality `D` (an integer) or a shape tuple that can be
            resolved to `D`.
        """
        num_unit = in_shape * 4

        input_o = Input(in_shape)
        input_i = Input(in_shape)
        inputs = Concatenate()([input_o, input_i])
        
        input_c = Input(in_shape)

        x1 = Dense(num_unit, activation='relu')(input_c)
        x1 = Dense(num_unit, activation='relu')(x1)
        outputs_c = Dense(in_shape, activation='sigmoid')(x1)
        self.CHOOSE = keras.Model(inputs=input_c, outputs=outputs_c, name='CHOOSE')

        x0 = Dense(num_unit)(inputs)
        x0 = Dense(num_unit)(x0)
        outputs = Dense(in_shape)(x0)
        self.MASK = keras.Model(inputs=[input_o, input_i], outputs=outputs, name='MASK')

        return

    
    def defineMaskApply(self, in_shape):
        """
        Build the mask applier network.
        All tensors are `(B, D)` and `O' = O + choice ⊙ mask` (⊙ element-wise product).

        Parameters
        ----------
        in_shape : int or Tuple[int]
            Input dimensionality `D`.           
        """
        inputs = [Input(in_shape, name='input_img'), Input(in_shape, name='input_mask'),
                  Input(in_shape, name='input_choice')] 
        mid_output = Multiply()([inputs[1], inputs[2]])

        outputs = Add()([inputs[0], mid_output])
        self.MASKAPPLY = keras.Model(inputs=inputs, outputs=outputs)

        return
    
    
    def loss_fn(self, data):
        """
        Proxy function that compute the transformation relating to the input data and subsequently compute the loss score.
        
        Parameters
        ----------
        data: Sequence[tf.Tensor]
            `[O, R]` with shapes `(B, D)` each (outliers, references).
        
        Returns
        -------
        int
            loss function value.
        """
        patches, mask, choose = self(data)
        return self.compute_loss(data, patches, mask, choose).numpy()

    
    def compute_loss(self, data, patches, mask, choose):
        """
        Compute the composite loss for the masking/choice model.

        This loss combines three per-sample terms:
        1) a proximity term that pulls the patched outlier toward its reference in the selected subspace, 
        2) a contrastive term that favors subspaces where the outlier deviates from normals (using `self.normal_dist`), 
        3) a sparsity term on the choice vector to prefer compact explanations.

        Parameters
        ----------
        data : Sequence[tf.Tensor] or Tuple[tf.Tensor, tf.Tensor]
            Pair `[data_o, data_i]` where:
              - `data_o` : tf.Tensor, shape `(B, D)`
                    Batch of outlier samples **O**.
              - `data_i` : tf.Tensor, shape `(B, D)`
                    Batch of reference/normal samples **R** (e.g., kNN of **O**).
        patches : tf.Tensor
            Patched/counterfactual samples **O'**, shape `(B, D)`, produced by the mask applier.
        mask : tf.Tensor
            Real-valued mask magnitudes, shape `(B, D)`.
        choose : tf.Tensor
            Real-valued per-feature selectors in `[0, 1]`, shape `(B, D)`. Acts as a soft/binary **choice** of dimensions composing the explanation subspace.

        Returns
        -------
        tf.Tensor
            Scalar loss (0-D tensor): `mean( α1 * margin_n + α2 * sample_distance + α3 * ndim_loss )`.

        Notes
        -----
        Let `Δ = R − O` (element-wise), `D` the feature count, and `α = self.loss_weights`.

        Per-sample components (all shape `(B,)`):
        - **Sparsity / dimensionality**
        - **Proximity in the chosen subspace** (weighted L2, normalized by `sqrt(D)`)
        - **Contrast vs. normal samples** (favoring subspaces with larger normal dispersion)

        Final scalar loss:
        `loss = mean( α[0] * margin_n + α[1] * sample_distance + α[2] * ndim_loss )`

        References
        ----------
        - Angiulli, Fassetti, Nisticò, Palopoli (2024). *Explaining outliers and anomalous groups via
          subspace density contrastive loss*, Machine Learning. https://doi.org/10.1007/s10994-024-06618-8
        - Angiulli, Fassetti, Nisticò, Palopoli (2025). *Explaining evolving outliers for uncovering key
          aspects of the green comparative advantage*, Array. https://doi.org/10.1016/j.array.2025.100518
        """
        data_o = data[0]
        data_i = data[1]
        
        ndim_loss = tf.sqrt(tf.reduce_sum(choose ** 2, axis=1))
        margin_n = tf.sqrt(tf.reduce_sum(((patches - data_i) ** 2) * choose, axis=1)) / np.sqrt(data_o.shape[1]) 
        differences = (- data_o + data_i)
        differences_red = tf.reduce_sum((differences ** 2) * (choose ** 2), axis=1)
        normal_dist = tf.sqrt(tf.reduce_sum(self.normal_dist * (choose), axis=1))
        sample_distance = normal_dist / (differences_red + 1e-4)
        
        loss = tf.reduce_mean(self.loss_weights[0] * margin_n +
                              self.loss_weights[1] * sample_distance +
                              self.loss_weights[2] * ndim_loss)
        
        return loss
    
    
    def train_step(self, data):
        """
        """
        x, y = data
        
        with tf.GradientTape() as tape:
             
            patches, mask, choose = self(x)
            loss = self.compute_loss(x, patches, mask, choose)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(y, patches)
        return {m.name: m.result() for m in self.metrics}

                                
        
    