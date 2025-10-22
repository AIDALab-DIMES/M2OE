from math import ceil

from tensorflow.python.keras.layers import ReLU, BatchNormalization

from .Explainer import Explainer
from .Explainer import _get_medoid
from ..models.AETabularMM_SC import TabularMM_SC

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Add, Multiply, Activation, Concatenate
import numpy as np

from sklearn.neighbors import NearestNeighbors
from mlxtend.frequent_patterns import fpmax
from sklearn.cluster import DBSCAN

import pandas as pd
import numpy as np
from tqdm import tqdm



class TabularGroupExplainer(Explainer):
    """
    Group explainer for multiple tabular outliers (shared-choice explanations).

    This explainer discovers compact, *shared* subspaces that jointly explain a set
    of outliers. It wraps a shared-choice masking model (`TabularMM_SC`) so that all
    samples in a group use the **same per-feature choice vector**, while keeping the
    **mask** (magnitude of change) sample-conditioned. At a high level, it:

    1) builds a **reference set** `RS` for each outlier,
    2) trains base shared-choice models and evaluates **cross-loss** to find
       mutually-explainable outliers,
    3) **merges** the most compatible outliers into groups and retrains a shared
       model per group,
    4) mines clusters `RS[:, dims]` with `cl_algo` (default: DBSCAN) to identify 
       distinct local explanations,
    5) for each cluster medoid, generates a **counterfactual patch** by applying the
       learned mask only on the shared chosen set of features.

    Parameters
    ----------
    normal_data : np.ndarray
        Normal data of shape `(N, D)` used for the explanation.
    loss_weights : Sequence[float], optional, default=[1.0, 1.0, 0.5]
        Three non-negative loss weights `[alpha1, alpha2, alpha3]` for: subspace
        contrast, patch proximity, and choice sparsity.
    lr : float, optional, default=1e-3
        Learning rate for the optimizer used when fitting the masking model(s).
    epochs : int, optional, default= 30
        Number of training epochs for each (re)trained group model.
    bs : int, optional, default= 16
        Mini-batch size used during training.
    rs_selector : type or object, optional (default: `sklearn.neighbors.NearestNeighbors`)
        samples selector used to build reference sets. Must expose `fit`/`kneighbors`.
        If a **type** is provided, it is instantiated internally.
    threshold : float, optional, default= 0.5
        Binarization threshold in `[0, 1]` applied to the choice output.
    cl_algo : sklearn.cluster, optional, default= `DBSCAN(eps=0.10)`
        Clustering algorithm used on `RS[:, dims]` for each discovered subspace.
        Must implement `fit_predict`.

    Attributes
    ----------
    in_shape : int
        Feature dimensionality `D` inferred from `normal_data.shape[1]`.
    loss_weights : Sequence[float]
        Weights used by all group models.
    lr, epochs, bs : Union[float, int]
        Optimizer and training hyperparameters used during (re)training.
    threshold : float
        Effective choice threshold used during explanation generation.
    cl_algo : sklearn.cluster
        Clustering algorithm used to obtain medoid-based counterfactuals.
    rs_selector : object
        Fitted selector used to retrieve reference sets.

    See Also
    --------
    TabularExplainer
        Single-sample explainer (no sharing).
    TabularSequentialExplainer
        Preference-guided explainer for **evolving** outliers.
    TabularMM_SC
        Shared-choice masking model used internally for group explanations.

    References
    ----------
    - Angiulli, Fassetti, Nisticò, Palopoli (2024). *Explaining outliers and
      anomalous groups via subspace density contrastive loss*, Machine Learning.
      https://doi.org/10.1007/s10994-024-06618-8
    """

    def __init__(self, loss_weights=[1.0, 1.0, 0.5], lr=0.001, epochs=30, bs=16, rs_selector=NearestNeighbors, threshold=0.5, cl_algo=DBSCAN(eps=0.10)): 
        super(TabularGroupExplainer, self).__init__(rs_selector=NearestNeighbors, threshold=threshold)
        self.in_shape = None
        
        self.loss_weights = loss_weights
        self.lr = lr
        self.epochs = epochs
        self.bs = bs
        
        self.explainers = []
        self.outliers = {}
        self.choices = []
        self.exps = []
        self.rs = {}
        self.exp_id = {}
        
        self._couples = None
        self.out = None
        self.cl_algo = cl_algo

    
        
    def _explain(self, exp_nn, sample, threshold=0.5, binary=True):
        """
        Generate the per-sample **choice** and corresponding **counterfactual patch**.

        This routine queries the trained masking model to obtain:
        (1) a per-feature **choice** vector (soft or binarized), and
        (2) the **patched** sample obtained by applying the learned mask only on the
            selected features of the outlier.

        Parameters
        ----------
        sample : Sequence[tf.Tensor] | Tuple[tf.Tensor, tf.Tensor]
            Pair `[O, R]` where:
              - `O` : outlier sample, shape `(B, D)`.
              - `R` : reference/normal sample matched to `O`, shape `(B, D)`.
            The underlying model expects both to have identical shapes.
        threshold : float, optional, default=0.5
            Threshold in `[0, 1]` used to filter the choice output.
        binary : bool, optional, default=True
            If `True`, the choice is binarized to `{0, 1}`.
            If `False`, the choice remains **soft** in `[0, 1]`.

        Returns
        -------
        patches : np.ndarray
            Counterfactual sample(s) `O'` of shape `(B, D)` created by applying the mask
            in the chosen subspace, i.e., `O' = O + (choice ⊙ mask)`.
        choose : np.ndarray
            Per-feature choice vector of shape `(B, D)`:
              - If `binary=True`, entries are in `{0, 1}`.
              - If `binary=False`, entries are in `[0, 1]`.

        """
        ones_input = np.ones_like(sample[0])
        mask = exp_nn.MASK(sample)
        choose = exp_nn.CHOOSE(ones_input)

        if binary:
            choose = np.where(choose.numpy() > threshold, 1, 0)
        else:
            choose = choose.numpy()
            choose = np.where(choose > threshold, choose, 0)
        patches = exp_nn.MASKAPPLY([sample[0], mask, choose])
        choose = np.where(choose > threshold, 1, 0)

        return patches.numpy(), choose
    
    
    def _explanation_generation(self, exp_nn, out, rs, choose):
        
        # Normal samples clusters computation
        labels = self.cl_algo.fit_predict(rs[:, choose])
        
        exps = []
        dims = np.argwhere(choose>0)[0]
        
        # Cluster medoids computation
        if len(set(labels[labels>=0]))>0:
            for j in range(len(set(labels[labels>=0]))):
                mds = _get_medoid(rs[labels==j])
                ptc, _ = self._explain(exp_nn, [out, mds], threshold=self.threshold, binary=False)
                
                
                # Add explanation
                exps.append((dims, ptc))
        else:
            # Only a unique cluster if no clusters found
            mds = _get_medoid(rs)
            ptc, _ = self._explain(exp_nn, [out, mds], threshold=self.threshold, binary=False)
            
            # Add explanation
            exps.append((dims, ptc))
                
        return exps
    
    
    def _single_explanation(self, ds, num_tries, norm_samples):
        t = 0
        choose = []
        
        while len(choose)==0 and t<num_tries:
            # Explanator creation
            exp_nn = TabularMM_SC(norm_samples, self.loss_weights, self.lr, self.epochs, self.bs)

            # Explainer set up and training
            opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
            exp_nn.compile(optimizer=opt, run_eagerly=True)
            exp_nn.fit(ds, ds[1], batch_size=self.bs, epochs=self.epochs, verbose=0)

            # Alternative explanation collection
            _, choose = self._explain(exp_nn, ds, threshold=self.threshold, binary=True)
            choose = choose[0]
            exps = self._explanation_generation(exp_nn, ds[0][:1], ds[1], choose)
        if len(choose)==0:
            raise Exception("Explanation process failed, try with a lower value for alpha3!") 
        
        return exp_nn, choose, exps
    
    
    def compute_explanation(self, out, norm_samples, n_neigh=30, num_groups=1, num_tries=3):
        """
        Compute explanations for a set of outliers by discovering/merging groups that
        share a compact subspace-level explanation.

        Parameters
        ----------
        outs : np.ndarray
            Outliers array `(M, D)`.
        norm_samples : np.ndarray
            Normal samples data used for the explanation `(N, D)`.
        n_neigh : int, optional, default=30
            Reference set size `k` at each snapshot. If None all data samples are considered.
        num_groups : int, optional, default=1
            Target number of groups after merging.
        num_tries : int, optional, default=3
            Number of restarts in case of explanation failure.

        Returns
        -------
        List[List[List[Tuple[np.ndarray, np.ndarray]]]]
            For each resulting group, a list of `(dims, patched)` pair is produced for each outlier belonging to that group:
              - `dims`: 1-D array of feature indices for the **shared** choice.
              - `patched`: representative counterfactual `(1, D)` for the each outlier belonging to the group (e.g., medoid-based).
            Multiple `(dims, patched)` pairs can be produced per outliers if several normal samples clusters are found.
        """
        self.in_shape = norm_samples.shape[1]
        self.out = out
        
        self.explainers = [] # Explainers from 0 to out.shape[0] are base explainers, the following ones are aggregated explainars 
        self.outliers = {}
        self.choices = []
        self.rs = {}
        self.num_groups = num_groups
        self.max_lev = self.out.shape[0] - num_groups
        
        exps = []
        chooses = []
        
        # Build base explainers
        for i in tqdm(range(len(self.out))):

            # Reference Set retrival
            if n_neigh is not None:
                rs = self._rs_selection(self.out[i:i+1], norm_samples, n_neigh)
            else:
                rs = norm_samples
            # Dataset creation
            mm_data = self._combine_data(self.out[i], rs)
            self.rs[i] = rs
            
            # Explanation computation
            exp_nn, choose, exp = self._single_explanation(mm_data, num_tries, norm_samples)
            self.exps.append(exp)
            self.choices.append(choose)
            self.explainers.append(exp_nn)
            
 
        self.outliers = {0: [[i] for i in range(self.out.shape[0])]}
        self.exp_id = {0: {i: i for i in range(self.out.shape[0])}}
        
        
        # Build the hierarchical explanation merging tree
        for it in tqdm(range(self.out.shape[0]-num_groups)):
            best_fit = -1
            best_s = -1
            best_j = -1
            
            for s in range(len(self.outliers[it])):
                for j in range(len(self.outliers[it])):
                    if s != j:
                        x_normal = self.rs[self.outliers[it][j][0]]
                        mm_data_o = np.full_like(x_normal, fill_value=self.out[self.outliers[it][j][0]])
                        mm_data_i = x_normal

                        for op in range(1, len(self.outliers[it][j])):
                            x_normal = self.rs[self.outliers[it][j][op]]

                            mm_data_o = np.append(mm_data_o, np.full_like(x_normal, self.out[self.outliers[it][j][op]:self.outliers[it][j][op]+1]), axis=0)
                            mm_data_i = np.append(mm_data_i, x_normal, axis=0)
                        mm_data = [mm_data_o, mm_data_i]

                        score = self.explainers[self.exp_id[it][s]].loss_fn(mm_data)
                        if score < best_fit or best_fit == -1:
                            best_fit = score
                            best_s = s
                            best_j = j
            
            ext_s = self.outliers[it][best_s].copy()
            ext_s.extend(self.outliers[it][best_j])
            self.exp_id[it+1] = {}
            
            outs_up = []
            for s in range(len(self.outliers[it])):
                if s != best_j:
                    new_id = s if s < best_j else s-1
                    if s != best_s:
                        outs_up.append(self.outliers[it][s])
                        self.exp_id[it+1][new_id] = self.exp_id[it][s]
                    else:
                        outs_up.append(ext_s)
                        self.exp_id[it+1][new_id] = len(self.explainers)
            self.outliers[it+1] = outs_up
            
            # Train the new explainer
            x_normal = self.rs[self.outliers[it][best_s][0]]
            mm_data_o = np.full_like(x_normal, fill_value=self.out[self.outliers[it][best_s][0]])
            mm_data_i = x_normal

            for op in range(1, len(self.outliers[it][best_s])):
                x_normal = self.rs[self.outliers[it][best_s][op]]

                mm_data_o = np.append(mm_data_o, np.full_like(x_normal, self.out[self.outliers[it][best_s][op]]), axis=0)
                mm_data_i = np.append(mm_data_i, x_normal, axis=0)

            for op in range(1, len(self.outliers[it][best_j])):
                x_normal = self.rs[self.outliers[it][best_j][op]]

                mm_data_o = np.append(mm_data_o, np.full_like(x_normal, self.out[self.outliers[it][best_j][op]]), axis=0)
                mm_data_i = np.append(mm_data_i, x_normal, axis=0)

            mm_data = [mm_data_o, mm_data_i]
            
            # Explanation computation
            exp_nn, choose, exp = self._single_explanation(mm_data, num_tries, norm_samples)
                
            self.exps.append(exp)
            self.explainers.append(exp_nn)
            self.choices.append(choose)
            
        # Compute final explanation
        exp = []
        for i in range(len(self.outliers[it+1])):
            exp.append(self.exps[self.exp_id[it+1][i]])
            
        return exp


