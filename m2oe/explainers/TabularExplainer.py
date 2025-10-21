from .Explainer import Explainer
from .Explainer import _get_medoid
from ..models.AETabularMM import TabularMM

from mlxtend.frequent_patterns import fpmax
from sklearn.cluster import DBSCAN

from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import numpy as np
import pandas as pd


class TabularExplainer(Explainer):
    """
    Tabular explainer for explaining an evolving outlier.

    This explainer wraps a sample-conditioned masking model (`TabularMM`) to
    produce compact, subspace-level explanations of a tabular outlier. Given an
    outlier `out ∈ ℝ^D` and a reference population of normals, it:

    1) builds a **reference set** `RS ∈ ℝ^{k×D}`,
    2) trains the masking model on paired inputs `[O_rep, RS]` with `O_rep` being
       `out` replicated to `(k, D)`,
    3) reads per-feature **choice** vectors for each pair and mines **frequent
       choices** (subspaces) via `fpmax`,
    4) clusters reference points restricted to each chosen subspace using
       `cl_algo` (default: DBSCAN),
    5) for each cluster medoid, generates a **counterfactual patch** `out'` by
       applying the learned **mask** only on the chosen features,
    6) returns a list of `(dims, patched)` explanations.

    Parameters
    ----------
    normal_data : np.ndarray
        Normal/reference data array of shape `(N, D)` used to compute the model’s
        per-feature dispersion vector `normal_dist` `(D,)`.
    loss_weights : Sequence[float], default=[1.0, 1.2, 0.3]
        Three non-negative weights `[alpha1, alpha2, alpha3]` for the loss terms:
        subspace contrast, patch proximity, and choice sparsity.
    lr : float, default=0.001
        Learning rate for the optimizer used during the internal model training.
    epochs : int, default=30
        Number of training epochs used when fitting the masking model.
    bs : int, default=16
        Mini-batch size used when fitting the masking model.
    rs_selector : type or object, optional, default=`sklearn.neighbors.NearestNeighbors`
        Reference set selector used to build the reference set. It must expose a `fit` and
        `kneighbors`-like interface. If a **type** is provided, it will be
        instantiated internally with sensible defaults.
    threshold : float, optional, default=0.5
        Binarization threshold in `[0, 1]` applied to the choice output when
        computing explanations.
    cl_algo : sklearn.cluster, optional, default=`DBSCAN(eps=0.10)`
        Clustering algorithm used on `RS[:, dims]` to discover distinct local
        modes for each frequent choice `dims`. Must implement `fit_predict`.

    Attributes
    ----------
    in_shape : int
        Feature dimensionality `D`.
    exp_nn : TabularMM
        The underlying masking model that learns **choice** and **mask** and
        outputs `patches` (`(B, D)`), `masks` (`(B, D)`), and `choices` (`(B, D)`).
    out : Optional[np.ndarray]
        Stores the last explained outlier, shape `(1, D)` or `(D,)`.
    loss_weights : Sequence[float]
        Loss weights used for all time steps.
    lr, epochs, bs : Union[float, int]
        Optimizer and training hyperparameters reused at each step.
    cl_algo : sklearn.base.ClusterMixin
        The clustering algorithm actually used during explanation generation.
    threshold : float, default=0.5
        Effective threshold used to filter/binarize the choice output.
    rs_selector : object, default=
        Fitted reference set selector used to retrieve the reference set.

    See Also
    --------
    TabularGroupExplainer
        Learns **shared** choices across multiple outliers (group explanations).
    TabularSequentialExplainer
        Preference-guided variant for **evolving** outliers across time/snapshots.
    TabularMM
        Sample-conditioned masking model for tabular data.


    References
    ----------
    - Angiulli, Fassetti, Nisticò, Palopoli (2024). *Explaining outliers and anomalous
      groups via subspace density contrastive loss*, Machine Learning.
      https://doi.org/10.1007/s10994-024-06618-8
    - Angiulli, Fassetti, Nisticò, Palopoli (2025). *Explaining evolving outliers for
      uncovering key aspects of the green comparative advantage*, Array.
      https://doi.org/10.1016/j.array.2025.100518
    """

    def __init__(self, loss_weights=[1.0, 1.2, 0.3], lr=0.001, epochs=30, bs=16, rs_selector=NearestNeighbors(), threshold=0.5, cl_algo=DBSCAN(eps=0.10)):
        super(TabularExplainer, self).__init__(rs_selector=NearestNeighbors, threshold=threshold)
        self.in_shape = None
        
        self.loss_weights = loss_weights
        self.lr = lr
        self.epochs = epochs
        self.bs = bs
        
        self.exp_nn = None
        self.out = None
        self.cl_algo = cl_algo
        

    
    def _explain(self, sample, threshold=0.5, binary=True):
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
        mask = self.exp_nn.MASK(sample)
        choose = self.exp_nn.CHOOSE(sample)

        if binary:
            choose = np.where(choose.numpy() > threshold, 1, 0)
        else:
            choose = choose.numpy()
            choose = np.where(choose > threshold, choose, 0)
        patches = self.exp_nn.MASKAPPLY([sample[0], mask, choose])
        choose = np.where(choose > threshold, 1, 0)

        return patches.numpy(), choose
    
    
    def _explanation_generation(self, out, rs, chooses):
        
        # Frequent itemset search
        frequent_itemsets = fpmax(pd.DataFrame(chooses).astype(bool), min_support = 0.25)
        exps = []
        
        for fit in frequent_itemsets['itemsets'].to_numpy():
            dims = list(fit)
            
            x_ok = chooses[:, dims].sum(axis=1) == len(dims)
            
            # Normal samples clusters computation
            labels = self.cl_algo.fit_predict(rs[x_ok][:, dims])
            
            # Cluster medoids computation
            if len(set(labels[labels>=0]))>0:
                for j in range(len(set(labels[labels>=0]))):
                    mds = _get_medoid(rs[x_ok][labels==j])
                    ptc, _ = self._explain([out, mds], threshold=self.threshold, binary=False)
                    
                    
                    # Add explanation
                    exps.append((dims, ptc))
            else:
                # Only a unique cluster if no clusters found
                mds = _get_medoid(rs[x_ok])
                ptc, _ = self._explain([out, mds], threshold=self.threshold, binary=False)
                
                # Add explanation
                exps.append((dims, ptc))
                
        return exps
    
    
    def compute_explanation(self, out, norm_samples, n_neigh=30, num_tries=3):
        """
        Compute explanations for a **single** outlier tabular sample.

        Parameters
        ----------
        out : np.ndarray
            Outlier `(1, D)`.
        norm_samples : np.ndarray
            Normal data `(N, D)` used for both reference set selection and statistic vector computation.
        n_neigh : int, optional, default=30
            Reference set size `k` at each snapshot. If None all data samples are considered.
        num_tries : int, optional, default=3
            Number of restarts in case of explanation failure.

        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray]]
            A list of `(dims, patched)` pairs where:
              - `dims`: 1-D array of feature indices (length `m`) representing the chosen subspace.
              - `patched`: a counterfactual sample of shape `(1, D)` (only `dims` differ from `out`).
        """
        self.in_shape = norm_samples.shape[1]
        self.out = out.copy()
        
        # Reference Set retrival
        if n_neigh is not None:
            rs = self._rs_selection(self.out, norm_samples, n_neigh)
        else:
            rs = norm_samples
        # Dataset creation
        mm_data = self._combine_data(self.out, rs)
        
        
        # Explainer net set up and training
        t = 0
        choose = []
        
        while len(choose)==0 and t<num_tries:
            self.exp_nn= TabularMM(norm_samples, self.loss_weights, self.lr, self.epochs, self.bs)
            opt = tf.keras.optimizers.Adam(learning_rate=self.exp_nn.lr)
            self.exp_nn.compile(optimizer=opt, run_eagerly=True)
            self.exp_nn.fit(mm_data, rs, batch_size=self.exp_nn.bs, epochs=self.exp_nn.epochs, verbose=0)

            # Alternative explanation collection
            _, choose = self._explain(mm_data, threshold=self.threshold, binary=True)
        if len(choose)==0:
            raise Exception("Explanation process failed, try with a lower value for alpha3!") 
        
        # Explanation generation
        exps = self._explanation_generation(out, rs, choose)
        
        return exps