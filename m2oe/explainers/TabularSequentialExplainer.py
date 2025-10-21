from .Explainer import Explainer
from .Explainer import _get_medoid
from ..models.AETabularMM_Pref import TabularMM_Pref

from mlxtend.frequent_patterns import fpmax
from sklearn.cluster import OPTICS

from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm


class TabularSequentialExplainer(Explainer):
    """
    Sequential explainer for **evolving** (time-indexed) tabular outliers.

    This explainer builds one preference-guided masking model per snapshot
    (`TabularMM_Pref`) and encourages **temporal stability** by *reducing the
    sparsity penalty* on features selected at the previous step(s). Given an
    outlier sequence and a matching sequence of normal populations, it:

    1) for each time step `t = 0..T-1`, forms a **reference set**
       `RS_t ∈ ℝ^{k×D}` for each outlier,
    2) trains a `TabularMM_Pref` using a per-feature **preference** vector
       `p_t ∈ ℝ^D` (values `< 1` reduce sparsity penalty, promoting re-selection),
    3) thresholds the model’s **choice** to get the subspace `dims_t`,
    4) clusters `RS_t[:, dims_t]` via `cl_algo` (default: OPTICS) and, for each
       cluster medoid, generates a **counterfactual patch** `out'_t`,
    5) updates `p_{t+1}` by setting entries in `dims_t` to `pref_fact` and the
       rest to `1.0`,
    6) returns the per-step lists of `(dims_t, out'_t)` explanations.

    Parameters
    ----------
    normal_data : np.ndarray
        Array of shape `(T, N, D)`. The first array dimension contains the `T` time steps, 
        the second one the normal samples, finally, the last one contains the outlier's features.
    loss_weights : Sequence[float], default=[1.0, 1.2, 0.3]
        Three non-negative loss weights `[alpha1, alpha2, alpha3]` for the loss terms:
        subspace contrast, patch proximity, and choice sparsity.
    lr : float, default=0.001
        Learning rate for the optimizer used when training each step-specific model.
    epochs : int, default=30
        Number of training epochs per time step.
    bs : int, default=16
        Mini-batch size used to fit each step-specific model.
    rs_selector : type or object, optional (default: `sklearn.neighbors.NearestNeighbors`)
        selector used to build each `RS_t`. Must expose `fit`/`kneighbors`.
        If a **type** is provided, it will be instantiated internally.
    threshold : float, optional, default=0.5
        Binarization threshold in `[0, 1]` applied to the choice output.
    pref_fact : float, optional, default=0.5
        Preference value assigned to **previously-selected** features for the next
        step (`< 1` lowers the sparsity penalty and promotes re-selection).
    cl_algo : sklearn.cluster, optional (default: `OPTICS()`)
        Clustering algorithm used on `RS_t[:, dims_t]` to identify local modes.
        Must implement `fit_predict`.

    Attributes
    ----------
    in_shape : int
        Feature dimensionality `D`.
    loss_weights : Sequence[float]
        Loss weights used for all time steps.
    lr, epochs, bs : Union[float, int]
        Optimizer and training hyperparameters reused at each step.
    explainers : List[TabularMM_Pref]
        The per-step preference-guided masking models.
    out : Optional[np.ndarray]
        Stores the last explained outlier sequence; expected shape `(T, 1, D)`.
    pref_fact : float
        Preference value applied to features selected at the previous step.
    cl_algo : sklearn.cluster
        Clustering algorithm used during explanation generation.
    threshold : float
        Effective choice threshold used during explanation.
    rs_selector : object
        Fitted reference set selector used to retrieve per-step reference sets.


    See Also
    --------
    TabularExplainer
        Single-sample explainer (no preference carry-over).
    TabularGroupExplainer
        Learns shared choices across multiple outliers at a single snapshot.
    TabularMM_Pref
        Preference-guided masking model used internally at each step.

    References
    ----------
    - Angiulli, Fassetti, Nisticò, Palopoli (2025). *Explaining evolving outliers
      for uncovering key aspects of the green comparative advantage*, Array.
      https://doi.org/10.1016/j.array.2025.100518
    """

    def __init__(self, loss_weights, lr, epochs, bs, rs_selector=NearestNeighbors(), threshold=0.5, pref_fact=0.5, cl_algo = OPTICS()):
        super(TabularSequentialExplainer, self).__init__(rs_selector=NearestNeighbors, threshold=threshold)
        self.in_shape = None
        
        self.loss_weights = loss_weights
        self.lr = lr
        self.epochs = epochs
        self.bs = bs
        
        self.explainers = []
        self.out = None
        self.pref_fact = pref_fact
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
    
    
    def _single_explanation(self, ds, num_tries, norm_samples, preferences):
        t = 0
        choose = []
        
        while len(choose)==0 and t<num_tries:
            # Explanator creation
            exp_nn = TabularMM_Pref(preferences, norm_samples, self.loss_weights, self.lr, self.epochs, self.bs)

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
    
    
    def compute_explanation(self, out, norm_samples, n_neigh=30, num_tries=3):
        """
        Compute explanations for an **evolving** outlier across `T` snapshots, softly
        preferring features used at previous steps (preference-guided sparsity).

        Parameters
        ----------
        out: np.ndarray
            Array of shape `(T, D)`, The first array dimension contains the `T` time steps, while  the last one contains the outlier's features.
        norm_samples : np.ndarray
            Array of shape `(T, N, D)`. The first array dimension contains the `T` time steps, the second one the normal samples, finally, the last one contains the outlier's features.
        n_neigh : int, optional, default=30
            Reference set size `k` at each snapshot. If None all data samples are considered.
        num_tries : int, optional, default=3
            Number of restarts in case of explanation failure.

        Returns
        -------
        List[List[Tuple[np.ndarray, np.ndarray]]]
            A list of length `T`; each item is a list of `(dims, patched)` as in the single-sample case.
        """
        self.in_shape = norm_samples.shape[-1]
        self.out = out.copy()
        exp = []
        preferences = np.ones(self.in_shape, dtype=np.float32)
        
        for ts in tqdm(range(len(self.out))):
            # Reference Set retrival
            if n_neigh is not None:
                rs = self._rs_selection(self.out[ts:ts+1], norm_samples[ts], n_neigh)
            else:
                rs = norm_samples[ts]
            # Dataset creation
            mm_data = self._combine_data(self.out[ts:ts+1], rs)

            exp_nn, choose, exps = self._single_explanation(mm_data, num_tries, norm_samples[ts], preferences)
            self.explainers.append(exp_nn)        
            
            feats = np.argwhere(choose>0).reshape(-1)
            
            preferences = np.ones(self.in_shape, dtype=np.float32)
            preferences[feats] = self.pref_fact
            
            exp.append(exps)
            
        return exp