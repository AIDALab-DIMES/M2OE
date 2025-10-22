import numpy as np

from sklearn.neighbors import NearestNeighbors


def _get_medoid(vX):
    vMean = np.mean(vX, axis=0)
    return vX[np.argmin([sum((x - vMean)**2) for x in vX])].reshape(1, -1)


class Explainer:
    
    def __init__(self, rs_selector=NearestNeighbors(), threshold=0.5):
        
        self.rs_selector = rs_selector 
        self.threshold = threshold
        
        
    def _rs_selection(self, out, norm_samples, n_neigh):
        """
        Select the reference set from normals. Uses the explainer's `rs_selector`.

        Parameters
        ----------
        out : np.ndarray
            Outlier sample, shape `(1, D)`.
        norm_samples : np.ndarray
            Normal data, shape `(N, D)`.
        n_neigh : int
            Number of neighbors `k` to retrieve.

        Returns
        -------
        np.ndarray
            Reference set of shape `(k, D)`.
        """
        
        # Fit the Reference Set selector
        near_neigh = self.rs_selector(n_neighbors=n_neigh)
        near_neigh.fit(norm_samples)
        
        # Extract the Reference Set
        _, y_rs = near_neigh.kneighbors(out)
        rs = norm_samples[y_rs[0]]
        
        return rs
    
    def _combine_data(self, out, rs):
        """
        Replicate the outlier to match the reference set and build model inputs.

        Parameters
        ----------
        out : np.ndarray
            Outlier sample `(1, D)`.
        rs : np.ndarray
            Reference set `(k, D)`.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A pair `[O, R]` where `O` is the outlier replicated to shape `(k, D)` and
            `R` is `rs` `(k, D)`.
        """
        
        # Create a tuple containing the replicated outlier and reference data
        mm_data_o = np.full_like(rs, fill_value=out)
        mm_data = [mm_data_o, rs]
        
        return mm_data
    
    
    def compute_explanation(self, out, norm_samples, n_neigh):
        """
        Abstract method to compute choice–mask explanations.

        Parameters
        ----------
        out : np.ndarray
            Outlier sample.
        norm_samples : np.ndarray
            Normal data.
        n_neigh : int
            Size of the k-NN reference set `k`.

        """
        raise NotImplementedError('subclass must override compute_explanation!')