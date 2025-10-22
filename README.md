# $\text{M}^2 \text{OE}$ - Masking Models for Outlier Explanation

This repository provides a Python implementation of the **Masking Models for Outlier Explanation** ($\text{M}^2 \text{OE}$) method and its extensions, namely, $\text{M}^2\text{OE-groups} and $\text{M}^2\text{OE}_e$, tailored to explain groups of outliers and evolving outliers, respectively. Given the normal samples of a tabular dataset, defined over continuous features, and an outlier, or a group of outliers, either a sequential outlier belonging to the dataset, the method learns *which* features to change (**choice**) and *by how much* (**mask**) to transform an outlier into a nearby normal point. The transformation is applied only to the chosen subspace, producing an interpretable **counterfactual**. Multiple explanations can be outputed for the same outlier to figure out alternative ways to restore the outlier.


The package includes ready-to-use explainers for:
- **Single outliers** - per-sample explanations
- **Groups of outliers** - shared feature choices across a set of outliers and indipendent patches
- **Evolving outliers** - explanation for evolving outliers, with preference-guided stability

All those explainers work with continuous tabular data.

---

## Reference

Further information about the here implemented approaches are reported in the following papers. If you you our work, please cite them.

### Single outliers

```
@article{angiulli2024explaining,
  title={Explaining outliers and anomalous groups via subspace density contrastive loss},
  author={Angiulli, Fabrizio and Fassetti, Fabio and Nistic{\`o}, Simona and Palopoli, Luigi},
  journal={Machine Learning},
  volume={113},
  number={10},
  pages={7565--7589},
  year={2024},
  publisher={Springer}
}

@inproceedings{angiulli2023counterfactuals,
  title={Counterfactuals explanations for outliers via subspaces density contrastive loss},
  author={Angiulli, Fabrizio and Fassetti, Fabio and Nistic{\'o}, Simona and Palopoli, Luigi},
  booktitle={International Conference on Discovery Science},
  pages={159--173},
  year={2023},
  organization={Springer}
}
```

### Group Explanations

```
@article{angiulli2024explaining,
  title={Explaining outliers and anomalous groups via subspace density contrastive loss},
  author={Angiulli, Fabrizio and Fassetti, Fabio and Nistic{\`o}, Simona and Palopoli, Luigi},
  journal={Machine Learning},
  volume={113},
  number={10},
  pages={7565--7589},
  year={2024},
  publisher={Springer}
}
```

### Explanations for evolving tabular data

```
@article{angiulli2025explaining,
  title={Explaining evolving outliers for uncovering key aspects of the green comparative advantage},
  author={Angiulli, Fabrizio and Fassetti, Fabio and Nistic{\`o}, Simona and Palopoli, Luigi},
  journal={Array},
  pages={100518},
  year={2025},
  publisher={Elsevier}
}
```


---

## Installation

You can install our package by cloning this repository or using pip.

```bash
# clone
git clone https://github.com/AIDALab-DIMES/M2OE.git
cd m2oe
pip install -e .
```
```bash
# pip
pip install m2oe
```

---
## Simple usages exaples

### Single outlier (TabularExplainer)

```python
import numpy as np
import m2oe
import m2oe.explainers.TabularExplainer as TE

# Create a random dataset (values in the [0,1] range) with 20 features
X = np.random.rand(100,20).astype(np.float32)
# Create a simple outlier
X[0,[4,5,6]] = 2
out = X[:1] # Normal samples

exp = TE.TabularExplainer([1.0, 1.2, 0.3], 0.001, 30, 16)
res = exp.compute_explanation(out, X[1:], 30)

for dims, patched in res:
    print("chosen features:", dims)          # indices of the chosen subspace
    print("counterfactual patch:", patched)  # patched sample o' (same shape as outlier)
```

### Groups of outliers (TabularGroupExplainer)

```python
import numpy as np
import m2oe
import m2oe.explainers.TabularGroupExplainer as TGE

# Create a random dataset (values in the [0,1] range) with 20 features
X = np.random.rand(100,20).astype(np.float32)
# Create three simple outliers
X[0,[4,5,6]] = 2
X[1,[3,4,5]] = 2
X[2,[5,6,7]] = 2

out = X[:3] # Normal samples
exp = TGE.TabularGroupExplainer([1.0, 1.0, 0.5], 0.001, 30, 16)
res = exp.compute_explanation(out, X[3:], 30)

# For each resulting group, you get a list of (dims, patched) per outlier -- the set of dims is unique for each group
for g, expls_for_group in enumerate(res):
    print(f"Group {g}:")
    print("shared chosen features:", dims)
    for dims, patched in expls_for_group[0]:  # explanations for the first outlier in group g
        print("group counterfactual patch (example outlier):", patched)
```


### Evolving outliers (TabularSequentialExplainer)

```python
import numpy as np
import m2oe
import m2oe.explainers.TabularSequentialExplainer as TSE

# Create a data collection including three (T=3) snapshots with 20 features
X = np.random.rand(3, 100, 20).astype(np.float32)
# Generate a simple outlier
X[:,0,[4,5,6]] = 2

out = X[:, 0] # Normal samples
exp = TSE.TabularSequentialExplainer([1.0, 1.0, 0.5], 0.001, 30, 16)
res = exp.compute_explanation(out, X[:, 1:], 30)

# res is a list of length T; each item is a list of (dims_t, patched_t) -- the set of dims is unique for each snapshot
for t, exps_t in enumerate(seq_exps):
    for dims_t, patched_t in exps_t:
        print(f"[t={t}] chosen features:", dims_t, " patched:", patched_t)
```


---

## Package structure

This module consists of two sub-packages: one (`models/`), which groups all the alternative neural networks used to produce the patches, and another (`explainers/`) containing the explainers, responsible for managing the entire explanation pipeline. The current structure is as follows.

- `models/`
  - `MaskingModel` (abstract base)
  - `TabularMM` - single-sample masking model
  - `TabularMM_SC` - shared-choice masking model (groups)
  - `TabularMM_Pref` - preference-guided shared-choice masking model (sequences)
- `explainers/`
  - `TabularExplainer` - single outlier
  - `TabularGroupExplainer` - multiple outliers with shared choices
  - `TabularSequentialExplainer` - evolving outlier across time

If you want to extend our module you can either 
> add a new neural architecture by creating a new class in the `models/` folder (must inherit from the MaskingModel abstract class) 
or 
> create a new explanation pipeline by creating a new class in the `explainers/` folder (must inherit from the Explainer abstract class)


---


## Need help?
If you find any bugs, have questions, need help modifying $\text{M}^2\text{OE}$, or want to get in touch, feel free to write us an [email](mailto:simona.nistico@dimes.unical.it)!





