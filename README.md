# $\text{M}^2 \text{OE}$ - Masking Models for Outlier Explanation

> Transformation‑based, counterfactual explanations for **outliers**, **anomalous groups**, and **evolving outliers** in tabular data.

This repository provides a Python implementation of **Masking Models for Outlier Explanation** ($\text{M}^2 \text{OE}$). Given tabular data, the library learns *which* features to change (**choice**) and *by how much* (**mask**) to transform an outlier into a nearby normal point. The transformation is applied only on the chosen subspace, producing an interpretable **counterfactual patch**.


The package includes ready‑to‑use explainers for:
- **Single outliers** (per‑sample explanations)
- **Groups of outliers** (shared feature choices across a set)
- **Evolving outliers** (time‑indexed sequences, with preference‑guided stability)

---

## Reference

Further information about this paper are reported in the following papers. If you you our work, please cite them.

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

## Package structure

- `models/`
  - `MaskingModel` (abstract base)
  - `TabularMM` - single-sample masking model
  - `TabularMM_SC` - shared-choice masking model (groups)
  - `TabularMM_Pref` - preference-guided masking model (sequences)
- `explainers/`
  - `TabularExplainer` - single outlier
  - `TabularGroupExplainer` - multiple outliers with shared choices
  - `TabularSequentialExplainer` - evolving outlier across time



