[← back to README](../../README.en.md)

# Hybrid classifier (CNN + classical features + SVM)

<!-- When editing this file, also update ../05_classificador_hibrido.md -->

[🇧🇷 Português](../05_classificador_hibrido.md) | **🇺🇸 English**

In addition to the CNN, the project trains an **SVM** classifier that
combines two kinds of information: the **embeddings** learned by the CNN and
**classical features** from image processing. It is compared with the CNN
itself on the same test set.

```
fixed split (data_split/split_manifest.csv: training / validation / test)
   │
   ├─ original image ───→ segmentation (segment_image) ──→ classical leaf features (147)
   │                                                        → StandardScaler ─────────────┐
   │                                                                                      ├─→ SVM (RBF) → class
   └─ segmented image ──→ trained CNN ──→ GAP embedding (1280)                             │
                                            → StandardScaler → PCA (64/128/256) ──────────┘
```

Implementation: [`src/processamento_imagens/features.py`](../../src/processamento_imagens/features.py) (features), [`src/ia/hibrido/extract_features.py`](../../src/ia/hibrido/extract_features.py)
(extracts and saves everything to `features_cache/`) and [`src/ia/hibrido/hybrid_classifier.py`](../../src/ia/hibrido/hybrid_classifier.py)
(training, evaluation and comparison).

**Classical features** (all computed only on the leaf pixels, using the
segmentation mask):

| Group | No. | What it captures |
|---|---|---|
| Color: H (32 bins), S (16), V (16) histograms | 64 | distribution of hue, saturation and brightness of the leaf (yellowing, spots, paleness) |
| Color: moments (mean, std. deviation, skewness) in HSV and Lab | 18 | mean color of the leaf and how spread out/skewed it is |
| Lesion proportion | 1 | fraction of the leaf in lesion tones (`LOWER_LESION`/`UPPER_LESION`): how much of the leaf is spotted |
| GLCM texture (32 levels; distances 1, 2, 3; 0°/45°/90°/135°) | 36 | contrast, dissimilarity, homogeneity, energy, correlation and ASM; mean and std. deviation across angles (rotation invariant) |
| Uniform LBP texture (P=8, R=1) and (P=16, R=2) | 28 | texture micro-patterns (dots, edges, fine spots) at two scales |

In the GLCM, the background is encoded as level 0 and row/column 0 of the
matrix is discarded before computing the properties, so that only pairs of
leaf pixels count.

Three SVMs are compared with the CNN:

| Model | Input |
|---|---|
| `SVM_classicas` | classical features only |
| `SVM_cnn` | CNN embeddings only (with PCA) |
| `SVM_hibrido` | embeddings (with PCA) + classical features |
| `CNN` | the CNN's own softmax (reference) |

Design decisions:

- **Why an SVM with RBF kernel**: it works well with a few hundred dense,
  standardized features, separates the classes non-linearly and, with
  `class_weight="balanced"`, compensates for the class imbalance.
- **Why PCA on the CNN block**: the embeddings have 1280 dimensions and the
  classical features, 147. Without reduction, the RBF kernel distance would
  be dominated by the embeddings and the classical features would carry
  almost no weight. PCA balances the two blocks.
- **Why choose the hyperparameters on the validation set**: the CNN was
  trained on the training images, so the training embeddings are "too
  easy". Cross-validation within the training set would overestimate every
  candidate and bias the choice. Therefore, `GridSearchCV` uses a
  `PredefinedSplit`: it fits on the training set, measures macro-F1 on the
  validation set and, with the best hyperparameters (C, gamma and number of
  PCA components), refits on training + validation. **The test set is never
  used for training or for choosing hyperparameters.**
- **McNemar's test**: compares two classifiers on the same test images,
  looking only at the cases where they disagree (one is right and the other
  is wrong). If the difference were due to chance, these cases would split
  half and half; the exact binomial test gives the p-value (p < 0.05
  indicates a statistically significant difference). It is applied between
  `CNN` and `SVM_hibrido` and between `SVM_cnn` and `SVM_hibrido`.

Outputs in [`reports/ia/hibrido/`](../../reports/ia/hibrido/): metrics and per-class report for each model,
grid search results, the `SVM_hibrido` confusion matrix,
`comparativo.csv`/`comparativo.png` and `mcnemar.txt`. The best hybrid
pipeline is saved in `models/svm_hybrid.joblib`.

## Hybrid classifier results

All models are evaluated on the same test set (2,725 images).

**Main protocol** (hyperparameters chosen on the validation set; SVMs
refit on training + validation, 15,438 images):

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| CNN | 0.949 | 0.950 | 0.949 |
| SVM_classicas | 0.946 | 0.922 | 0.946 |
| SVM_cnn | 0.966 | 0.960 | 0.965 |
| **SVM_hibrido** | **0.979** | **0.972** | **0.979** |

**Control: SVMs fit on the training set only.** In the main protocol, the
SVMs use ~21% more data than the CNN (15,438 vs. 12,714 images; the CNN
only used the validation set for early stopping). To separate the effect of
the fusion from the effect of this extra data, the control
(`python -m ia.hibrido.hybrid_classifier --control-train-only`) reuses the
same hyperparameters and trains the SVMs only on the 12,714 training
images:

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| CNN | 0.949 | 0.950 | 0.949 |
| SVM_classicas | 0.941 | 0.915 | 0.940 |
| SVM_cnn | 0.965 | 0.960 | 0.965 |
| **SVM_hibrido** | **0.976** | **0.971** | **0.976** |

**McNemar's test** (cases in which only one of the two models is right):

| Protocol | Pair (A vs. B) | Discordant cases | Only A right | Only B right | p-value |
|---|---|---|---|---|---|
| Main | CNN vs. SVM_hibrido | 111 | 15 | 96 | 1.2e-15 |
| Main | SVM_cnn vs. SVM_hibrido | 52 | 8 | 44 | 4.0e-07 |
| Control | CNN vs. SVM_hibrido | 102 | 14 | 88 | 2.8e-14 |
| Control | SVM_cnn vs. SVM_hibrido | 43 | 6 | 37 | 1.6e-06 |

In every pair, `SVM_hibrido` is right more often in the discordant cases,
with p < 0.05.

**Interpretation** (based on the control, in which all models learn only
from the training set):

- **Gain from the classical features** (`SVM_cnn` vs. `SVM_hibrido`: same
  embeddings, same protocol, only the presence of the classical features
  changes): macro-F1 rises from 0.960 to 0.971 and accuracy from 0.965 to
  0.976. In the 43 cases where the two disagree, the hybrid is right in 37
  (p = 1.6e-06). The classical features bring information that the
  embeddings alone do not have.
- **CNN vs. SVM_hibrido**: the advantage holds in the control. The hybrid's
  accuracy drops only from 0.979 to 0.976 without the validation data, and
  remains above the CNN (0.949), with 88 vs. 14 discordant cases
  (p = 2.8e-14). Therefore, the advantage does not come from the ~21% extra
  data. Part of it comes from replacing the CNN's softmax head with an SVM
  on the same embeddings (`SVM_cnn` already reaches 0.965 accuracy, vs.
  0.949 for the CNN; this pair was not tested with McNemar), and the rest
  from the classical features (previous item).
- **Why the classical features alone have a lower macro-F1** (0.922 vs.
  0.946 accuracy in the main protocol): macro-F1 gives the same weight to
  every class, and `Potato___healthy` has only 106 training images (23 in
  the test set). On this class, `SVM_classicas` has a recall of 0.52
  (F1 0.67); without it, the macro-F1 of the other 12 classes would be about
  0.94. In the hybrid, the CNN embeddings recover a good part of this class:
  recall of 0.87 in the main protocol and 0.91 in the control (F1 0.93 and
  0.95). It is still below the CNN, which gets all 23 images right (recall
  1.00). With only 23 images, each image is worth ~0.04 of recall.

**Best hyperparameters** (chosen by macro-F1 on the validation set):

| Model | C | gamma | PCA components | Macro-F1 on validation |
|---|---|---|---|---|
| SVM_classicas | 100 | 0.01 | — | 0.924 |
| SVM_cnn | 10 | scale | 256 | 0.968 |
| SVM_hibrido | 10 | scale | 128 | 0.973 |

Plots: [`reports/ia/hibrido/comparativo.png`](../../reports/ia/hibrido/comparativo.png) (main protocol) and
[`reports/ia/hibrido/controle_refit_treino/comparativo.png`](../../reports/ia/hibrido/controle_refit_treino/comparativo.png) (control), each with
a full-scale panel and a zoomed-in panel (axis starting at 0.85), where the
differences between the models become visible.

**Limitation**: PlantVillage consists of laboratory photos (uniform
background, a single isolated leaf, controlled lighting). The results of all
models, CNN and SVMs, may not hold on field photos, with complex
backgrounds, multiple leaves and variable lighting (*domain shift*).
Validation on field images is required before any practical use.

## Figures

![Comparison in the main protocol](../../reports/ia/hibrido/comparativo.png)

![Comparison in the control](../../reports/ia/hibrido/controle_refit_treino/comparativo.png)

![Hybrid SVM confusion matrix](../../reports/ia/hibrido/confusion_matrix_SVM_hibrido.png)

The live demonstration, on a single image, is done with `python src/demo_ia.py` (see [How to reproduce](07_how_to_reproduce.md#ai-demo-demo_iapy)).
