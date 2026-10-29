[← back to README](../../README.en.md)

# CNN (AI integration)

<!-- When editing this file, also update ../04_cnn.md -->

[🇧🇷 Português](../04_cnn.md) | **🇺🇸 English**

The output of the image processing pipeline (segmented images) feeds a
classification model based on **transfer learning** (implemented in
[`src/ia/cnn/train_model.py`](../../src/ia/cnn/train_model.py)). The full project pipeline is:

```
Raw image
   → Preprocessing (resize, blur, HSV)
   → Leaf segmentation (background removal)
   → [AI branch]   → Data augmentation (flip, rotation, zoom, contrast; training only)
                    → MobileNetV2 model (pretrained on ImageNet)
                    → Custom classification head (GlobalAveragePooling + Dropout + Dense)
                    → Fine-tuning of the last layers of the base
   → Class prediction (healthy or disease type)
```

Training steps:

1. **Loading**: the images in `data_processed/` are split by
   [`src/dados/split_dataset.py`](../../src/dados/split_dataset.py) into 70% training / 15% validation / 15% test
   (stratified by class, seed 42), with the split recorded in
   `data_split/split_manifest.csv`. The validation set is used only for
   early stopping; the final metrics are computed on the test set, which is
   never used in training.
2. **Data augmentation**: random horizontal and vertical flips, rotation,
   zoom and contrast, applied by layers inside the model itself. They are
   only active during training; in validation, test and the demo the image
   passes through unchanged.
3. **Balanced class weight**: the dataset is imbalanced (~21x between
   `Tomato__Tomato_YellowLeaf__Curl_Virus`, with 3208 images, and
   `Potato___healthy`, with 152). Each class receives a weight inversely
   proportional to its frequency in the training set
   (`compute_class_weight("balanced")`), so that small classes are not
   ignored.
4. **Transfer learning**: MobileNetV2 is loaded with ImageNet weights and
   initially frozen; only the classification head (13 classes) is trained,
   for up to 10 epochs with learning rate 1e-3.
5. **Fine-tuning**: the last 50 layers of MobileNetV2 are unfrozen and
   retrained for up to 20 epochs with learning rate 1e-4, allowing the model
   to specialize in the specific characteristics of leaf diseases. The
   **BatchNormalization layers stay frozen**: if they were unlocked, they
   would start updating their statistics with the training batches, which
   destabilized the model at the start of fine-tuning in the previous run
   ([`reports/ia/historico/cnn_run1/`](../../reports/ia/historico/cnn_run1/)).
6. **Early stopping and ReduceLROnPlateau**: in each stage, training stops
   when `val_loss` (validation loss) stops improving for 4 epochs, and the
   learning rate is multiplied by 0.3 whenever it goes 2 epochs without
   improving (minimum 1e-6). At the end of each stage, the model keeps the
   weights from the epoch with the lowest `val_loss`, even if the stage
   reaches the epoch limit without early stopping being triggered.

## CNN results

Metrics on the **test set**: 2,725 images that were never used for training
or model selection (the validation set is used only for early stopping and
learning rate reduction).

**Model evolution**

| | Baseline v1 | Run 1 | Run 2 (current) |
|---|---|---|---|
| Measured on | validation (3,632 images) | test (2,725 images) | test (2,725 images) |
| Accuracy | 0.88 | 0.815 | **0.949** |
| Macro-F1 | 0.86 | 0.796 | **0.950** |
| Recall `Tomato_Early_blight` | 0.59 | 0.51 | **0.86** |
| Recall `Tomato__Target_Spot` | 0.73 | 0.64 | **0.88** |
| Recall `Tomato_Late_blight` | 0.87 | 0.62 | **0.92** |

- **Baseline v1**: old segmentation (green-only mask, which erased the
  lesions), 80/20 split with no test set, and metric measured on the same
  validation set used for early stopping, hence optimistic and not directly
  comparable with the following runs. The baseline report only has two
  decimal places.
- **Run 1**: fixed segmentation, 70/15/15 split with a separate test set,
  data augmentation and balanced class weight. Fine-tuning, however,
  destabilized the model because the unlocked BatchNormalization layers
  started updating their statistics: in the first fine-tuning epoch the
  training loss jumped from 0.53 to 1.40 (spike visible in
  [`reports/ia/historico/cnn_run1/training_history.png`](../../reports/ia/historico/cnn_run1/training_history.png)).
- **Run 2**: BatchNormalization frozen during fine-tuning, 50 layers
  unlocked, learning rate 1e-4 with ReduceLROnPlateau and up to 20
  fine-tuning epochs. In practice, stage 1 ran for 10 epochs (best: 7) and
  fine-tuning for 18 epochs (best: 14), with the learning rate reduced down
  to 2.7e-6.

The recalls of `Tomato_Early_blight` and `Tomato__Target_Spot` remain the
lowest among these classes: the two diseases produce similar spots.

The full report (precision/recall/F1 per class), the confusion matrix, the
metrics and the CNN training history are in [`reports/ia/cnn/`](../../reports/ia/cnn/); the
hybrid classifier results are in [`reports/ia/hibrido/`](../../reports/ia/hibrido/) (see
[Hybrid classifier results](05_hybrid_classifier.md#hybrid-classifier-results)).

## Figures

![CNN training history](../../reports/ia/cnn/training_history.png)

![CNN confusion matrix on the test set](../../reports/ia/cnn/confusion_matrix.png)

The live demonstration, on a single image, is done with `python src/demo_ia.py` (see [How to reproduce](07_how_to_reproduce.md#ai-demo-demo_iapy)).
