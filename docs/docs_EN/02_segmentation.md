[← back to README](../../README.en.md)

# Image Processing pipeline and segmentation

<!-- When editing this file, also update ../02_segmentacao.md -->

[🇧🇷 Português](../02_segmentacao.md) | **🇺🇸 English**

The goal of the project is to determine, from a leaf image, whether the
plant is healthy or shows some kind of disease. The image processing
pipeline follows the steps below (1 to 4 in
[`src/processamento_imagens/preprocessing.py`](../../src/processamento_imagens/preprocessing.py); 5 in [`src/processamento_imagens/features.py`](../../src/processamento_imagens/features.py)):

1. **Loading and resizing**: each image is loaded and standardized to
   224x224 pixels, ensuring a uniform input for the following steps.
2. **Smoothing / noise removal**: a Gaussian Blur filter (5x5 kernel) is
   applied, reducing capture noise and small imperfections that could
   hinder segmentation.
3. **Color space conversion (BGR → HSV)**: OpenCV reads images in BGR; the
   HSV space separates hue (color), saturation and brightness information,
   which makes color-based segmentation easier.
4. **Leaf segmentation**: the mask is the union (`cv2.bitwise_or`) of two
   HSV color ranges (`cv2.inRange`): the green tones of the leaf and the
   lesion tones (brown/yellow spots, H ~8–35). Morphological operations
   (opening and closing) are then applied to clean up noise. Finally, with
   `cv2.findContours` we keep only the largest external contour (the leaf)
   and fill it entirely (`cv2.drawContours` with `cv2.FILLED`), so that
   everything inside the leaf is included in the mask and the background is
   removed. If no contour is found, or if the largest one covers less than
   5% of the image, the color mask is used as a fallback.
   - **Reason for the change**: the previous version used only the green
     mask, and the disease lesions (which are not green) were left out of
     it, becoming black "holes" in the segmented image — in other words,
     the disease symptom was exactly what was being erased. The comparison
     between the two approaches can be generated with
     `python -m processamento_imagens.segmentation_report --comparacao` (saved to
     [`reports/pi/segmentacao/segmentation_comparison.png`](../../reports/pi/segmentacao/segmentation_comparison.png)).
   - **Choice of the lesion mask thresholds**: the hue (H ~8–35) was fixed
     and we tested the saturation (S) and brightness (V) minimums on the
     same 12 images (seed 42), with
     `python -m processamento_imagens.segmentation_report --comparacao --s <S> --v <V> --out <arquivo>`:

     | S | V | Lesions on the edge | Background leakage | Image |
     |---|---|---|---|---|
     | 40 | 30 | Dark edge lesions cut off | No | [`reports/pi/segmentacao/estudo_limiares/segmentation_S40_V30.png`](../../reports/pi/segmentacao/estudo_limiares/segmentation_S40_V30.png) |
     | 25 | 20 | Recovered | Yes: the beige/gray background has a hue in the brown range, and only saturation separates the two | [`reports/pi/segmentacao/estudo_limiares/segmentation_S25_V20.png`](../../reports/pi/segmentacao/estudo_limiares/segmentation_S25_V20.png) |
     | **40** | **20** | **Recovered (chosen)** | **No** | [`reports/pi/segmentacao/segmentation_comparison.png`](../../reports/pi/segmentacao/segmentation_comparison.png) |
     | 35 | 20 | Recovered | Shadow/background start to leak in | [`reports/pi/segmentacao/estudo_limiares/segmentation_S35_V20.png`](../../reports/pi/segmentacao/estudo_limiares/segmentation_S35_V20.png) |

   - **Known limitation**: edge lesions with saturation as low as the
     background's (e.g., the cut-off in one of the
     `Potato___Early_blight` images) cannot be separated by HSV
     thresholding alone. A future improvement would be to segment the leaf
     with GrabCut or with a segmentation model (e.g., U-Net).
   - **Objective verification** (`python -m processamento_imagens.segmentation_report`, on
     all 18,163 images; results in [`reports/pi/segmentacao/`](../../reports/pi/segmentacao/)):
     - The segmentation steps are in `etapas_segmentacao.png`. The script
       checks that the final mask reimplemented for the figure is
       identical to the one from `segment_leaf` on every image.
     - **Holes**: the old mask (green only) would have internal holes in
       10,604 images (mean of 2.15 per image), mostly in the lesions. With
       the new mask, only 2 images have any hole.
     - **Fraction of the image covered by the leaf**: median of 0.42. The
       fallback (largest contour < 5% of the image) was triggered on 5
       images.
     - **Suspicious cases**: 100 images (0.6%), 49 with leaf < 10% of the
       image and 51 with leaf > 95%. The 12 most suspicious ones are in
       `casos_suspeitos.png`. Inspecting the cases reveals two kinds of
       real failure:
       - **Background leakage** (leaf > 95%): beige/pinkish backgrounds
         with saturation ≥ 40 fall within the lesion range, and the filled
         contour swallows the whole image. Concentrated in
         `Tomato__Tomato_YellowLeaf__Curl_Virus` (32) and
         `Tomato_Spider_mites_Two_spotted_spider_mite` (11).
       - **Whitish leaves** (low saturation): the green mask fragments and
         only a piece of the leaf is kept. This is the cause of 4 of the 5
         fallbacks.
       - Not every case with leaf < 10% is a failure: in a sample of 8,
         about half were leaves that really were small in the photo,
         segmented correctly.
     - Both kinds of failure reinforce the future improvement mentioned
       above (GrabCut or U-Net).
5. **Classical feature extraction** (complementary, in [`src/processamento_imagens/features.py`](../../src/processamento_imagens/features.py)):
   147 color, lesion proportion and texture (GLCM and LBP) descriptors,
   computed only on the leaf pixels (segmentation mask). This step was
   separated from `preprocessing.py` (which only segments) and is used by
   the hybrid classifier (see [Hybrid classifier](05_hybrid_classifier.md)).

## Figures

Segmentation steps (`python -m processamento_imagens.segmentation_report`):

![Segmentation steps](../../reports/pi/segmentacao/etapas_segmentacao.png)

Old vs. new mask (`python -m processamento_imagens.segmentation_report --comparacao`):

![Old vs. new mask comparison](../../reports/pi/segmentacao/segmentation_comparison.png)

Objective verification on all images:

![Segmentation quality](../../reports/pi/segmentacao/qualidade_segmentacao.png)

![Suspicious cases](../../reports/pi/segmentacao/casos_suspeitos.png)

## Intensity histogram (P1 material)

The script `src/processamento_imagens/p1/histogram_analysis.py` generates
[`reports/pi/p1_histograma/histograma_exemplo.png`](../../reports/pi/p1_histograma/histograma_exemplo.png), which
compares the intensity histogram of an image before and after a min-max
normalization. This is P1 material, kept only for the record: this
normalization **is not part of the final pipeline** (the `preprocessing.py`
steps are 1 to 4 above, with no intensity normalization).

The live demonstration of these steps, on a single image, is done with `python src/demo_pi.py` (see [How to reproduce](07_how_to_reproduce.md#image-processing-demo-demo_pipy)).
