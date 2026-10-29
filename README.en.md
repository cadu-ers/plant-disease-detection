# Plant Leaf Disease Detection

<!-- When editing this file, also update README.md -->

[🇧🇷 Português](README.md) | **🇺🇸 English**

Image processing pipeline integrated with an AI model (transfer learning) to
identify diseases in tomato and potato leaves, based on the PlantVillage
dataset.

## Team members

- Carlos Eduardo Rodrigues Silva
- Daniel Lucarelli Cerri
- Melck Silva de Oliveira Nascimento
- Murilo Moretto Marques
- Vinicius dos Santos

## Problem and goal

Diseases in tomato and potato leaves show up as spots, yellowing and texture
changes. The goal is, from a photo of the leaf, to tell whether the plant is
healthy or which of the 11 diseases it has (13 classes, 18,163 PlantVillage
images), segmenting the leaf without erasing the lesions themselves.

## Pipeline

```
BLOCK 1 — Image Processing (runs standalone)                   src/run_pi.py
  image → preprocessing (resize 224x224, 5x5 Gaussian blur, BGR→HSV)
        → segmentation (green + lesion mask, filled external contour)
        → FFT analysis (spectra, filters, convolution theorem)
        → feature extraction (features.py: color, lesion, GLCM, LBP;
                              fft_analysis.py: FFT descriptors)
                              │
                              ▼  data_processed/, data_split/
BLOCK 2 — Artificial Intelligence (uses block 1 outputs)       src/run_ia.py
  classification: MobileNetV2 CNN (transfer learning)
                  hybrid SVM (CNN embeddings + classical features)
        → diagnosis (leaf class)
```

The Image Processing block never depends on the AI block. The classical
features are defined in `features.py` and computed in bulk by
`extract_features.py` (block 2), together with the CNN embeddings. The expert
system (IF... THEN rules) runs separately, with facts in JSON.

## Main results

| Result | Value | Details |
|---|---|---|
| Holes in the mask (old green-only vs. new) | 10,604 vs. 2 images with holes (mean 2.15 vs. 0.0002 per image) | [segmentation](docs/docs_EN/02_segmentation.md) |
| Convolution theorem (blur in space vs. in frequency) | max error 8.5e-14 gray levels | [FFT](docs/docs_EN/03_fft.md) |
| CNN on test set (2,725 images) | accuracy 0.949, macro-F1 0.950 | [CNN](docs/docs_EN/04_cnn.md) |
| Hybrid SVM on test set | accuracy 0.979, macro-F1 0.972 (train-only control: 0.976 / 0.971) | [hybrid classifier](docs/docs_EN/05_hybrid_classifier.md) |

## How to run

```bash
pip install -r requirements.txt
python src/run_pi.py          # Image Processing block
python src/run_ia.py          # AI block (trains the CNN only if no model exists)
python src/demo_pi.py         # Image Processing demo
python src/demo_ia.py         # AI demo
```

Orchestrator options, manual script-by-script execution, Docker and tests:
[docs/docs_EN/07_how_to_reproduce.md](docs/docs_EN/07_how_to_reproduce.md).

## Structure

| Group | Scripts in `src/` |
|---|---|
| Orchestration | `run_pi.py`, `run_ia.py`, `run_all.py` (and `common/orchestration.py`) |
| Demo | `demo_pi.py` (Image Processing), `demo_ia.py` (AI) |
| Common | `common/`: `config.py`, `utils.py`, `orchestration.py` |
| Data | `dados/`: `download_dataset.py`, `split_dataset.py`, `data_overview.py` |
| Image Processing | `processamento_imagens/`: `preprocessing.py`, `segmentation_report.py`, `fft_analysis.py`, `features.py`, `p1/histogram_analysis.py` (P1) |
| Artificial Intelligence | `ia/cnn/`: `model_utils.py`, `train_model.py`; `ia/hibrido/`: `extract_features.py`, `hybrid_classifier.py`; `ia/expert_system/` |
| Tests | `tests/` |

Scripts at the root of `src/` run with `python src/<script>.py`; those in
subfolders run as modules (e.g. `python -m dados.split_dataset`), with `src/`
on the `PYTHONPATH` (see [manual execution](docs/docs_EN/07_how_to_reproduce.md#manual-execution-script-by-script)).

Outputs: `reports/pi/` and `reports/ia/` (figures and metrics for each block,
indexed in [reports/pi/LEIAME.md](reports/pi/LEIAME.md) (in Portuguese) and
[reports/ia/LEIAME.md](reports/ia/LEIAME.md) (in Portuguese)), `models/` (models),
`data_processed/` and `data_split/` (segmented images and split),
`features_cache/` (features).
Full tree in [docs/docs_EN/07_how_to_reproduce.md](docs/docs_EN/07_how_to_reproduce.md#repository-structure).

## Technical documentation

1. [Data](docs/docs_EN/01_data.md): dataset, format and resolution, class imbalance, split
2. [Segmentation](docs/docs_EN/02_segmentation.md): image processing pipeline, thresholds, objective verification, limitations
3. [FFT](docs/docs_EN/03_fft.md): frequency domain, magnitude vs. phase, convolution theorem, descriptors
4. [CNN](docs/docs_EN/04_cnn.md): AI integration, training stages, results
5. [Hybrid classifier](docs/docs_EN/05_hybrid_classifier.md): classical features, SVM, McNemar, control
6. [Expert system](docs/docs_EN/06_expert_system.md): rules, certainty factors, examples
7. [How to reproduce](docs/docs_EN/07_how_to_reproduce.md): requirements, orchestrators, manual execution, Docker, structure

Full README prior to the reorganization: [docs/historico/README_v1_completo.md](docs/historico/README_v1_completo.md) (in Portuguese).
