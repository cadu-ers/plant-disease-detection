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
                              │
                              ▼  models/
BLOCK 3 — Diagnosis application (CLI)                   src/diagnostico_cli.py
  draw one image from the TEST split → segmentation
        → MobileNetV2 CNN (13 probabilities) + leaf measurements (medidas_lesao.py)
        → EXPERT SYSTEM (IF... THEN rules with certainty factors,
                         knowledge base in knowledge_base.yaml) → DECIDES the diagnosis
        → generative AI (Gemini): plain-language report, checked by the guardrail;
          conversation after the reveal → symptoms → the expert system recomputes
```

The Image Processing block never depends on the AI block. The classical
features are defined in `features.py` and computed in bulk by
`extract_features.py` (block 2), together with the CNN embeddings. The expert
system (IF... THEN rules) is built into the block 3 CLI and can still run on
its own, with facts in JSON (`python -m ia.expert_system.diagnose`).

## Main results

| Result | Value | Details |
|---|---|---|
| Holes in the mask (old green-only vs. new) | 10,604 vs. 2 images with holes (mean 2.15 vs. 0.0002 per image) | [segmentation](docs/docs_EN/02_segmentation.md) |
| Convolution theorem (blur in space vs. in frequency) | max error 8.5e-14 gray levels | [FFT](docs/docs_EN/03_fft.md) |
| CNN on test set (2,725 images) | accuracy 0.949, macro-F1 0.950 | [CNN](docs/docs_EN/04_cnn.md) |
| Hybrid SVM on test set | accuracy 0.979, macro-F1 0.972 (train-only control: 0.976 / 0.971) | [hybrid classifier](docs/docs_EN/05_hybrid_classifier.md) |

## Diagnosis application (CLI)

`python src/diagnostico_cli.py` opens a demo menu (the interface is in
Portuguese). In each analysis it:

1. Draws an image from the **test** split and shows only the split name: the
   file and the true class stay hidden.
2. Shows the diagnosis: the CNN's top 3 classes, the leaf measurements, the
   rules that fired, the open questions and the expert system's conclusion.
3. Shows the plain-language report written by the generative AI, already
   checked by the guardrail.
4. After you press Enter, reveals the image and the true class, compares them
   with the CNN and the expert system, and saves the figures to
   `rotina_demo/cli/<folder>/`.
5. If the diagnosis was not conclusive, starts the conversation: you look at the
   leaf (`original.png`), answer in free text, confirm what was understood and
   see the expert system's **BEFORE x AFTER**, with a new report.

| Option | Effect |
|---|---|
| `--seed N` | repeats the draw for seed N (reproducible draw) |
| `--sem-llm` | runs offline, with the expert system only: no report and no conversation |
| `--sem-cache` | the report and the conversation always call the API (evaluation runs) |

## Generative AI and guardrail

**The expert system makes the diagnosis.** The generative AI (Google Gemini,
through its API, in `src/ia/generativa/`) only explains and converses:

- **Report:** the LLM receives only facts the engine has already produced
  (hypothesis, CF, rules that fired, measurements, open questions and the
  knowledge base's recommendations). It never sees the image, the file path or
  the true class.
- **Guardrail:** the expert system checks the LLM's text for a hypothesis
  different from the engine's, diseases not in the facts, percentages, doses or
  products, irreversible actions without a confirmation condition, and missing
  accents. If the text fails, the LLM tries again with the list of violations;
  if the problem persists, the CLI shows the engine's own text. Every report is
  logged (approved, corrected or rejected).
- **Conversation:** after the reveal, the user answers in free text and the LLM
  turns the answer into symptoms, each with the literal excerpt from the answer,
  which the code checks. The user confirms, the engine recomputes and the CLI
  shows the BEFORE x AFTER. The LLM never picks the disease.
- **Client:** disk cache, retries and a fallback model. Inputs seen before work
  offline, from the cache.

## How to run

```bash
pip install -r requirements.txt
python src/run_pi.py          # Image Processing block
python src/run_ia.py          # AI block (trains the CNN only if no model exists)
python src/demo_pi.py         # Image Processing demo
python src/demo_ia.py         # AI demo
python src/diagnostico_cli.py # diagnosis application (CLI)
```

Python 3.11. Modules in subfolders run with `src/` on the `PYTHONPATH`. For the
report and the conversation, the API key lives **only** in the `GEMINI_API_KEY`
environment variable; never write it to a file. Without the key, the CLI runs
without the report.

```bash
setx GEMINI_API_KEY "your-key"                      # Windows (then open a new terminal)
export GEMINI_API_KEY="your-key"                    # Git Bash
PYTHONPATH=src python -m ia.generativa.testar_conexao   # tests the key
PYTHONPATH=src python -m pytest -q                  # 480 tests
```

The tests never call Gemini: the client gets a fake transport.

Git does **not** store what the CLI needs to run: `models/plant_disease_model.keras`,
`data_split/split_manifest.csv` and the original images in `data/` (see
`.gitignore`). Anyone who clones the repository must generate them with
`run_pi.py` and `run_ia.py` or get these files from the team.

Orchestrator options, manual script-by-script execution, Docker and tests:
[docs/docs_EN/07_how_to_reproduce.md](docs/docs_EN/07_how_to_reproduce.md).

## Limitations

- The knowledge base is a starting point and needs to be validated by an agronomist.
- The knowledge base has no counter-evidence rules: answering "no mottling" does
  not lower the mosaic hypothesis.
- The guardrail ensures the text is faithful to the expert system, not that the diagnosis is correct.
- The model was trained only on PlantVillage (laboratory images).

## Structure

| Group | Scripts in `src/` |
|---|---|
| Orchestration | `run_pi.py`, `run_ia.py`, `run_all.py` (and `common/orchestration.py`) |
| Demo | `demo_pi.py` (Image Processing), `demo_ia.py` (AI) |
| Common | `common/`: `config.py`, `utils.py`, `orchestration.py` |
| Data | `dados/`: `download_dataset.py`, `split_dataset.py`, `data_overview.py` |
| Image Processing | `processamento_imagens/`: `preprocessing.py`, `segmentation_report.py`, `fft_analysis.py`, `features.py`, `p1/histogram_analysis.py` (P1) |
| Artificial Intelligence | `ia/cnn/`: `model_utils.py`, `train_model.py`; `ia/hibrido/`: `extract_features.py`, `hybrid_classifier.py`; `ia/inferencia.py`; `ia/expert_system/` |
| Diagnosis application | `diagnostico_cli.py`; `aplicacao/`: `pipeline.py`, `contrato.py`, `sorteio.py`, `exportacao.py`; `cli/`: `menu.py`, `formatacao.py` |
| Generative AI | `ia/generativa/`: `cliente.py`, `laudo.py`, `guardrail.py`, `conversa.py`, `prompts.py`, `testar_conexao.py` |
| Tests | `tests/` |

Scripts at the root of `src/` run with `python src/<script>.py`; those in
subfolders run as modules (e.g. `python -m dados.split_dataset`), with `src/`
on the `PYTHONPATH` (see [manual execution](docs/docs_EN/07_how_to_reproduce.md#manual-execution-script-by-script)).

Outputs: `reports/pi/` and `reports/ia/` (figures and metrics for each block,
indexed in [reports/pi/LEIAME.md](reports/pi/LEIAME.md) (in Portuguese) and
[reports/ia/LEIAME.md](reports/ia/LEIAME.md) (in Portuguese)), `models/` (models),
`data_processed/` and `data_split/` (segmented images and split),
`features_cache/` (features), `rotina_demo/` (CLI figures, LLM cache and the
guardrail and conversation logs; not in Git).
Full tree in [docs/docs_EN/07_how_to_reproduce.md](docs/docs_EN/07_how_to_reproduce.md#repository-structure).

## Technical documentation

1. [Data](docs/docs_EN/01_data.md): dataset, format and resolution, class imbalance, split
2. [Segmentation](docs/docs_EN/02_segmentation.md): image processing pipeline, thresholds, objective verification, limitations
3. [FFT](docs/docs_EN/03_fft.md): frequency domain, magnitude vs. phase, convolution theorem, descriptors
4. [CNN](docs/docs_EN/04_cnn.md): AI integration, training stages, results
5. [Hybrid classifier](docs/docs_EN/05_hybrid_classifier.md): classical features, SVM, McNemar, control
6. [Expert system](docs/docs_EN/06_expert_system.md): rules, certainty factors, examples
7. [How to reproduce](docs/docs_EN/07_how_to_reproduce.md): requirements, orchestrators, manual execution, Docker, structure
8. [AI usage](docs/uso_de_ia.md) (in Portuguese): log of the AI tools used during development

Full README prior to the reorganization: [docs/historico/README_v1_completo.md](docs/historico/README_v1_completo.md) (in Portuguese).
