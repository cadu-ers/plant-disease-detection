[← voltar ao README](../README.md)

# Como reproduzir

## Requisitos

As dependências do projeto estão listadas em [`requirements.txt`](../requirements.txt).
Para instalar:

```bash
pip install -r requirements.txt
```

Para baixar o data set e prepará-las em `data/`:

```bash
python src/download_dataset.py
```

Para rodar também os testes automatizados, instale `requirements-dev.txt`
(inclui `requirements.txt` e o `pytest`):

```bash
pip install -r requirements-dev.txt
```

## Execução por blocos (orquestradores)

O projeto tem dois blocos. O de **Processamento de Imagens (PI)** roda sozinho;
o de **Inteligência Artificial (IA)** usa as saídas do bloco de PI, mas o bloco
de PI nunca depende do de IA.

```bash
python src/run_pi.py          # bloco 1: Processamento de Imagens
python src/run_ia.py          # bloco 2: IA (treina a CNN só se não houver modelo)
python src/run_all.py         # os dois blocos em sequência
```

Cada orquestrador mostra a lista de etapas no início, roda cada etapa em um
subprocesso, pula etapas cujas saídas já existem ("[pulado] já existe"), para na
primeira etapa com erro (mostrando o comando para reproduzi-lo) e termina com uma
tabela-resumo `etapa | status | tempo`.

| Script | Etapas | Opções |
|---|---|---|
| `run_pi.py` | download_dataset (só com `data/` vazia) → preprocessing → split_dataset → data_overview → segmentation_report → segmentation_report `--comparacao` → fft_analysis → histogram_analysis | `--force`: refaz tudo (o download continua só com `data/` vazia); `--so-figuras`: regera só data_overview, segmentation_report e fft_analysis (exige `data_processed/` e `data_split/`) |
| `run_ia.py` | train_model (só se `models/plant_disease_model.keras` não existir; ~1h30 em CPU) → extract_features → hybrid_classifier → hybrid_classifier `--control-train-only` | `--treinar`: treina a CNN de novo; `--force`: refaz extração e classificador híbrido; `--sem-hibrido`: para depois da CNN |
| `run_all.py` | `run_pi.py` e depois `run_ia.py` | `--force`: repassado aos dois |

O `run_ia.py` verifica antes se `data_processed/` e
`data_split/split_manifest.csv` existem; se não existirem, para com a mensagem
"Rode primeiro o bloco de Processamento de Imagens: python src/run_pi.py".

A demonstração (`python src/demo.py`) continua separada dos orquestradores.

## Execução manual, script por script

```bash
pip install -r requirements.txt
python src/download_dataset.py
python src/preprocessing.py
python src/split_dataset.py
python src/train_model.py
python src/extract_features.py
python src/hybrid_classifier.py
python src/demo.py
```

Material de processamento de imagens (figuras para a apresentação; usam o
manifest do split. Só a comparação com SVM do `fft_analysis.py` usa
`features_cache/handcrafted.csv`, do bloco de IA, e é pulada se o arquivo não existir):

```bash
python src/data_overview.py
python src/segmentation_report.py
python src/fft_analysis.py
```

Para um teste rápido do classificador híbrido (poucas imagens e grade
reduzida): `python src/extract_features.py --limit 5` seguido de
`python src/hybrid_classifier.py --quick`.

Também fazem parte do bloco de PI a comparação máscara antiga x nova e o
histograma da P1:

```bash
python src/segmentation_report.py --comparacao
python src/histogram_analysis.py
```

Sistema especialista (independente do pipeline acima; detalhes na seção
[Sistema especialista](06_sistema_especialista.md)) e testes:

```bash
python src/expert_system/diagnose.py src/expert_system/exemplos/late_blight_tomate.json
python src/expert_system/diagnose.py src/expert_system/exemplos/late_blight_tomate.json --sem-cnn
python src/expert_system/diagnose.py src/expert_system/exemplos/ambiguo_early_target.json --interativo
python -m pytest -q
```

## Docker

Com Docker Desktop e o Docker Compose instalados, construa a imagem:

```bash
docker compose build
```

Execute cada etapa do pipeline em um container. Os arquivos gerados ficam no
workspace, pois o projeto é montado em `/app`:

```bash
docker compose run --rm app python src/download_dataset.py
docker compose run --rm app python src/preprocessing.py
docker compose run --rm app python src/split_dataset.py
docker compose run --rm app python src/train_model.py
docker compose run --rm app python src/extract_features.py
docker compose run --rm app python src/hybrid_classifier.py
docker compose run --rm app python src/demo.py
```

Os orquestradores também funcionam no container:

```bash
docker compose run --rm app python src/run_pi.py
docker compose run --rm app python src/run_ia.py
```

O `Dockerfile` instala só o `requirements.txt` (sem o `pytest`).

## Estrutura do repositório

```
.
├── data/                  # implementar o dataset nessa pasta!
├── docs/                  # documentação técnica (este arquivo e os demais)
│   └── historico/         # README completo anterior à reorganização (registro)
├── features_cache/        # features clássicas, embeddings da CNN e descritores FFT (não versionado)
├── models/
│   ├── plant_disease_model.keras  # modelo atual (gerado por train_model.py, não versionado)
│   ├── class_names.json    # ordem das classes usada no treino
│   └── svm_hybrid.joblib   # melhor pipeline do classificador híbrido (gerado por hybrid_classifier.py)
├── reports/
│   ├── baseline_v1/        # relatórios do modelo antigo + README explicando o que mudou
│   ├── cnn_run1/           # relatórios da rodada 1 da CNN + README com o diagnóstico
│   ├── dados/              # exemplos por classe e resumo do dataset (data_overview.py)
│   ├── fft/                # análise no domínio da frequência (fft_analysis.py)
│   ├── cnn/                # saídas do train_model.py (relatório, matriz de confusão, métricas, histórico)
│   ├── hybrid/             # saídas do hybrid_classifier.py (métricas, comparativo, McNemar, controle)
│   ├── segmentation/       # comparações, etapas e verificação objetiva da segmentação
│   ├── split_summary.csv   # contagem de imagens por classe x split
│   └── histograma_exemplo.png  # saída do histogram_analysis.py (material da P1)
├── src/
│   │   # Orquestração
│   ├── run_pi.py           # bloco 1: Processamento de Imagens, etapa por etapa
│   ├── run_ia.py           # bloco 2: Inteligência Artificial (usa as saídas do bloco 1)
│   ├── run_all.py          # roda run_pi.py e depois run_ia.py
│   ├── orchestration.py    # funções comuns dos orquestradores (subprocesso, plano, resumo)
│   │   # Comuns
│   ├── config.py           # constantes compartilhadas (caminhos, tamanhos, seed, limiares HSV)
│   ├── utils.py            # funções auxiliares (listar classes/imagens, sortear imagem, matriz de confusão)
│   │   # Bloco 1: Processamento de Imagens
│   ├── download_dataset.py # baixa o PlantVillage e copia as classes de tomate/batata para data/
│   ├── preprocessing.py    # pipeline de processamento de imagens (segmentação -> data_processed/)
│   ├── split_dataset.py    # divisão estratificada treino/val/teste (70/15/15) -> data_split/ + manifest
│   ├── data_overview.py    # exemplos por classe e resumo de resolução/formato (reports/dados/)
│   ├── segmentation_report.py # etapas, verificação objetiva e comparação antiga x nova (reports/segmentation/)
│   ├── fft_analysis.py     # FFT: espectros, magnitude x fase, teorema da convolução, descritores
│   ├── features.py         # features clássicas da folha (cor, lesão, GLCM, LBP) para o classificador híbrido
│   ├── histogram_analysis.py  # histograma antes/depois da normalização (material da P1, fora do pipeline)
│   │   # Bloco 2: Inteligência Artificial
│   ├── model_utils.py      # arquitetura do modelo usada no treino (augmentation + MobileNetV2 + cabeça)
│   ├── train_model.py      # treino da CNN
│   ├── extract_features.py # extrai features clássicas + embeddings da CNN -> features_cache/
│   ├── hybrid_classifier.py  # SVMs (clássicas / CNN / híbrido) x CNN no teste + McNemar + controle
│   ├── expert_system/      # sistema especialista: base de conhecimento (YAML), motor de inferência e diagnose.py
│   │   # Demo e testes
│   ├── demo.py             # rotina de demonstração (predição em uma imagem)
│   └── tests/              # testes automatizados do sistema especialista (pytest)
├── requirements.txt        # dependências para rodar o projeto
├── requirements-dev.txt    # requirements.txt + pytest
└── README.md
```
