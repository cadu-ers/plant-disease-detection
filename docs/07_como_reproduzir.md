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

## Demonstração do Processamento de Imagens (`demo_pi.py`)

Mostra, em uma imagem, todo o bloco de PI: as etapas da segmentação, a FFT e
os histogramas HSV com as faixas e limiares reais do `config.py`, além das
medidas da imagem comparadas com a mediana da classe. Não usa nada do bloco de
IA (não importa `tensorflow`, não lê `models/` nem `features_cache/`, exceto o
`fft_features.csv` gerado pelo `fft_analysis.py`).

```bash
python src/demo_pi.py                              # imagem aleatória do TESTE (manifest)
python src/demo_pi.py --classe Tomato_Early_blight # aleatória do teste dessa classe
python src/demo_pi.py caminho/da/imagem.jpg        # imagem específica
python src/demo_pi.py --seed 7                     # sorteio reprodutível
python src/demo_pi.py --sem-janela                 # só salva, não abre janela
```

As opções podem ser combinadas (ex.: `--classe Tomato_Early_blight --seed 42
--sem-janela`); só não é possível usar um caminho de imagem junto com
`--classe`. Para uma imagem informada, a classe real e o split vêm do
manifest; se ela não estiver no manifest, a classe vem do nome da pasta (se for
uma classe de `data/`) ou fica desconhecida.

A figura tem 3 linhas:

1. **Segmentação**: original, blur gaussiano, canal H, máscara verde, máscara
   de lesão, máscara final (contorno preenchido) e folha segmentada. O script
   confere que a máscara final é idêntica à de `segment_image`.
2. **FFT**: imagem preparada (como no `fft_analysis.py`), log da magnitude,
   passa-baixa (D0 = 15), passa-alta e um quadro com as medidas.
3. **Histogramas (espaço HSV da imagem suavizada)**: canal H da imagem inteira
   e só da folha, com as faixas de matiz verde, de lesão e a faixa H 25–35, que
   pertence às duas; canais S e V da folha e do fundo, com os limiares mínimos
   S e V da máscara de lesão.

Medidas (no terminal e no quadro da figura): fração da imagem ocupada pela
folha, proporção de lesão na folha, % dos pixels da folha nas faixas verde e de
lesão e razão de energia alta/baixa frequência (corte em 0.25 de Nyquist). A
fração de folha, a proporção de lesão e a razão alta/baixa vêm acompanhadas da
mediana da classe real, calculada a partir de
`reports/pi/segmentacao/qualidade_segmentacao.csv` e
`features_cache/fft_features.csv`; se algum deles não existir, aparece só o
valor da imagem.

Saídas em `rotina_demo/pi/`: `resultado_<timestamp>.png` (figura),
`resultado_<timestamp>.json` (caminho, classe real, split e medidas) e
`ultima.json` (cópia do último resultado, usada pelo `demo_ia --ultima-pi`).

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
│   ├── pi/                 # bloco 1: Processamento de Imagens (índice em pi/LEIAME.md)
│   │   ├── dados/          # exemplos por classe, resumo do dataset e split_summary.csv
│   │   ├── segmentacao/    # etapas, comparação antiga x nova e verificação objetiva da segmentação
│   │   │   └── estudo_limiares/  # comparações com outros limiares S/V da máscara de lesão
│   │   ├── fft/            # análise no domínio da frequência (fft_analysis.py)
│   │   └── p1_histograma/  # histograma antes/depois da normalização (material da P1)
│   └── ia/                 # bloco 2: Inteligência Artificial (índice em ia/LEIAME.md)
│       ├── cnn/            # saídas do train_model.py (relatório, matriz de confusão, métricas, histórico)
│       ├── hibrido/        # saídas do hybrid_classifier.py (métricas, comparativo, McNemar, controle)
│       └── historico/      # rodadas antigas: baseline_v1/ e cnn_run1/, cada uma com README
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
│   ├── data_overview.py    # exemplos por classe e resumo de resolução/formato (reports/pi/dados/)
│   ├── segmentation_report.py # etapas, verificação objetiva e comparação antiga x nova (reports/pi/segmentacao/)
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
│   ├── demo_pi.py          # demonstração do Processamento de Imagens (segmentação, FFT, histogramas HSV)
│   ├── demo.py             # rotina de demonstração (predição em uma imagem)
│   └── tests/              # testes automatizados do sistema especialista (pytest)
├── requirements.txt        # dependências para rodar o projeto
├── requirements-dev.txt    # requirements.txt + pytest
└── README.md
```
