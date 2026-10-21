# Detecção de Doenças em Folhas de Plantas

Pipeline de processamento de imagens integrado a um modelo de IA (transfer
learning) para identificação de doenças em folhas de tomate e batata, a
partir do dataset PlantVillage.

## Integrantes da equipe

- Carlos Eduardo Rodrigues Silva
- Daniel Lucarelli Cerri
- Melck Silva de Oliveira Nascimento
- Murilo Moretto Marques
- Vinicius dos Santos

## Problema e objetivo

Doenças em folhas de tomate e batata aparecem como manchas, amarelamento e
alterações de textura. O objetivo é, a partir de uma foto da folha, dizer se a
planta está saudável ou qual das 11 doenças ela tem (13 classes, 18.163 imagens
do PlantVillage), segmentando a folha sem apagar justamente as lesões.

## Pipeline

```
BLOCO 1 — Processamento de Imagens (roda sozinho)              src/run_pi.py
  imagem → pré-processamento (resize 224x224, blur gaussiano 5x5, BGR→HSV)
         → segmentação (máscara verde + lesão, contorno externo preenchido)
         → análise FFT (espectros, filtros, teorema da convolução)
         → extração de características (features.py: cor, lesão, GLCM, LBP;
                                         fft_analysis.py: descritores FFT)
                              │
                              ▼  data_processed/, data_split/
BLOCO 2 — Inteligência Artificial (usa as saídas do bloco 1)    src/run_ia.py
  classificação: CNN MobileNetV2 (transfer learning)
                 SVM híbrido (embeddings da CNN + características clássicas)
         → diagnóstico (classe da folha)
```

O bloco de Processamento de Imagens nunca depende do bloco de IA. As
características clássicas são definidas em `features.py` e calculadas em
massa pelo `extract_features.py` (bloco 2), junto com os embeddings da CNN. O
sistema especialista (regras SE... ENTÃO) roda à parte, com fatos em JSON.

## Resultados principais

| Resultado | Valor | Detalhes |
|---|---|---|
| Buracos na máscara (antiga só verde x nova) | 10.604 x 2 imagens com buracos (média 2,15 x 0,0002 por imagem) | [segmentação](docs/02_segmentacao.md) |
| Teorema da convolução (blur no espaço x na frequência) | erro máximo 8,5e-14 níveis de cinza | [FFT](docs/03_fft.md) |
| CNN no teste (2.725 imagens) | acurácia 0.949, F1-macro 0.950 | [CNN](docs/04_cnn.md) |
| SVM híbrido no teste | acurácia 0.979, F1-macro 0.972 (controle só com treino: 0.976 / 0.971) | [classificador híbrido](docs/05_classificador_hibrido.md) |

## Como rodar

```bash
pip install -r requirements.txt
python src/run_pi.py          # bloco de Processamento de Imagens
python src/run_ia.py          # bloco de IA (treina a CNN só se não houver modelo)
python src/demo_pi.py         # demonstração do Processamento de Imagens
python src/demo_ia.py         # demonstração da IA
```

Opções dos orquestradores, execução manual script por script, Docker e testes:
[docs/07_como_reproduzir.md](docs/07_como_reproduzir.md).

## Estrutura

| Grupo | Scripts em `src/` |
|---|---|
| Orquestração | `run_pi.py`, `run_ia.py`, `run_all.py` (e `common/orchestration.py`) |
| Demo | `demo_pi.py` (Processamento de Imagens), `demo_ia.py` (IA) |
| Comuns | `common/`: `config.py`, `utils.py`, `orchestration.py` |
| Dados | `dados/`: `download_dataset.py`, `split_dataset.py`, `data_overview.py` |
| Processamento de Imagens | `processamento_imagens/`: `preprocessing.py`, `segmentation_report.py`, `fft_analysis.py`, `features.py`, `p1/histogram_analysis.py` (P1) |
| Inteligência Artificial | `ia/cnn/`: `model_utils.py`, `train_model.py`; `ia/hibrido/`: `extract_features.py`, `hybrid_classifier.py`; `ia/expert_system/` |
| Testes | `tests/` |

Os scripts da raiz de `src/` rodam com `python src/<script>.py`; os das
subpastas rodam como módulo (ex.: `python -m dados.split_dataset`), com `src/`
no `PYTHONPATH` (ver [execução manual](docs/07_como_reproduzir.md#execução-manual-script-por-script)).

Saídas: `reports/pi/` e `reports/ia/` (figuras e métricas de cada bloco, com
índices em [reports/pi/LEIAME.md](reports/pi/LEIAME.md) e
[reports/ia/LEIAME.md](reports/ia/LEIAME.md)), `models/` (modelos),
`data_processed/` e `data_split/` (imagens segmentadas e divisão),
`features_cache/` (features).
Árvore completa em [docs/07_como_reproduzir.md](docs/07_como_reproduzir.md#estrutura-do-repositório).

## Documentação técnica

1. [Dados](docs/01_dados.md): dataset, formato e resolução, desbalanceamento, split
2. [Segmentação](docs/02_segmentacao.md): pipeline de PI, limiares, verificação objetiva, limitações
3. [FFT](docs/03_fft.md): domínio da frequência, magnitude x fase, teorema da convolução, descritores
4. [CNN](docs/04_cnn.md): integração com IA, etapas do treino, resultados
5. [Classificador híbrido](docs/05_classificador_hibrido.md): features clássicas, SVM, McNemar, controle
6. [Sistema especialista](docs/06_sistema_especialista.md): regras, fatores de certeza, exemplos
7. [Como reproduzir](docs/07_como_reproduzir.md): requisitos, orquestradores, execução manual, Docker, estrutura

README completo anterior à reorganização: [docs/historico/README_v1_completo.md](docs/historico/README_v1_completo.md).
