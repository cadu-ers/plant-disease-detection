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

## Descrição detalhada do projeto — Pipeline de Processamento de Imagens

O projeto tem como objetivo identificar, a partir de uma imagem de folha,
se a planta está saudável ou apresenta algum tipo de doença. O pipeline de
processamento de imagens segue as etapas abaixo (1 a 4 em
`src/preprocessing.py`; a 5 em `src/features.py`):

1. **Leitura e redimensionamento**: cada imagem é carregada e padronizada
   para 224x224 pixels, garantindo uma entrada uniforme para as etapas
   seguintes.
2. **Suavização / remoção de ruído**: aplicação de um filtro Gaussian Blur
   (kernel 5x5), reduzindo ruídos de captura e pequenas imperfeições que
   poderiam atrapalhar a segmentação.
3. **Conversão de espaço de cor (BGR → HSV)**: o OpenCV lê as imagens em
   BGR; o espaço HSV separa informação de matiz (cor), saturação e brilho,
   o que facilita a segmentação por cor.
4. **Segmentação da folha**: a máscara é a união (`cv2.bitwise_or`) de
   duas faixas de cor em HSV (`cv2.inRange`): os tons de verde da folha e
   os tons de lesão (manchas marrons/amarelas, H ~8–35). Em seguida, são
   aplicadas operações morfológicas (abertura e fechamento) para limpar
   ruídos. Por fim, com `cv2.findContours` mantemos apenas o maior
   contorno externo (a folha) e o preenchemos por inteiro
   (`cv2.drawContours` com `cv2.FILLED`), de modo que tudo o que está
   dentro da folha entra na máscara e o fundo é removido. Se nenhum
   contorno for encontrado ou se o maior ocupar menos de 5% da imagem,
   usa-se a máscara de cor como fallback.
   - **Motivo da mudança**: a versão anterior usava apenas a máscara de
     verde, e as lesões das doenças (que não são verdes) ficavam fora
     dela, virando "buracos" pretos na imagem segmentada — ou seja, o
     sintoma da doença era justamente o que estava sendo apagado. A
     comparação entre as duas abordagens pode ser gerada com
     `python src/check_segmentation.py` (salva em
     `reports/segmentation/segmentation_comparison.png`).
   - **Escolha dos limiares da máscara de lesão**: o matiz (H ~8–35) foi
     fixado e testamos os mínimos de saturação (S) e brilho (V) nas mesmas
     12 imagens (seed 42), com
     `python src/check_segmentation.py --s <S> --v <V> --out <arquivo>`:

     | S | V | Lesões na borda | Vazamento de fundo | Imagem |
     |---|---|---|---|---|
     | 40 | 30 | Lesões escuras da borda cortadas | Não | `reports/segmentation/segmentation_S40_V30.png` |
     | 25 | 20 | Recuperadas | Sim: o fundo bege/cinza tem matiz na faixa do marrom, e só a saturação separa os dois | `reports/segmentation/segmentation_S25_V20.png` |
     | **40** | **20** | **Recuperadas (escolhido)** | **Não** | `reports/segmentation/segmentation_comparison.png` |
     | 35 | 20 | Recuperadas | Sombra/fundo começam a entrar | `reports/segmentation/segmentation_S35_V20.png` |

   - **Limitação conhecida**: lesões de borda com saturação tão baixa
     quanto a do fundo (ex.: o recorte em uma das imagens de
     `Potato___Early_blight`) não são separáveis apenas por limiar de HSV.
     Uma melhoria futura seria segmentar a folha com GrabCut ou com um
     modelo de segmentação (ex.: U-Net).
5. **Extração de features clássicas** (complementar, em `src/features.py`):
   147 descritores de cor, proporção de lesão e textura (GLCM e LBP),
   calculados só nos pixels da folha (máscara da segmentação). Essa etapa
   foi separada do `preprocessing.py` (que só segmenta) e é usada pelo
   classificador híbrido (ver seção "Classificador híbrido" abaixo).

## Descrição detalhada da integração com IA — Pipeline completo

A saída do pipeline de processamento de imagens (imagens segmentadas)
alimenta um modelo de classificação baseado em **transfer learning**
(implementado em `src/train_model.py`). O pipeline completo do projeto é:

```
Imagem bruta
   → Pré-processamento (resize, blur, HSV)
   → Segmentação da folha (remoção de fundo)
   → [ramo IA]     → Data augmentation (flip, rotação, zoom, contraste; só no treino)
                    → Modelo MobileNetV2 (pré-treinada em ImageNet)
                    → Camada de classificação customizada (GlobalAveragePooling + Dropout + Dense)
                    → Fine-tuning das últimas camadas da base
   → Predição da classe (saudável ou tipo de doença)
```

Etapas do treino:

1. **Carregamento**: as imagens de `data_processed/` são divididas por
   `src/split_dataset.py` em 70% treino / 15% validação / 15% teste
   (estratificado por classe, seed 42), com a divisão registrada em
   `data_split/split_manifest.csv`. A validação é usada só no early
   stopping; as métricas finais são calculadas no teste, que nunca entra
   no treino.
2. **Data augmentation**: flip horizontal e vertical, rotação, zoom e
   contraste aleatórios, aplicados por camadas dentro do próprio modelo.
   Elas só atuam no treino; na validação, no teste e na demo a imagem
   passa sem alteração.
3. **Class weight balanceado**: o dataset é desbalanceado (~21x entre
   `Tomato__Tomato_YellowLeaf__Curl_Virus`, com 3208 imagens, e
   `Potato___healthy`, com 152). Cada classe recebe um peso inversamente
   proporcional à sua frequência no treino (`compute_class_weight("balanced")`),
   para que as classes pequenas não sejam ignoradas.
4. **Transfer learning**: a MobileNetV2 é carregada com pesos do ImageNet
   e inicialmente congelada; apenas a camada de classificação (13 classes)
   é treinada, por até 10 épocas com learning rate 1e-3.
5. **Fine-tuning**: as últimas 50 camadas da MobileNetV2 são descongeladas
   e re-treinadas por até 20 épocas com learning rate 1e-4, permitindo que
   o modelo se especialize nas características específicas das doenças de
   folhas. As camadas **BatchNormalization continuam congeladas**: se
   fossem destravadas, passariam a atualizar suas estatísticas com os
   batches do treino, o que desestabilizou o modelo no início do
   fine-tuning na rodada anterior (`reports/cnn_run1/`).
6. **Early stopping e ReduceLROnPlateau**: em cada etapa, o treino é
   interrompido quando a `val_loss` (loss de validação) para de melhorar
   por 4 épocas, e o learning rate é multiplicado por 0.3 sempre que ela
   fica 2 épocas sem melhorar (mínimo 1e-6). Ao fim de cada etapa, o
   modelo fica com os pesos da época de menor `val_loss`, mesmo que a
   etapa chegue ao limite de épocas sem o early stopping disparar.

### Resultados

> Resultados abaixo são do modelo baseline (reports/baseline_v1/); serão atualizados com o modelo corrigido.

O modelo atingiu **89% de acurácia** no conjunto de validação. Classes
com sintomas visuais bem distintos (ex: `Tomato_YellowLeaf_Curl_Virus`,
`Potato_Early_blight`) obtiveram F1-score acima de 0.92. Já classes com
sintomas visualmente semelhantes entre si (ex: `Tomato_Early_blight` e
`Tomato__Target_Spot`) apresentaram mais confusão, com recall mais baixo
(0.58 e 0.75, respectivamente) — um comportamento esperado dado que essas
doenças compartilham padrões de manchas parecidos.

O gráfico de acurácia por época e o relatório completo de métricas
(precision/recall/F1 por classe) estão disponíveis em `reports/`.

## Classificador híbrido (CNN + features clássicas + SVM)

Além da CNN, o projeto treina um classificador **SVM** que combina dois tipos
de informação: os **embeddings** aprendidos pela CNN e **features clássicas**
de processamento de imagens. Ele é comparado com a própria CNN no mesmo
conjunto de teste.

```
split fixo (data_split/split_manifest.csv: treino / validação / teste)
   │
   ├─ imagem original ──→ segmentação (segment_image) ──→ features clássicas da folha (147)
   │                                                        → StandardScaler ─────────────┐
   │                                                                                      ├─→ SVM (RBF) → classe
   └─ imagem segmentada ─→ CNN treinada ─→ embedding GAP (1280)                            │
                                            → StandardScaler → PCA (64/128/256) ──────────┘
```

Implementação: `src/features.py` (features), `src/extract_features.py`
(extrai e salva tudo em `features_cache/`) e `src/hybrid_classifier.py`
(treino, avaliação e comparativo).

**Features clássicas** (todas calculadas só nos pixels da folha, usando a
máscara da segmentação):

| Grupo | Nº | O que captura |
|---|---|---|
| Cor: histogramas H (32 bins), S (16), V (16) | 64 | distribuição de matiz, saturação e brilho da folha (amarelamento, manchas, palidez) |
| Cor: momentos (média, desvio, assimetria) em HSV e Lab | 18 | cor média da folha e quão espalhada/assimétrica ela é |
| Proporção de lesão | 1 | fração da folha nos tons de lesão (`LOWER_LESION`/`UPPER_LESION`): quanto da folha está manchada |
| Textura GLCM (32 níveis; distâncias 1, 2, 3; 0°/45°/90°/135°) | 36 | contraste, dissimilaridade, homogeneidade, energia, correlação e ASM; média e desvio entre os ângulos (invariante à rotação) |
| Textura LBP uniforme (P=8, R=1) e (P=16, R=2) | 28 | micro-padrões de textura (pontos, bordas, manchas finas) em duas escalas |

No GLCM, o fundo é codificado como nível 0 e a linha/coluna 0 da matriz é
descartada antes de calcular as propriedades, para que só pares de pixels da
folha contem.

Três SVMs são comparados com a CNN:

| Modelo | Entrada |
|---|---|
| `SVM_classicas` | só features clássicas |
| `SVM_cnn` | só embeddings da CNN (com PCA) |
| `SVM_hibrido` | embeddings (com PCA) + features clássicas |
| `CNN` | softmax da própria CNN (referência) |

Decisões de projeto:

- **Por que SVM com kernel RBF**: funciona bem com algumas centenas de
  features densas e padronizadas, separa as classes de forma não linear e,
  com `class_weight="balanced"`, compensa o desbalanceamento entre as classes.
- **Por que PCA no bloco da CNN**: os embeddings têm 1280 dimensões e as
  features clássicas, 147. Sem redução, a distância do kernel RBF seria
  dominada pelos embeddings e as features clássicas quase não pesariam. A PCA
  equilibra os dois blocos.
- **Por que escolher os hiperparâmetros na validação**: a CNN foi treinada
  nas imagens de treino, então os embeddings do treino ficam "fáceis demais".
  Uma validação cruzada dentro do treino superestimaria todos os candidatos e
  enviesaria a escolha. Por isso, o `GridSearchCV` usa um `PredefinedSplit`:
  ajusta no treino, mede o F1-macro na validação e, com os melhores
  hiperparâmetros (C, gamma e nº de componentes da PCA), reajusta em
  treino + validação. **O teste nunca é usado para treinar nem para escolher
  hiperparâmetros.**
- **Teste de McNemar**: compara dois classificadores nas mesmas imagens de
  teste, olhando só os casos em que eles discordam (um acerta e o outro erra).
  Se a diferença fosse acaso, esses casos se dividiriam meio a meio; o teste
  binomial exato dá o p-valor (p < 0,05 indica diferença estatisticamente
  significativa). É aplicado entre `CNN` e `SVM_hibrido` e entre `SVM_cnn` e
  `SVM_hibrido`.

Saídas em `reports/hybrid/`: métricas e relatório por classe de cada modelo,
resultados do grid search, matriz de confusão do `SVM_hibrido`,
`comparativo.csv`/`comparativo.png` e `mcnemar.txt`. O melhor pipeline
híbrido fica em `models/svm_hybrid.joblib`.

### Resultados do classificador híbrido

Resultados: (preencher após rodar)

## Sistema especialista

O projeto também possui um sistema especialista em `src/expert_system/`.
Ele recebe fatos sobre a planta e aplica regras do tipo **SE... ENTÃO...**
para ranquear doenças, calcular fatores de certeza e explicar o resultado.

A base de conhecimento está em `src/expert_system/knowledge_base.yaml` e
contém os sintomas, as 13 doenças, as regras, os fatores de certeza e as
recomendações gerais. O motor também pode receber probabilidades produzidas
pela CNN por meio do campo `cnn_prior`.

### Executar um diagnóstico de exemplo

Na raiz do projeto, execute:

```bash
python src/expert_system/diagnose.py src/expert_system/exemplos/late_blight_tomate.json
```

Para executar sem utilizar a evidência da CNN:

```bash
python src/expert_system/diagnose.py src/expert_system/exemplos/late_blight_tomate.json --sem-cnn
```

Para utilizar o modo interativo:

```bash
python src/expert_system/diagnose.py src/expert_system/exemplos/ambiguo_early_target.json --interativo
```

O resultado apresenta a hipótese principal, as evidências das regras, as
hipóteses alternativas, os sintomas pendentes e recomendações gerais. O
valor exibido como **CF** é um fator de certeza do sistema, não uma
probabilidade estatística calibrada.

Atualmente, o sistema especialista pode ser executado de forma independente
com fatos informados em JSON. A integração automática entre imagem,
segmentação, extração de medidas, CNN e sistema especialista é uma etapa
futura do projeto.

### Executar os testes

```bash
python -m pytest -q
```

Os testes cobrem o carregamento da base de conhecimento, o mapeamento das
classes, o cálculo dos fatores de certeza, as regras de diagnóstico, as
explicações e o modo interativo.

## Requisitos

As dependências do projeto estão listadas em [`requirements.txt`](./requirements.txt).
Para instalar:

```bash
pip install -r requirements.txt
```

Para baixar o data set e prepará-las em `data/`:

```bash
python src/download_dataset.py
```

## Dataset

- [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)

Utilizamos um subconjunto do dataset, com as classes de **tomate** e
**batata** (13 classes no total).

## Estrutura do repositório

```
.
├── data/                  # implementar o dataset nessa pasta!
├── features_cache/        # features clássicas + embeddings da CNN (extract_features.py, não versionado)
├── models/
│   ├── plant_disease_model.keras  # modelo atual (gerado por train_model.py, não versionado)
│   ├── class_names.json    # ordem das classes usada no treino
│   └── svm_hybrid.joblib   # melhor pipeline do classificador híbrido (gerado por hybrid_classifier.py)
├── reports/
│   ├── baseline_v1/        # relatórios do modelo antigo + README explicando o que mudou
│   ├── cnn/                # saídas do train_model.py (relatório, matriz de confusão, métricas, histórico)
│   ├── hybrid/             # saídas do hybrid_classifier.py (métricas, comparativo, McNemar)
│   ├── segmentation/       # comparações de segmentação (check_segmentation.py)
│   ├── split_summary.csv   # contagem de imagens por classe x split
│   └── histograma_exemplo.png  # saída do histogram_analysis.py
├── src/
│   ├── config.py           # constantes compartilhadas (caminhos, tamanhos, seed, limiares HSV)
│   ├── utils.py            # funções auxiliares (listar classes/imagens, sortear imagem, matriz de confusão)
│   ├── download_dataset.py # baixa o PlantVillage e copia as classes de tomate/batata para data/
│   ├── preprocessing.py    # pipeline de processamento de imagens (segmentação -> data_processed/)
│   ├── split_dataset.py    # divisão estratificada treino/val/teste (70/15/15) -> data_split/ + manifest
│   ├── features.py         # features clássicas da folha (cor, lesão, GLCM, LBP) para o classificador híbrido
│   ├── extract_features.py # extrai features clássicas + embeddings da CNN -> features_cache/
│   ├── hybrid_classifier.py  # SVMs (clássicas / CNN / híbrido) x CNN no teste + McNemar
│   ├── check_segmentation.py  # compara a máscara antiga e a nova (reports/segmentation/)
│   ├── model_utils.py      # arquitetura do modelo usada no treino (augmentation + MobileNetV2 + cabeça)
│   ├── train_model.py      # treino do modelo de IA
│   ├── demo.py             # rotina de demonstração (predição em uma imagem)
│   ├── histogram_analysis.py  # script de documentação (histograma antes/depois da normalização)
│   ├── expert_system/      # sistema especialista: base de conhecimento (YAML), motor de inferência e diagnose.py
│   └── tests/              # testes automatizados do sistema especialista (pytest)
├── requirements.txt
└── README.md
```

## Como executar

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

Para um teste rápido do classificador híbrido (poucas imagens e grade
reduzida): `python src/extract_features.py --limit 5` seguido de
`python src/hybrid_classifier.py --quick`.

Sistema especialista (independente do pipeline acima; detalhes na seção
[Sistema especialista](#sistema-especialista)) e testes:

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
