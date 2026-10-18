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
   histograma de cor (H, S, V) e descritores de textura via matriz de
   co-ocorrência (GLCM) — contraste, homogeneidade, energia e correlação.
   Essa etapa foi separada do `preprocessing.py` (que agora só segmenta) e
   será usada pelo classificador híbrido (features clássicas + CNN).

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
├── models/
│   ├── plant_disease_model.keras  # modelo atual (gerado por train_model.py, não versionado)
│   ├── class_names.json    # ordem das classes usada no treino
│   └── baseline_v1/        # modelo antigo, antes das correções (não versionado)
├── reports/
│   ├── baseline_v1/        # relatórios do modelo antigo + README explicando o que mudou
│   ├── cnn/                # saídas do train_model.py (relatório, matriz de confusão, métricas, histórico)
│   ├── segmentation/       # comparações de segmentação (check_segmentation.py)
│   ├── split_summary.csv   # contagem de imagens por classe x split
│   └── histograma_exemplo.png  # saída do histogram_analysis.py
├── src/
│   ├── config.py           # constantes compartilhadas (caminhos, tamanhos, seed, limiares HSV)
│   ├── utils.py            # funções auxiliares para listar classes/imagens e sortear uma imagem
│   ├── download_dataset.py # baixa o PlantVillage e copia as classes de tomate/batata para data/
│   ├── preprocessing.py    # pipeline de processamento de imagens (segmentação -> data_processed/)
│   ├── split_dataset.py    # divisão estratificada treino/val/teste (70/15/15) -> data_split/ + manifest
│   ├── features.py         # extração de features clássicas (cor + textura) para o classificador híbrido
│   ├── check_segmentation.py  # compara a máscara antiga e a nova (reports/segmentation/)
│   ├── model_utils.py      # arquitetura do modelo (augmentation + MobileNetV2 + cabeça)
│   ├── train_model.py      # treino do modelo de IA
│   ├── demo.py             # rotina de demonstração (predição em uma imagem)
│   └── histogram_analysis.py  # script de documentação (histograma antes/depois da normalização)
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
python src/demo.py
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
docker compose run --rm app python src/demo.py
```