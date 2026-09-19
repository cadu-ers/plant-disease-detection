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
processamento de imagens (implementado em `src/preprocessing.py`) segue as
seguintes etapas:

1. **Leitura e redimensionamento**: cada imagem é carregada e padronizada
   para 224x224 pixels, garantindo uma entrada uniforme para as etapas
   seguintes.
2. **Conversão de espaço de cor (RGB → HSV)**: o espaço HSV separa
   informação de matiz (cor), saturação e brilho, o que facilita a
   segmentação por cor em comparação ao RGB.
3. **Suavização / remoção de ruído**: aplicação de um filtro Gaussian Blur
   (kernel 5x5), reduzindo ruídos de captura e pequenas imperfeições que
   poderiam atrapalhar a segmentação.
4. **Segmentação da folha**: utiliza uma máscara de cor em HSV
   (`cv2.inRange`) para isolar os tons de verde característicos da folha,
   seguida de operações morfológicas (abertura e fechamento) para limpar
   ruídos da máscara. O resultado é a folha isolada do fundo da imagem.
   - **Limitação observada**: como a segmentação é baseada em uma faixa
     fixa de cor, regiões da própria folha saudável com brilho, sombra ou
     pequenos furos de inseto também podem ser removidas (aparecendo como
     "buracos" pretos na imagem segmentada), mesmo sem indicar doença.
     Optamos por manter essa abordagem por ser simples e eficiente,
     mantendo esses "buracos" como uma limitação conhecida do método.
     Na próxima entrega (P2), pretendemos aprimorar essa etapa utilizando
     preenchimento de contornos (`cv2.fillPoly`) para manter apenas o
     contorno externo da folha como máscara.
5. **Extração de features clássicas** (complementar): histograma de cor
   (H, S, V) e descritores de textura via matriz de co-ocorrência (GLCM) —
   contraste, homogeneidade, energia e correlação — salvos em
   `features.csv` para fins de análise.

## Descrição detalhada da integração com IA — Pipeline completo

A saída do pipeline de processamento de imagens (imagens segmentadas)
alimenta um modelo de classificação baseado em **transfer learning**
(implementado em `src/train_model.py`). O pipeline completo do projeto é:

```
Imagem bruta
   → Pré-processamento (resize, HSV, blur)
   → Segmentação da folha (remoção de fundo)
   → [ramo IA]     → Modelo MobileNetV2 (pré-treinada em ImageNet)
                    → Camada de classificação customizada (GlobalAveragePooling + Dropout + Dense)
                    → Fine-tuning das últimas camadas da base
   → Predição da classe (saudável ou tipo de doença)
```

Etapas do treino:

1. **Carregamento**: as imagens de `data_processed/` são divididas
   automaticamente em 80% treino / 20% validação.
2. **Transfer learning**: a MobileNetV2 é carregada com pesos do ImageNet
   e inicialmente congelada; apenas a camada de classificação (13 classes)
   é treinada.
3. **Fine-tuning**: as últimas ~30 camadas da MobileNetV2 são
   descongeladas e re-treinadas com uma taxa de aprendizado bem menor,
   permitindo que o modelo se especialize nas características específicas
   das doenças de folhas.
4. **Early stopping**: o treino é interrompido automaticamente caso a
   acurácia de validação pare de melhorar, evitando overfitting e tempo de
   treino desnecessário.

### Resultados

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

## Dataset

- [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)

Utilizamos um subconjunto do dataset, com as classes de **tomate** e
**batata** (13 classes no total).

## Estrutura do repositório

```
.
├── data/                  # dataset original (não versionado, ver .gitignore)
├── data_processed/        # imagens após o pipeline de PI (não versionado)
├── models/                # modelo treinado (não versionado)
├── reports/               # gráfico de treino e relatório de classificação
├── src/
│   ├── preprocessing.py    # pipeline de processamento de imagens
│   ├── model_utils.py      # arquitetura do modelo (compartilhada entre treino e demo)
│   ├── train_model.py      # treino do modelo de IA
│   ├── demo.py             # rotina de demonstração (predição em uma imagem)
│   └── histogram_analysis.py  # script de documentação (histograma antes/depois da normalização)
├── features.csv           # features clássicas extraídas (não versionado)
├── requirements.txt
└── README.md
```

## Como executar

```bash
pip install -r requirements.txt
python src/preprocessing.py
python src/train_model.py
python src/demo.py
```

## Observações técnicas

**Carregamento do modelo (`demo.py`)**: o modelo é salvo em formato `.h5`
ao final do treino. Esse formato, ao usar operações do TensorFlow
aplicadas diretamente sobre os tensores dentro da arquitetura (no caso, o
`preprocess_input` da MobileNetV2), gera camadas internas que o Keras não
consegue desserializar de volta via `load_model()`. Para contornar isso
sem precisar retreinar, `demo.py` reconstrói a arquitetura do modelo em
código (função `build_model()` em `model_utils.py`, reaproveitada também
pelo treino) e carrega apenas os pesos salvos, por nome de camada
(`model.load_weights(..., by_name=True)`), em vez de carregar o modelo
completo.