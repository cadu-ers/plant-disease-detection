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
   - **Verificação objetiva** (`python src/segmentation_report.py`, em
     todas as 18.163 imagens; resultados em `reports/segmentation/`):
     - As etapas da segmentação estão em `etapas_segmentacao.png`. O
       script confere que a máscara final reimplementada para a figura é
       idêntica à do `segment_leaf` em todas as imagens.
     - **Buracos**: a máscara antiga (só verde) teria buracos internos em
       10.604 imagens (média de 2,15 por imagem), em geral nas lesões. Na
       máscara nova, só 2 imagens têm algum buraco.
     - **Fração da imagem ocupada pela folha**: mediana de 0,42. O
       fallback (maior contorno < 5% da imagem) foi acionado em 5 imagens.
     - **Casos suspeitos**: 100 imagens (0,6%), sendo 49 com folha < 10% da
       imagem e 51 com folha > 95%. As 12 mais suspeitas estão em
       `casos_suspeitos.png`. Inspecionando os casos, há dois tipos de
       falha real:
       - **Fundo que vaza** (folha > 95%): fundos bege/rosados com
         saturação ≥ 40 passam na faixa de lesão, e o contorno preenchido
         engole a imagem inteira. Concentrado em
         `Tomato__Tomato_YellowLeaf__Curl_Virus` (32) e
         `Tomato_Spider_mites_Two_spotted_spider_mite` (11).
       - **Folhas esbranquiçadas** (pouca saturação): a máscara verde se
         fragmenta e só um pedaço da folha é mantido. É a causa de 4 dos 5
         fallbacks.
       - Nem todo caso com folha < 10% é falha: numa amostra de 8, cerca de
         metade eram folhas realmente pequenas na foto, segmentadas
         corretamente.
     - Os dois tipos de falha reforçam a melhoria futura citada acima
       (GrabCut ou U-Net).
5. **Extração de features clássicas** (complementar, em `src/features.py`):
   147 descritores de cor, proporção de lesão e textura (GLCM e LBP),
   calculados só nos pixels da folha (máscara da segmentação). Essa etapa
   foi separada do `preprocessing.py` (que só segmenta) e é usada pelo
   classificador híbrido (ver seção "Classificador híbrido" abaixo).

## FFT (domínio da frequência)

Implementado em `src/fft_analysis.py`, com figuras em `reports/fft/`.

**Preparação de cada imagem**:
1. Tom de cinza da imagem original 224x224, antes do blur, porque o blur
   apagaria as altas frequências.
2. Fundo preenchido com a média de cinza da folha. O degrau entre a folha
   e o fundo preto criaria um "+" enorme no espectro.
3. Subtração da média, para o termo DC não dominar as baixas frequências.
4. Janela de Hann 2D, que reduz o vazamento espectral causado pelas bordas
   da imagem.
5. `np.fft.fft2` + `fftshift`.

**Baixas x altas frequências**: o centro do espectro (baixas frequências)
representa a forma geral da folha e as variações lentas de tom, como manchas
grandes. A periferia (altas frequências) representa bordas, nervuras e
textura fina. Em `fft_saudavel_vs_doente.png`, o passa-baixa gaussiano
(D0 = 15) deixa só a silhueta e as manchas grandes, e o passa-alta realça
bordas, nervuras e o contorno das lesões. No espectro radial por classe
(`espectro_radial_por_classe.png`), as duas classes **saudáveis** têm a
maior razão de energia alta/baixa frequência, e
`Tomato__Tomato_YellowLeaf__Curl_Virus` a menor. A energia em alta
frequência vem principalmente da textura fina e nítida da folha; as lesões
são manchas de tamanho médio e não aumentam essa razão.

**Magnitude x fase** (`fft_magnitude_vs_fase.png`): a reconstrução só com a
fase mantém a estrutura da folha (posição das bordas e das lesões). A
reconstrução só com a magnitude perde a estrutura: a magnitude diz "quanto"
de cada frequência existe, mas não onde.

**Teorema da convolução com o blur do pipeline**
(`blur_espaco_vs_frequencia.png`): aplicar o `GaussianBlur` 5x5 do
`preprocessing.py` no espaço, com borda circular, dá o mesmo resultado que
multiplicar a FFT da imagem pela FFT do kernel (com zero-padding para
224x224). O erro máximo absoluto é de 8,5e-14 níveis de cinza, ou seja,
erro de arredondamento. A FFT do kernel mostra que o blur é um filtro
passa-baixa.

**FFT como descritor de textura**: para todas as imagens, 11 descritores do
espectro de potência (energia normalizada em 8 faixas radiais espaçadas em
log, razão alta/baixa, centroide espectral e entropia espectral), em
`features_cache/fft_features.csv`. SVM (RBF, C = 10, `class_weight`
balanceado), treinado só no treino e avaliado no teste:

| Features | Nº | Acurácia | F1-macro |
|---|---|---|---|
| Só FFT | 11 | 0.380 | 0.330 |
| Só clássicas | 147 | 0.937 | 0.915 |
| Clássicas + FFT | 158 | 0.939 | 0.920 |

Sozinha, a FFT radial discrimina mal as 13 classes: ela resume a textura
em poucos números e não tem informação de cor. Somada às features
clássicas, o ganho é **pequeno** (+0.002 de acurácia, cerca de 5 imagens de
2.725, e +0.006 de F1-macro), então não a incorporamos ao classificador
híbrido. O resultado "só clássicas" difere um pouco do `SVM_classicas` do
classificador híbrido porque aqui os hiperparâmetros são fixos (C = 10),
sem a busca na validação.

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

### Resultados da CNN

Métricas no **conjunto de teste**: 2.725 imagens que nunca foram usadas no
treino nem na seleção de modelo (a validação serve só para o early stopping e
para a redução do learning rate).

**Evolução do modelo**

| | Baseline v1 | Rodada 1 | Rodada 2 (atual) |
|---|---|---|---|
| Medido em | validação (3.632 imagens) | teste (2.725 imagens) | teste (2.725 imagens) |
| Acurácia | 0.88 | 0.815 | **0.949** |
| F1-macro | 0.86 | 0.796 | **0.950** |
| Recall `Tomato_Early_blight` | 0.59 | 0.51 | **0.86** |
| Recall `Tomato__Target_Spot` | 0.73 | 0.64 | **0.88** |
| Recall `Tomato_Late_blight` | 0.87 | 0.62 | **0.92** |

- **Baseline v1**: segmentação antiga (máscara só de verde, que apagava as
  lesões), divisão 80/20 sem conjunto de teste e métrica medida na própria
  validação usada no early stopping, por isso otimista e não comparável
  diretamente com as rodadas seguintes. O relatório do baseline só tem duas
  casas decimais.
- **Rodada 1**: segmentação corrigida, divisão 70/15/15 com teste separado,
  data augmentation e class weight balanceado. O fine-tuning, porém,
  desestabilizou o modelo porque as camadas BatchNormalization destravadas
  passaram a atualizar suas estatísticas: na primeira época do fine-tuning a
  loss de treino saltou de 0.53 para 1.40 (pico visível em
  `reports/cnn_run1/training_history.png`).
- **Rodada 2**: BatchNormalization congelado no fine-tuning, 50 camadas
  destravadas, learning rate 1e-4 com ReduceLROnPlateau e até 20 épocas de
  fine-tuning. Na prática, a etapa 1 rodou 10 épocas (melhor: 7) e o
  fine-tuning 18 épocas (melhor: 14), com o learning rate reduzido até 2.7e-6.

Os recalls de `Tomato_Early_blight` e `Tomato__Target_Spot` continuam os mais
baixos entre essas classes: as duas doenças produzem manchas parecidas.

O relatório completo (precision/recall/F1 por classe), a matriz de confusão,
as métricas e o histórico de treino da CNN estão em `reports/cnn/`; os
resultados do classificador híbrido estão em `reports/hybrid/` (ver
[Resultados do classificador híbrido](#resultados-do-classificador-híbrido)).

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

Todos os modelos são avaliados no mesmo conjunto de teste (2.725 imagens).

**Protocolo principal** (hiperparâmetros escolhidos na validação; SVMs
reajustados em treino + validação, 15.438 imagens):

| Modelo | Acurácia | F1-macro | F1-weighted |
|---|---|---|---|
| CNN | 0.949 | 0.950 | 0.949 |
| SVM_classicas | 0.946 | 0.922 | 0.946 |
| SVM_cnn | 0.966 | 0.960 | 0.965 |
| **SVM_hibrido** | **0.979** | **0.972** | **0.979** |

**Controle: SVMs ajustados só no treino.** No protocolo principal, os SVMs
usam ~21% a mais de dados que a CNN (15.438 contra 12.714 imagens; a CNN só
usou a validação para o early stopping). Para separar o efeito da fusão do
efeito desses dados extras, o controle (`python src/hybrid_classifier.py
--control-train-only`) reusa os mesmos hiperparâmetros e treina os SVMs só nas
12.714 imagens de treino:

| Modelo | Acurácia | F1-macro | F1-weighted |
|---|---|---|---|
| CNN | 0.949 | 0.950 | 0.949 |
| SVM_classicas | 0.941 | 0.915 | 0.940 |
| SVM_cnn | 0.965 | 0.960 | 0.965 |
| **SVM_hibrido** | **0.976** | **0.971** | **0.976** |

**Teste de McNemar** (casos em que só um dos dois modelos acerta):

| Protocolo | Par (A x B) | Casos discordantes | Só A acerta | Só B acerta | p-valor |
|---|---|---|---|---|---|
| Principal | CNN x SVM_hibrido | 111 | 15 | 96 | 1.2e-15 |
| Principal | SVM_cnn x SVM_hibrido | 52 | 8 | 44 | 4.0e-07 |
| Controle | CNN x SVM_hibrido | 102 | 14 | 88 | 2.8e-14 |
| Controle | SVM_cnn x SVM_hibrido | 43 | 6 | 37 | 1.6e-06 |

Em todos os pares o `SVM_hibrido` acerta mais nos casos discordantes, com
p < 0,05.

**Interpretação** (com base no controle, em que todos os modelos aprendem só
com o treino):

- **Ganho das features clássicas** (`SVM_cnn` x `SVM_hibrido`: mesmos
  embeddings, mesmo protocolo, só muda a presença das features clássicas): o
  F1-macro sobe de 0.960 para 0.971 e a acurácia de 0.965 para 0.976. Nos 43
  casos em que os dois discordam, o híbrido acerta 37 (p = 1.6e-06). As
  features clássicas trazem informação que os embeddings sozinhos não têm.
- **CNN x SVM_hibrido**: a vantagem se mantém no controle. A acurácia do
  híbrido cai só de 0.979 para 0.976 sem os dados de validação, e continua
  acima da CNN (0.949), com 88 contra 14 casos discordantes (p = 2.8e-14).
  Portanto, a vantagem não vem dos ~21% de dados extras. Parte dela vem de
  trocar a cabeça softmax da CNN por um SVM sobre os mesmos embeddings
  (`SVM_cnn` já fica em 0.965 de acurácia, contra 0.949 da CNN; esse par não
  passou por McNemar), e o restante das features clássicas (item anterior).
- **Por que as clássicas sozinhas têm F1-macro menor** (0.922 contra 0.946 de
  acurácia no protocolo principal): o F1-macro dá o mesmo peso a todas as
  classes, e `Potato___healthy` tem só 106 imagens de treino (23 no teste).
  Nessa classe, o `SVM_classicas` tem recall de 0.52 (F1 0.67); sem ela, o
  F1-macro das outras 12 classes seria cerca de 0.94. No híbrido, os embeddings da CNN
  recuperam boa parte dessa classe: recall de 0.87 no protocolo principal e
  0.91 no controle (F1 0.93 e 0.95). Ainda fica abaixo da CNN, que acerta as 23
  imagens (recall 1.00). Com só 23 imagens, cada imagem equivale a ~0.04 de
  recall.

**Melhores hiperparâmetros** (escolhidos pelo F1-macro na validação):

| Modelo | C | gamma | Componentes da PCA | F1-macro na validação |
|---|---|---|---|---|
| SVM_classicas | 100 | 0.01 | — | 0.924 |
| SVM_cnn | 10 | scale | 256 | 0.968 |
| SVM_hibrido | 10 | scale | 128 | 0.973 |

Gráficos: `reports/hybrid/comparativo.png` (protocolo principal) e
`reports/hybrid/controle_refit_treino/comparativo.png` (controle), cada um com
um painel em escala completa e um painel com zoom (eixo começando em 0.85),
onde as diferenças entre os modelos ficam visíveis.

**Limitação**: o PlantVillage é formado por fotos de laboratório (fundo
uniforme, uma folha isolada, iluminação controlada). Os resultados de todos os
modelos, CNN e SVMs, podem não se manter em fotos de campo, com fundo
complexo, várias folhas e iluminação variável (*domain shift*). Validar em
imagens de campo é necessário antes de qualquer uso prático.

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
│   ├── dados/              # exemplos por classe e resumo do dataset (data_overview.py)
│   ├── fft/                # análise no domínio da frequência (fft_analysis.py)
│   ├── cnn/                # saídas do train_model.py (relatório, matriz de confusão, métricas, histórico)
│   ├── hybrid/             # saídas do hybrid_classifier.py (métricas, comparativo, McNemar)
│   ├── segmentation/       # comparações, etapas e verificação objetiva da segmentação
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
│   ├── segmentation_report.py # etapas da segmentação + verificação objetiva (reports/segmentation/)
│   ├── fft_analysis.py     # FFT: espectros, magnitude x fase, teorema da convolução, descritores
│   ├── data_overview.py    # exemplos por classe e resumo de resolução/formato (reports/dados/)
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

Material de processamento de imagens (figuras para a apresentação; usam o
manifest do split e, no caso da FFT, também `features_cache/handcrafted.csv`):

```bash
python src/data_overview.py
python src/segmentation_report.py
python src/fft_analysis.py
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
