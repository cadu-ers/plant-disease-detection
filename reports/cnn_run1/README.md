# CNN — rodada 1 (outubro/2026)

Primeira rodada com as correções (segmentação nova, split 70/15/15 com teste,
data augmentation e class_weight balanceado). Fica preservada para comparar com
a rodada seguinte.

## Configuração

| Etapa | Épocas | Camadas treináveis | Learning rate |
|---|---|---|---|
| 1. Cabeça (base congelada) | 8 | só a cabeça | 1e-3 |
| 2. Fine-tuning | 5 | últimas 30 da MobileNetV2 | 1e-5 |

EarlyStopping em `val_loss`, paciência 3, restaurando os melhores pesos.

## Resultado (conjunto de teste, 2725 imagens)

- Acurácia: **0.815**
- F1-macro: **0.796**

Detalhes em `classification_report.txt`, `confusion_matrix.png` e `metrics_cnn.json`.

## Diagnóstico (ver `training_history.png`)

1. **Salto na entrada do fine-tuning**: na 1ª época da etapa 2, a loss de treino
   subiu de 0.53 para 1.40 e a acurácia de treino caiu de 0.81 para 0.65. As
   camadas BatchNormalization destravadas passaram a atualizar suas estatísticas
   (o `training=False` na chamada da base não bastou nesta versão do Keras).
2. **Fine-tuning curto**: com 5 épocas, a etapa 2 terminou com a acurácia de
   validação ainda subindo.

Correções na rodada seguinte: BatchNorm congelado no fine-tuning, 50 camadas
destravadas com lr 1e-4, mais épocas (10 + 20) e ReduceLROnPlateau.
