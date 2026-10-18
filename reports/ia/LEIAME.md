# Resultados do bloco de Inteligência Artificial

[← voltar ao README](../../README.md) · [resultados de Processamento de Imagens](../pi/LEIAME.md)

Índice de `reports/ia/`. Gerado por `python src/run_ia.py` (detalhes em
[docs/04_cnn.md](../../docs/04_cnn.md) e
[docs/05_classificador_hibrido.md](../../docs/05_classificador_hibrido.md)).
Todas as métricas são do conjunto de teste (2.725 imagens), exceto o baseline.

## Números principais

| Modelo | Acurácia | F1-macro | Fonte |
|---|---|---|---|
| CNN (rodada 2, atual) | 0.949 | 0.950 | [`metrics_cnn.json`](cnn/metrics_cnn.json) |
| SVM_hibrido (protocolo principal) | 0.979 | 0.972 | [`comparativo.csv`](hibrido/comparativo.csv) |
| SVM_hibrido (controle: só treino) | 0.976 | 0.971 | [`comparativo.csv`](hibrido/controle_refit_treino/comparativo.csv) |
| CNN rodada 1 | 0.815 | 0.796 | [`metrics_cnn.json`](historico/cnn_run1/metrics_cnn.json) |
| Baseline v1 (medido na validação) | 0.88 | 0.86 | [`classification_report.txt`](historico/baseline_v1/classification_report.txt) |

McNemar CNN x SVM_hibrido: 111 casos discordantes, 96 a favor do híbrido
(p = 1,2e-15); no controle, 102 casos, 88 a favor do híbrido (p = 2,8e-14)
([`mcnemar.txt`](hibrido/mcnemar.txt),
[controle](hibrido/controle_refit_treino/mcnemar.txt)).

## Figuras e tabelas

| Arquivo | O que mostra | Item |
|---|---|---|
| [`cnn/training_history.png`](cnn/training_history.png) | Acurácia e loss por época nas duas etapas do treino da CNN. | Resultados |
| [`cnn/confusion_matrix.png`](cnn/confusion_matrix.png) | Matriz de confusão da CNN no teste, normalizada por linha. | Resultados |
| [`cnn/classification_report.txt`](cnn/classification_report.txt) | Precision, recall e F1 por classe da CNN no teste. | Resultados |
| [`hibrido/comparativo.png`](hibrido/comparativo.png) | CNN x SVMs (clássicas, CNN, híbrido): acurácia e F1-macro, com painel de zoom. | Resultados |
| [`hibrido/controle_refit_treino/comparativo.png`](hibrido/controle_refit_treino/comparativo.png) | O mesmo comparativo com os SVMs ajustados só no treino (controle). | Resultados |
| [`hibrido/confusion_matrix_SVM_hibrido.png`](hibrido/confusion_matrix_SVM_hibrido.png) | Matriz de confusão do SVM híbrido no teste. | Resultados |
| [`hibrido/mcnemar.txt`](hibrido/mcnemar.txt) | Teste de McNemar: CNN x SVM_hibrido e SVM_cnn x SVM_hibrido. | Resultados |
| [`historico/baseline_v1/`](historico/baseline_v1/README.md) | Primeiro modelo (segmentação antiga, sem teste): métrica otimista. | Limitações |
| [`historico/cnn_run1/`](historico/cnn_run1/README.md) | Rodada 1 da CNN: o fine-tuning desestabilizou por causa do BatchNorm. | Limitações |
