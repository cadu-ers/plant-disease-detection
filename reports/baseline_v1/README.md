# Baseline v1 (modelo de setembro/2026)

Resultados do primeiro modelo, **antes** das correções. Servem de "antes" para
comparar com o modelo corrigido.

Como esse modelo foi treinado:

- **Segmentação antiga**: máscara HSV só de verde. As lesões (manchas
  marrons/amarelas) ficavam fora da máscara e eram removidas da imagem, ou seja,
  o modelo via a folha sem o sintoma.
- **Split 80/20 (treino/validação), sem conjunto de teste.**
- **Sem data augmentation e sem class_weight** (desbalanceamento de ~21x entre
  classes).
- **Acurácia de 0.88 medida na VALIDAÇÃO**, o mesmo conjunto usado no early
  stopping. Por isso é uma estimativa otimista.

Arquivos:

| Arquivo | Conteúdo |
|---|---|
| `classification_report.txt` | precision/recall/F1 por classe (validação, 3632 imagens) |
| `training_history.png` | acurácia de treino e validação por época |
| `resultado_20260919_172330.png` / `.txt` | exemplo da rotina de demonstração |

O arquivo do modelo não foi mantido no repositório (não é usado por nenhum script);
apenas os relatórios desta rodada foram preservados.
