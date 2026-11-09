Contexto do projeto plant-disease-detection (Python 3.11, Windows + Git Bash):

- Diagnóstico de doenças em folhas de tomate e batata (PlantVillage, 13 classes).
- Pipeline: segmentação (src/processamento_imagens) -> CNN MobileNetV2 (src/ia/cnn,
  src/ia/inferencia.py) -> medidas da folha (medidas_lesao.py) -> SISTEMA ESPECIALISTA
  (src/ia/expert_system: regras + fatores de certeza, base em knowledge_base.yaml)
  -> CLI (src/diagnostico_cli.py, src/cli/, src/aplicacao/).
- Camada de IA generativa em src/ia/generativa/: cliente.py (Gemini, cache, novas
  tentativas, modelo reserva), prompts.py, laudo.py (laudo em linguagem simples).
- Quem DECIDE o diagnóstico é o sistema especialista. O LLM só explica e conversa.

Regras:
1. Rodar com PYTHONPATH=src. Testes: python -m pytest -q (todos devem passar).
2. Nunca chamar a API do Gemini nos testes: usar o parâmetro `transporte` falso do
   ClienteLLM. A chave fica só na variável GEMINI_API_KEY; nunca escrever a chave.
3. Contrato anti-vazamento: antes do Enter de revelação, nada na tela nem no prompt do
   LLM pode conter caminho, nome de arquivo, nome de classe do dataset (Tomato_...,
   Potato_...), split ou classe real. O ResultadoDiagnostico não tem esses dados.
4. Texto da CLI: só ASCII e letras acentuadas do português (console do Windows).
5. Código, comentários e mensagens em português do Brasil, no estilo dos arquivos
   existentes (docstring no topo explicando o módulo).
6. docs/ e docs/docs_EN/ são mantidos em par: mudou um, avise sobre o outro.
7. Não fazer commit nem push; eu reviso e commito.
