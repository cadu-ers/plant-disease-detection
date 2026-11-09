[← voltar ao README](../README.md)

# Registro de uso de IA

Exigência do Projeto Prático de IA Generativa: **todo** uso de ferramentas de IA
(inclusive para escrever texto, código e relatório) deve ser descrito e
contextualizado. Este arquivo é a fonte para a seção de ética do relatório e dos
slides.

## Como registrar

Cada integrante acrescenta uma linha **no mesmo dia** em que usou a ferramenta:

- **Ferramenta**: nome e, se souber, o modelo (ex.: Claude, Gemini 3.8 Flash, ChatGPT, Cursor).
- **Para quê**: o que a ferramenta fez (análise, código, texto, imagem...).
- **Prompt principal**: resumo do pedido (o texto completo pode ir em `docs/prompts/`).
- **Revisão humana**: quem revisou e o que foi conferido ou alterado à mão.

## Dois papéis diferentes da IA neste projeto

1. **IA como componente da aplicação** (é o objeto do trabalho): o Gemini, chamado
   por `src/ia/generativa/`, explica o diagnóstico do sistema especialista em
   linguagem simples e converte respostas livres do usuário em sintomas. Os prompts
   ficam no código e são documentados no relatório.
2. **IA como ferramenta de desenvolvimento**: assistentes usados pelo grupo para
   analisar, planejar, programar e escrever. Registrados na tabela abaixo.

## Registro

| Data | Quem | Ferramenta | Para quê | Prompt principal (resumo) | Revisão humana |
|---|---|---|---|---|---|
| 03/11/2026 | Cadu | Claude (app Claude) | Análise do estado do repositório e to-do list; plano de ação em fases | "analise o código, veja onde paramos e monte um to-do list / plano de ação (sem mexer no código)" | _preencher_ |
| 03/11/2026 | Cadu | Claude (app Claude) | Mapeamento do que referencia Docker no projeto, para remoção | "como tirar o Docker sem quebrar nada?" | _preencher_ |
| 03/11/2026 | Cadu | Claude (app Claude) | Guia `COMO_RODAR.md` para rodar o projeto em outra máquina | "me mande o passo a passo em .md para meu amigo rodar" | _preencher_ |
| 08/11/2026 | Cadu | Claude (app Claude) | Análise do PR #2 (CLI de diagnóstico): como se integra com CNN, segmentação e sistema especialista | "o sistema que meu amigo fez funciona junto com o que eu fiz?" | _preencher_ |
| 08/11/2026 | Cadu | Claude (app Claude) | Verificação de vazamento dos pesos da CNN (datas, git log, diff do commit 14ca44c) | "checar se os pesos da CNN vazaram para o teste" | Cadu rodou os comandos git e enviou a saída |
| 08/11/2026 | Cadu | Claude (app Claude) | Proposta de arquitetura da camada de IA generativa e cronograma até 13/11 | "como integrar IA generativa ao nosso projeto?" | _preencher_ |
| 09/11/2026 | Cadu | Claude (app Claude) | Código de `src/ia/generativa/cliente.py`, `testar_conexao.py` e `src/tests/test_generativa_cliente.py`; trechos de `common/config.py` e `requirements.txt`; este arquivo | "vamos começar a fazer a IA generativa" | _preencher_ |
| 09/11/2026 | Cadu | Claude (app Claude) | Novas tentativas automáticas e modelo reserva no cliente (depois dos erros 503/504 do Gemini 3.8 Flash); troca do modelo principal para o 3.5 Flash-Lite | "o teste de conexão deu 503/504" | Cadu testou os dois modelos na própria máquina |
| 09/11/2026 | Cadu | Claude (app Claude) | Laudo em linguagem simples: `src/ia/generativa/prompts.py` (prompt de sistema v1), `laudo.py` (resumo só com fatos do motor, fallback, checagem de doenças fora dos fatos), integração no CLI (`cli/menu.py`, `cli/formatacao.py`, `diagnostico_cli.py --sem-llm`) e testes | "passo 2: laudo do LLM lado a lado com o sistema especialista" | _preencher_ |
| _data_ | _integrante do PR #2_ | _ferramenta usada, se houver_ | CLI de diagnóstico (`src/aplicacao/`, `src/cli/`, `medidas_lesao.py`, `inferencia.py`) | | |

<!-- Ao final do projeto, criar também docs/docs_EN/ai_usage.md (regra de manter PT e EN sincronizados). -->
