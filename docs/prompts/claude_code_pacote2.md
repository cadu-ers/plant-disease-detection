# Prompts para o Claude Code: pacote 2 (correções depois do primeiro laudo real)

Use na branch `feature/ia-generativa`. Rode **um prompt por vez**, na ordem. Depois de
cada um: leia o diff, rode `python -m pytest -q` e só então faça o commit e passe para o
próximo. Registre cada uso em `docs/uso_de_ia.md` (exigência de ética do trabalho).

Dica: comece cada prompt pedindo só o **plano** ("mostre o plano antes de editar") e
aprove antes de ele mexer nos arquivos.

---

## Prompt 0: contexto do projeto (uma vez por sessão, ou salve como CLAUDE.md)

> Crie o arquivo `CLAUDE.md` na raiz com o conteúdo abaixo (ou só leia, se já existir) e confirme que entendeu.

```text
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
```

---

## Prompt 1: corrigir o bug do YAML e impedir que volte

```text
Bug na base de conhecimento: em src/ia/expert_system/knowledge_base.yaml, duas
recomendações começam com "Não há cura: ..." (doenças tomate_yellow_leaf_curl, perto
da linha 226, e tomate_mosaic, perto da linha 236). Por causa do ": ", o YAML lê cada
uma como um dicionário {'Não há cura': '...'} em vez de texto, e a CLI mostra
"{'Não há cura': ...}" na explicação.

Faça:
1. Corrija as duas linhas colocando o texto entre aspas duplas, sem mudar o conteúdo.
2. Procure no YAML inteiro qualquer outro texto (recomendacoes, pergunta, descricao,
   porque, nome, agente) que o YAML esteja lendo como algo que não é string, e corrija
   do mesmo jeito.
3. Em src/ia/expert_system/kb.py, na validação estrita que já existe, faça o
   carregamento levantar ValueError com mensagem clara (doença/sintoma/regra + campo)
   se algum desses campos de texto não for str.
4. Crie testes em src/tests/test_expert_system.py: (a) a KB real carrega e todos esses
   campos são str; (b) uma KB mínima com uma recomendação "Texto: outro" sem aspas
   levanta ValueError apontando a doença.
5. Rode python -m pytest -q e mostre o resultado.
```

**Pronto quando:** a explicação do mosaico mostra "Não há cura: remover..." como texto e todos os testes passam.

---

## Prompt 2: prompt v2 do laudo, `--seed` no CLI e formato

```text
O primeiro laudo real (mosaico, seed 594525008) mostrou 3 problemas no prompt v1
(src/ia/generativa/prompts.py) e no laudo (src/ia/generativa/laudo.py):

A) O motor disse INCONCLUSIVO, mas o laudo mandou "remova e destrua as plantas" como
   ordem. Ação irreversível em diagnóstico incerto é perigosa.
B) O "Por quê" só disse "a pontuação ficou em 0,49". O real motivo: a rede neural deu
   ~100% para mosaico, mas NENHUMA regra disparou (as medidas mostram a folha quase toda
   verde). O laudo deveria dizer de onde veio a evidência.
C) Formato: título do bloco numa linha e texto com recuo torto.

Faça:
1. prompts.py: separe em SISTEMA_BASE (identidade, público, tom, limites que valem para
   qualquer tarefa: nunca diagnosticar sozinho, nunca indicar produto/dose/marca,
   sempre recomendar agrônomo, não sair do tema, dizer quando não sabe) e
   TAREFA_LAUDO (regras do laudo). SISTEMA_LAUDO = SISTEMA_BASE + TAREFA_LAUDO.
   VERSAO_PROMPT_LAUDO = 2. Mantenha o texto v1 em um comentário ou constante
   SISTEMA_LAUDO_V1, para o relatório.
2. Regras novas na TAREFA_LAUDO:
   - Se "situacao" não for "conclusivo": recomendações SEMPRE condicionais ("se o
     diagnóstico for confirmado por um técnico, ..."), nunca como ordem; ações
     irreversíveis (destruir, arrancar, eliminar plantas) só com essa condição; a
     prioridade é responder às perguntas pendentes.
   - O "Por quê" deve dizer de onde veio a evidência, usando o novo campo
     "origem_da_evidencia" (abaixo), em linguagem simples.
   - Formato: cada bloco começa com o título sozinho numa linha ("O que o sistema
     encontrou:", "Por quê:", "Perguntas para você:", "O que fazer:") seguido de um
     parágrafo curto; perguntas uma por linha começando com "- ".
3. laudo.py, resumir_diagnostico: acrescente "origem_da_evidencia" com
   {"forca_pela_rede_neural": cf_cnn da hipótese principal (2 casas),
    "forca_pelas_regras": cf_regras (2 casas),
    "regras_que_dispararam": quantas regras a favor da hipótese principal}
   (os valores estão em diag.resultado.melhor.cf_cnn / cf_regras). Sem hipótese: null.
4. cli/formatacao.py, formatar_laudo: imprima títulos de bloco com recuo 4 e os
   parágrafos com recuo 6, sem recuo pendurado extra (não use _quebrar_linhas para o
   laudo; quebre por bloco). Linhas "- " das perguntas com recuo 6.
5. diagnostico_cli.py: aceite --seed N (inteiro) para repetir um sorteio; repasse via
   Dependencias(gerar_seed=lambda: N). Mantenha --sem-llm. Recusar opções desconhecidas
   (já existe; atualize a mensagem).
6. Atualize/crie testes (sem rede): o resumo tem origem_da_evidencia correta; o prompt
   de sistema contém as regras novas; --seed repassa a seed; formatar_laudo com blocos.
7. Rode python -m pytest -q.
```

**Pronto quando:** `python src/diagnostico_cli.py --seed 594525008` mostra a mesma folha de mosaico com o laudo v2. Salve a tela como `docs/exemplos/laudo_v2_mosaico.txt`, ao lado do v1.

---

## Prompt 3: guardrail v2 (sistema especialista controlando o LLM)

```text
Quero usar o sistema especialista como guardrail do LLM no laudo. Hoje existe só uma
checagem que AVISA quando o texto cita doença fora do diagnóstico
(laudo.doencas_fora_dos_fatos). Implemente o guardrail v2 em um módulo novo
src/ia/generativa/guardrail.py, usado por laudo.py:

1. Saída estruturada: gerar_laudo passa a usar cliente.gerar_json com um JSON Schema:
   {
     "o_que_encontrou": str, "por_que": str, "o_que_fazer": str,
     "perguntas": [str],
     "hipotese_citada": str | null,          # nome exato da hipótese principal ou null
     "situacao_declarada": "conclusivo" | "inconclusivo" | "sem_hipotese",
     "recomendacoes_usadas": [int],          # índices de resumo["recomendacoes"]
     "perguntas_usadas": [int]               # índices de resumo["perguntas_pendentes"]
   }
   Ajuste a TAREFA_LAUDO para pedir esse formato. O texto final é montado em código a
   partir dos blocos (mesmos títulos do prompt 2).
2. validar_laudo(dados, resumo, kb) -> lista de violações (texto curto cada), checando:
   - hipotese_citada é a hipótese principal do resumo (ou null quando não há);
   - situacao_declarada == resumo["situacao"];
   - índices de recomendacoes_usadas e perguntas_usadas existem;
   - nenhuma doença fora dos fatos nos textos (reusar doencas_fora_dos_fatos);
   - sem "%" nos textos; sem padrões de dose/produto (ex.: números com ml, g, kg, L,
     "/ha", "dose", "fungicida", "inseticida", "bactericida");
   - se situacao != "conclusivo": "o_que_fazer" não pode ter ação irreversível
     (destruir/destrua, arrancar/arranque, eliminar/elimine planta(s)) sem um marcador
     de condição ("se for confirmad", "caso seja confirmad", "após a confirmação",
     "depois de confirmad");
   - se há perguntas pendentes e situacao != "conclusivo": "perguntas" não vazia.
3. Laço: 1ª geração -> validar. Se houver violações, gerar de novo UMA vez,
   acrescentando ao prompt "Sua resposta anterior violou: ..." com a lista. Se ainda
   violar: Laudo com fonte="motor" (texto do sistema especialista) e aviso com as
   violações. Laudo ganha os campos violacoes (tuple), tentativas (int) e
   status_guardrail ("aprovado" | "corrigido" | "reprovado").
4. Registro para o relatório: acrescentar uma linha JSON por laudo em
   rotina_demo/guardrail_log.jsonl (não versionado): data/hora, modelo, versão do
   prompt, status_guardrail, tentativas, violações. NUNCA caminho, classe real ou seed.
   Falha ao gravar o log não pode quebrar o laudo.
5. cli/formatacao.py: mostrar uma linha "Guardrail: aprovado na 1a tentativa" /
   "corrigido na 2a tentativa (violações: ...)" / "reprovado: mostrando o texto do
   sistema especialista (violações: ...)".
6. Testes (sem rede, transporte falso): resposta válida aprovada; resposta com doença
   fora dos fatos corrigida na 2a; duas respostas ruins -> reprovado e fonte motor;
   ação irreversível sem condição em caso inconclusivo é violação, com condição não é;
   "%" e dose são violações; o log não contém caminho nem nome de classe do dataset;
   JSON inválido conta como violação.
7. Rode python -m pytest -q.
```

**Pronto quando:** o CLI mostra a linha do guardrail e o `rotina_demo/guardrail_log.jsonl` vai juntando linhas, que dão os números da visão crítica (quantos aprovados, corrigidos e reprovados).

---

## Prompt 4: corrigir o aviso falso sobre a CNN

> Este texto é do colega do PR #2: combine com ele antes.

```text
O aviso AVISO_PESOS_LOCAIS (src/ia/inferencia.py) e o "Lembrete" da tela de revelação
(src/cli/formatacao.py, formatar_revelacao) dizem que os pesos da CNN foram gravados
antes da divisão 70/15/15 e com a segmentação antiga. Isso foi verificado e é FALSO:

- data_processed/ e data_split/ foram gerados em 18/10 ~12:20, com a segmentação
  corrigida (commit "checkpoint 1: segmentação corrigida", 12:26);
- o commit 14ca44c (12:39) só moveu os limiares para config.py, sem mudar valores;
- o modelo (rodada 2) foi gravado em 18/10 15:36, treinado só com data_split/train,
  validação no early stopping e teste só na avaliação (acurácia 0.949, F1 0.950,
  reports/ia/cnn/metrics_cnn.json);
- o demo_ia reproduziu as predições do features_cache em 26/26 imagens.

Faça:
1. Reescreva AVISO_PESOS_LOCAIS com um texto curto e verdadeiro, por exemplo:
   "CNN rodada 2: treinada só com o conjunto de treino (divisão 70/15/15) e com a
   segmentação atual; acurácia de 0,949 no conjunto de teste. A saída do softmax é uma
   pontuação relativa, não uma probabilidade calibrada."
   Mantenha o nome da constante (os testes a importam).
2. Ajuste o "Lembrete" da revelação para algo como: "Lembrete: esta comparação é de UMA
   imagem, só para demonstração. A avaliação da CNN está em reports/ia/LEIAME.md."
3. Atualize a docstring de inferencia.py e qualquer teste que verifique o texto antigo.
4. Rode python -m pytest -q.
```

---

## Verificação final do pacote

```text
Rode python -m pytest -q e depois:
python src/diagnostico_cli.py --seed 594525008
Confira: explicação sem "{'Não há cura'...}", aviso da CNN novo, laudo v2 com
recomendações condicionais (o caso é inconclusivo), "Por quê" citando rede neural x
regras, linha do guardrail. Liste tudo que mudou desde o último commit.
```
