# Prompts para o Claude Code: pacote 3 (conversa: resposta livre → sintomas → motor recalcula)

Use na branch `feature/ia-generativa`, **depois** de commitar o pacote 2 (com a correção dos
acentos). Um prompt por vez: peça o plano, aprove, leia o diff, rode `python -m pytest -q`.
Registre o uso em `docs/uso_de_ia.md`.

Resultado esperado (conferido no motor, com uma folha sintética de mosaico e a CNN dando
~100% para mosaico):

| Resposta do usuário | Resultado do motor |
|---|---|
| nenhuma | Mosaico, CF 0,49, inconclusivo |
| "tem mosqueado" (mosaico_claro_escuro = 1) | Mosaico, CF 0,90, **conclusivo** |
| mosqueado + folhas deformadas | Mosaico, CF 0,96, **conclusivo** |
| "não tem mosqueado" (= 0) | Mosaico, CF 0,49, inconclusivo (a base não tem regra contrária: limitação para o relatório) |

---

## Prompt 5: módulo da conversa (sem mexer na CLI ainda)

```text
Passo 3 da IA generativa: o usuário responde às perguntas pendentes em TEXTO LIVRE, o LLM
converte em sintomas, o guardrail valida e o SISTEMA ESPECIALISTA recalcula. O LLM nunca
escolhe a doença.

Crie src/ia/generativa/conversa.py (com docstring no estilo do projeto) com:

1. sintomas_para_perguntar(diag, kb, maximo=4) -> tuple de Sintoma:
   - só sintomas com origem "usuario" na KB (o contrato aplicacao/contrato.py só aceita
     esses em sintomas_usuario);
   - primeiro o de proxima_pergunta(diag.resultado, kb), depois os pendentes da hipótese
     principal (diag.resultado.melhor.pendentes), sem repetir, em ordem estável;
   - vazio se diag.status != "ok" ou não houver hipótese.

2. interpretar_resposta(texto_usuario, sintomas, cliente) -> Interpretacao:
   - cliente.gerar_json com um JSON Schema em que, para CADA sintoma perguntado, há
     {"valor": number 0..1 ou null, "trecho": string ou null}. null = o usuário não falou
     disso ou disse que não sabe. "trecho" = pedaço LITERAL da resposta do usuário que
     justifica o valor.
   - Prompt em prompts.py: TAREFA_CONVERSA (somada ao SISTEMA_BASE) e
     VERSAO_PROMPT_CONVERSA = 1. O LLM recebe SÓ a lista de sintomas (id, descrição,
     pergunta) e a resposta do usuário. NÃO recebe a hipótese, a CNN nem o nome de
     nenhuma doença (para não induzir a resposta). Escala: 1 = sim/claramente presente,
     0 = não/ausente, valores intermediários só se o usuário disser "um pouco", "talvez",
     "em algumas folhas" (ex.: 0.5); na dúvida, null.
   - Guardrail da conversa (funções puras, testáveis):
     a) ids fora da lista perguntada são descartados (violação registrada);
     b) valor fora de [0, 1] ou não numérico é descartado;
     c) valor não nulo exige "trecho" que exista na resposta do usuário (comparação sem
        acento e sem diferença de maiúsculas, ignorando espaços extras); se não existir,
        o sintoma é descartado (o LLM afirmou algo que o usuário não disse);
     d) texto do usuário vazio ou só espaços: não chama o LLM, devolve sem sintomas.
   - Interpretacao: sintomas (dict id -> float, só os aceitos), trechos (dict id -> str),
     descartados (tuple de (id, motivo)), modelo, veio_do_cache, aviso (quando o LLM está
     indisponível: sem sintomas, nunca levanta exceção).

3. reavaliar(diag_anterior, amostra_segmentada, mascara, divisao, sintomas, deps/preditor,
   kb) -> novo ResultadoDiagnostico, chamando diagnosticar_amostra(...,
   sintomas_usuario=sintomas). Reaproveite o preditor já carregado.

4. Log: uma linha por interpretação em rotina_demo/conversa_log.jsonl (caminho em
   common/config.py): data/hora, modelo, versão do prompt, ids perguntados, sintomas
   aceitos, descartados com motivo, do_cache. SEM o texto do usuário, sem caminho, classe
   real ou seed. Ligado só pela CLI (nos testes, desligado). Falha ao gravar não quebra.

5. Testes em src/tests/test_generativa_conversa.py (transporte falso, sem rede):
   - "Sim, tem umas manchas verde claro e escuro, tipo mosaico" -> mosaico_claro_escuro=1
     aceito com trecho; o novo diagnóstico do mosaico (preditor falso com ~100% mosaico)
     fica conclusivo;
   - o LLM inventa folha_deformada=1 com trecho que não está na resposta -> descartado;
   - id que não foi perguntado -> descartado; valor 1.7 -> descartado; null -> omitido;
   - resposta vazia não chama o transporte;
   - sem chave -> aviso, nenhum sintoma, nenhuma exceção;
   - o prompt enviado ao LLM não contém nome de doença da KB, "Tomato_", "Potato_",
     caminho nem a palavra "CNN";
   - sintomas_para_perguntar só devolve origem "usuario".
6. Rode python -m pytest -q.
```

---

## Prompt 6: conversa na CLI (depois da revelação)

```text
Ligue a conversa (src/ia/generativa/conversa.py) na CLI. Ela entra DEPOIS do Enter de
revelação, porque só então o usuário pode olhar a folha (as figuras são salvas em
rotina_demo/cli/<pasta>/original.png; a revelação já mostra a pasta).

Fluxo em cli/menu.py, analisar_uma, depois de formatar_revelacao e ANTES de
"Analisar outra?":
1. Se deps.conversar for None (padrão, para os testes) ou o diagnóstico já for
   conclusivo ou não houver sintomas_para_perguntar: pular.
2. Mostrar: "Olhe a folha em <pasta>/original.png e responda em texto livre às perguntas
   abaixo (Enter vazio para pular):" e as perguntas numeradas.
3. Ler a resposta; interpretar; mostrar "Entendi:" com cada sintoma aceito em linguagem
   simples (descrição + sim/não/parcial + o trecho entre aspas) e os descartados com o
   motivo ("não encontrei isso na sua resposta").
4. Perguntar "Confirma? (s/n)". Só com "s" o motor recalcula (o humano confirma o que o
   LLM entendeu).
5. Mostrar "ANTES x DEPOIS": hipótese, CF e situação (conclusiva ou não) antes e depois,
   e as regras que passaram a disparar. Depois gerar o laudo novo (com guardrail) do
   diagnóstico recalculado e mostrar.
6. Dependencias ganha conversar: Optional[Callable] = None; diagnostico_cli.py liga a
   conversa junto com o laudo (desligada com --sem-llm). Atualize o teste que confere os
   campos de Dependencias.
7. Textos novos em cli/formatacao.py (funções puras), só ASCII + acentos do português.
8. Ctrl+C/EOF continuam encerrando sem traceback; erro na conversa nunca derruba a
   análise (mensagem genérica com o tipo do erro).
9. Testes em test_cli.py com entradas simuladas: fluxo completo com conversa (resposta
   -> "s" -> antes x depois mostra inconclusiva -> conclusiva); "n" na confirmação não
   recalcula; Enter vazio pula; conversa desligada não pergunta nada; diagnóstico
   conclusivo não pergunta.
10. Rode python -m pytest -q.
```

**Teste real depois do prompt 6:**
```
rm -rf rotina_demo/llm_cache
python src/diagnostico_cli.py --seed 594525008
```
`1` → leia o diagnóstico → `Enter` → abra o `original.png` da pasta indicada → responda
algo como *"sim, a folha tem manchas verde claro e verde escuro, e está um pouco
deformada"* → `s` → veja o ANTES x DEPOIS. Essa é a **demo principal da apresentação**.
