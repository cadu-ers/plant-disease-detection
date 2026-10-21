[← voltar ao README](../README.md)

# Sistema especialista

O projeto também possui um sistema especialista em [`src/ia/expert_system/`](../src/ia/expert_system/).
Ele recebe fatos sobre a planta e aplica regras do tipo **SE... ENTÃO...**
para ranquear doenças, calcular fatores de certeza e explicar o resultado.

A base de conhecimento está em [`src/ia/expert_system/knowledge_base.yaml`](../src/ia/expert_system/knowledge_base.yaml) e
contém os sintomas, as 13 doenças, as regras, os fatores de certeza e as
recomendações gerais. O motor também pode receber probabilidades produzidas
pela CNN por meio do campo `cnn_prior`.

## Executar um diagnóstico de exemplo

Na raiz do projeto, com `src/` no `PYTHONPATH` (ver
[execução manual](07_como_reproduzir.md#execução-manual-script-por-script)), execute:

```bash
export PYTHONPATH=src            # no PowerShell: $env:PYTHONPATH = "src"
python -m ia.expert_system.diagnose src/ia/expert_system/exemplos/late_blight_tomate.json
```

Para executar sem utilizar a evidência da CNN:

```bash
python -m ia.expert_system.diagnose src/ia/expert_system/exemplos/late_blight_tomate.json --sem-cnn
```

Para utilizar o modo interativo:

```bash
python -m ia.expert_system.diagnose src/ia/expert_system/exemplos/ambiguo_early_target.json --interativo
```

O resultado apresenta a hipótese principal, as evidências das regras, as
hipóteses alternativas, os sintomas pendentes e recomendações gerais. O
valor exibido como **CF** é um fator de certeza do sistema, não uma
probabilidade estatística calibrada.

Atualmente, o sistema especialista pode ser executado de forma independente
com fatos informados em JSON. A integração automática entre imagem,
segmentação, extração de medidas, CNN e sistema especialista é uma etapa
futura do projeto.

## Executar os testes

```bash
python -m pytest -q
```

Os testes cobrem o carregamento da base de conhecimento, o mapeamento das
classes, o cálculo dos fatores de certeza, as regras de diagnóstico, as
explicações e o modo interativo.

Documentação do próprio módulo: [`src/ia/expert_system/README.md`](../src/ia/expert_system/README.md).
