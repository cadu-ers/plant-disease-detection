# Sistema especialista de diagnóstico (tomate e batata)

Módulo independente do pipeline de imagem e da CNN: recebe **fatos** (medidas das
máscaras, sintomas informados, probabilidades da CNN) e devolve hipóteses
ranqueadas com **explicação**.

## Como rodar

```bash
pip install pyyaml pytest            # pyyaml deve entrar no requirements.txt
python src/expert_system/diagnose.py src/expert_system/exemplos/late_blight_tomate.json
python src/expert_system/diagnose.py src/expert_system/exemplos/ambiguo_early_target.json --interativo
python src/expert_system/diagnose.py fatos.json --sem-cnn      # só regras (para a ablação)
pytest                                                          # na raiz do repositório
```

## Contrato de entrada (`fatos`)

| Campo | Significado |
|---|---|
| `cultura` | `"tomate"` ou `"batata"`. Se faltar, é inferida do `cnn_prior`. |
| `medidas` | Medidas das máscaras (`frac_marrom`, `frac_lesao_total`, `n_lesoes_pequenas`, `dist_borda_media_lesoes`, `frac_lesoes_com_halo`, `frac_amarelo`). São convertidas em graus de sintoma pelos trapézios da base. |
| `sintomas` | Graus 0..1 informados direto (ex.: `{"aneis_concentricos": 1.0}`). Têm prioridade sobre `medidas`. |
| `cnn_prior` | Probabilidades da CNN, com os nomes das pastas do dataset ou chaves canônicas. |

Sintoma ausente = **desconhecido** (não é o mesmo que 0 = ausente). Regras que
dependem de um sintoma desconhecido não disparam, e o modo interativo pergunta
o sintoma mais útil para separar as hipóteses líderes.

## Arquivos

- `knowledge_base.yaml`: sintomas, 13 doenças e regras `SE ... ENTÃO ...` (é aqui que se edita o conhecimento).
- `kb.py`: carrega e valida o YAML (erros de digitação viram `ValueError` claro).
- `engine.py`: inferência com fatores de certeza (CF estilo MYCIN) e detecção de conflito CNN x regras.
- `busca.py`: fila de prioridade das hipóteses e escolha da próxima pergunta.
- `explicacao.py`: por quê, por que não, conflitos e recomendações.
- `classes.py`: nomes do dataset -> chaves canônicas.
- `fuzzy.py`: trapézios das medidas.

## O que é provisório

- **Trapézios** dos sintomas de imagem são chutes iniciais: calibre com a
  distribuição das medidas por classe quando o extrator de features estiver pronto.
- *Early blight* x *Target spot* têm regras com as mesmas condições (R01 e R05):
  hoje o sistema só sinaliza a ambiguidade. Separá-las exige conhecimento
  adicional, não mais regras iguais.

## Próximos passos

1. Extrator de features sobre as máscaras de folha e lesão
2. Função que devolve o vetor completo de probabilidades da CNN 
3. Avaliação no conjunto de teste: só CNN x só regras x híbrido.
