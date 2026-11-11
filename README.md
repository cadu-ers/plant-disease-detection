# Detecção de Doenças em Folhas de Plantas

<!-- Ao editar este arquivo, atualize também o README.en.md -->

**🇧🇷 Português** | [🇺🇸 English](README.en.md)

Pipeline de processamento de imagens integrado a um modelo de IA (transfer
learning) para identificação de doenças em folhas de tomate e batata, a
partir do dataset PlantVillage.

## Integrantes da equipe

- Carlos Eduardo Rodrigues Silva
- Daniel Lucarelli Cerri
- Melck Silva de Oliveira Nascimento
- Murilo Moretto Marques
- Vinicius dos Santos

## Problema e objetivo

Doenças em folhas de tomate e batata aparecem como manchas, amarelamento e
alterações de textura. O objetivo é, a partir de uma foto da folha, dizer se a
planta está saudável ou qual das 11 doenças ela tem (13 classes, 18.163 imagens
do PlantVillage), segmentando a folha sem apagar justamente as lesões.

## Pipeline

```
BLOCO 1 — Processamento de Imagens (roda sozinho)              src/run_pi.py
  imagem → pré-processamento (resize 224x224, blur gaussiano 5x5, BGR→HSV)
         → segmentação (máscara verde + lesão, contorno externo preenchido)
         → análise FFT (espectros, filtros, teorema da convolução)
         → extração de características (features.py: cor, lesão, GLCM, LBP;
                                         fft_analysis.py: descritores FFT)
                              │
                              ▼  data_processed/, data_split/
BLOCO 2 — Inteligência Artificial (usa as saídas do bloco 1)    src/run_ia.py
  classificação: CNN MobileNetV2 (transfer learning)
                 SVM híbrido (embeddings da CNN + características clássicas)
         → diagnóstico (classe da folha)
                              │
                              ▼  models/
BLOCO 3 — Aplicação de diagnóstico (CLI)                src/diagnostico_cli.py
  sorteio de uma imagem do split de TESTE → segmentação
         → CNN MobileNetV2 (13 probabilidades) + medidas da folha (medidas_lesao.py)
         → SISTEMA ESPECIALISTA (regras SE... ENTÃO com fatores de certeza,
                                 base em knowledge_base.yaml) → DECIDE o diagnóstico
         → IA generativa (Gemini): laudo em linguagem simples, validado pelo guardrail;
           conversa depois da revelação → sintomas → o especialista recalcula
```

O bloco de Processamento de Imagens nunca depende do bloco de IA. As
características clássicas são definidas em `features.py` e calculadas em
massa pelo `extract_features.py` (bloco 2), junto com os embeddings da CNN. O
sistema especialista (regras SE... ENTÃO) está integrado à CLI do bloco 3 e
continua podendo rodar isolado, com fatos em JSON (`python -m ia.expert_system.diagnose`).

## Resultados principais

| Resultado | Valor | Detalhes |
|---|---|---|
| Buracos na máscara (antiga só verde x nova) | 10.604 x 2 imagens com buracos (média 2,15 x 0,0002 por imagem) | [segmentação](docs/02_segmentacao.md) |
| Teorema da convolução (blur no espaço x na frequência) | erro máximo 8,5e-14 níveis de cinza | [FFT](docs/03_fft.md) |
| CNN no teste (2.725 imagens) | acurácia 0.949, F1-macro 0.950 | [CNN](docs/04_cnn.md) |
| SVM híbrido no teste | acurácia 0.979, F1-macro 0.972 (controle só com treino: 0.976 / 0.971) | [classificador híbrido](docs/05_classificador_hibrido.md) |

## Aplicação de diagnóstico (CLI)

`python src/diagnostico_cli.py` abre um menu de demonstração. Em cada análise:

1. Sorteia uma imagem do split de **teste** e mostra só a divisão: o arquivo e a
   classe real ficam escondidos.
2. Mostra o diagnóstico: as 3 classes mais prováveis da CNN, as medidas da
   folha, as regras que dispararam, as perguntas pendentes e a conclusão do
   sistema especialista.
3. Mostra o laudo em linguagem simples gerado pela IA generativa, já validado
   pelo guardrail.
4. Depois do Enter, revela a imagem e a classe real, compara com a CNN e com o
   especialista e salva as figuras em `rotina_demo/cli/<pasta>/`.
5. Se o diagnóstico não foi conclusivo, abre a conversa: você olha a folha
   (`original.png`), responde em texto livre, confirma o que foi entendido e vê
   o **ANTES x DEPOIS** do sistema especialista, com um laudo novo.

| Opção | Efeito |
|---|---|
| `--seed N` | repete o sorteio da seed N (sorteio reprodutível) |
| `--sem-llm` | roda offline, só com o sistema especialista: sem laudo e sem conversa |
| `--sem-cache` | o laudo e a conversa sempre chamam a API (rodadas de avaliação) |

## IA generativa e guardrail

**Quem decide o diagnóstico é o sistema especialista.** A IA generativa (Google
Gemini, via API, em `src/ia/generativa/`) só explica e conversa:

- **Laudo:** o LLM recebe só os fatos que o motor já produziu (hipótese, CF,
  regras que dispararam, medidas, pendências e recomendações da base). Nunca
  recebe a imagem, o caminho nem a classe real.
- **Guardrail:** o sistema especialista valida o texto do LLM: hipótese
  diferente da do motor, doenças fora dos fatos, porcentagens, dose ou produto,
  ação irreversível sem condição de confirmação e falta de acentos. Se reprovar,
  o LLM tenta de novo sabendo das violações; se o problema persistir, a CLI
  mostra o texto do próprio motor. Cada laudo é registrado em log (aprovado,
  corrigido ou reprovado).
- **Conversa:** depois da revelação, o usuário responde em texto livre e o LLM
  converte a resposta em sintomas, cada um com o trecho literal da resposta, que
  o código confere. O usuário confirma, o motor recalcula e a CLI mostra o ANTES
  x DEPOIS. O LLM nunca escolhe a doença.
- **Cliente:** cache em disco, novas tentativas e modelo reserva. Entradas já
  vistas funcionam sem internet, a partir do cache.

## Como rodar

```bash
pip install -r requirements.txt
python src/run_pi.py          # bloco de Processamento de Imagens
python src/run_ia.py          # bloco de IA (treina a CNN só se não houver modelo)
python src/demo_pi.py         # demonstração do Processamento de Imagens
python src/demo_ia.py         # demonstração da IA
python src/diagnostico_cli.py # aplicação de diagnóstico (CLI)
```

Python 3.11. Os módulos das subpastas rodam com `src/` no `PYTHONPATH`. Para o
laudo e a conversa, a chave da API fica **só** na variável de ambiente
`GEMINI_API_KEY`; nunca a escreva em arquivo. Sem a chave, a CLI segue sem laudo.

```bash
setx GEMINI_API_KEY "sua-chave"                     # Windows (depois abra um terminal novo)
export GEMINI_API_KEY="sua-chave"                   # Git Bash
PYTHONPATH=src python -m ia.generativa.testar_conexao   # testa a chave
PYTHONPATH=src python -m pytest -q                  # 480 testes
```

Os testes nunca chamam o Gemini: o cliente recebe um transporte falso.

O Git **não** guarda o que a CLI precisa para rodar: `models/plant_disease_model.keras`,
`data_split/split_manifest.csv` e as imagens originais em `data/` (ver
`.gitignore`). Quem clona o repositório precisa gerá-los com `run_pi.py` e
`run_ia.py` ou receber esses arquivos.

Opções dos orquestradores, execução manual script por script, Docker e testes:
[docs/07_como_reproduzir.md](docs/07_como_reproduzir.md).

## Limitações

- A base de conhecimento é um ponto de partida e precisa ser validada por um agrônomo.
- A base não tem regras de evidência contrária: responder "não tem mosqueado" não
  reduz a hipótese de mosaico.
- O guardrail garante fidelidade ao sistema especialista, não que o diagnóstico seja verdadeiro.
- O modelo foi treinado só com o PlantVillage (imagens de laboratório).

## Estrutura

| Grupo | Scripts em `src/` |
|---|---|
| Orquestração | `run_pi.py`, `run_ia.py`, `run_all.py` (e `common/orchestration.py`) |
| Demo | `demo_pi.py` (Processamento de Imagens), `demo_ia.py` (IA) |
| Comuns | `common/`: `config.py`, `utils.py`, `orchestration.py` |
| Dados | `dados/`: `download_dataset.py`, `split_dataset.py`, `data_overview.py` |
| Processamento de Imagens | `processamento_imagens/`: `preprocessing.py`, `segmentation_report.py`, `fft_analysis.py`, `features.py`, `p1/histogram_analysis.py` (P1) |
| Inteligência Artificial | `ia/cnn/`: `model_utils.py`, `train_model.py`; `ia/hibrido/`: `extract_features.py`, `hybrid_classifier.py`; `ia/inferencia.py`; `ia/expert_system/` |
| Aplicação de diagnóstico | `diagnostico_cli.py`; `aplicacao/`: `pipeline.py`, `contrato.py`, `sorteio.py`, `exportacao.py`; `cli/`: `menu.py`, `formatacao.py` |
| IA generativa | `ia/generativa/`: `cliente.py`, `laudo.py`, `guardrail.py`, `conversa.py`, `prompts.py`, `testar_conexao.py` |
| Testes | `tests/` |

Os scripts da raiz de `src/` rodam com `python src/<script>.py`; os das
subpastas rodam como módulo (ex.: `python -m dados.split_dataset`), com `src/`
no `PYTHONPATH` (ver [execução manual](docs/07_como_reproduzir.md#execução-manual-script-por-script)).

Saídas: `reports/pi/` e `reports/ia/` (figuras e métricas de cada bloco, com
índices em [reports/pi/LEIAME.md](reports/pi/LEIAME.md) e
[reports/ia/LEIAME.md](reports/ia/LEIAME.md)), `models/` (modelos),
`data_processed/` e `data_split/` (imagens segmentadas e divisão),
`features_cache/` (features), `rotina_demo/` (figuras da CLI, cache do LLM e
logs do guardrail e da conversa; fora do Git).
Árvore completa em [docs/07_como_reproduzir.md](docs/07_como_reproduzir.md#estrutura-do-repositório).

## Documentação técnica

1. [Dados](docs/01_dados.md): dataset, formato e resolução, desbalanceamento, split
2. [Segmentação](docs/02_segmentacao.md): pipeline de PI, limiares, verificação objetiva, limitações
3. [FFT](docs/03_fft.md): domínio da frequência, magnitude x fase, teorema da convolução, descritores
4. [CNN](docs/04_cnn.md): integração com IA, etapas do treino, resultados
5. [Classificador híbrido](docs/05_classificador_hibrido.md): features clássicas, SVM, McNemar, controle
6. [Sistema especialista](docs/06_sistema_especialista.md): regras, fatores de certeza, exemplos
7. [Como reproduzir](docs/07_como_reproduzir.md): requisitos, orquestradores, execução manual, Docker, estrutura
8. [Uso de IA](docs/uso_de_ia.md): registro do uso de ferramentas de IA no desenvolvimento

README completo anterior à reorganização: [docs/historico/README_v1_completo.md](docs/historico/README_v1_completo.md).
