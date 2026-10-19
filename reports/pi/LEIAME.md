# Resultados do bloco de Processamento de Imagens

[← voltar ao README](../../README.md) · [resultados de IA](../ia/LEIAME.md)

Índice das figuras e tabelas de `reports/pi/`, na ordem da apresentação. A
coluna "Item" indica o item da orientação do professor que cada figura atende.
Todas são geradas por `python src/run_pi.py` (detalhes em
[docs/02_segmentacao.md](../../docs/02_segmentacao.md) e
[docs/03_fft.md](../../docs/03_fft.md)).

## Números principais

| Medida | Valor | Fonte |
|---|---|---|
| Imagens | 18.163 em 13 classes, todas 256x256, JPEG, RGB | [`resumo_dados.csv`](dados/resumo_dados.csv) |
| Split treino / validação / teste | 12.714 / 2.724 / 2.725 | [`split_summary.csv`](dados/split_summary.csv) |
| Imagens com buracos na máscara: antiga (só verde) x nova | 10.604 x 2 (média de 2,15 x 0,0002 buracos por imagem) | [`qualidade_segmentacao.csv`](segmentacao/qualidade_segmentacao.csv) |
| Fração da imagem ocupada pela folha (mediana) | 0,42 | [`qualidade_segmentacao.csv`](segmentacao/qualidade_segmentacao.csv) |
| Fallbacks (maior contorno < 5% da imagem) | 5 | [`qualidade_segmentacao.csv`](segmentacao/qualidade_segmentacao.csv) |
| Casos suspeitos (folha < 10% ou > 95% da imagem) | 100 (49 + 51) | [`qualidade_segmentacao.csv`](segmentacao/qualidade_segmentacao.csv) |
| Teorema da convolução: erro máximo (blur no espaço x na frequência) | 8,5e-14 níveis de cinza | painel de diferença absoluta de [`blur_espaco_vs_frequencia.png`](fft/blur_espaco_vs_frequencia.png) (também impresso pelo `fft_analysis.py`) |
| Impacto da segmentação na etapa seguinte: recall de `Tomato_Early_blight` | 0,59 (máscara antiga, baseline) → 0,86 (máscara nova, CNN corrigida) | [resultados de IA](../ia/LEIAME.md) |

## Parâmetros do bloco

Todos ficam em [`src/config.py`](../../src/config.py) e nas constantes do topo de
[`src/fft_analysis.py`](../../src/fft_analysis.py).

| Etapa | Parâmetro | Valor | Por quê |
|---|---|---|---|
| Pré-processamento | Tamanho | 256x256 → 224x224 | Entrada uniforme: máscaras, espectros e medidas comparáveis |
| Pré-processamento | Blur gaussiano | kernel 5x5 | Reduz o ruído que viraria pontos soltos na máscara |
| Segmentação | Faixa verde (HSV) | H 25–90, S ≥ 40, V ≥ 40 | Tecido saudável da folha |
| Segmentação | Faixa de lesão (HSV) | H 8–35, S ≥ 40, V ≥ 20 | Manchas marrons e amarelas; S ≥ 40 segura o fundo bege, V ≥ 20 recupera lesões escuras |
| Segmentação | Morfologia | abertura + fechamento, kernel 5x5 | Remove pontos soltos e fecha falhas pequenas |
| Segmentação | Contorno | maior contorno externo, preenchido | Tudo dentro da folha entra na máscara, inclusive as lesões |
| Segmentação | Fallback | maior contorno < 5% da imagem | Evita jogar fora a folha quando o contorno falha |
| FFT | Preparação | cinza da original, fundo = média da folha, sem DC, janela de Hann 2D | Evita o "+" do degrau folha/fundo e o vazamento espectral pelas bordas |
| FFT | Passa-baixa gaussiano | D0 = 15 | Mostra o que fica só com as baixas frequências |
| FFT | Corte alta/baixa | 0,25 da frequência de Nyquist | Razão de energia usada no espectro por classe |
| FFT | Descritores | 8 faixas radiais (log), razão alta/baixa, centroide, entropia | 11 números de textura por imagem |

## Como gerar cada figura

| Figuras | Comando |
|---|---|
| Tudo de uma vez | `python src/run_pi.py` (pula o que já existe; `--force` refaz; `--so-figuras` só regera as figuras) |
| `dados/*` | `python src/data_overview.py` (o `split_summary.csv` sai do `python src/split_dataset.py`) |
| `segmentacao/etapas_segmentacao.png`, `qualidade_segmentacao.*`, `casos_suspeitos.png` | `python src/segmentation_report.py` |
| `segmentacao/segmentation_comparison.png` | `python src/segmentation_report.py --comparacao` |
| `segmentacao/estudo_limiares/*` | `python src/segmentation_report.py --comparacao --s <S> --v <V> --out <arquivo>` |
| `fft/*` | `python src/fft_analysis.py` (a comparação com SVM só roda se `features_cache/handcrafted.csv` existir) |
| `p1_histograma/*` | `python src/histogram_analysis.py` |

SVM com descritores FFT ([`fft_svm.csv`](fft/fft_svm.csv); treinado só no treino, avaliado no teste):

| Features | Nº | Acurácia | F1-macro |
|---|---|---|---|
| Só FFT | 11 | 0.380 | 0.330 |
| Só clássicas | 147 | 0.937 | 0.915 |
| Clássicas + FFT | 158 | 0.939 | 0.920 |

## 1. Dados

| Figura | O que mostra | Item |
|---|---|---|
| [`dados/exemplos_por_classe.png`](dados/exemplos_por_classe.png) | Uma imagem original de cada uma das 13 classes, com o nº de imagens por classe. | Dados de entrada |
| [`dados/resumo_dados.csv`](dados/resumo_dados.csv) | Por classe: nº de imagens, resolução, extensão, formato real e modo de cor. | Dados de entrada |
| [`dados/split_summary.csv`](dados/split_summary.csv) | Contagem de imagens por classe em treino, validação e teste (70/15/15). | Dados de entrada |

## 2. Segmentação

| Figura | O que mostra | Item |
|---|---|---|
| [`segmentacao/etapas_segmentacao.png`](segmentacao/etapas_segmentacao.png) | As etapas em sequência (original, blur, canal H, máscaras verde e de lesão, união, morfologia, contorno preenchido, segmentada) em duas folhas com lesões grandes. | Segmentação |
| [`segmentacao/segmentation_comparison.png`](segmentacao/segmentation_comparison.png) | Máscara antiga (só verde) x nova (verde + lesão, contorno preenchido) em 12 imagens: a antiga apaga as lesões. | Segmentação |
| [`segmentacao/estudo_limiares/segmentation_S40_V30.png`](segmentacao/estudo_limiares/segmentation_S40_V30.png) | Limiar S=40, V=30: lesões escuras da borda ficam cortadas. | Segmentação |
| [`segmentacao/estudo_limiares/segmentation_S25_V20.png`](segmentacao/estudo_limiares/segmentation_S25_V20.png) | Limiar S=25, V=20: recupera as lesões, mas o fundo bege/cinza vaza. | Segmentação |
| [`segmentacao/estudo_limiares/segmentation_S35_V20.png`](segmentacao/estudo_limiares/segmentation_S35_V20.png) | Limiar S=35, V=20: sombra e fundo começam a entrar (o escolhido, S=40 e V=20, é o da comparação acima). | Segmentação |

## 3. Verificação objetiva da segmentação

| Figura | O que mostra | Item |
|---|---|---|
| [`segmentacao/qualidade_segmentacao.png`](segmentacao/qualidade_segmentacao.png) | Fração de folha por classe e nº médio de buracos na máscara antiga x nova, em todas as 18.163 imagens. | Resultados |
| [`segmentacao/qualidade_segmentacao.csv`](segmentacao/qualidade_segmentacao.csv) | Por imagem: fração de folha, proporção de lesão, fallback e buracos (antiga x nova). | Resultados |
| [`segmentacao/casos_suspeitos.png`](segmentacao/casos_suspeitos.png) | As 12 segmentações mais suspeitas: folhas esbranquiçadas que caem no fallback e fundo bege que passa na faixa de lesão. | Limitações |

## 4. FFT (domínio da frequência)

| Figura | O que mostra | Item |
|---|---|---|
| [`fft/fft_saudavel_vs_doente.png`](fft/fft_saudavel_vs_doente.png) | Folha saudável x com Septoria: imagem preparada, magnitude, fase, passa-baixa e passa-alta. | FFT |
| [`fft/fft_magnitude_vs_fase.png`](fft/fft_magnitude_vs_fase.png) | Reconstrução só com magnitude x só com fase: a fase guarda a estrutura da folha. | FFT |
| [`fft/blur_espaco_vs_frequencia.png`](fft/blur_espaco_vs_frequencia.png) | Teorema da convolução com o blur 5x5 do pipeline: espaço e frequência dão o mesmo resultado (erro 8,5e-14). | FFT |
| [`fft/espectro_radial_por_classe.png`](fft/espectro_radial_por_classe.png) | Espectro radial médio por classe e razão de energia alta/baixa frequência: as classes saudáveis têm a maior razão. | FFT / Resultados |
| [`fft/fft_svm.png`](fft/fft_svm.png) | SVM com FFT, com features clássicas e com as duas: a FFT sozinha discrimina mal e soma pouco às clássicas. | Resultados |

## 5. Material da P1

| Figura | O que mostra | Item |
|---|---|---|
| [`p1_histograma/histograma_exemplo.png`](p1_histograma/histograma_exemplo.png) | Histograma de intensidade antes e depois de uma normalização min-max. Material da P1: essa normalização não faz parte do pipeline final. | Dados de entrada (P1) |

## Limitações (resumo)

- Segmentação por limiar de HSV: 100 imagens suspeitas (0,6%), com dois tipos
  de falha real, fundo que vaza e folhas esbranquiçadas
  ([`casos_suspeitos.png`](segmentacao/casos_suspeitos.png)).
- PlantVillage tem fotos de laboratório (fundo uniforme, folha isolada); os
  resultados podem não se manter em fotos de campo.

A demonstração ao vivo do bloco em uma imagem é feita com `python src/demo_pi.py`
(saídas em `rotina_demo/pi/`, fora de `reports/`).
