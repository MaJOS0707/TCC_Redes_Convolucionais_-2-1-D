# Redes Convolucionais (2+1)D Aplicadas à Detecção de Vídeos de Inteligência Artificial

**Autor:** Matheus Jun Onishi da Silva  
**Orientador:** Prof. Dr. Douglas Rodrigues Pinto  
**Instituição:** Universidade Federal Fluminense — Departamento de Estatística  
**Ano:** 2026  

📄 [Monografia completa (PDF)](monografia_exemplo.pdf)

---

## Resumo

Este trabalho investiga o uso de **Redes Neurais Convolucionais (2+1)D** para diferenciar automaticamente vídeos gerados por Inteligência Artificial de vídeos reais. A arquitetura (2+1)D fatora a convolução espaço-temporal em duas etapas — uma convolução espacial 2D seguida de uma convolução temporal 1D — reduzindo o custo computacional em relação às CNNs 3D tradicionais e mantendo a capacidade de capturar padrões de movimento.

---

## Motivação

Com a popularização de geradores de vídeo por IA (Sora, Wan, EasyAnimate, etc.), cresce o risco de desinformação, conteúdo enganoso e fraude em plataformas digitais. Ferramentas automáticas de detecção tornam-se essenciais para apoiar a moderação de conteúdo em escala.

---

## Base de Dados — AIGVDBench

| Característica | Valor |
|---|---|
| Total de vídeos | +440.000 |
| Modelos geradores | 31 |
| Vídeos reais (fonte) | OpenVid-HD |
| Armazenamento total | ≈ 378 GB |

**Seleção utilizada neste trabalho:** modelos de 2025 (Open-Sora, RepVideo, AccVideo, EasyAnimate, Wan…), totalizando ~176.000 vídeos (162 mil IA + 14 mil reais), com classificação binária IA vs. Real.

---

## Metodologia

### Seleção de Frames

Os vídeos foram amostrados com **passo fixo P = 4**, usando a fórmula:

```
L = 1 + (N_frames − 1) × P
```

Onde `L` é o número de frames necessários e `N_frames` é a quantidade de frames selecionados. Vídeos com menos frames que o necessário recebem **frames vazios** (zeros) ao final.

### Pré-processamento

- Redimensionamento para **112 × 112 px**
- Modos disponíveis: `stretch`, `center_crop`, `pad`
- Canais: **RGB (3 canais)** ou **tons de cinza (1 canal)**
- Normalização para o intervalo [0, 1]

### Arquitetura (2+1)D

```
Entrada (T × 112 × 112 × C)
    │
Conv2Plus1D (16 filtros, kernel 3×7×7)  ← espacial + temporal
BatchNorm + ReLU
ResizeVideo (56×56)
Bloco Residual (16 filtros)
ResizeVideo (28×28)
Bloco Residual (32 filtros)
ResizeVideo (14×14)
Bloco Residual (64 filtros)
ResizeVideo (7×7)
Bloco Residual (128 filtros)
GlobalAveragePooling3D
Dense (num_classes)
    │
Saída (IA / Real)
```

Cada **bloco residual** aplica duas convoluções (2+1)D com conexão de atalho (*skip connection*), garantindo estabilidade no treinamento.

### Configurações de Treinamento

| Parâmetro | Valor |
|---|---|
| N° de frames avaliados | 1 a 15 |
| Passo (frame step) | 4 |
| Resolução | 112 × 112 px |
| Canais | RGB e Tons de Cinza |
| Épocas | 25 |
| Learning Rate | 0,01 |
| Batch Size | 20 |
| Ambiente | Google Colab (GPU T4) |
| Total de modelos treinados | 30 |

---

## Resultados

### Avaliação Inicial (5.400 vídeos)

Os melhores resultados foram obtidos com **14 e 15 frames**:

| N frames | Treino RGB (%) | Teste RGB (%) | Treino Cinza (%) | Teste Cinza (%) |
|---|---|---|---|---|
| 14 | 98,17 | 98,93 | 98,33 | 99,14 |
| 15 | 98,70 | 99,14 | 98,77 | 99,00 |

### Aprendizado por Atalho (*Shortcut Learning*)

A análise exploratória revelou que os vídeos de IA na amostra inicial tinham **significativamente menos frames** do que os reais:

| Conjunto | Mediana IA | Mediana Real |
|---|---|---|
| Treino | 49 frames | 148 frames |
| Teste | 49 frames | 148 frames |

O modelo pode ter aprendido a distinguir as classes pelo **número de frames** (característica estrutural), em vez de padrões visuais de geração. Vale notar que, em um cenário real, esse atalho poderia ser útil na prática, pois essa diferença estrutural tende a persistir enquanto os geradores não forem aprimorados nesse aspecto.

### Reavaliação com Amostra Filtrada

Aplicando um **limiar de ≥ 57 frames** por vídeo (eliminando vídeos curtos), a amostra passou de 5.400 para **3.744 vídeos**. O menor vídeo filtrado tinha 81 frames.

Os melhores resultados após a filtragem:

| N frames | Treino RGB (%) | Teste RGB (%) | Treino Cinza (%) | Teste Cinza (%) |
|---|---|---|---|---|
| 14 | 96,11 | 95,36 | 95,90 | 95,86 |
| 15 | 96,48 | 95,50 | 96,43 | 95,43 |

A redução na acurácia de ~99% para ~95% após a filtragem reforça a hipótese de **aprendizado por atalho** nos modelos da avaliação inicial.

---

## Conclusões

- A rede (2+1)D mostrou-se uma **alternativa promissora** para detecção de vídeos de IA, com acurácia superior ao acaso mesmo após a remoção do atalho estrutural.
- A acurácia alta **não é garantia de aprendizado correto** — a análise exploratória foi essencial para identificar o fenômeno de shortcut learning.
- A constante evolução dos geradores tende a tornar o problema **progressivamente mais difícil**, exigindo atualização contínua dos detectores.

---

## Estrutura do Repositório

```
├── modelo_2plus1d.py       # Código principal (treinamento + avaliação)
├── monografia_exemplo.pdf  # Monografia completa
└── README.md
```

---

## Dependências

```bash
pip install tensorflow keras opencv-python einops openpyxl matplotlib
```

> O código foi desenvolvido para rodar no **Google Colab** com acesso ao Google Drive. Ajuste os caminhos `DATASET_DIR` e `RESULTADOS_DIR` conforme seu ambiente.

---

## Citação

```
Silva, Matheus Jun Onishi da. Redes Convolucionais (2+1)D Aplicadas à Detecção 
de Vídeos de Inteligência Artificial. Monografia (Bacharelado em Estatística) — 
Universidade Federal Fluminense, Niterói, 2026.
```
