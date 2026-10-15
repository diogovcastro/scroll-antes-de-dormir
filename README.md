# scroll-antes-de-dormir — Impacto do Tempo de Tela no Sono

Projeto de **mineração de dados** que investiga como os hábitos de uso do celular antes de dormir se relacionam com a qualidade do sono, e constrói modelos de classificação para **prever a categoria de débito de sono** de uma pessoa.

> 🚧 Projeto em desenvolvimento, feito durante a disciplina de Mineração de Dados.

## 🎯 Objetivo

Prever a variável alvo **`sleep_debt_category`** a partir de atributos demográficos, de rotina e de uso de telas, comparando três algoritmos de classificação:

- **K-Nearest Neighbors (KNN)**
- **Naive Bayes**
- **Árvore de Decisão**

### Classes da variável alvo

| Classe | Significado | Registros | % |
|---|---|---:|---:|
| `Optimal Recovery` | Sono reparador, sem débito | 1.387 | 16,3% |
| `Mild Deficit` | Débito leve | 2.004 | 23,6% |
| `Moderate Debt` | Débito moderado | 4.462 | 52,5% |
| `Severe Sleep Debt` | Débito severo | 647 | 7,6% |

As classes são **ordenadas** (de melhor para pior) e **desbalanceadas**. Mais da metade dos registros está em `Moderate Debt`, o que influencia a separação treino/teste (estratificação) e a escolha das métricas de avaliação.

## 📊 Conjunto de dados

- **Fonte:** [Sleep Debt & Screen Time: Late Night Phone Habits](https://www.kaggle.com/datasets/samartalwar/sleep-debt-and-screen-time-late-night-phone-habits) (Kaggle), por [Samar Talwar](https://www.kaggle.com/samartalwar)
- **Licença:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- **Tamanho:** 8.500 registros × 18 colunas
- **Natureza:** dados **sintéticos**, gerados por uma simulação baseada em relações biofísicas (supressão de melatonina pelo brilho e pelo tempo de tela, atenuada pelo filtro de luz azul, e diferentes níveis de estímulo para cada tipo de app). Por isso, as conclusões refletem as regras da simulação, não necessariamente o comportamento de pessoas reais.

### Dicionário de variáveis

A coluna **Momento** indica quando a informação é obtida em relação à noite de sono. Isso importa porque atributos medidos *durante* o sono ou *no dia seguinte* podem conter informação da própria resposta (*data leakage*), e o uso deles como preditores será avaliado na etapa de validação. O próprio autor do dataset sugere priorizar os atributos comportamentais (cafeína, app, tempo de tela) em vez das medidas fisiológicas.

| Variável | Tipo | Descrição | Domínio | Momento |
|---|---|---|---|---|
| `user_id` | Identificador | Identificador alfanumérico único de cada participante | `USR-00001` … | — |
| `age` | Numérico (inteiro) | Idade do participante em anos (concentrada na faixa de 18 a 45) | 18 – 65 | Perfil |
| `gender` | Categórico nominal | Identidade de gênero autodeclarada | `Female`, `Male`, `Non-Binary` | Perfil |
| `occupation_type` | Categórico nominal | Categoria geral de ocupação, que influencia a obrigação de horário para acordar | `Corporate 9-to-5`, `Remote Tech`, `Student`, `Healthcare / Shift Worker`, `Freelance / Creative` | Perfil |
| `chronotype` | Categórico ordinal | Preferência circadiana biológica (cronotipo) | `Morning Lark`, `Intermediate`, `Night Owl` | Perfil |
| `bedtime_phone_minutes` | Numérico (inteiro) | Total de minutos de uso ativo do celular na cama antes de tentar dormir | 1 – 180 | Antes de dormir |
| `primary_bedtime_app` | Categórico nominal | Categoria de aplicativo mais usada durante o uso do celular na cama | `TikTok / Reels`, `YouTube`, `Instagram / Reddit`, `Streaming (Netflix/Hulu)`, `Messaging / Chat`, `News / Reading` | Antes de dormir |
| `screen_brightness_pct` | Numérico (inteiro) | Brilho da tela configurado no aparelho, em porcentagem | 10 – 100 | Antes de dormir |
| `blue_light_filter_active` | Binário | Indica se o filtro de luz azul (tons quentes) estava ativo | 1 = ativo, 0 = inativo | Antes de dormir |
| `caffeine_post_5pm_mg` | Numérico (inteiro) | Estimativa de cafeína consumida após as 17h (mg) | 0 – 250 | Antes de dormir |
| `physical_activity_min` | Numérico (inteiro) | Total de minutos de exercício físico intencional feito mais cedo no mesmo dia | 0 – 112 | Antes de dormir |
| `sleep_latency_min` | Numérico (contínuo) | Estimativa de minutos para adormecer depois de largar o aparelho | 6 – 123,3 | Durante o sono |
| `total_sleep_hours` | Numérico (contínuo) | Total de horas de sono calculado até o despertador tocar | 3,2 – 9,8 | Durante o sono |
| `deep_sleep_pct` | Numérico (contínuo) | Estimativa da % do sono total em sono profundo (ondas lentas) | 8,1 – 28 | Durante o sono |
| `rem_sleep_pct` | Numérico (contínuo) | Estimativa da % do sono total na fase REM (*Rapid Eye Movement*) | 9,6 – 27 | Durante o sono |
| `morning_alarm_snoozes` | Numérico (inteiro) | Número de vezes que o participante adiou o despertador pela manhã | 0 – 7 | Dia seguinte |
| `next_day_fatigue_score` | Numérico (contínuo) | Cansaço subjetivo no dia seguinte, em escala de 1,0 a 10,0. É o alvo de regressão sugerido pelo autor, e aqui é usado apenas como atributo | 1 – 10 | Dia seguinte |
| **`sleep_debt_category`** | **Categórico ordinal (alvo)** | **Alvo multiclasse que define a gravidade do débito de sono** | ver [classes](#classes-da-variável-alvo) | Alvo |

> As descrições seguem a documentação das colunas publicada pelo autor no Kaggle. Os domínios foram observados nos dados brutos e serão confirmados na etapa de validação.

## 🔄 Etapas do projeto

### 1. Validação dos dados
Auditoria da base bruta, **sem alterar os dados**. Verifica os tipos de cada atributo, os valores ausentes por coluna, as duplicatas, os valores mal formatados (ex.: espaços extras, variações de escrita), os valores fora do domínio esperado (ex.: horas negativas) e a consistência entre colunas. Também analisa a cardinalidade das categóricas, a distribuição da variável alvo e possíveis vazamentos de informação.

### 2. Limpeza dos dados
Aplica as correções **determinísticas** encontradas na validação: padronizar a formatação, remover duplicatas, ajustar os tipos e tratar os valores impossíveis. A base limpa é salva em `data/processed/`.

### 3. Separação treino/teste
Divide a base limpa em treino e teste, com **estratificação** pela variável alvo por causa do desbalanceamento das classes. A partir daqui, o conjunto de teste só é usado na avaliação final.

### 4. Análise exploratória (EDA)
Feita **apenas no conjunto de treino**. Inclui as distribuições dos atributos, a identificação e análise de outliers, as correlações entre variáveis numéricas e categóricas e a relação de cada atributo com a variável alvo, com visualizações gráficas. As conclusões guiam as decisões de pré-processamento.

### 5. Pré-processamento
Transformações que **aprendem com os dados** (imputação, tratamento de outliers, codificação, normalização, discretização, redução de dimensionalidade). Ficam encapsuladas em *pipelines* do scikit-learn, ajustados somente no treino para evitar *data leakage*. Cada algoritmo tem o seu próprio pré-processamento, já que eles têm necessidades diferentes (ex.: o KNN exige normalização, a árvore de decisão não).

### 6. Modelagem e validação
Treina os três classificadores e um *baseline*. Usa **validação cruzada k-fold estratificada** no treino para comparar os modelos e ajustar os hiperparâmetros, e faz uma avaliação final única no conjunto de teste (*holdout*). As métricas são a acurácia, a acurácia balanceada, o F1-score e a matriz de confusão.

### 7. Resultados
Compara o desempenho dos modelos, analisa os atributos mais importantes e apresenta as conclusões e limitações do estudo.

## 📈 Resultados

> Em breve.

| Modelo | Acurácia | Acurácia balanceada | F1 (macro) |
|---|---|---|---|
| Baseline | — | — | — |
| KNN | — | — | — |
| Naive Bayes | — | — | — |
| Árvore de Decisão | — | — | — |

## 📁 Estrutura do repositório

```
scroll-antes-de-dormir/
├── data/
│   ├── raw/          # dados originais, sem alterações
│   └── processed/    # dados limpos e divididos
├── notebooks/        # notebooks de cada etapa do projeto
├── src/              # código reutilizável
├── reports/
│   └── figures/      # gráficos gerados
├── pyproject.toml
├── uv.lock
├── LICENSE
└── README.md
```

## 🛠️ Tecnologias

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/):** gerenciamento do ambiente e das dependências
- **Jupyter:** notebooks de análise
- **pandas / NumPy:** manipulação dos dados
- **scikit-learn:** pré-processamento, modelos e validação
- **Matplotlib / Seaborn:** visualização

## ⚙️ Preparação do ambiente

### 1. Instalar o uv

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clonar o repositório e instalar as dependências

```bash
git clone https://github.com/diogovcastro/scroll-antes-de-dormir.git
cd scroll-antes-de-dormir
uv sync
```

O `uv sync` cria o ambiente virtual (`.venv`) e instala exatamente as versões registradas no `uv.lock`. Se a versão do Python exigida não estiver instalada, o uv baixa ela automaticamente.

### 3. Abrir os notebooks

```bash
uv run jupyter lab
```

Execute os notebooks na ordem numérica.

## 📄 Licença

- **Código:** [MIT](LICENSE)
- **Dados:** o conjunto de dados [Sleep Debt & Screen Time: Late Night Phone Habits](https://www.kaggle.com/datasets/samartalwar/sleep-debt-and-screen-time-late-night-phone-habits), de [Samar Talwar](https://www.kaggle.com/samartalwar), é distribuído sob a licença [Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/). Os dados em `data/processed/` são derivados do original (limpeza e divisão em treino/teste) e são distribuídos sob a mesma licença CC BY-SA 4.0.
