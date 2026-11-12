
# ARC-AGI-1 Solver com Síntese de Código e Autorreflexão

**Autor:** Thiago Aschebrock Baptista  
**Modelo de Referência:** `gemini-3.5-flash-lite`

Sistema autônomo em Python desenvolvido para resolver tarefas de raciocínio abstrato e indução espacial do benchmark **ARC-AGI-1 (*Abstraction and Reasoning Corpus*)**, comparando experimentalmente três paradigmas de inferência com Modelos de Linguagem (LLMs): predição direta (*End-to-End*), síntese de programas (*Code Execution*) e refinamento iterativo com autorreflexão (*Reflection* / *Pass@k*).

---

## 🎯 Problema e Arquitetura

O benchmark **ARC-AGI** avalia a capacidade de sistemas de IA inferirem regras lógicas e transformações matriciais a partir de poucos pares de demonstração (`train`) e generalizarem para grades inéditas (`test`).

Para superar as limitações da predição direta de tokens em tarefas espaciais, o solver implementa um pipeline modular:

1. **Predição Direta (`end-to-end`):** O LLM analisa os exemplos e gera diretamente as matrizes de saída em JSON.
2. **Síntese de Código (`code-execution`):** O LLM sintetiza uma função Python que expressa a regra lógica, valida-a nos exemplos de treino e aplica o código às entradas de teste.
3. **Reflexão Iterativa e *Pass@k* (`reflection`):** Executa múltiplas amostras (`--samples k`) com até 3 ciclos de autocorreção por amostra. Sempre que o código falha em algum exemplo de treino, o erro e a divergência matricial são retroalimentados ao modelo para refinamento da hipótese.

## 📁 Estrutura do Projeto

```text
arc-agi-llm-solver-reflection/
├── solver/
│   ├── main.py              # Orquestrador CLI, controle de métricas e experimentos
│   ├── llm_client.py        # Cliente de comunicação com a API do LLM (Google GenAI)
│   ├── prompt_manager.py    # Engenharia de prompts e parsers de saída/código
│   ├── code_executor.py     # Sandbox seguro para execução e teste nos exemplos de treino
│   └── validator.py         # Validação exata das matrizes contra o gabarito ARC-AGI
├── tarefas/                 # Arquivos JSON e lista de tarefas do benchmark ARC-AGI-1 (tarefas.txt)
├── resultados/              # Resumos consolidados em JSON (Fim a Fim, Código e Reflexão)
├── runs/                    # Logs detalhados, prompts e respostas por execução e tarefa
├── relatorio.pdf            # Relatório técnico compilado no formato IJCAI (2 páginas)
├── requirements.txt         # Dependências Python do projeto
├── .env.example             # Modelo de configuração da chave GOOGLE_API_KEY
└── README.md                # Documentação do repositório

```

* **`solver/`**: Núcleo da aplicação contendo toda a lógica de geração de prompts, chamadas assíncronas/síncronas ao modelo, execução isolada de código gerado e validação matricial.
* **`tarefas/`**: Base de entrada utilizada nos experimentos, contendo a lista de identificadores (`tarefas.txt`) e os arquivos JSON do benchmark ARC-AGI-1.
* **`resultados/`**: Métricas agregadas de cada experimento (`fim_a_fim_summary.json`, `geracao_codigo_summary.json` e `reflexao_summary.json`), facilitando a comparação direta de acurácia e consumo de tokens.
* **`runs/`**: Histórico granular organizado por experimento (`<label>/<tarefa>/`), armazenando o prompt enviado (`prompt.txt`), a resposta/código produzido (`answer.json`) e a telemetria individual (`log.json`).

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.12+** & **`uv`** (gerenciamento rápido de pacotes e ambientes virtuais)
* **Google GenAI SDK (`gemini-3.5-flash-lite`)**
* **NumPy** (manipulação e validação matricial)
* **JSON / Logging estruturado** (telemetria de latência, tentativas e tokens de raciocínio)

---

## 🚀 Como Executar

### 1. Instalação e Configuração do Ambiente

Clone o repositório e configure sua chave da API do Google no arquivo `.env` na raiz do projeto:

```bash
git clone https://github.com/ThiagoABaptista/arc-agi-llm-solver-reflection
cd arc-agi-llm-solver-reflection

```

Crie o arquivo `.env`:

```env
GOOGLE_API_KEY=sua_chave_de_api_aqui

```

Instale as dependências com `uv` (ou `pip install -r requirements.txt`):

```powershell
uv sync

```

### 2. Abordagem A: Fim a Fim (*End-to-End*)

```powershell
uv run --active solver/main.py tarefas/tarefas.txt --mode end-to-end --provider google --model gemini-3.5-flash-lite --label fim_a_fim

```

### 3. Abordagem B: Geração de Código (*Code Generation*)

```powershell
uv run --active solver/main.py tarefas/tarefas.txt --mode code-execution --provider google --model gemini-3.5-flash-lite --label geracao_codigo

```

### 4. Abordagem C: Geração de Código com Reflexão e *Pass@k* (Nível 3)

```powershell
uv run --active solver/main.py tarefas/tarefas.txt --mode reflection --provider google --model gemini-3.5-flash-lite --samples 3 --temperature 0.2 --label reflexao

```

---

## 📊 Resultados e Artefatos Gerados

* **`resultados/*_summary.json`**: Métricas consolidadas de cada abordagem (`accuracy_percentage`, `train_100_tasks`, duração total e consumo detalhado de tokens de entrada, raciocínio e saída).
* **`runs/<label>/<tarefa>/`**: Artefatos individuais por tarefa contendo `prompt.txt`, `answer.json` e `log.json`.
* **`relatorio.pdf`**: Análise comparativa dos resultados experimentais e discussão técnica no padrão acadêmico IJCAI.
