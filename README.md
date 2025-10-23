# Projeto de IC: Aprendizado Federado com PyTorch e Flower

Este projeto é uma implementação de um sistema de Aprendizado Federado (FL) usando PyTorch e a biblioteca Flower. O objetivo é simular um cenário onde 10 clientes treinam um modelo de Rede Neural Convolucional (CNN) no dataset MNIST de forma descentralizada.

```bash
\venv\Scripts\Activate.ps1
```

```bash
deactivate
```

# Primeiro, atualize o pip
python -m pip install --upgrade pip

# Instale as bibliotecas principais (PyTorch versão CPU)
pip install torch torchvision torchaudio

# (OU se preferir TensorFlow versão CPU)
pip install tensorflow

# Instale as outras dependências
pip install numpy
pip install flwr


## Estrutura dos Arquivos

* `data_loader.py`: Carrega e normaliza o dataset MNIST.
* `data_partitioner.py`: Divide o dataset de treino em N partições (uma para cada cliente).
* `model_definition.py`: Define a arquitetura da `SimpleCNN` usada no projeto.
* `baseline_training.py`: Script para treinar o modelo de forma centralizada (nosso baseline).
* `fl_client.py`: Define a classe `FlowerClient` que contém a lógica de treino/avaliação do cliente.
* `server.py`: O servidor FL que orquestra as rodadas de agregação.
* `client_runner.py`: O script que simula um cliente individual que se conecta ao servidor.

## Como Instalar

1.  Clone este repositório.
2.  Crie e ative um ambiente virtual (ex: `python -m venv venv` e `.\venv\Scripts\activate`).
3.  Instale as dependências:

```bash
pip install -r requirements.txt
```

    *(Não se esqueça de ter criado o `requirements.txt` com `pip freeze > requirements.txt`)*

## Como Rodar a Simulação

Esta simulação requer **dois terminais** abertos na pasta do projeto (e com o `venv` ativado).

**Terminal 1: Iniciar o Servidor**

Primeiro, inicie o servidor. Ele ficará "travado" esperando os clientes se conectarem.

```bash
python server.py
```

Com certeza! Você executou a simulação de Aprendizado Federado com sucesso e obteve um resultado excelente (97.30% de acurácia, muito perto do seu baseline de 98.16%!).

O seu projeto de 8 semanas está funcionalmente completo. O próximo e último passo, de acordo com o seu "Plano de Ação IC (2 Meses)", é a Semana 8: Conclusão e Próximos Passos.

O foco agora é documentar o que você fez para que outras pessoas (ou você mesmo no futuro) possam entender e rodar seu projeto.

Tarefas da Semana 8:
Módulo de Software: README.md.

Meta Principal: Criar o "manual de instruções" do seu projeto.

O README.md é um arquivo de texto (escrito em formato Markdown) que fica na pasta principal do seu projeto e é a primeira coisa que alguém vê no GitHub.

Como Começar (Seu README.md)
Aqui está um modelo excelente para você. Crie um novo arquivo chamado README.md (exatamente assim, com letras maiúsculas) na sua pasta otimizacao-rede-fl e cole o texto abaixo nele.

Você pode editá-lo diretamente no VS Code (ele até tem um botão de "preview" para ver como fica bonito).

(Copie e cole tudo abaixo disto no seu novo arquivo README.md)

Markdown

# Projeto de IC: Aprendizado Federado com PyTorch e Flower

Este projeto é uma implementação de um sistema de Aprendizado Federado (FL) usando PyTorch e a biblioteca Flower. O objetivo é simular um cenário onde 10 clientes treinam um modelo de Rede Neural Convolucional (CNN) no dataset MNIST de forma descentralizada.

## Estrutura dos Arquivos

* `data_loader.py`: Carrega e normaliza o dataset MNIST.
* `data_partitioner.py`: Divide o dataset de treino em N partições (uma para cada cliente).
* `model_definition.py`: Define a arquitetura da `SimpleCNN` usada no projeto.
* `baseline_training.py`: Script para treinar o modelo de forma centralizada (nosso baseline).
* `fl_client.py`: Define a classe `FlowerClient` que contém a lógica de treino/avaliação do cliente.
* `server.py`: O servidor FL que orquestra as rodadas de agregação.
* `client_runner.py`: O script que simula um cliente individual que se conecta ao servidor.

## Como Instalar

1.  Clone este repositório.
2.  Crie e ative um ambiente virtual (ex: `python -m venv venv` e `.\venv\Scripts\activate`).
3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
    *(Não se esqueça de ter criado o `requirements.txt` com `pip freeze > requirements.txt`)*

## Como Rodar a Simulação

Esta simulação requer **dois terminais** abertos na pasta do projeto (e com o `venv` ativado).

**Terminal 1: Iniciar o Servidor**

Primeiro, inicie o servidor. Ele ficará "travado" esperando os clientes se conectarem.

```bash
python server.py
Terminal 2: Iniciar os Clientes

**Enquanto o servidor está rodando, abra um segundo terminal e execute o seguinte comando (PowerShell) para iniciar 10 clientes em paralelo:**

```bash
foreach ($i in 0..9) { Start-Process python -ArgumentList "client_runner.py $i" }
```