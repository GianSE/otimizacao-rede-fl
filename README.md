# Projeto de IC: Aprendizado Federado com PyTorch e Flower

Este projeto é uma implementação de um sistema de Aprendizado Federado (FL) usando PyTorch e a biblioteca Flower. O objetivo é simular um cenário onde 10 clientes treinam um modelo de Rede Neural Convolucional (CNN) no dataset MNIST de forma descentralizada.

```bash
\venv\Scripts\Activate.ps1
```

```bash
deactivate
```

## Estrutura dos Arquivos

* `baixa_modelo.py`: Script de configuração (executa uma vez para baixar o modelo).
* `client_runner.py`: Quem inicia os clientes.
* `data_loader.py`: Carrega os dados processados do disco.
* `fl_client.py`: A lógica de treino (loop do PyTorch).
* `model_definition.py`: Carrega o modelo GPT-2.
* `preparar_dados.py`: Script de configuração (executa uma vez para criar o dataset).
* `server.py`: O cérebro da operação.

* `exec_train.bat`: Executa venv, server e cliente em sequencia

## Como Instalar

1.  Clone este repositório.
2.  Crie e ative um ambiente virtual (ex: `python -m venv venv` e `.\venv\Scripts\activate`).
3.  Instale as dependências:
```bash
pip install -r requirements.txt
```

## Como Rodar a Simulação

Esta simulação requer **dois terminais** abertos na pasta do projeto (e com o `venv` ativado).

**Terminal 1: Iniciar o Servidor**

Primeiro, inicie o servidor. Ele ficará "travado" esperando os clientes se conectarem.

```bash
python server.py
```

**Terminal 2: Iniciar os Clientes**

Enquanto o servidor está rodando, abra um segundo terminal e execute o seguinte comando (PowerShell) para iniciar 10 clientes em paralelo:

```bash
foreach ($i in 0..9) { Start-Process python -ArgumentList "client_runner.py $i" }
```

Sequencia de scripts pra usar o projeto

```bash
# 1. Cria o ambiente virtual
python -m venv venv

# 2. Ativa o ambiente
./venv/scripts/activate

# 3. Instala as dependências listadas
pip install -r requirements.txt

# 4. Baixa o modelo distilgpt2 e salva na pasta './meu_modelo_torch'
python baixar_modelo.py

# 5. Baixa o dataset, processa e salva na pasta './meu_dataset_processado'
python preparar_dados.py

# 6. Executa o launcher automático
./exec_train.py
```