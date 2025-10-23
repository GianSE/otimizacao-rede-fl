# Projeto de IC: Aprendizado Federado com PyTorch e Flower

Este projeto é uma implementação de um sistema de Aprendizado Federado (FL) usando PyTorch e a biblioteca Flower. O objetivo é simular um cenário onde 10 clientes treinam um modelo de Rede Neural Convolucional (CNN) no dataset MNIST de forma descentralizada.

```bash
\venv\Scripts\Activate.ps1
```

```bash
deactivate
```

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