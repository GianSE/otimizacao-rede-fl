# Hack para o erro de DLL (SEMPRE no topo)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import torch
import flwr as fl

# --- Importando todas as suas novas peças (V2) ---
from data_loader import load_data
from model_definition import load_model_and_tokenizer
from fl_client import FlowerClient # Reutilizando sua classe V2!

# --- 1. Pegar o ID do Cliente ---
if len(sys.argv) < 2:
    raise Exception("Erro: Forneça o ID do cliente (um número de 0 a 9)")
    
CLIENT_ID = int(sys.argv[1])
NUM_CLIENTS = 10

if not 0 <= CLIENT_ID < NUM_CLIENTS:
     raise Exception(f"Erro: O ID do cliente deve estar entre 0 e {NUM_CLIENTS-1}")

print(f"[Cliente {CLIENT_ID}] Iniciando...")

# --- 2. Definir Dispositivo ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. Carregar Modelo Pré-treinado ---
print(f"[Cliente {CLIENT_ID}] Carregando modelo GenAI...")
# Carregamos o modelo, mas não precisamos do tokenizer aqui
model, _ = load_model_and_tokenizer()
model.to(DEVICE)

# --- 4. Carregar e Particionar os Dados ---
print(f"[Cliente {CLIENT_ID}] Carregando e particionando dados...")
# Carrega todos os dados (eles vêm do cache do Hugging Face)
train_dataset, test_dataset = load_data()

# Particiona os dados de TREINO
# .shard() é o jeito do Hugging Face de dividir o dataset
# Cada cliente pega sua própria "fatia"
my_train_data = train_dataset.shard(
    num_shards=NUM_CLIENTS, 
    index=CLIENT_ID
)

# Todos os clientes usam o mesmo dataset de TESTE (validação)
my_test_data = test_dataset 

print(f"[Cliente {CLIENT_ID}] Dados prontos. Amostras de treino: {len(my_train_data)}")

# --- 5. Instanciar e Iniciar o Cliente Flower ---
print(f"[Cliente {CLIENT_ID}] Conectando ao servidor em 127.0.0.1:8080...")

# Cria a instância da sua classe FlowerClient (V2)
client_app = FlowerClient(model, my_train_data, my_test_data)

# Inicia o cliente (usando a nova API 'start_client' para evitar warnings)
fl.client.start_client(
    server_address="127.0.0.1:8080", # Endereço do servidor
    client=client_app.to_client(),  # Converte o NumPyClient para um Client
)

print(f"[Cliente {CLIENT_ID}] Finalizado.")