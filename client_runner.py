# Hack para o erro de DLL (SEMPRE no topo)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import torch
import flwr as fl

# Importar suas peças
from data_loader import load_mnist_data
from data_partitioner import partition_data
from model_definition import SimpleCNN
from fl_client import FlowerClient # Reutilizando sua classe!

# --- 1. Pegar o ID do Cliente ---
# Vamos passar o ID (0, 1, 2...) pela linha de comando
if len(sys.argv) < 2:
    raise Exception("Erro: Forneça o ID do cliente (um número de 0 a 9)")
    
CLIENT_ID = int(sys.argv[1])
if not 0 <= CLIENT_ID < 10:
     raise Exception("Erro: O ID do cliente deve estar entre 0 e 9")

print(f"[Cliente {CLIENT_ID}] Iniciando...")

# --- 2. Definir Dispositivo ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. Carregar Modelo ---
model = SimpleCNN().to(DEVICE)

# --- 4. Carregar a Partição de Dados deste Cliente ---
print(f"[Cliente {CLIENT_ID}] Carregando dados...")
# Carrega todos os dados
train_dataset, test_dataset = load_mnist_data()
# Particiona
NUM_CLIENTS = 10
client_partitions = partition_data(train_dataset, num_clients=NUM_CLIENTS)

# Pega apenas a partição deste cliente
my_train_data = client_partitions[CLIENT_ID]

# (Cada cliente vai avaliar no dataset de teste INTEIRO)
my_test_data = test_dataset 

# --- 5. Instanciar e Iniciar o Cliente Flower ---
print(f"[Cliente {CLIENT_ID}] Conectando ao servidor em 127.0.0.1:8080...")

# Cria a instância da sua classe FlowerClient
client_app = FlowerClient(model, my_train_data, my_test_data)

# Inicia o cliente NumPy
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080", # Endereço do servidor
    client=client_app,
)

print(f"[Cliente {CLIENT_ID}] Finalizado.")