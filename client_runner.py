import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import flwr as fl

# Importa a nova função apply_lora
from data_loader import load_data
from model_definition import load_model_and_tokenizer, apply_lora 
from fl_client import FlowerClient

# 1. Configuração Inicial
if len(sys.argv) < 2:
    raise Exception("Erro: Forneça o ID do cliente")
CLIENT_ID = int(sys.argv[1])
NUM_CLIENTS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[Cliente {CLIENT_ID}] Iniciando com LoRA...")

# 2. Carregar Modelo Base
model, _ = load_model_and_tokenizer()

# 3. APLICAR LORA
print(f"[Cliente {CLIENT_ID}] Aplicando adaptadores LoRA...")
model = apply_lora(model)
model.to(DEVICE)

# 4. Dados
all_data = load_data()
train_dataset = all_data["train"].shard(num_shards=NUM_CLIENTS, index=CLIENT_ID)
test_dataset = all_data["test"]

# 5. Iniciar Cliente (COM PROTEÇÃO DE REDE)
print(f"[Cliente {CLIENT_ID}] Conectando ao servidor...")
fl.client.start_client(
    server_address="127.0.0.1:8080", 
    client=FlowerClient(model, train_dataset, test_dataset).to_client(),
    # Voltamos com essa linha para garantir que o Windows não reclame do tamanho do buffer
    grpc_max_message_length=1024*1024*1024 
)