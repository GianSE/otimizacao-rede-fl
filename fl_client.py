# fl_client.py
# Hack para o erro de DLL (SEMPRE no topo)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import flwr as fl
import math
from torch.utils.data import DataLoader
from torch.optim import AdamW                 
from transformers import get_scheduler

# Importando suas peças
from model_definition import load_model_and_tokenizer

# Define o dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Lógica de Treino/Teste ---

def train(model, train_loader, epochs):
    """Treina o modelo de linguagem no dataset local do cliente."""
    model.train()
    
    # Otimizador padrão para Transformers
    optimizer = AdamW(model.parameters(), lr=5e-5) 
    
    # Agendador de taxa de aprendizado
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    print(f"Iniciando treino local por {epochs} época(s)...")
    
    for epoch in range(epochs):
        # Usamos enumerate para ter o número do passo (step)
        for step, batch in enumerate(train_loader):
            # Envia o batch para o dispositivo
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            # 1. Zera os gradientes
            optimizer.zero_grad()
            
            # 2. Forward pass (CRUCIAL: Tem que ser feito a cada passo!)
            outputs = model(**batch)
            loss = outputs.loss
            
            # 3. Backward pass
            loss.backward()
            
            # 4. Atualiza os pesos
            optimizer.step()
            lr_scheduler.step()
            
            # Imprime progresso a cada 10 passos para você ver que não travou
            if step % 10 == 0:
                print(f"Época {epoch} | Passo {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

    print(f"Treino local concluído. Perda (Loss) final: {loss.item():.4f}")

def test(model, test_loader):
    """
    Avalia o modelo no dataset de teste local.
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    print("Iniciando avaliação local...")
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1
            
    # Calcula a Perda (Loss) média
    if num_batches > 0:
        avg_loss = total_loss / num_batches
    else:
        avg_loss = 0.0
    
    # Perplexidade
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")
        
    print(f"Avaliação local concluída. Loss: {avg_loss:.4f}, PPL: {perplexity:.4f}")
    
    return avg_loss, perplexity, len(test_loader.dataset)

# --- Definição do Cliente Flower ---

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, test_dataset):
        self.model = model
        # Batch size 4 para economizar memória na CPU
        self.train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=4)

    def get_parameters(self, config):
        # print("[Cliente] Enviando parâmetros...") # Comentado para poluir menos
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # print("[Cliente] Recebendo parâmetros...")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("\n[Cliente] Iniciando treinamento...")
        self.set_parameters(parameters) 
        
        # Chama a função de treino corrigida
        train(self.model, self.train_loader, epochs=1) 
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        print("[Cliente] Iniciando avaliação...")
        self.set_parameters(parameters) 
        loss, perplexity, num_examples = test(self.model, self.test_loader)
        return loss, num_examples, {"perplexity": perplexity}

# Bloco de teste local (não usado pelo Flower, mas útil para debug)
if __name__ == '__main__':
    print("Este arquivo deve ser usado via client_runner.py")