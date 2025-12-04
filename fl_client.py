import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import flwr as fl
import math
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from collections import OrderedDict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, epochs):
    """Loop de treino padrão."""
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-4) # Learning rate um pouco maior para LoRA
    
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            if step % 10 == 0:
                print(f"Época {epoch} | Passo {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

def test(model, test_loader):
    """Loop de avaliação padrão."""
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")
        
    print(f"Avaliação: Loss {avg_loss:.4f} | PPL {perplexity:.4f}")
    return avg_loss, perplexity, len(test_loader.dataset)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, test_dataset):
        self.model = model
        self.train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=4)

    def get_parameters(self, config):
        """
        RETORNA APENAS OS PESOS TREINÁVEIS (LoRA).
        Isso reduz o envio de 300MB para ~1MB.
        """
        # Filtra apenas os parâmetros que requerem gradiente (LoRA)
        return [val.cpu().numpy() for _, val in self.model.state_dict().items() if "lora" in _]

    def set_parameters(self, parameters):
        """
        ATUALIZA APENAS OS PESOS TREINÁVEIS (LoRA).
        """
        # Pega as chaves (nomes) apenas das camadas LoRA
        keys = [k for k in self.model.state_dict().keys() if "lora" in k]
        
        # Cria um dicionário zipando nomes e valores
        params_dict = zip(keys, parameters)
        
        # Converte para tensores
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        # Carrega com strict=False, pois só estamos atualizando uma parte do modelo
        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        print("\n[LoRA Client] Iniciando treino leve...")
        self.set_parameters(parameters)
        train(self.model, self.train_loader, epochs=1)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, perplexity, num_examples = test(self.model, self.test_loader)
        return loss, num_examples, {"perplexity": perplexity}