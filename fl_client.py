# Hack para o erro de DLL (SEMPRE no topo)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import flwr as fl
import math
from torch.utils.data import DataLoader
from torch.optim import AdamW                 # <-- CORREÇÃO AQUI
from transformers import get_scheduler       # Otimizadores para Transformers

# Importando suas peças
from model_definition import load_model_and_tokenizer

# Define o dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Lógica de Treino/Teste (Adaptada para GenAI) ---

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
        for batch in train_loader:
            # Envia o batch para o dispositivo
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            # 1. Zera os gradientes
            optimizer.zero_grad()
            
            # 2. Forward pass (Predição)
            outputs = model(**batch)
            loss = outputs.loss
            
            # 3. Backward pass (Cálculo dos gradientes)
            loss.backward()
            
            # 4. Atualiza os pesos
            optimizer.step()
            lr_scheduler.step()

    print(f"Treino local concluído. Perda (Loss) final: {loss.item():.4f}")

def test(model, test_loader):
    """
    Avalia o modelo no dataset de teste local.
    Métrica: Perplexidade (quanto menor, melhor).
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
    avg_loss = total_loss / num_batches
    
    # Perplexidade é o exponencial da perda média
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")
        
    print(f"Avaliação local concluída. Perda Média: {avg_loss:.4f}, Perplexidade: {perplexity:.4f}")
    
    return avg_loss, perplexity, len(test_loader.dataset)

# --- Definição do Cliente Flower (Adaptado) ---

class FlowerClient(fl.client.NumPyClient):
    """
    Define a lógica do cliente FL para o modelo de linguagem.
    """
    def __init__(self, model, train_dataset, test_dataset):
        self.model = model
        
        # BATCH_SIZE precisa ser pequeno (ex: 4 ou 8)
        self.train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=4)

    def get_parameters(self, config):
        """Retorna os pesos atuais do modelo."""
        print("[Cliente] Enviando parâmetros para o servidor.")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Atualiza o modelo local com os pesos vindos do servidor."""
        print("[Cliente] Recebendo parâmetros do servidor.")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Treina o modelo localmente."""
        print("[Cliente] Iniciando 'fit' (treinamento local)...")
        self.set_parameters(parameters) 
        
        train(self.model, self.train_loader, epochs=1) 
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Avalia o modelo localmente."""
        print("[Cliente] Iniciando 'evaluate' (avaliação local)...")
        self.set_parameters(parameters) 
        
        loss, perplexity, num_examples = test(self.model, self.test_loader)
        
        # Flower espera a 'loss' primeiro
        return loss, num_examples, {"perplexity": perplexity}


if __name__ == '__main__':
    # Teste rápido para o fl_client.py
    
    print("--- Testando o fl_client.py ---")
    
    print("Carregando modelo e tokenizer...")
    model, _ = load_model_and_tokenizer()
    model.to(DEVICE)
    
    print("Carregando dados (versão de teste)...")
    from data_loader import load_data
    train_data, test_data = load_data()
    
    # Pega subconjuntos pequenos para um teste rápido
    dummy_train = train_data.select(range(20))
    dummy_test = test_data.select(range(10))

    print("Iniciando cliente de teste...")
    client_teste = FlowerClient(model, dummy_train, dummy_test)
    
    print("\nTestando client.fit()...")
    params_iniciais = client_teste.get_parameters(config={})
    params_novos, num_exemplos, _ = client_teste.fit(params_iniciais, config={})
    print(f"Fit concluído. {num_exemplos} amostras processadas.")

    print("\nTestando client.evaluate()...")
    loss, num_ex, metrics = client_teste.evaluate(params_novos, config={})
    print(f"Evaluate concluído. Loss: {loss:.4f}, Perplexity: {metrics['perplexity']:.4f}")
    
    print("\n--- Teste de fl_client.py concluído com sucesso! ---")