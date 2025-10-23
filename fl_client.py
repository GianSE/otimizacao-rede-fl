import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl # Flower

# Importando o que você já fez
from model_definition import SimpleCNN

# Define o dispositivo (vai usar CPU no seu caso)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Lógica de Treino/Teste (Adaptada da Semana 5) ---
# Estas são as funções que o *cliente* vai usar em seus *dados locais*

def train(model, train_loader, epochs):
    """Treina o modelo no dataset local do cliente."""
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Mesmo LR do baseline
    
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

def test(model, test_loader):
    """Avalia o modelo no dataset de teste local do cliente."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    correct = 0
    total_loss = 0.0
    num_examples = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            total_loss += loss_fn(output, target).item() * len(target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_examples += len(target)
            
    accuracy = correct / num_examples
    avg_loss = total_loss / num_examples
    return avg_loss, accuracy, num_examples

# --- Definição do Cliente Flower ---

class FlowerClient(fl.client.NumPyClient):
    """
    Define a lógica do cliente FL.
    """
    def __init__(self, model, train_dataset, test_dataset):
        self.model = model
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    def get_parameters(self, config):
        """Retorna os pesos atuais do modelo."""
        print("[Cliente] Enviando parâmetros para o servidor.")
        # Pega os parâmetros do modelo PyTorch e converte para uma lista NumPy
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Atualiza o modelo local com os pesos vindos do servidor."""
        print("[Cliente] Recebendo parâmetros do servidor.")
        # Converte a lista NumPy de volta para o state_dict do PyTorch
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        1. Recebe os parâmetros do servidor.
        2. Treina o modelo localmente.
        3. Retorna os novos parâmetros para o servidor.
        """
        print("[Cliente] Treinando localmente...")
        self.set_parameters(parameters) # Atualiza o modelo com os pesos globais
        
        # Treina o modelo (ex: 1 época local)
        train(self.model, self.train_loader, epochs=1) 
        
        # Retorna os parâmetros atualizados e o número de exemplos de treino
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """
        1. Recebe os parâmetros do servidor.
        2. Avalia o modelo nos dados de teste locais.
        3. Retorna os resultados da avaliação.
        """
        print("[Cliente] Avaliando localmente...")
        self.set_parameters(parameters) # Atualiza com os pesos globais
        
        # Avalia
        loss, accuracy, num_examples = test(self.model, self.test_loader)
        
        # Retorna os resultados para o servidor agregar
        return loss, num_examples, {"accuracy": accuracy}


# O 'if __name__ == "__main__":' NÃO VAI RODAR NADA AINDA.
# Este arquivo é apenas uma definição de classe.
# Vamos usá-lo na Semana 7.

if __name__ == '__main__':
    print("Este é o script 'fl_client.py'.")
    print("Ele define a classe FlowerClient, mas não inicia um cliente.")
    print("Ele será importado e usado pelo script 'run_simulation.py' (Semana 7).")