import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Importando o que você já fez:
from data_loader import load_mnist_data
from model_definition import SimpleCNN

def train(model, train_loader, optimizer, loss_fn, device):
    """
    Executa uma época de treinamento.
    """
    model.train() # Coloca o modelo em modo de treinamento
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Envia dados e rótulos para o dispositivo (CPU ou GPU)
        data, target = data.to(device), target.to(device)
        
        # 1. Zera os gradientes
        optimizer.zero_grad()
        
        # 2. Faz a predição (forward pass)
        output = model(data)
        
        # 3. Calcula a perda (loss)
        loss = loss_fn(output, target)
        
        # 4. Calcula os gradientes (backward pass)
        loss.backward()
        
        # 5. Atualiza os pesos
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Batch [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item():.6f}")

def test(model, test_loader, loss_fn, device):
    """
    Avalia o modelo no dataset de teste.
    """
    model.eval() # Coloca o modelo em modo de avaliação (desliga dropout, etc.)
    test_loss = 0
    correct = 0
    
    # 'with torch.no_grad()' desliga o cálculo de gradientes
    # para economizar memória e acelerar a avaliação
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Soma a perda do batch
            test_loss += loss_fn(output, target).item()
            
            # Pega o índice da classe com maior probabilidade (a predição)
            pred = output.argmax(dim=1, keepdim=True)
            
            # Compara a predição com o rótulo verdadeiro
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f"\nResultado do Teste:")
    print(f"  Perda média: {test_loss:.4f}")
    print(f"  Acurácia: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
    return accuracy

if __name__ == '__main__':
    
    # --- Configurações ---
    NUM_EPOCHS = 5 # Quantas vezes vamos rodar o dataset completo
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    
    # Define o dispositivo (vai usar CPU no seu caso)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # --- 1. Carregar Dados (Semana 2) ---
    train_dataset, test_dataset = load_mnist_data()
    
    # DataLoader prepara os dados em "lotes" (batches)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 2. Carregar Modelo (Semana 4) ---
    model = SimpleCNN().to(device)
    
    # --- 3. Definir Otimizador e Perda ---
    # Função de perda (ótima para classificação)
    loss_fn = nn.CrossEntropyLoss()
    
    # Otimizador (Adam é uma escolha robusta e popular)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 4. Loop de Treinamento ---
    print("\n--- Iniciando Treinamento Centralizado (Baseline) ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- Época {epoch}/{NUM_EPOCHS} ---")
        train(model, train_loader, optimizer, loss_fn, device)
        test(model, test_loader, loss_fn, device)
        
    print("--- Treinamento Baseline Concluído ---")