import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Uma Rede Neural Convolucional (CNN) simples para classificação no MNIST.
    Arquitetura: [Conv -> ReLU -> Pool] -> [Conv -> ReLU -> Pool] -> [FC -> ReLU] -> [FC (Output)]
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Camada Convolucional 1
        # Entrada: 1 canal (imagem P&B), Saída: 10 filtros, Kernel: 5x5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        
        # Camada Convolucional 2
        # Entrada: 10 canais (da camada anterior), Saída: 20 filtros, Kernel: 5x5
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        
        # Camada de Dropout para regularização (ajuda a evitar overfitting)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        
        # Camada Linear 1 (Totalmente Conectada)
        # A entrada (320) é calculada a partir da saída da conv2 após o pooling
        # (20 filtros * 4 * 4 = 320)
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        
        # Camada Linear 2 (Saída)
        # Entrada: 50, Saída: 10 (uma para cada dígito, 0-9)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        # x tem formato [batch_size, 1, 28, 28]
        
        # Passa pela conv1, aplica ReLU e Max Pooling (2x2)
        # Formato muda para [batch_size, 10, 12, 12]
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        # Passa pela conv2, aplica Dropout, ReLU e Max Pooling (2x2)
        # Formato muda para [batch_size, 20, 4, 4]
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        # "Achata" o tensor para a camada linear
        # Formato muda para [batch_size, 320]
        x = x.view(-1, 320) # O -1 infere o tamanho do batch
        
        # Passa pela camada linear 1 com ReLU
        # Formato muda para [batch_size, 50]
        x = F.relu(self.fc1(x))
        
        # Passa pela camada de saída (fc2)
        # Formato muda para [batch_size, 10]
        # (Não aplicamos Softmax aqui, pois a função de perda CrossEntropyLoss faz isso)
        x = self.fc2(x)
        
        return x

if __name__ == '__main__':
    # Teste rápido para verificar se a rede está funcionando
    model = SimpleCNN()
    print("--- Arquitetura do Modelo ---")
    print(model)
    
    # Cria uma "imagem" falsa (1 imagem, 1 canal, 28x28)
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # Passa a imagem pela rede
    output = model(dummy_input)
    
    print("\n--- Teste de Execução ---")
    print(f"Dimensão da saída: {output.shape}") # Esperado: [1, 10]
    print(f"Valores de saída (logits): {output.data}")