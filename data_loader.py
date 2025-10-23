import torch
from torchvision import datasets, transforms

def load_mnist_data():
    """
    Baixa e carrega o dataset MNIST, aplicando transformações.
    """
    
    # Define a transformação:
    # 1. Converte a imagem para Tensor
    # 2. Normaliza os dados (pixels de 0-1 para -1 a 1)
    #    Isso ajuda o modelo a aprender mais rápido.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
        # (0.5,) é a média e (0.5,) é o desvio padrão para 1 canal (preto e branco)
    ])
    
    # Baixa o dataset de treino (se não existir na pasta ./data)
    train_dataset = datasets.MNIST(
        root='./data',  # Pasta onde os dados serão salvos
        train=True,     # Indica que é o conjunto de treino
        download=True,  # Baixa se não estiver na pasta 'root'
        transform=transform # Aplica a transformação que definimos
    )
    
    # Baixa o dataset de teste
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,    # Indica que é o conjunto de teste
        download=True,
        transform=transform
    )
    
    print(f"Dataset MNIST carregado.")
    print(f"Amostras de treino: {len(train_dataset)}")
    print(f"Amostras de teste: {len(test_dataset)}")
    
    return train_dataset, test_dataset

if __name__ == '__main__':
    # Este código só roda quando você executa 'python data_loader.py'
    # É bom para testar se o módulo funciona
    train_data, test_data = load_mnist_data()
    
    # Exemplo: pegar a primeira imagem do treino
    image, label = train_data[0]
    print(f"Dimensões da 1ª imagem: {image.shape}") # Deve ser [1, 28, 28] (Canal, Altura, Largura)
    print(f"Rótulo da 1ª imagem: {label}")