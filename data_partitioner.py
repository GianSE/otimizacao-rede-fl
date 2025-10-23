import torch
from torch.utils.data import DataLoader, random_split
from data_loader import load_mnist_data # Importa sua função da Semana 2

def partition_data(dataset, num_clients=10):
    """
    Divide um dataset em N partições para simular clientes FL.
    
    Para o MNIST, a divisão mais simples é dividir igualmente 
    o número de amostras (ex: 60.000 amostras / 10 clientes = 6.000 por cliente).
    """
    
    # Calcula o tamanho de cada partição
    num_samples = len(dataset)
    samples_per_client = num_samples // num_clients
    
    # Garante que todas as amostras sejam usadas
    # (o último cliente fica com o resto)
    lengths = [samples_per_client] * (num_clients - 1)
    lengths.append(num_samples - sum(lengths)) # Adiciona o restante ao último cliente
    
    # 'random_split' é a forma mais fácil do PyTorch de fazer isso
    # Ele divide o dataset nos tamanhos que definimos em 'lengths'
    client_datasets = random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))
    
    # Retorna uma lista, onde cada item é o 'Subset' (subconjunto) de um cliente
    return client_datasets

if __name__ == '__main__':
    # 1. Carrega os dados (da Semana 2)
    train_data, test_data = load_mnist_data()
    
    # 2. Particiona os dados (Tarefa da Semana 3)
    NUM_CLIENTES = 10
    print(f"Dividindo o dataset de treino para {NUM_CLIENTES} clientes...")
    
    client_partitions = partition_data(train_data, num_clients=NUM_CLIENTES)
    
    # 3. Testa o resultado
    print(f"Total de partições criadas: {len(client_partitions)}")
    
    # Verifica o tamanho da primeira e da última partição
    print(f"Tamanho da partição do Cliente 1: {len(client_partitions[0])}")
    print(f"Tamanho da partição do Cliente {NUM_CLIENTES}: {len(client_partitions[-1])}")