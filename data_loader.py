# data_loader.py (Versão Otimizada)
import torch
from datasets import load_from_disk # <--- A mágica está aqui

# Nome da pasta onde salvamos os dados
PASTA_DATASET = "./meu_dataset_processado"

def load_data():
    """
    Carrega o dataset já processado do disco.
    Zero processamento, zero conflito, velocidade máxima.
    """
    print(f"Carregando dados processados de '{PASTA_DATASET}'...")
    
    try:
        # Carrega o objeto DatasetDict inteiro do disco
        tokenized_datasets = load_from_disk(PASTA_DATASET)
        
        # Define o formato PyTorch (caso tenha se perdido no save/load)
        tokenized_datasets.set_format("torch")
        
        train_dataset = tokenized_datasets["train"]
        test_dataset = tokenized_datasets["validation"]
        
        print(f"Dados carregados! Treino: {len(train_dataset)}, Teste: {len(test_dataset)}")
        return train_dataset, test_dataset
        
    except FileNotFoundError:
        raise Exception(f"Erro: A pasta '{PASTA_DATASET}' não existe. Rode 'python preparar_dados.py' primeiro!")

if __name__ == '__main__':
    # Teste rápido
    train, test = load_data()
    print(f"Shape do primeiro item: {train[0]['input_ids'].shape}")