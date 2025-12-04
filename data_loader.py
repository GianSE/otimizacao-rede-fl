import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# Vamos importar o tokenizer que definimos no passo 2
from model_definition import load_model_and_tokenizer

# Define um tamanho fixo para os blocos de texto
BLOCK_SIZE = 128

def load_data():
    """
    Carrega o dataset 'blended_skill_talk', o processa e o tokeniza.
    """
    
    # 1. Carrega o Tokenizer (do model_definition.py)
    _, tokenizer = load_model_and_tokenizer()

    # 2. Carrega o dataset 'blended_skill_talk'
    print("Carregando o dataset 'blended_skill_talk'...")
    raw_datasets = load_dataset("blended_skill_talk")
    
    def preprocess_function(examples):
        """
        Função de pré-processamento que será aplicada a cada amostra.
        """
        
        separator = tokenizer.eos_token 
        texts = []

        # CORREÇÃO AQUI:
        # Iteramos pelo batch usando os nomes corretos das colunas
        # 'previous_utterance' (contexto) e 'free_messages' (conversa) são listas.
        for i in range(len(examples["previous_utterance"])):
            context_list = examples["previous_utterance"][i]
            message_list = examples["free_messages"][i] # Corrigido de 'free_text' para 'free_messages'
            
            # Junta o contexto + a conversa em uma única lista
            full_dialog_list = context_list + message_list
            
            # Transforma a lista de falas em um único texto
            # ex: "Fala1<|eos|>" + "Fala2<|eos|>" + "Fala3<|eos|>"
            full_text = separator.join(full_dialog_list) + separator
            texts.append(full_text)

        # 3. Tokeniza os textos
        tokenized_output = tokenizer(
            texts,
            padding="max_length", # Preenche até o BLOCK_SIZE
            truncation=True,      # Trunca se for maior que o BLOCK_SIZE
            max_length=BLOCK_SIZE,
            return_tensors="pt"
        )
        
        # Os 'labels' são os próprios 'input_ids'
        tokenized_output["labels"] = tokenized_output["input_ids"].clone()
        
        return tokenized_output

    # 4. Aplica a função de pré-processamento
    print("Processando e tokenizando o dataset...")
    
    column_names = raw_datasets["train"].column_names 
    
    tokenized_datasets = raw_datasets.map(
        preprocess_function, 
        batched=True, 
        remove_columns=column_names 
    )

    # 5. Define o formato final para o PyTorch
    tokenized_datasets.set_format("torch")

    # 6. Pega os splits de treino e teste
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["validation"] 
    
    print(f"Dataset processado. Amostras de treino: {len(train_dataset)}, Amostras de teste: {len(test_dataset)}")
    
    return train_dataset, test_dataset

if __name__ == '__main__':
    # Teste rápido para verificar se o data_loader funciona
    
    train_data, test_data = load_data()
    
    print("\n--- Testando o Data Loader ---")
    print(f"Tipo de dados do treino: {type(train_data)}")
    
    primeira_amostra = train_data[0]
    
    print(f"\nChaves da primeira amostra: {primeira_amostra.keys()}")
    print(f"Shape dos input_ids: {primeira_amostra['input_ids'].shape}") 
    print(f"Shape dos labels: {primeira_amostra['labels'].shape}")     
    
    _, tokenizer = load_model_and_tokenizer()
    print("\n--- Exemplo de Amostra (Decodificada) ---")
    print(tokenizer.decode(primeira_amostra['input_ids'], skip_special_tokens=False))