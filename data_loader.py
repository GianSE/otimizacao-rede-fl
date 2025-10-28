import tensorflow as tf
from datasets import load_dataset

# Vamos importar o tokenizer que definimos (agora carrega o modelo TF)
from model_definition import load_model_and_tokenizer

# Define um tamanho fixo para os blocos de texto
BLOCK_SIZE = 128

def load_data():
    """
    Carrega o dataset 'blended_skill_talk', o processa e o tokeniza
    para o formato TensorFlow.
    """
    
    # 1. Carrega o Tokenizer
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

        for i in range(len(examples["previous_utterance"])):
            context_list = examples["previous_utterance"][i]
            message_list = examples["free_messages"][i]
            
            full_dialog_list = context_list + message_list
            
            full_text = separator.join(full_dialog_list) + separator
            texts.append(full_text)

        # 3. Tokeniza os textos (retornando Tensores de TensorFlow)
        tokenized_output = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=BLOCK_SIZE,
            return_tensors="tf" # <-- MUDANÇA PRINCIPAL AQUI
        )
        
        # Os 'labels' são os próprios 'input_ids'
        # Usamos tf.identity em vez de .clone()
        tokenized_output["labels"] = tf.identity(tokenized_output["input_ids"])
        
        return tokenized_output

    # 4. Aplica a função de pré-processamento
    print("Processando e tokenizando o dataset...")
    
    column_names = raw_datasets["train"].column_names 
    
    tokenized_datasets = raw_datasets.map(
        preprocess_function, 
        batched=True, 
        remove_columns=column_names 
    )

    # 5. Define o formato final para o TensorFlow
    tokenized_datasets.set_format("tf") # <-- MUDANÇA PRINCIPAL AQUI

    # 6. Pega os splits de treino e teste
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["validation"] 
    
    print(f"Dataset processado. Amostras de treino: {len(train_dataset)}, Amostras de teste: {len(test_dataset)}")
    
    return train_dataset, test_dataset

if __name__ == '__main__':
    # Teste rápido
    train_data, test_data = load_data()
    
    print("\n--- Testando o Data Loader (TF) ---")
    print(f"Tipo de dados do treino: {type(train_data)}")
    
    primeira_amostra = train_data[0]
    
    print(f"\nChaves da primeira amostra: {primeira_amostra.keys()}")
    print(f"Shape dos input_ids: {primeira_amostra['input_ids'].shape}") 
    print(f"Shape dos labels: {primeira_amostra['labels'].shape}")     
    print(f"Tipo do tensor: {type(primeira_amostra['input_ids'])}")