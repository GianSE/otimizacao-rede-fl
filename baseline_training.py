# baseline_training_tf.py

import tensorflow as tf
import math

# Importando as funções refatoradas para TensorFlow
from data_loader import load_data
from model_definition import load_model_and_tokenizer

# --- Helper (copiado do fl_client.py) ---
# Necessário para converter o dataset HF para o formato do Keras

def create_tf_dataset(dataset, batch_size, shuffle=False):
    """
    Converte um dataset do Hugging Face (com formato "tf") 
    em um tf.data.Dataset pronto para o Keras (model.fit).
    """
    
    def format_batch_for_keras(batch):
        # Os modelos TF da Hugging Face esperam as features
        # como um dicionário no primeiro argumento (x)
        features = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"]
        }
        # E os rótulos como o segundo argumento (y)
        labels = batch["labels"]
        return features, labels
    
    if shuffle:
        dataset = dataset.shuffle(seed=42)
        
    tf_ds = dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        batch_size=batch_size,
        collate_fn=None 
    )
    
    return tf_ds.map(format_batch_for_keras)

# --- Fim do Helper ---

if __name__ == '__main__':
    
    # --- Configurações ---
    NUM_EPOCHS = 3 # O treino centralizado pode rodar por mais épocas
    BATCH_SIZE = 8 # Maior que no FL, já que não temos 10 modelos ao mesmo tempo
    LEARNING_RATE = 5e-5 # Padrão para Transformers
    
    print("--- Iniciando Treinamento Centralizado (Baseline) com TensorFlow ---")

    # --- 1. Carregar Dados ---
    print("Carregando e processando dados...")
    train_dataset, test_dataset = load_data()
    
    # Converte os datasets para tf.data.Dataset (formato Keras)
    tf_train_data = create_tf_dataset(train_dataset, BATCH_SIZE, shuffle=True)
    tf_test_data = create_tf_dataset(test_dataset, BATCH_SIZE, shuffle=False)

    print(f"Dados prontos. Amostras de treino: {len(train_dataset)}, Amostras de teste: {len(test_dataset)}")

    # --- 2. Carregar Modelo ---
    print("Carregando modelo GenAI (TensorFlow)...")
    model, _ = load_model_and_tokenizer()
    
    # --- 3. Compilar o Modelo (Keras) ---
    print("Compilando o modelo Keras...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # Modelos da HF no Keras calculam a perda internamente
    # quando 'labels' são fornecidos.
    model.compile(optimizer=optimizer)
    
    # --- 4. Loop de Treinamento (Keras .fit) ---
    print(f"\n--- Iniciando Treinamento por {NUM_EPOCHS} épocas ---")
    
    model.fit(
        tf_train_data,
        epochs=NUM_EPOCHS,
        validation_data=tf_test_data,
        verbose=1
    )
        
    print("\n--- Treinamento Baseline Concluído ---")

    # --- 5. Avaliação Final ---
    print("Avaliando o modelo final no dataset de teste...")
    
    # .evaluate() retorna a perda (loss)
    final_loss = model.evaluate(tf_test_data)
    
    try:
        final_perplexity = math.exp(final_loss)
    except OverflowError:
        final_perplexity = float("inf")

    print("\n--- Resultado Final (Baseline) ---")
    print(f"  Perda (Loss) Média: {final_loss:.4f}")
    print(f"  Perplexidade: {final_perplexity:.4f} (Menor é melhor)")