import flwr as fl
from flwr.common import Metrics

def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """
    Função para agregar as métricas de avaliação (como acurácia).
    Ela faz uma média ponderada da acurácia, baseada em quantos
    exemplos de teste cada cliente usou.
    """
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Calcula a acurácia agregada
    aggregated_accuracy = sum(accuracies) / sum(examples)
    
    # Retorna o dicionário de métricas para o servidor
    return {"accuracy": aggregated_accuracy}

print("Iniciando o Servidor FL (Versão Corrigida) em 127.0.0.1:8080")
print("Aguardando 10 clientes se conectarem...")

# Define a estratégia (FedAvg)
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,           # Usar 100% dos clientes para treino
    fraction_evaluate=1.0,    # Usar 100% dos clientes para avaliação
    min_available_clients=10,   # Esperar TODOS os 10 clientes conectarem
    min_fit_clients=10,         # FORÇAR o uso de 10 clientes para treino
    min_evaluate_clients=10,    # FORÇAR o uso de 10 clientes para avaliação
    
    # CORREÇÃO: Diz ao servidor como calcular a média da acurácia
    evaluate_metrics_aggregation_fn=weighted_average, 
)

# Inicia o servidor!
history = fl.server.start_server(
    server_address="127.0.0.1:8080", 
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)

print("Servidor finalizado.")
print("\n--- Histórico de Acurácia ---")
print(history.metrics_distributed)

# Mostra a acurácia final
final_round = history.metrics_distributed["accuracy"][-1][0]
final_accuracy = history.metrics_distributed["accuracy"][-1][1]
print(f"\nAcurácia final (Rodada {final_round}): {final_accuracy * 100:.2f}%")
print(f"Baseline (Treino Centralizado) foi: 98.16%")