import flwr as fl
from flwr.common import Metrics

def weighted_average_perplexity(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """
    Função de agregação para a Perplexidade (Perplexity).
    Calcula a média ponderada da perplexidade enviada pelos clientes.
    """
    
    # Extrai a perplexidade (ppl) e o número de exemplos de cada cliente
    perplexities = [num_examples * m["perplexity"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Calcula a perplexidade agregada (média ponderada)
    aggregated_perplexity = sum(perplexities) / sum(examples)
    
    # Retorna o dicionário de métricas para o servidor
    print(f"Perplexidade média agregada nesta rodada: {aggregated_perplexity:.4f}")
    return {"perplexity": aggregated_perplexity}

print("Iniciando o Servidor FL (Versão GenAI) em 127.0.0.1:8080")
print("Aguardando 2 clientes se conectarem...")

# Define a estratégia (FedAvg)
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,           # Usar 100% dos clientes para treino
    fraction_evaluate=1.0,    # Usar 100% dos clientes para avaliação
    min_available_clients=2,   # Esperar TODOS os 10 clientes conectarem
    min_fit_clients=2,         # FORÇAR o uso de 10 clientes para treino
    min_evaluate_clients=2,    # FORÇAR o uso de 10 clientes para avaliação
    
    # DIZ AO SERVIDOR PARA USAR NOSSA NOVA FUNÇÃO DE MÉTRICA:
    evaluate_metrics_aggregation_fn=weighted_average_perplexity, 
)

# Inicia o servidor!
history = fl.server.start_server(
    server_address="127.0.0.1:8080", 
    config=fl.server.ServerConfig(num_rounds=3), # 3 rodadas (GenAI é mais lento)
    strategy=strategy,
    grpc_max_message_length=1024*1024*1024
)

print("Servidor finalizado.")
print("\n--- Histórico de Perplexidade (Menor é Melhor) ---")

# Extrai o histórico de métricas
ppl_history = history.metrics_distributed.get("perplexity")

if ppl_history:
    print(ppl_history)
    # Mostra a perplexidade final
    final_round, final_ppl = ppl_history[-1]
    print(f"\nPerplexidade final (Rodada {final_round}): {final_ppl:.4f}")
else:
    print("Nenhuma métrica de perplexidade foi registrada.")