# Celda 4: Configuración para entornos limitados (como Colab)
import os
import json

# Crear configuración mínima para Colab
colab_config = {
    "world_size": 1,
    "rank": 0,
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 1,
    "num_layers": 4,  # Reducido para Colab
    "hidden_size": 256,  # Reducido para Colab
    "num_attention_heads": 4,
    "seq_length": 512,
    "max_position_embeddings": 512,
    "use_cpu_initialization": False,
    "micro_batch_size": 1,  # Muy pequeño para Colab
    "global_batch_size": 1,
    "lr": 0.0001,
    "train_iters": 100,
    "save": "./checkpoints",
    "load": None,
    "exit_interval": 50
}

# Guardar configuración
with open('colab_config.json', 'w') as f:
    json.dump(colab_config, f, indent=2)

print("Configuración para Colab creada")
