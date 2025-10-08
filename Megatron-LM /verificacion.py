# Verificar instalación con Bittensor
import torch
import bittensor as bt
import megatron
from megatron import get_args, initialize_megatron

print("=== VERIFICACIÓN BITENSOR + MEGATRON ===")
print(f"Bittensor version: {bt.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
