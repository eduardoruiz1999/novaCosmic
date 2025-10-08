# Celda 5: Implementación de modelo pequeño adaptado para Colab
import torch
import torch.nn as nn
from megatron.model import GPTModel
from megatron.mpu import initialize_model_parallel
from megatron.initialize import initialize_megatron

class ColabMegatronWrapper:
    def __init__(self):
        self.model = None
        self.initialized = False
    
    def initialize_for_colab(self):
        """Inicializar Megatron para entorno Colab"""
        try:
            # Configuración mínima
            sys.argv = [
                'dummy_script.py',
                '--num-layers', '4',
                '--hidden-size', '256',
                '--num-attention-heads', '4',
                '--seq-length', '128',
                '--max-position-embeddings', '128',
                '--micro-batch-size', '1',
                '--global-batch-size', '1',
                '--tensor-model-parallel-size', '1',
                '--pipeline-model-parallel-size', '1'
            ]
            
            initialize_megatron()
            self.initialized = True
            print("Megatron inicializado exitosamente en Colab")
            
        except Exception as e:
            print(f"Error en inicialización completa: {e}")
            self._initialize_minimal()
    
    def _initialize_minimal(self):
        """Inicialización mínima alternativa"""
        try:
            # Inicialización manual de componentes esenciales
            from megatron import mpu
            mpu.initialize_model_parallel(1, 1)
            
            self.initialized = True
            print("Inicialización mínima completada")
        except Exception as e:
            print(f"Error en inicialización mínima: {e}")
    
    def create_small_model(self, vocab_size=10000):
        """Crear modelo pequeño para Colab"""
        if not self.initialized:
            self.initialize_for_colab()
        
        try:
            from megatron.model import GPTModel
            from megatron import get_args
            
            # Configurar args manualmente
            args = get_args()
            args.hidden_size = 256
            args.ffn_hidden_size = 1024
            args.num_layers = 4
            args.num_attention_heads = 4
            args.seq_length = 128
            args.max_position_embeddings = 128
            args.vocab_size = vocab_size
            
            # Crear modelo
            self.model = GPTModel(
                num_tokentypes=0,
                parallel_output=True
            )
            
            print(f"Modelo pequeño creado: {sum(p.numel() for p in self.model.parameters()):,} parámetros")
            return self.model
            
        except Exception as e:
            print(f"Error creando modelo: {e}")
            return self._create_fallback_model(vocab_size)
    
    def _create_fallback_model(self, vocab_size):
        """Modelo de respaldo usando PyTorch nativo"""
        print("Creando modelo de respaldo con PyTorch...")
        
        class MiniGPT(nn.Module):
            def __init__(self, vocab_size, hidden_size=256, num_layers=4, num_heads=4):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=num_heads,
                        dim_feedforward=hidden_size * 4,
                        batch_first=True
                    ) for _ in range(num_layers)
                ])
                self.output = nn.Linear(hidden_size, vocab_size)
                
            def forward(self, x):
                x = self.embedding(x)
                for layer in self.layers:
                    x = layer(x)
                return self.output(x)
        
        self.model = MiniGPT(vocab_size)
        print(f"Modelo de respaldo creado: {sum(p.numel() for p in self.model.parameters()):,} parámetros")
        return self.model

# Inicializar wrapper
megatron_wrapper = ColabMegatronWrapper()
model = megatron_wrapper.create_small_model()
