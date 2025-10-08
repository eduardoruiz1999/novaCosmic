# Configuración específica para Bittensor
def setup_bittensor_megatron():
    """Configurar Megatron-LM para trabajar con Bittensor"""
    
    # Configuración de Bittensor
    config = bt.config(
        config = {
            "megatron": {
                "model_size": "small",  # 'small', 'medium', 'large'
                "use_bittensor": True,
                "subtensor_network": "finney",  # o 'test'
                "auto_update": True
            }
        }
    )
    
    # Inicializar Bittensor
    subtensor = bt.subtensor(network=config.megatron.subtensor_network)
    wallet = bt.wallet()
    metagraph = subtensor.metagraph(config.netuid)
    
    print("✅ Bittensor inicializado correctamente")
    print(f"   Red: {config.megatron.subtensor_network}")
    print(f"   Metagraph: {metagraph.n} nodos")
    
    return config, subtensor, wallet, metagraph

# Ejecutar configuración
try:
    bt_config, subtensor, wallet, metagraph = setup_bittensor_megatron()
except Exception as e:
    print(f"❌ Error en configuración Bittensor: {e}")
