def run_bittensor_megatron_demo():
    """Demo de integración Bittensor + Megatron"""
    print("🚀 INICIANDO DEMO BITENSOR + MEGATRON")
    
    # 1. Configuración inicial
    config = bt.config({
        "megatron": {
            "model_size": "small",
            "use_bittensor": True,
            "subtensor_network": "test"  # Usar red de prueba
        }
    })
    
    # 2. Inicializar modelo
    model = BittensorMegatronModel(config)
    model.initialize_model()
    
    # 3. Conectar a Bittensor
    model.connect_to_bittensor_network()
    
    # 4. Datos de prueba
    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.data = torch.randint(0, 1000, (size, 512))
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.data[idx]
    
    dataset = TestDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    
    # 5. Entrenamiento de prueba
    print("🧠 Iniciando entrenamiento de prueba...")
    model.train_with_bittensor(dataloader, epochs=1)
    
    # 6. Inferencia de prueba
    print("🔮 Probando inferencia...")
    test_input = torch.randint(0, 1000, (1, 10))
    with torch.no_grad():
        if hasattr(model.model, 'forward'):
            output = model.model(test_input)
            print(f"✅ Inferencia exitosa - Shape: {output.shape}")
        else:
            print("⚠️  No se pudo probar inferencia completa")
    
    print("🎯 Demo completado exitosamente!")
    return model

# Ejecutar demo (solo si la instalación fue exitosa)
if installation_ok:
    try:
        demo_model = run_bittensor_megatron_demo()
    except Exception as e:
        print(f"❌ Error en demo: {e}")
