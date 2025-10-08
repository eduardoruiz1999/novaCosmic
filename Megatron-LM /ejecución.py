# Celda 11: Script de ejecución automática
def setup_megatron_colab():
    """Configuración automática completa para Megatron en Colab"""
    print("🚀 Iniciando configuración de Megatron-LM en Colab...")
    
    # 1. Verificar entorno
    monitor_resources()
    
    # 2. Inicializar Megatron
    wrapper = ColabMegatronWrapper()
    model = wrapper.create_small_model()
    
    # 3. Probar modelo
    if model is not None:
        print("✅ Megatron-LM configurado exitosamente en Colab")
        
        # Mostrar resumen del modelo
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Modelo creado con {total_params:,} parámetros")
        
        # Probar inferencia básica
        try:
            test_input = torch.randint(0, 1000, (1, 10))
            with torch.no_grad():
                output = model(test_input)
            print(f"✅ Inferencia de prueba exitosa - Output shape: {output.shape}")
        except Exception as e:
            print(f"⚠️  Inferencia de prueba falló: {e}")
    
    else:
        print("❌ Error en la configuración de Megatron-LM")
    
    return model

# Ejecutar configuración automática
final_model = setup_megatron_colab()
