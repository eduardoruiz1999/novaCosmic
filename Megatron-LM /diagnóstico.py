def troubleshoot_bittensor_issues():
    """Diagnóstico de problemas comunes"""
    print("=== DIAGNÓSTICO BITENSOR ===")
    
    issues_found = []
    
    # 1. Versiones compatibles
    try:
        import bittensor
        import megatron
        print("✅ Versiones compatibles")
    except ImportError as e:
        issues_found.append(f"Import error: {e}")
    
    # 2. Configuración de red
    try:
        subtensor = bt.subtensor(network='test')
        print("✅ Conexión a red de prueba: OK")
    except Exception as e:
        issues_found.append(f"Conexión a red: {e}")
    
    # 3. Wallet y autenticación
    try:
        wallet = bt.wallet()
        print("✅ Wallet configuration: OK")
    except Exception as e:
        issues_found.append(f"Wallet: {e}")
    
    # 4. Recursos GPU
    if not torch.cuda.is_available():
        issues_found.append("GPU no disponible - requerida para Megatron")
    else:
        print("✅ GPU resources: OK")
    
    # Reportar problemas
    if issues_found:
        print("❌ Problemas encontrados:")
        for issue in issues_found:
            print(f"   - {issue}")
    else:
        print("🎉 No se encontraron problemas")
    
    return len(issues_found) == 0

# Ejecutar diagnóstico
is_healthy = troubleshoot_bittensor_issues()
