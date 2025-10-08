import bittensor as bt
import asyncio

async def auto_detect_network():
    networks = [
        ("fantom", "wss://wsapi.fantom.network"),
        ("fantom-testnet", "wss://wsapi.testnet.fantom.network"),
        ("nakamoto", "wss://entrypoint-finney.opentensor.ai:443"),
        ("local", "ws://127.0.0.1:9946")
    ]
    
    for network_name, endpoint in networks:
        try:
            print(f"Probando {network_name} en {endpoint}...")
            
            async with bt.AsyncSubtensor(
                network=network_name,
                chain_endpoint=endpoint,
                timeout=10  # Timeout más corto para pruebas
            ) as sub:
                
                block = await sub.get_current_block()
                print(f"✅ {network_name} conectado - Bloque: {block}")
                
                # Aquí puedes ejecutar tu lógica específica
                total_subnets = await sub.get_total_subnets()
                print(f"Subnets en {network_name}: {total_subnets}")
                
        except Exception as e:
            print(f"❌ {network_name} falló: {e}")

await auto_detect_network()
