import bittensor as bt
import asyncio

# Diferentes endpoints de Fantom
FANTOM_ENDPOINTS = [
    "wss://wsapi.fantom.network",
    "https://rpcapi.fantom.network",
    "wss://fantom.publicnode.com",
    "https://fantom-rpc.publicnode.com"
]

async def fetch_fantom_data():
    for endpoint in FANTOM_ENDPOINTS:
        try:
            print(f"Intentando conectar a: {endpoint}")
            
            async with bt.AsyncSubtensor(
                network="fantom",
                chain_endpoint=endpoint
            ) as sub:
                
                # Verificar conexión
                current_block = await sub.get_current_block()
                total_subnets = await sub.get_total_subnets()
                
                print(f"✅ Conexión exitosa a {endpoint}")
                print(f"Bloque actual en Fantom: {current_block}")
                print(f"Total de subnets: {total_subnets}")
                break
                
        except Exception as e:
            print(f"❌ Error con {endpoint}: {e}")
            continue

await fetch_fantom_data()
