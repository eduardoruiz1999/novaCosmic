import bittensor as bt
import asyncio

# Configurar para usar la red Fantom
config = bt.subtensor.config()
config.subtensor.network = "fantom"  # o "fantom-testnet" para testnet
config.subtensor.chain_endpoint = "wss://wsapi.fantom.network"  # Endpoint de Fantom

async def fetch_subtensor_data():
    # Inicializar Subtensor con configuración personalizada
    async with bt.AsyncSubtensor(
        network="fantom",
        chain_endpoint="wss://wsapi.fantom.network"  # Endpoint WebSocket de Fantom
    ) as sub:
        
        # Fetch multiple pieces of information concurrently
        total_subnets, current_block, delegate_identities = await asyncio.gather(
            sub.get_total_subnets(),
            sub.get_current_block(),
            sub.get_delegate_identities()
        )

        print(f"Total number of subnets: {total_subnets}")
        print(f"Current block: {current_block}")
        print("Delegate identities:")
        for hotkey, identity in delegate_identities.items():
            print(f"  {hotkey}: {identity.name} ({identity.url})")

await fetch_subtensor_data()
