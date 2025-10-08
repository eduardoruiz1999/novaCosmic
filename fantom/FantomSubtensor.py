import bittensor as bt
import asyncio

class FantomSubtensor:
    def __init__(self):
        self.network = "fantom"
        self.endpoints = [
            "wss://wsapi.fantom.network",
            "https://rpcapi.fantom.network"
        ]
    
    async def connect(self):
        for endpoint in self.endpoints:
            try:
                self.subtensor = await bt.AsyncSubtensor(
                    network=self.network,
                    chain_endpoint=endpoint
                ).__aenter__()
                print(f"✅ Conectado a Fantom via {endpoint}")
                return True
            except Exception as e:
                print(f"❌ Error con {endpoint}: {e}")
        return False
    
    async def get_network_info(self):
        if not hasattr(self, 'subtensor'):
            await self.connect()
        
        tasks = [
            self.subtensor.get_total_subnets(),
            self.subtensor.get_current_block(),
            self.subtensor.get_delegate_identities()
        ]
        
        return await asyncio.gather(*tasks)
    
    async def close(self):
        if hasattr(self, 'subtensor'):
            await self.subtensor.__aexit__(None, None, None)

# Uso
async def main():
    fantom = FantomSubtensor()
    try:
        if await fantom.connect():
            total_subnets, current_block, delegates = await fantom.get_network_info()
            
            print(f"Fantom - Total subnets: {total_subnets}")
            print(f"Fantom - Current block: {current_block}")
            print("Delegates en Fantom:")
            for hotkey, identity in delegates.items():
                print(f"  {hotkey}: {identity.name}")
    finally:
        await fantom.close()

await main()
