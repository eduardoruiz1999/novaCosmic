import bittensor as bt
import asyncio
from web3 import Web3

async def fetch_bittensor_data():
    async with bt.AsyncSubtensor() as sub:
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

async def fetch_fantom_data():
    # Conectarse a un nodo de Fantom (usando un RPC público o tu propio nodo)
    w3 = Web3(Web3.HTTPProvider('https://rpc.ftm.tools'))
    if w3.is_connected():
        block_number = w3.eth.block_number
        print(f"Fantom current block: {block_number}")
    else:
        print("Failed to connect to Fantom")

async def main():
    await asyncio.gather(
        fetch_bittensor_data(),
        fetch_fantom_data()
    )

await main()
