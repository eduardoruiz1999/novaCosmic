import bittensor as bt
import asyncio
from web3 import Web3
import json

async def connect_evm_with_wallet():
    # Configurar conexión a Subtensor EVM
    w3 = Web3(Web3.HTTPProvider('https://lite.chain.opentensor.ai'))
    
    if not w3.is_connected():
        print("❌ No se pudo conectar a Subtensor EVM")
        return None
    
    print("✅ Conectado a Subtensor EVM")
    print(f"🔗 Chain ID: {w3.eth.chain_id}")
    print(f"📦 Último bloque: {w3.eth.block_number}")
    
    return w3

async def check_evm_balance(w3, evm_address):
    """Verificar balance de una dirección EVM en Subtensor"""
    try:
        # Verificar si la dirección es válida
        if not Web3.is_address(evm_address):
            print("❌ Dirección EVM inválida")
            return None
        
        # Convertir a checksum address
        checksum_address = Web3.to_checksum_address(evm_address)
        
        # Obtener balance en wei
        balance_wei = w3.eth.get_balance(checksum_address)
        
        # Convertir a TAO (1 TAO = 10^9 tao, 1 tao = 10^9 wei)
        balance_tao = balance_wei / 10**18
        
        print(f"💰 Balance de {checksum_address}:")
        print(f"   - WEI: {balance_wei}")
        print(f"   - TAO: {balance_tao}")
        
        return balance_tao
        
    except Exception as e:
        print(f"❌ Error al verificar balance: {e}")
        return None

# Configuración para MetaMask/EVMs
def setup_evm_wallet(private_key=None):
    """Configurar wallet EVM desde private key"""
    w3 = Web3(Web3.HTTPProvider('https://lite.chain.opentensor.ai'))
    
    if private_key:
        try:
            # Crear cuenta desde private key
            account = w3.eth.account.from_key(private_key)
            print(f"✅ Wallet EVM configurada: {account.address}")
            return w3, account
        except Exception as e:
            print(f"❌ Error con private key: {e}")
    
    return w3, None

async def main():
    # 1. Conectar a Subtensor EVM
    w3 = await connect_evm_with_wallet()
    if not w3:
        return
    
    # 2. Ejemplo: Verificar balance de una dirección específica
    example_address = "0x742d35Cc6634C0532925a3b8Dc9F1a37cD7e8b5C"  # Reemplaza con tu address
    await check_evm_balance(w3, example_address)
    
    # 3. Configurar con private key (OPCIONAL - solo si tienes)
    # PRECAUCIÓN: Nunca expongas tu private key en código production
    # private_key = "0x..."  # Tu private key aquí
    # w3, account = setup_evm_wallet(private_key)
    
    # 4. También puedes usar Bittensor nativo para comparar
    print("\n--- Comparación con Bittensor Nativo ---")
    await check_bittensor_native()

async def check_bittensor_native():
    """Verificar usando Bittensor nativo (SS58)"""
    try:
        async with bt.AsyncSubtensor() as sub:
            # Usar una coldkey pública de ejemplo
            example_coldkey = "5DkQ4t6sYudbq2uBxqpsPdF5kZ8vqjvNrjWm5cRZ7QzXqQH"
            
            balance = await sub.get_balance(example_coldkey)
            print(f"💰 Balance nativo Bittensor: {balance} TAO")
            
    except Exception as e:
        print(f"❌ Error con Bittensor nativo: {e}")

# Ejecutar
await main()
