import os
from web3 import Web3
from eth_account import Account
import getpass

class EVMWalletManager:
    def __init__(self, rpc_url="https://lite.chain.opentensor.ai"):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.account = None
        
    def create_new_wallet(self):
        """Crear nueva wallet EVM de forma segura"""
        # Generar nueva cuenta
        new_account = Account.create()
        
        print("🆕 NUEVA WALLET EVM CREADA:")
        print(f"📍 Address: {new_account.address}")
        print(f"🔑 Private Key: {new_account.key.hex()}")
        print("\n⚠️  GUARDA ESTA INFORMACIÓN EN UN LUGAR SEGURO!")
        print("⚠️  EL PRIVATE KEY ES SENSIBLE Y NO DEBE COMPARTIRSE!")
        
        return new_account
    
    def load_wallet_from_private_key(self, private_key=None):
        """Cargar wallet desde private key de forma segura"""
        if not private_key:
            # Pedir private key de forma segura
            private_key = getpass.getpass("🔐 Ingresa tu private key (0x...): ")
        
        try:
            # Validar formato
            if private_key.startswith('0x'):
                private_key = private_key[2:]
                
            self.account = Account.from_key(private_key)
            print(f"✅ Wallet cargada: {self.account.address}")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando wallet: {e}")
            return False
    
    def get_balance(self):
        """Obtener balance de la wallet"""
        if not self.account:
            print("❌ Primero carga una wallet")
            return None
        
        try:
            balance_wei = self.w3.eth.get_balance(self.account.address)
            balance_tao = balance_wei / 10**18
            return balance_tao
        except Exception as e:
            print(f"❌ Error obteniendo balance: {e}")
            return None
    
    def send_transaction(self, to_address, amount_tao):
        """Enviar transacción (ejemplo básico)"""
        if not self.account:
            print("❌ Primero carga una wallet")
            return False
        
        try:
            # Convertir a wei
            amount_wei = int(amount_tao * 10**18)
            
            # Construir transacción
            transaction = {
                'to': Web3.to_checksum_address(to_address),
                'value': amount_wei,
                'gas': 21000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'chainId': 964  # Chain ID de Subtensor EVM
            }
            
            # Firmar transacción
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.key)
            
            # Enviar transacción
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            print(f"✅ Transacción enviada: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            print(f"❌ Error en transacción: {e}")
            return False

# Uso del gestor de wallets
async def demo_evm_wallet():
    wallet_mgr = EVMWalletManager()
    
    # Conectar a la red
    if not wallet_mgr.w3.is_connected():
        print("❌ No conectado a Subtensor EVM")
        return
    
    print("✅ Conectado a Subtensor EVM")
    
    # Opción 1: Crear nueva wallet
    # new_acc = wallet_mgr.create_new_wallet()
    
    # Opción 2: Cargar wallet existente (ejemplo)
    # wallet_mgr.load_wallet_from_private_key("0x_tu_private_key_aqui")
    
    # Ver balance
    # balance = wallet_mgr.get_balance()
    # if balance is not None:
    #     print(f"💰 Balance: {balance} TAO")

# Ejecutar demo
await demo_evm_wallet()
