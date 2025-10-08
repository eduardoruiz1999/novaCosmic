from web3 import Web3
import asyncio

class SubtensorEVMContract:
    def __init__(self, contract_address, abi_path=None):
        self.w3 = Web3(Web3.HTTPProvider('https://lite.chain.opentensor.ai'))
        self.contract_address = Web3.to_checksum_address(contract_address)
        
        if abi_path and os.path.exists(abi_path):
            with open(abi_path, 'r') as f:
                abi = json.load(f)
        else:
            # ABI básico para funciones comunes
            abi = []  # Aquí iría el ABI del contrato específico
        
        self.contract = self.w3.eth.contract(address=self.contract_address, abi=abi)
    
    async def read_contract_data(self, function_name, *args):
        """Leer datos del contrato (llamada sin gas)"""
        try:
            function = getattr(self.contract.functions, function_name)
            result = function(*args).call()
            return result
        except Exception as e:
            print(f"❌ Error leyendo contrato: {e}")
            return None
    
    async def write_contract(self, function_name, private_key, *args, value=0):
        """Escribir en el contrato (transacción con gas)"""
        try:
            account = self.w3.eth.account.from_key(private_key)
            
            function = getattr(self.contract.functions, function_name)
            transaction = function(*args).build_transaction({
                'from': account.address,
                'chainId': 964,
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(account.address),
                'value': value
            })
            
            signed_txn = self.w3.eth.account.sign_transaction(transaction, private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            return tx_hash.hex()
            
        except Exception as e:
            print(f"❌ Error escribiendo en contrato: {e}")
            return None

# Ejemplo de uso
async def interact_with_contract():
    # Dirección de ejemplo de un contrato en Subtensor EVM
    contract_address = "0x1234567890123456789012345678901234567890"  # Reemplazar con contrato real
    
    contract_handler = SubtensorEVMContract(contract_address)
    
    # Leer datos del contrato
    # data = await contract_handler.read_contract_data("someFunction")
    # print(f"📊 Datos del contrato: {data}")

await interact_with_contract()
