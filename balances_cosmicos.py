import asyncio
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import bittensor as bt
from bittensor.core.subtensor import Subtensor
from bittensor.core.async_subtensor import AsyncSubtensor
from bittensor.utils.balance import Balance
import subprocess
import json

class CosmicWalletTier(Enum):
    NOVICE = "novice"           # 0-100 TAO
    EXPLORER = "explorer"       # 100-1,000 TAO  
    GUARDIAN = "guardian"       # 1,000-10,000 TAO
    CELESTIAL = "celestial"     # 10,000-100,000 TAO
    COSMIC = "cosmic"           # 100,000+ TAO

@dataclass
class CosmicWallet:
    """Wallet cósmica con capacidades extendidas para el sistema solar"""
    name: str
    coldkey: str
    hotkey: str
    balance: Balance
    tier: CosmicWalletTier
    creation_date: float
    last_activity: float
    cosmic_importance: float = 0.0

@dataclass 
class ImmunityConfig:
    """Configuración de periodos de inmunidad cósmica"""
    commit_reveal_period: int  # En bloques
    tempo: int                 # Factor de tiempo cósmico
    old_immunity_period: int   # Periodo anterior de inmunidad

class CosmicBalanceManager:
    """
    Gestor avanzado de balances para el ecosistema cósmico.
    Combina operaciones síncronas y asíncronas para máximo rendimiento.
    """
    
    def __init__(self, network: str = "finney"):
        self.network = network
        self.sync_subtensor = Subtensor(network)
        self.wallet_registry: Dict[str, CosmicWallet] = {}
        
    async def initialize_async(self):
        """Inicialización asíncrona del gestor cósmico"""
        self.async_subtensor = AsyncSubtensor(self.network)
        return self

    async def __aenter__(self):
        await self.initialize_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'async_subtensor'):
            await self.async_subtensor.close()

    def create_cosmic_wallet(self, wallet_name: str, hotkey_name: str) -> Tuple[bool, str]:
        """
        Crea un nuevo wallet cósmico usando btcli.
        
        Args:
            wallet_name: Nombre del coldkey cósmico
            hotkey_name: Nombre del hotkey cósmico
            
        Returns:
            Tuple[bool, str]: (éxito, mensaje)
        """
        try:
            # Ejecutar comando btcli para crear wallet
            command = [
                "btcli", "wallet", "create",
                "--wallet.name", wallet_name,
                "--wallet.hotkey", hotkey_name,
                "--no_prompt"  # Ejecución no interactiva
            ]
            
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            
            if result.returncode == 0:
                bt.logging.info(f"✅ Wallet cósmico creado: {wallet_name}/{hotkey_name}")
                
                # Obtener direcciones del wallet creado
                coldkey_addr = self._extract_coldkey_address(wallet_name)
                hotkey_addr = self._extract_hotkey_address(wallet_name, hotkey_name)
                
                # Registrar en el sistema cósmico
                cosmic_wallet = CosmicWallet(
                    name=wallet_name,
                    coldkey=coldkey_addr,
                    hotkey=hotkey_addr,
                    balance=Balance(0),
                    tier=CosmicWalletTier.NOVICE,
                    creation_date=time.time(),
                    last_activity=time.time()
                )
                
                self.wallet_registry[wallet_name] = cosmic_wallet
                return True, f"Wallet cósmico {wallet_name} creado exitosamente"
            else:
                error_msg = f"Error creando wallet: {result.stderr}"
                bt.logging.error(f"❌ {error_msg}")
                return False, error_msg
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Error en comando btcli: {e.stderr}"
            bt.logging.error(f"❌ {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Excepción creando wallet: {e}"
            bt.logging.error(f"❌ {error_msg}")
            return False, error_msg

    def _extract_coldkey_address(self, wallet_name: str) -> str:
        """Extrae la dirección del coldkey del wallet"""
        try:
            # En implementación real, leeríamos del archivo del wallet
            # Por ahora simulamos una dirección
            return f"5SimulatedColdkey{wallet_name[:10]}"
        except Exception as e:
            bt.logging.warning(f"Error extrayendo coldkey: {e}")
            return "5UnknownColdkeyAddress"

    def _extract_hotkey_address(self, wallet_name: str, hotkey_name: str) -> str:
        """Extrae la dirección del hotkey del wallet"""
        try:
            # En implementación real, leeríamos del archivo del wallet
            return f"5SimulatedHotkey{hotkey_name[:10]}"
        except Exception as e:
            bt.logging.warning(f"Error extrayendo hotkey: {e}")
            return "5UnknownHotkeyAddress"

    async def get_comprehensive_balances_async(self, addresses: List[str]) -> Dict[str, Dict]:
        """
        Obtiene información comprehensiva de balances de forma asíncrona.
        
        Args:
            addresses: Lista de direcciones cósmicas a consultar
            
        Returns:
            Dict con información completa de cada dirección
        """
        async with AsyncSubtensor(self.network) as async_sub:
            # Ejecutar múltiples consultas concurrentemente
            balance_task = async_sub.get_balance(*addresses)
            delegated_task = async_sub.get_delegated(*addresses)
            staked_task = asyncio.gather(*[
                async_sub.get_total_stake_for_coldkey(addr) for addr in addresses
            ])
            
            balances, delegated, staked_list = await asyncio.gather(
                balance_task, delegated_task, staked_task
            )
            
            # Procesar resultados
            comprehensive_data = {}
            for i, address in enumerate(addresses):
                balance = balances.get(address, Balance(0))
                delegated_info = delegated.get(address, {})
                total_staked = staked_list[i] if i < len(staked_list) else Balance(0)
                
                # Calcular valor total
                total_value = balance + total_staked
                
                # Determinar tier cósmico
                tier = self._calculate_cosmic_tier(total_value.tao)
                
                comprehensive_data[address] = {
                    "address": address,
                    "balance": balance,
                    "total_staked": total_staked,
                    "delegated": delegated_info,
                    "total_value": total_value,
                    "cosmic_tier": tier,
                    "cosmic_importance": self._calculate_cosmic_importance(total_value.tao),
                    "timestamp": time.time()
                }
            
            return comprehensive_data

    def _calculate_cosmic_tier(self, tao_balance: float) -> CosmicWalletTier:
        """Calcula el tier cósmico basado en el balance"""
        if tao_balance >= 100000:
            return CosmicWalletTier.COSMIC
        elif tao_balance >= 10000:
            return CosmicWalletTier.CELESTIAL
        elif tao_balance >= 1000:
            return CosmicWalletTier.GUARDIAN
        elif tao_balance >= 100:
            return CosmicWalletTier.EXPLORER
        else:
            return CosmicWalletTier.NOVICE

    def _calculate_cosmic_importance(self, tao_balance: float) -> float:
        """Calcula la importancia cósmica basada en el balance"""
        # Función logarítmica para importancia (suaviza diferencias grandes)
        if tao_balance <= 0:
            return 0.0
        
        importance = min(1.0, (tao_balance ** 0.3) / 10)
        return importance

    async def get_parallel_balances(self, addresses: List[str]) -> Dict[str, Balance]:
        """
        Obtiene balances de múltiples direcciones en paralelo.
        Versión optimizada del ejemplo original.
        """
        async with AsyncSubtensor(self.network) as async_sub:
            return await async_sub.get_balance(*addresses)

    def get_parallel_balances_sync(self, addresses: List[str]) -> Dict[str, Balance]:
        """
        Obtiene balances de múltiples direcciones de forma síncrona.
        """
        balances = {}
        for address in addresses:
            try:
                balance = self.sync_subtensor.get_balance(address)
                balances[address] = balance
            except Exception as e:
                bt.logging.error(f"Error obteniendo balance síncrono para {address}: {e}")
                balances[address] = Balance(0)
        
        return balances

    async def massive_concurrent_query(self, coldkey_addresses: List[str]) -> Dict:
        """
        Consulta masiva concurrente inspirada en btcli wallets.
        Ejemplo avanzado de operaciones paralelas.
        """
        async with AsyncSubtensor(self.netney) as async_sub:
            # Obtener hash de bloque para consistencia
            block_hash = await async_sub.get_block_hash()
            
            # Obtener todas las subredes existentes
            total_subnets = await async_sub.get_total_subnets(block_hash=block_hash)
            all_netuids = list(range(total_subnets + 1))
            
            # Consulta masiva concurrente
            balances, all_neurons, all_delegates = await asyncio.gather(
                async_sub.get_balance(*coldkey_addresses, block_hash=block_hash),
                asyncio.gather(*[
                    async_sub.neurons_lite(netuid=netuid, block_hash=block_hash)
                    for netuid in all_netuids
                ]),
                asyncio.gather(*[
                    async_sub.get_delegated(coldkey)
                    for coldkey in coldkey_addresses
                ]),
            )
            
            # Procesar resultados
            total_neurons = sum(len(neurons) for neurons in all_neurons if neurons)
            total_delegated = sum(
                sum(delegated.values()) if delegated else Balance(0)
                for delegated in all_delegates
            )
            
            return {
                "balances": balances,
                "total_neurons": total_neurons,
                "total_subnets": total_subnets,
                "total_delegated": total_delegated,
                "coldkey_count": len(coldkey_addresses),
                "block_hash": block_hash,
                "timestamp": time.time()
            }

class CosmicImmunityCalculator:
    """
    Calculadora de periodos de inmunidad para el sistema cósmico.
    Basado en la fórmula: new_immunity_period = (new_commit_reveal_period x tempo - old_commit_reveal_period x tempo) + old_immunity_period
    """
    
    def __init__(self, config: ImmunityConfig):
        self.config = config
        
    def calculate_new_immunity_period(self, new_commit_reveal_period: int) -> int:
        """
        Calcula el nuevo periodo de inmunidad usando la fórmula cósmica.
        
        Args:
            new_commit_reveal_period: Nuevo periodo commit-reveal en bloques
            
        Returns:
            int: Nuevo periodo de inmunidad en bloques
        """
        try:
            # Aplicar fórmula: new_immunity = (new_commit * tempo - old_commit * tempo) + old_immunity
            new_immunity = (
                (new_commit_reveal_period * self.config.tempo) -
                (self.config.commit_reveal_period * self.config.tempo) +
                self.config.old_immunity_period
            )
            
            # Asegurar que no sea negativo
            new_immunity = max(0, new_immunity)
            
            bt.logging.info(
                f"🛡️ Cálculo de Inmunidad Cósmica:\n"
                f"   Nuevo Commit-Reveal: {new_commit_reveal_period} bloques\n"
                f"   Viejo Commit-Reveal: {self.config.commit_reveal_period} bloques\n"
                f"   Tempo: {self.config.tempo}\n"
                f"   Vieja Inmunidad: {self.config.old_immunity_period} bloques\n"
                f"   Nueva Inmunidad: {new_immunity} bloques"
            )
            
            return new_immunity
            
        except Exception as e:
            bt.logging.error(f"❌ Error calculando inmunidad cósmica: {e}")
            return self.config.old_immunity_period  # Fallback al periodo anterior

    def update_config(self, new_commit_reveal_period: int, new_tempo: Optional[int] = None):
        """
        Actualiza la configuración y calcula nueva inmunidad.
        
        Args:
            new_commit_reveal_period: Nuevo periodo commit-reveal
            new_tempo: Nuevo tempo (opcional)
        """
        # Calcular nueva inmunidad
        new_immunity = self.calculate_new_immunity_period(new_commit_reveal_period)
        
        # Actualizar configuración
        self.config.old_immunity_period = new_immunity
        self.config.commit_reveal_period = new_commit_reveal_period
        if new_tempo is not None:
            self.config.tempo = new_tempo
            
        bt.logging.info(f"⚙️ Configuración cósmica actualizada - Nueva inmunidad: {new_immunity} bloques")

class CosmicEconomyDashboard:
    """
    Dashboard en tiempo real para la economía cósmica del sistema solar.
    """
    
    def __init__(self, balance_manager: CosmicBalanceManager):
        self.balance_manager = balance_manager
        self.dashboard_data = {}
        self.update_interval = 60  # segundos
        
    async def start_dashboard(self):
        """Inicia el dashboard de economía cósmica"""
        bt.logging.info("📊 Iniciando Dashboard de Economía Cósmica...")
        
        while True:
            try:
                await self._update_dashboard()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                bt.logging.error(f"❌ Error en dashboard cósmico: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _update_dashboard(self):
        """Actualiza los datos del dashboard"""
        # Direcciones cósmicas de ejemplo para monitoreo
        cosmic_addresses = [
            "5EhCvSxpFRgXRCaN5LH2wRCD5su1vKsnVfYfjzkqfmPoCy2G",
            "5CZrQzo3W6LGEopMw2zVMugPcwFBmQDYne3TJc9XzZbTX2WR",
            "5Dcmx3kNTKqExHoineVpBJ6HnD9JHApRs8y2GFBgPLCaYn8d",
        ]
        
        # Obtener datos comprehensivos
        comprehensive_data = await self.balance_manager.get_comprehensive_balances_async(cosmic_addresses)
        
        # Calcular métricas agregadas
        total_value = sum(data["total_value"].tao for data in comprehensive_data.values())
        avg_importance = sum(data["cosmic_importance"] for data in comprehensive_data.values()) / len(comprehensive_data)
        
        # Actualizar dashboard
        self.dashboard_data = {
            "timestamp": time.time(),
            "total_monitored_value": total_value,
            "average_cosmic_importance": avg_importance,
            "wallet_tiers": {
                tier.value: sum(1 for data in comprehensive_data.values() if data["cosmic_tier"] == tier)
                for tier in CosmicWalletTier
            },
            "detailed_balances": comprehensive_data,
            "network_health": self._assess_network_health(total_value, avg_importance)
        }
        
        bt.logging.info(
            f"🌌 Dashboard Actualizado - "
            f"Valor Total: {total_value:.2f} TAO, "
            f"Importancia Promedio: {avg_importance:.3f}, "
            f"Salud: {self.dashboard_data['network_health']}"
        )
    
    def _assess_network_health(self, total_value: float, avg_importance: float) -> str:
        """Evalúa la salud de la red cósmica"""
        if total_value > 1000 and avg_importance > 0.5:
            return "💚 Excelente"
        elif total_value > 100 and avg_importance > 0.3:
            return "💛 Buena"
        else:
            return "🧡 En Desarrollo"

# =============================================================================
# EJEMPLOS DE USO Y DEMOSTRACIONES
# =============================================================================

async def demo_advanced_balance_management():
    """Demostración avanzada de gestión de balances cósmicos"""
    bt.logging.info("💰 INICIANDO DEMOSTRACIÓN AVANZADA DE BALANCES CÓSMICOS")
    
    async with CosmicBalanceManager("finney") as balance_manager:
        # Direcciones cósmicas de ejemplo
        COLDKEY_PUBS = [
            "5EhCvSxpFRgXRCaN5LH2wRCD5su1vKsnVfYfjzkqfmPoCy2G",
            "5CZrQzo3W6LGEopMw2zVMugPcwFBmQDYne3TJc9XzZbTX2WR",
            "5Dcmx3kNTKqExHoineVpBJ6HnD9JHApRs8y2GFBgPLCaYn8d",
            "5DZaBZKKGZBGaevi42bYUK44tEuS3SYJ7GU3rQKYr7kjfa8v"
        ]
        
        # 1. Consulta básica de balances (como en el ejemplo original)
        bt.logging.info("1. Consulta Básica de Balances...")
        async_balances = await balance_manager.get_parallel_balances(COLDKEY_PUBS)
        
        for address, balance in async_balances.items():
            bt.logging.info(f"   📍 {address[:12]}...: {balance}")
        
        # 2. Consulta comprehensiva con tiers cósmicos
        bt.logging.info("2. Consulta Comprehensiva Cósmica...")
        comprehensive = await balance_manager.get_comprehensive_balances_async(COLDKEY_PUBS[:2])
        
        for address, data in comprehensive.items():
            bt.logging.info(
                f"   🌟 {address[:12]}... | "
                f"Tier: {data['cosmic_tier'].value} | "
                f"Importancia: {data['cosmic_importance']:.3f} | "
                f"Total: {data['total_value']}"
            )
        
        # 3. Consulta masiva concurrente
        bt.logging.info("3. Consulta Masiva Concurrente...")
        massive_data = await balance_manager.massive_concurrent_query(COLDKEY_PUBS[:2])
        
        bt.logging.info(
            f"   📊 Resultados Masivos:\n"
            f"      Subredes: {massive_data['total_subnets']}\n"
            f"      Neuronas Totales: {massive_data['total_neurons']}\n"
            f"      Direcciones: {massive_data['coldkey_count']}"
        )

def demo_immunity_calculations():
    """Demostración de cálculos de inmunidad cósmica"""
    bt.logging.info("🛡️ DEMOSTRACIÓN DE CÁLCULOS DE INMUNIDAD CÓSMICA")
    
    # Configuración inicial
    config = ImmunityConfig(
        commit_reveal_period=100,  # 100 bloques
        tempo=2,                   # Factor de tempo 2x
        old_immunity_period=50     # 50 bloques de inmunidad anterior
    )
    
    calculator = CosmicImmunityCalculator(config)
    
    # Escenario 1: Aumento moderado del periodo commit-reveal
    new_period_1 = 150
    new_immunity_1 = calculator.calculate_new_immunity_period(new_period_1)
    bt.logging.info(f"   Escenario 1 (+50 bloques): {new_immunity_1} bloques de inmunidad")
    
    # Escenario 2: Reducción del periodo commit-reveal
    new_period_2 = 75
    new_immunity_2 = calculator.calculate_new_immunity_period(new_period_2)
    bt.logging.info(f"   Escenario 2 (-25 bloques): {new_immunity_2} bloques de inmunidad")
    
    # Actualizar configuración
    calculator.update_config(new_period_1)
    bt.logging.info(f"   ✅ Configuración actualizada a {new_period_1} bloques")

async def demo_wallet_creation():
    """Demostración de creación de wallets cósmicos"""
    bt.logging.info("👛 DEMOSTRACIÓN DE CREACIÓN DE WALLETS CÓSMICOS")
    
    balance_manager = CosmicBalanceManager("finney")
    
    # Crear wallets cósmicos (nombres de ejemplo)
    wallets_to_create = [
        ("sol_central", "quantum_processor"),
        ("tierra_conciencia", "comms_hub"),
        ("jupiter_mind", "memory_controller"
