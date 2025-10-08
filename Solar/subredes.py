import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import bittensor as bt
from bittensor.core.subtensor import Subtensor
from bittensor.core.async_subtensor import AsyncSubtensor
from bittensor.utils.balance import Balance

@dataclass
class CosmicSubnet:
    """Representa una subred cósmica en el ecosistema Bittensor"""
    netuid: int
    exists: bool
    name: str = "Desconocida"
    neuron_count: int = 0
    emission: float = 0.0
    difficulty: str = "media"
    cosmic_importance: float = 0.0

class CosmicSubnetExplorer:
    """
    Explorador avanzado de subredes cósmicas para el sistema solar inteligente.
    Combina métodos síncronos y asíncronos para máximo rendimiento.
    """
    
    def __init__(self, network: str = "finney"):
        self.network = network
        self.sync_subtensor = Subtensor(network)
        self.subnet_cache: Dict[int, CosmicSubnet] = {}
        self.last_update: float = 0
        self.cache_ttl = 300  # 5 minutos
        
    def discover_all_subnets_sync(self, max_netuid: int = 50) -> List[CosmicSubnet]:
        """
        Descubre todas las subredes cósmicas usando métodos síncronos.
        Ideal para operaciones simples y scripts rápidos.
        """
        discovered_subnets = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Verificar existencia de subredes en paralelo
            futures = {
                executor.submit(self._check_subnet_exists_sync, netuid): netuid 
                for netuid in range(1, max_netuid + 1)
            }
            
            for future in futures:
                netuid = futures[future]
                try:
                    exists = future.result()
                    if exists:
                        subnet = self._get_subnet_details_sync(netuid)
                        discovered_subnets.append(subnet)
                        bt.logging.debug(f"🔍 Subred {netuid} descubierta: {subnet.name}")
                except Exception as e:
                    bt.logging.warning(f"❌ Error verificando subred {netuid}: {e}")
        
        # Ordenar por importancia cósmica
        discovered_subnets.sort(key=lambda x: x.cosmic_importance, reverse=True)
        
        bt.logging.info(f"🌌 Descubiertas {len(discovered_subnets)} subredes cósmicas")
        return discovered_subnets
    
    async def discover_all_subnets_async(self, max_netuid: int = 50) -> List[CosmicSubnet]:
        """
        Descubre todas las subredes cósmicas usando métodos asíncronos.
        Máximo rendimiento para aplicaciones en tiempo real.
        """
        start_time = time.time()
        
        async with AsyncSubtensor(self.network) as async_sub:
            # Obtener hash de bloque para consistencia
            block_hash = await async_sub.get_block_hash()
            
            # Verificar existencia de todas las subredes concurrentemente
            existence_tasks = [
                async_sub.subnet_exists(netuid, block_hash=block_hash) 
                for netuid in range(1, max_netuid + 1)
            ]
            existence_results = await asyncio.gather(*existence_tasks)
            
            # Filtrar subredes existentes y obtener detalles
            existing_netuids = [
                netuid for netuid, exists in enumerate(existence_results, 1) 
                if exists
            ]
            
            # Obtener detalles de todas las subredes existentes concurrentemente
            detail_tasks = [
                self._get_subnet_details_async(async_sub, netuid, block_hash)
                for netuid in existing_netuids
            ]
            discovered_subnets = await asyncio.gather(*detail_tasks)
            
            # Actualizar cache
            for subnet in discovered_subnets:
                self.subnet_cache[subnet.netuid] = subnet
            
            self.last_update = time.time()
            
            # Ordenar por importancia
            discovered_subnets.sort(key=lambda x: x.cosmic_importance, reverse=True)
            
            elapsed = time.time() - start_time
            bt.logging.info(
                f"🚀 Descubrimiento asíncrono completado: "
                f"{len(discovered_subnets)} subredes en {elapsed:.2f}s"
            )
            
            return discovered_subnets
    
    def _check_subnet_exists_sync(self, netuid: int) -> bool:
        """Verifica existencia de subred de forma síncrona"""
        try:
            return self.sync_subtensor.subnet_exists(netuid=netuid)
        except Exception as e:
            bt.logging.error(f"Error verificando subred {netuid}: {e}")
            return False
    
    def _get_subnet_details_sync(self, netuid: int) -> CosmicSubnet:
        """Obtiene detalles de subred de forma síncrona"""
        try:
            # Información básica de la subred
            neuron_count = len(self.sync_subtensor.neurons(netuid))
            emission = self.sync_subtensor.emission(netuid)
            
            # Determinar importancia cósmica
            cosmic_importance = self._calculate_cosmic_importance(
                neuron_count, emission, netuid
            )
            
            # Determinar dificultad
            difficulty = self._determine_difficulty(neuron_count, emission)
            
            # Nombre basado en el netuid (en implementación real, usar metadatos)
            subnet_name = self._get_subnet_cosmic_name(netuid)
            
            return CosmicSubnet(
                netuid=netuid,
                exists=True,
                name=subnet_name,
                neuron_count=neuron_count,
                emission=emission,
                difficulty=difficulty,
                cosmic_importance=cosmic_importance
            )
            
        except Exception as e:
            bt.logging.error(f"Error obteniendo detalles de subred {netuid}: {e}")
            return CosmicSubnet(netuid=netuid, exists=True, cosmic_importance=0.0)
    
    async def _get_subnet_details_async(self, async_sub: AsyncSubtensor, netuid: int, block_hash: str) -> CosmicSubnet:
        """Obtiene detalles de subred de forma asíncrona"""
        try:
            # Obtener neuronas y emisión concurrentemente
            neurons_task = async_sub.neurons_lite(netuid, block_hash)
            emission_task = async_sub.get_emission(netuid, block_hash)
            
            neurons, emission = await asyncio.gather(neurons_task, emission_task)
            
            neuron_count = len(neurons) if neurons else 0
            emission_value = emission.value if hasattr(emission, 'value') else 0.0
            
            # Calcular métricas cósmicas
            cosmic_importance = self._calculate_cosmic_importance(
                neuron_count, emission_value, netuid
            )
            
            difficulty = self._determine_difficulty(neuron_count, emission_value)
            subnet_name = self._get_subnet_cosmic_name(netuid)
            
            return CosmicSubnet(
                netuid=netuid,
                exists=True,
                name=subnet_name,
                neuron_count=neuron_count,
                emission=emission_value,
                difficulty=difficulty,
                cosmic_importance=cosmic_importance
            )
            
        except Exception as e:
            bt.logging.error(f"Error asíncrono en subred {netuid}: {e}")
            return CosmicSubnet(netuid=netuid, exists=True, cosmic_importance=0.0)
    
    def _calculate_cosmic_importance(self, neuron_count: int, emission: float, netuid: int) -> float:
        """Calcula la importancia cósmica de una subred"""
        # Factores de importancia
        neuron_factor = min(neuron_count / 1000, 1.0)  # Normalizar a 1000 neuronas
        emission_factor = min(emission / 1000, 1.0)    # Normalizar a 1000 emisión
        netuid_factor = 1.0 / (1 + abs(netuid - 1))    # Subredes bajas más importantes
        
        # Ponderación
        importance = (
            neuron_factor * 0.4 +
            emission_factor * 0.4 +
            netuid_factor * 0.2
        )
        
        return importance
    
    def _determine_difficulty(self, neuron_count: int, emission: float) -> str:
        """Determina la dificultad de participación en la subred"""
        if neuron_count < 50 or emission < 100:
            return "fácil"
        elif neuron_count < 200 or emission < 500:
            return "media"
        else:
            return "difícil"
    
    def _get_subnet_cosmic_name(self, netuid: int) -> str:
        """Asigna nombres cósmicos a las subredes basado en su netuid"""
        cosmic_names = {
            1: "Núcleo Central Cósmico",
            2: "Red de Conciencia Colectiva", 
            3: "Sistema de Sabiduría Distribuida",
            4: "Red de Computación Cuántica",
            5: "Ecosistema de Memoria Global",
            6: "Sistema de Comunicación Interestelar",
            7: "Red de Energía Tokenizada",
            8: "Sistema de Predicción Temporal",
            9: "Red de Simulación Multidimensional",
            10: "Ecosistema de Creación Colaborativa"
        }
        
        return cosmic_names.get(netuid, f"Subred Cósmica #{netuid}")

class CosmicBalanceManager:
    """
    Gestor avanzado de balances para el ecosistema cósmico.
    Soporta múltiples direcciones y métodos eficientes.
    """
    
    def __init__(self, network: str = "finney"):
        self.network = network
        self.sync_sub = Subtensor(network)
    
    async def get_cosmic_balances_async(self, addresses: List[str]) -> Dict[str, Balance]:
        """
        Obtiene balances de múltiples direcciones de forma asíncrona.
        """
        async with AsyncSubtensor(self.network) as async_sub:
            balances = await async_sub.get_balance(*addresses)
            return balances
    
    def get_cosmic_balances_sync(self, addresses: List[str]) -> Dict[str, Balance]:
        """
        Obtiene balances de múltiples direcciones de forma síncrona.
        """
        balances = {}
        for address in addresses:
            try:
                balance = self.sync_sub.get_balance(address)
                balances[address] = balance
            except Exception as e:
                bt.logging.error(f"Error obteniendo balance para {address}: {e}")
                balances[address] = Balance(0)
        
        return balances
    
    async def get_comprehensive_cosmic_info(self, coldkey_address: str) -> Dict:
        """
        Obtiene información comprehensiva de una dirección cósmica.
        """
        async with AsyncSubtensor(self.network) as async_sub:
            # Ejecutar múltiples consultas concurrentemente
            balance_task = async_sub.get_balance(coldkey_address)
            delegated_task = async_sub.get_delegated(coldkey_address)
            staked_task = async_sub.get_total_stake_for_coldkey(coldkey_address)
            
            balance, delegated, staked = await asyncio.gather(
                balance_task, delegated_task, staked_task,
                return_exceptions=True
            )
            
            # Manejar excepciones
            balance = balance if not isinstance(balance, Exception) else {}
            delegated = delegated if not isinstance(delegated, Exception) else {}
            staked = staked if not isinstance(staked, Exception) else Balance(0)
            
            return {
                "coldkey": coldkey_address,
                "balance": balance.get(coldkey_address, Balance(0)),
                "delegated": delegated,
                "total_staked": staked,
                "total_value": balance.get(coldkey_address, Balance(0)) + staked,
                "timestamp": time.time()
            }

class CosmicNetworkMonitor:
    """
    Monitor en tiempo real del ecosistema cósmico completo.
    """
    
    def __init__(self, network: str = "finney"):
        self.network = network
        self.explorer = CosmicSubnetExplorer(network)
        self.balance_manager = CosmicBalanceManager(network)
        self.monitoring_data = {}
    
    async def start_comprehensive_monitoring(self, interval: int = 60):
        """
        Inicia monitoreo comprehensivo del ecosistema cósmico.
        """
        bt.logging.info("🛰️ Iniciando monitoreo cósmico comprehensivo...")
        
        while True:
            try:
                start_time = time.time()
                
                # Monitorear subredes y balances concurrentemente
                subnets_task = self.explorer.discover_all_subnets_async()
                balances_task = self._monitor_key_balances()
                
                subnets, balance_info = await asyncio.gather(subnets_task, balances_task)
                
                # Actualizar datos de monitoreo
                self.monitoring_data = {
                    "timestamp": time.time(),
                    "subnets_count": len(subnets),
                    "total_neurons": sum(subnet.neuron_count for subnet in subnets),
                    "total_emission": sum(subnet.emission for subnet in subnets),
                    "top_subnets": subnets[:5],
                    "balance_info": balance_info,
                    "network_health": self._calculate_network_health(subnets)
                }
                
                elapsed = time.time() - start_time
                bt.logging.info(
                    f"📊 Monitoreo Cósmico - "
                    f"Subredes: {len(subnets)}, "
                    f"Neuronas: {self.monitoring_data['total_neurons']}, "
                    f"Tiempo: {elapsed:.2f}s"
                )
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                bt.logging.error(f"❌ Error en monitoreo cósmico: {e}")
                await asyncio.sleep(interval)
    
    async def _monitor_key_balances(self) -> Dict:
        """Monitorea balances de direcciones clave del ecosistema"""
        # Direcciones de ejemplo - en implementación real usar direcciones reales
        key_addresses = [
            "5EhCvSxpFRgXRCaN5LH2wRCD5su1vKsnVfYfjzkqfmPoCy2G",
            "5CZrQzo3W6LGEopMw2zVMugPcwFBmQDYne3TJc9XzZbTX2WR",
        ]
        
        try:
            balances = await self.balance_manager.get_cosmic_balances_async(key_addresses)
            return balances
        except Exception as e:
            bt.logging.error(f"Error monitoreando balances: {e}")
            return {}
    
    def _calculate_network_health(self, subnets: List[CosmicSubnet]) -> str:
        """Calcula la salud general de la red cósmica"""
        total_importance = sum(subnet.cosmic_importance for subnet in subnets)
        avg_neurons = sum(subnet.neuron_count for subnet in subnets) / len(subnets) if subnets else 0
        
        if total_importance > 5.0 and avg_neurons > 100:
            return "💚 Excelente"
        elif total_importance > 3.0 and avg_neurons > 50:
            return "💛 Buena"
        else:
            return "🧡 En desarrollo"

# =============================================================================
# EJEMPLOS DE USO Y DEMOSTRACIONES
# =============================================================================

async def demo_complete_exploration():
    """Demostración completa de exploración cósmica"""
    bt.logging.info("🌌 INICIANDO DEMOSTRACIÓN DE EXPLORACIÓN CÓSMICA")
    
    # Explorador de subredes
    explorer = CosmicSubnetExplorer("finney")
    
    # Método asíncrono (recomendado)
    bt.logging.info("1. Exploración Asíncrona...")
    subnets_async = await explorer.discover_all_subnets_async(max_netuid=20)
    
    for subnet in subnets_async[:3]:  # Mostrar primeras 3
        bt.logging.info(
            f"   📡 {subnet.name} (netuid={subnet.netuid}) - "
            f"Neuronas: {subnet.neuron_count}, "
            f"Importancia: {subnet.cosmic_importance:.2f}"
        )
    
    # Gestor de balances
    balance_manager = CosmicBalanceManager("finney")
    
    # Direcciones de ejemplo
    cosmic_addresses = [
        "5EhCvSxpFRgXRCaN5LH2wRCD5su1vKsnVfYfjzkqfmPoCy2G",
        "5CZrQzo3W6LGEopMw2zVMugPcwFBmQDYne3TJc9XzZbTX2WR",
    ]
    
    bt.logging.info("2. Consulta de Balances Asíncrona...")
    balances = await balance_manager.get_cosmic_balances_async(cosmic_addresses)
    
    for address, balance in balances.items():
        bt.logging.info(f"   💰 {address[:12]}...: {balance}")
    
    # Información comprehensiva
    bt.logging.info("3. Información Comprehensiva...")
    cosmic_info = await balance_manager.get_comprehensive_cosmic_info(cosmic_addresses[0])
    bt.logging.info(f"   📊 Balance Total: {cosmic_info['total_value']}")

async def demo_network_monitoring():
    """Demostración del sistema de monitoreo en tiempo real"""
    bt.logging.info("🛰️ INICIANDO DEMOSTRACIÓN DE MONITOREO EN TIEMPO REAL")
    
    monitor = CosmicNetworkMonitor("finney")
    
    # Ejecutar monitoreo por 2 ciclos
    monitoring_task = asyncio.create_task(monitor.start_comprehensive_monitoring(interval=30))
    
    # Esperar y mostrar resultados
    await asyncio.sleep(65)  # Esperar 2 ciclos + margen
    
    # Mostrar datos recolectados
    if monitor.monitoring_data:
        data = monitor.monitoring_data
        bt.logging.info("📈 DATOS DE MONITOREO CÓSMICO:")
        bt.logging.info(f"   Subredes Activas: {data['subnets_count']}")
        bt.logging.info(f"   Neuronas Totales: {data['total_neurons']}")
        bt.logging.info(f"   Salud de Red: {data['network_health']}")
        bt.logging.info(f"   Emisión Total: {data['total_emission']:.2f}")
    
    monitoring_task.cancel()

def demo_sync_methods():
    """Demostración de métodos síncronos para scripts simples"""
    bt.logging.info("⚡ INICIANDO DEMOSTRACIÓN DE MÉTODOS SÍNCRONOS")
    
    explorer = CosmicSubnetExplorer("finney")
    
    # Exploración síncrona
    subnets_sync = explorer.discover_all_subnets_sync(max_netuid=15)
    
    bt.logging.info(f"🔍 Encontradas {len(subnets_sync)} subredes (síncrono):")
    for subnet in subnets_sync[:5]:
        bt.logging.info(f"   🪐 {subnet.name} - Dificultad: {subnet.difficulty}")

if __name__ == "__main__":
    # Configurar logging
    bt.logging.set_debug(True)
    
    # Ejecutar demostraciones
    asyncio.run(demo_complete_exploration())
    
    # Demostración síncrona
    demo_sync_methods()
    
    # Descomentar para monitoreo en tiempo real
    # asyncio.run(demo_network_monitoring())
