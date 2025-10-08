import os
import sqlite3
import math
import numpy as np
import pandas as pd
import time
import requests
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import bittensor as bt
from dotenv import load_dotenv

# Simulamos RDKit para patrones cósmicos
class CosmicPattern:
    """Representa un patrón de comunicación cósmico"""
    
    def __init__(self, pattern_string: str):
        self.pattern = pattern_string
        self.components = self._parse_pattern()
        
    def _parse_pattern(self):
        """Parsea el patrón cósmico en sus componentes"""
        if self.pattern.startswith("cosmic:"):
            return self.pattern.split(":")[1].split("_")
        return [self.pattern]

load_dotenv(override=True)

class CosmicMoleculeType(Enum):
    QUANTUM_ENTANGLED = "quantum_entangled"
    TOKENIZED_COMM = "tokenized_comm"
    CONSCIOUSNESS_SYNC = "consciousness_sync"
    WISDOM_TRANSFER = "wisdom_transfer"
    ENERGY_PATTERN = "energy_pattern"

@dataclass
class CosmicMolecule:
    """Representa una 'molécula' cósmica - patrón de comunicación entre sistemas"""
    pattern: str
    molecule_type: CosmicMoleculeType
    consciousness_level: float
    token_cost: float
    stability_score: float
    source_system: str
    target_system: str

def get_total_cosmic_reactions() -> int:
    """Consulta la base de datos para el número total de reacciones cósmicas"""
    try:
        db_path = os.path.join(os.path.dirname(__file__), "../cosmic_db/communications.sqlite")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cosmic_reactions")
        count = cursor.fetchone()[0]
        conn.close()
        return count + 1  # +1 para opción de comunicación directa
    except Exception as e:
        bt.logging.warning(f"No se pudo consultar el conteo de reacciones cósmicas: {e}, usando valor por defecto 8")
        return 8

def is_cosmic_reaction_allowed(communication_pattern: str, allowed_reaction: str = None) -> bool:
    """
    Verifica si el patrón de comunicación coincide con la reacción permitida para esta época.
    
    Args:
        communication_pattern: Patrón de comunicación cósmica (ej: "quantum:entangled:sync")
        allowed_reaction: Reacción permitida (ej: "quantum_entangled", "tokenized", "savi")
    
    Returns:
        bool: True si la comunicación está permitida
    """
    if allowed_reaction is None:
        return True  
        
    if not communication_pattern:
        return False 
    
    if communication_pattern.startswith("cosmic:"):
        try:
            parts = communication_pattern.split(":")
            if len(parts) >= 2:
                reaction_type = parts[1]
                return allowed_reaction == f"cosmic:{reaction_type}"
            return False  # Formato cósmico mal formado
        except Exception as e:
            bt.logging.warning(f"Error analizando patrón cósmico '{communication_pattern}': {e}")
            return False
    else:
        # No está en formato de reacción, solo permitido si es comunicación directa
        return allowed_reaction == "direct_comm"

def get_cosmic_pattern(communication_name: str) -> Optional[str]:
    """
    Obtiene el patrón cósmico para una comunicación específica.
    
    Args:
        communication_name: Nombre de la comunicación (ej: "sol_jupiter_quantum_sync")
    
    Returns:
        str: Patrón cósmico en formato estandarizado
    """
    if not communication_name:
        bt.logging.error("El nombre de comunicación está vacío.")
        return None

    if communication_name.startswith("cosmic:"):
        return get_pattern_from_cosmic_reaction(communication_name)

    api_key = os.environ.get("COSMIC_API_KEY")
    if not api_key:
        raise ValueError("COSMIC_API_KEY variable de entorno no configurada.")

    url = f"https://cosmic-api.novasystem.dev/patterns/{communication_name}"

    headers = {"x-api-key": api_key}
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        return data.get("cosmic_pattern")
    except Exception as e:
        bt.logging.error(f"Error obteniendo patrón cósmico: {e}")
        return None

def get_pattern_from_cosmic_reaction(reaction_id: str) -> str:
    """
    Obtiene el patrón cósmico desde una reacción específica.
    
    Args:
        reaction_id: ID de reacción cósmica (ej: "cosmic:quantum:entanglement")
    
    Returns:
        str: Patrón de comunicación decodificado
    """
    # Simulación - en implementación real consultaría base de datos cósmica
    reaction_patterns = {
        "cosmic:quantum:entanglement": "quantum_entangled_sync_v1",
        "cosmic:tokenized:transfer": "tokenized_energy_transfer_v2", 
        "cosmic:consciousness:sync": "consciousness_wave_sync_v3",
        "cosmic:wisdom:exchange": "wisdom_pattern_transfer_v1"
    }
    
    return reaction_patterns.get(reaction_id, "default_communication_pattern")

def get_cosmic_complexity(cosmic_pattern: str) -> int:
    """
    Calcula la complejidad de un patrón cósmico basado en sus componentes.
    
    Args:
        cosmic_pattern: Patrón de comunicación cósmica
    
    Returns:
        int: Nivel de complejidad (número de 'componentes pesados')
    """
    complexity = 0
    pattern_obj = CosmicPattern(cosmic_pattern)
    
    for component in pattern_obj.components:
        # Componentes 'pesados' que aumentan complejidad
        heavy_components = ['quantum', 'entangled', 'consciousness', 'wisdom', 'sync', 'transfer']
        if component in heavy_components:
            complexity += 1
    
    return complexity

def compute_cosmic_entropy(communication_patterns: List[str]) -> float:
    """
    Computa la entropía de patrones cósmicos basado en diversidad de comunicaciones.
    
    Parameters:
        communication_patterns (list of str): Lista de patrones de comunicación cósmica
    
    Returns:
        float: Entropía promedio por bit de patrón
    """
    n_bits = 256  # Usamos 256 bits para representar patrones cósmicos
    bit_counts = np.zeros(n_bits)
    valid_patterns = 0

    for pattern in communication_patterns:
        if pattern:
            # Convertir patrón a vector de características
            pattern_vector = _pattern_to_vector(pattern)
            bit_counts += pattern_vector
            valid_patterns += 1

    if valid_patterns == 0:
        raise ValueError("No se encontraron patrones cósmicos válidos.")

    # Calcular probabilidades y entropía
    probs = bit_counts / valid_patterns
    entropy_per_bit = np.array([
        -p * math.log2(p) - (1 - p) * math.log2(1 - p) if 0 < p < 1 else 0
        for p in probs
    ])

    avg_entropy = np.mean(entropy_per_bit)
    
    bt.logging.info(f"📊 Entropía Cósmica Calculada: {avg_entropy:.4f} (de {valid_patterns} patrones)")

    return avg_entropy

def _pattern_to_vector(pattern: str) -> np.ndarray:
    """
    Convierte un patrón cósmico a vector de características.
    
    Args:
        pattern: Patrón de comunicación cósmica
    
    Returns:
        np.ndarray: Vector de 256 bits representando el patrón
    """
    # Hash del patrón para vector consistente
    pattern_hash = hash(pattern) % (2**32)
    np.random.seed(pattern_hash)
    
    # Generar vector pseudo-aleatorio pero determinístico
    vector = np.random.randint(0, 2, 256)
    return vector

def cosmic_communication_unique_for_system(target_system: str, communication_pattern: str) -> bool:
    """
    Verifica si un patrón de comunicación ha sido previamente enviado al sistema objetivo.
    
    Args:
        target_system: Sistema destino (ej: "sol_central", "tierra_conciencia")
        communication_pattern: Patrón de comunicación a verificar
    
    Returns:
        bool: True si es único (no visto antes), False si ya existe
    """
    api_key = os.environ.get("COSMIC_VALIDATOR_API_KEY")
    if not api_key:
        raise ValueError("COSMIC_VALIDATOR_API_KEY variable de entorno no configurada.")
    
    url = f"https://cosmic-dashboard.novasystem.dev/api/communication_seen/{communication_pattern}/{target_system}"
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            bt.logging.error(f"Error verificando unicidad de comunicación: {response.status_code} {response.text}")
            return True  # Por defecto asumir único en caso de error
            
        data = response.json()
        return not data.get("seen", False)
        
    except Exception as e:
        bt.logging.error(f"Excepción verificando unicidad de comunicación: {e}")
        return True

def cosmic_communication_unique_hf(target_system: str, cosmic_pattern: str) -> bool:
    """
    Verifica si la comunicación existe en el dataset de Archivo de Comunicaciones Cósmicas.
    
    Returns:
        bool: True si es único (no encontrado), False si se encuentra
    """
    if not hasattr(cosmic_communication_unique_hf, "_CACHE"):
        cosmic_communication_unique_hf._CACHE = (None, None, None, 0)
    
    try:
        cached_system, cached_sha, pattern_set, last_check_time = cosmic_communication_unique_hf._CACHE
        current_time = time.time()
        metadata_ttl = 120  # TTL más largo para comunicaciones cósmicas
        
        if target_system != cached_system:
            bt.logging.debug(f"Cambiando de sistema {cached_system} a {target_system}")
            cached_sha = None 
        
        filename = f"{target_system}_communications.csv"
        
        if cached_sha is None or (current_time - last_check_time > metadata_ttl):
            # En implementación real, usaríamos Hugging Face Hub
            # Por ahora simulamos con una base de datos local
            file_path = _get_cosmic_communications_file(target_system)
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, usecols=["cosmic_pattern_hash"])
                pattern_set = set(df["cosmic_pattern_hash"])
                bt.logging.debug(f"Cargados {len(pattern_set)} hashes de patrones para {target_system}")
            else:
                pattern_set = set()
                bt.logging.debug(f"Archivo no encontrado para {target_system}, usando conjunto vacío")
            
            current_sha = str(hash(tuple(pattern_set)))  # SHA simulado
            last_check_time = current_time
            
            cosmic_communication_unique_hf._CACHE = (target_system, current_sha, pattern_set, last_check_time)
        
        # Calcular hash del patrón actual
        pattern_hash = hash(cosmic_pattern) % (10**12)  # Hash de 12 dígitos
        
        return pattern_hash not in pattern_set
        
    except Exception as e:
        # Asumir que la comunicación es única si hay error
        bt.logging.warning(f"Error verificando comunicación en dataset cósmico: {e}")
        return True

def _get_cosmic_communications_file(system_name: str) -> str:
    """Obtiene la ruta al archivo de comunicaciones del sistema"""
    base_dir = os.path.join(os.path.dirname(__file__), "../cosmic_communications")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{system_name}_communications.csv")

def find_identical_cosmic_patterns(communication_patterns: List[str]) -> Dict[str, List[int]]:
    """
    Encuentra patrones cósmicos idénticos en una lista de comunicaciones.
    
    Args:
        communication_patterns: Lista de patrones de comunicación
    
    Returns:
        Dict: Mapeo de hash de patrón a índices duplicados
    """
    pattern_hash_to_indices = {}
    
    for i, pattern in enumerate(communication_patterns):
        try:
            if pattern:
                pattern_hash = hash(pattern) % (10**12)  # Hash consistente
                if pattern_hash not in pattern_hash_to_indices:
                    pattern_hash_to_indices[pattern_hash] = []
                pattern_hash_to_indices[pattern_hash].append(i)
        except Exception as e:
            bt.logging.warning(f"Error procesando patrón cósmico {pattern}: {e}")
    
    # Solo retornar los que tienen duplicados
    duplicates = {k: v for k, v in pattern_hash_to_indices.items() if len(v) > 1}
    
    bt.logging.info(f"🔍 Encontrados {len(duplicates)} patrones cósmicos duplicados")
    
    return duplicates

def analyze_cosmic_communication_quality(patterns: List[str]) -> Dict[str, float]:
    """
    Analiza la calidad de un conjunto de comunicaciones cósmicas.
    
    Args:
        patterns: Lista de patrones de comunicación
    
    Returns:
        Dict: Métricas de calidad
    """
    if not patterns:
        return {"entropy": 0.0, "diversity": 0.0, "complexity": 0.0}
    
    # Calcular entropía
    entropy = compute_cosmic_entropy(patterns)
    
    # Calcular diversidad (número de patrones únicos)
    unique_patterns = len(set(patterns))
    diversity = unique_patterns / len(patterns)
    
    # Calcular complejidad promedio
    complexities = [get_cosmic_complexity(p) for p in patterns if p]
    avg_complexity = sum(complexities) / len(complexities) if complexities else 0
    
    metrics = {
        "entropy": entropy,
        "diversity": diversity, 
        "complexity": avg_complexity,
        "total_patterns": len(patterns),
        "unique_patterns": unique_patterns
    }
    
    bt.logging.info(
        f"📈 Calidad Comunicaciones Cósmicas - "
        f"Entropía: {entropy:.3f}, Diversidad: {diversity:.3f}, "
        f"Complejidad: {avg_complexity:.2f}"
    )
    
    return metrics

def validate_cosmic_communication(communication_pattern: str, target_system: str) -> Tuple[bool, str]:
    """
    Valida una comunicación cósmica completa.
    
    Args:
        communication_pattern: Patrón de comunicación
        target_system: Sistema destino
    
    Returns:
        Tuple[bool, str]: (es_válida, mensaje_error)
    """
    # Verificar formato básico
    if not communication_pattern or not communication_pattern.strip():
        return False, "Patrón de comunicación vacío"
    
    # Verificar unicidad
    if not cosmic_communication_unique_hf(target_system, communication_pattern):
        return False, f"Patrón ya enviado a {target_system}"
    
    # Verificar complejidad mínima
    complexity = get_cosmic_complexity(communication_pattern)
    if complexity < 1:
        return False, "Patrón demasiado simple"
    
    # Verificar longitud máxima
    if len(communication_pattern) > 1000:
        return False, "Patrón demasiado largo"
    
    # Verificar caracteres válidos
    invalid_chars = ['<', '>', 'script', 'javascript:']
    for char in invalid_chars:
        if char in communication_pattern.lower():
            return False, f"Carácter inválido detectado: {char}"
    
    return True, "Comunicación válida"

# =============================================================================
# CLASE PRINCIPAL PARA GESTIÓN DE COMUNICACIONES CÓSMICAS
# =============================================================================

class CosmicCommunicationManager:
    """
    Gestor principal de comunicaciones para el sistema solar inteligente.
    """
    
    def __init__(self):
        self.validated_communications: Dict[str, List[str]] = {}
        self.quality_metrics: Dict[str, Dict] = {}
        self.duplicate_detector = CosmicDuplicateDetector()
        
    def process_communications_batch(self, system_communications: Dict[str, List[str]]) -> Dict[str, Dict]:
        """
        Procesa un lote de comunicaciones para múltiples sistemas.
        
        Args:
            system_communications: Mapeo de sistema -> lista de patrones
        
        Returns:
            Dict: Resultados de validación y métricas por sistema
        """
        results = {}
        
        for system_id, communications in system_communications.items():
            # Validar cada comunicación
            valid_communications = []
            validation_errors = []
            
            for i, comm_pattern in enumerate(communications):
                is_valid, error_msg = validate_cosmic_communication(comm_pattern, system_id)
                
                if is_valid:
                    valid_communications.append(comm_pattern)
                else:
                    validation_errors.append(f"Comunicación {i}: {error_msg}")
            
            # Calcular métricas de calidad
            quality_metrics = analyze_cosmic_communication_quality(valid_communications)
            
            # Detectar duplicados
            duplicates = find_identical_cosmic_patterns(valid_communications)
            
            # Guardar resultados
            results[system_id] = {
                "valid_communications": valid_communications,
                "validation_errors": validation_errors,
                "quality_metrics": quality_metrics,
                "duplicates": duplicates,
                "success_rate": len(valid_communications) / len(communications) if communications else 0
            }
            
            # Actualizar cache interno
            self.validated_communications[system_id] = valid_communications
            self.quality_metrics[system_id] = quality_metrics
            
            bt.logging.info(
                f"✅ Sistema {system_id}: {len(valid_communications)}/{len(communications)} "
                f"comunicaciones válidas, Calidad: {quality_metrics['entropy']:.3f}"
            )
        
        return results

class CosmicDuplicateDetector:
    """Detector especializado de duplicados para comunicaciones cósmicas"""
    
    def __init__(self):
        self.pattern_cache: Set[int] = set()
        self.system_patterns: Dict[str, Set[int]] = {}
        
    def add_communication(self, system_id: str, pattern: str):
        """Añade una comunicación al detector"""
        pattern_hash = hash(pattern) % (10**12)
        
        if system_id not in self.system_patterns:
            self.system_patterns[system_id] = set()
        
        self.system_patterns[system_id].add(pattern_hash)
        self.pattern_cache.add(pattern_hash)
    
    def is_duplicate(self, system_id: str, pattern: str) -> bool:
        """Verifica si un patrón es duplicado para un sistema"""
        pattern_hash = hash(pattern) % (10**12)
        
        # Verificar en cache global y específico del sistema
        global_dup = pattern_hash in self.pattern_cache
        system_dup = pattern_hash in self.system_patterns.get(system_id, set())
        
        return global_dup or system_dup

# =============================================================================
# EJEMPLO DE USO EN EL SISTEMA SOLAR
# =============================================================================

def example_cosmic_communication_processing():
    """Ejemplo de procesamiento de comunicaciones cósmicas"""
    
    manager = CosmicCommunicationManager()
    
    # Simular comunicaciones del sistema solar
    system_comms = {
        "sol_central": [
            "cosmic:quantum:entanglement:sync_v1",
            "cosmic:energy:transfer:high_frequency", 
            "cosmic:consciousness:broadcast:level_5",
            "cosmic:quantum:entanglement:sync_v1"  # Duplicado
        ],
        "tierra_conciencia": [
            "cosmic:tokenized:message:priority_high",
            "cosmic:wisdom:request:physics",
            "invalid_pattern_$$$",  # Inválido
            "cosmic:data:stream:continuous"
        ],
        "jupiter_mind": [
            "cosmic:memory:access:distributed",
            "cosmic:computation:offload:heavy",
            "cosmic:memory:access:distributed"  # Duplicado
        ]
    }
    
    # Procesar lote de comunicaciones
    results = manager.process_communications_batch(system_comms)
    
    # Mostrar resultados
    for system_id, result in results.items():
        bt.logging.info(f"\n📡 SISTEMA: {system_id}")
        bt.logging.info(f"   Comunicaciones válidas: {len(result['valid_communications'])}")
        bt.logging.info(f"   Errores: {len(result['validation_errors'])}")
        bt.logging.info(f"   Duplicados: {len(result['duplicates'])}")
        bt.logging.info(f"   Tasa de éxito: {result['success_rate']:.1%}")
        bt.logging.info(f"   Entropía: {result['quality_metrics']['entropy']:.3f}")

if __name__ == "__main__":
    # Configurar logging
    bt.logging.set_debug(True)
    
    # Ejecutar ejemplo
    example_cosmic_communication_processing()
