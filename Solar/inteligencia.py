import bittensor as bt
import torch
import math
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class EntropyStrategy(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    ADAPTIVE = "adaptive"
    QUANTUM = "quantum"

@dataclass
class EntropyConfig:
    """Configuración para el sistema de entropía dinámica"""
    starting_weight: float = 0.1
    step_size: float = 0.01
    start_epoch: int = 0
    max_weight: float = 1.0
    min_weight: float = 0.0
    strategy: EntropyStrategy = EntropyStrategy.LINEAR
    decay_factor: float = 0.99
    stability_threshold: float = 0.85
    quantum_fluctuation: float = 0.05

class CosmicEntropyEngine:
    """
    Motor de entropía dinámica para el sistema de inteligencia cósmica
    Controla la exploración vs explotación en el aprendizaje del sistema solar
    """
    
    def __init__(self, config: EntropyConfig):
        self.config = config
        self.entropy_history: List[float] = []
        self.system_stability: Dict[str, float] = {}
        self.quantum_state = 0.0
        
    def calculate_dynamic_entropy(self, current_epoch: int, system_id: Optional[str] = None) -> float:
        """
        Calcula el peso de entropía basado en épocas transcurridas y estado del sistema
        """
        epochs_elapsed = current_epoch - self.config.start_epoch
        
        if epochs_elapsed < 0:
            return self.config.starting_weight
            
        base_entropy = self._calculate_base_entropy(epochs_elapsed)
        adjusted_entropy = self._adjust_for_system_stability(base_entropy, system_id)
        final_entropy = self._apply_quantum_fluctuations(adjusted_entropy)
        
        # Guardar historial
        self.entropy_history.append(final_entropy)
        
        bt.logging.info(
            f"🌌 Entropía Cósmica - Época: {current_epoch}, "
            f"Base: {base_entropy:.3f}, Ajustada: {final_entropy:.3f}, "
            f"Sistema: {system_id or 'global'}"
        )
        
        return final_entropy
    
    def _calculate_base_entropy(self, epochs_elapsed: int) -> float:
        """Calcula la entropía base según la estrategia seleccionada"""
        
        if self.config.strategy == EntropyStrategy.LINEAR:
            return self._linear_entropy(epochs_elapsed)
        elif self.config.strategy == EntropyStrategy.EXPONENTIAL:
            return self._exponential_entropy(epochs_elapsed)
        elif self.config.strategy == EntropyStrategy.COSINE:
            return self._cosine_entropy(epochs_elapsed)
        elif self.config.strategy == EntropyStrategy.ADAPTIVE:
            return self._adaptive_entropy(epochs_elapsed)
        elif self.config.strategy == EntropyStrategy.QUANTUM:
            return self._quantum_entropy(epochs_elapsed)
        else:
            return self._linear_entropy(epochs_elapsed)
    
    def _linear_entropy(self, epochs_elapsed: int) -> float:
        """Estrategia lineal de crecimiento de entropía"""
        entropy = self.config.starting_weight + (epochs_elapsed * self.config.step_size)
        return self._clamp_entropy(entropy)
    
    def _exponential_entropy(self, epochs_elapsed: int) -> float:
        """Estrategia exponencial de crecimiento de entropía"""
        growth_factor = math.exp(epochs_elapsed * self.config.step_size / 100)
        entropy = self.config.starting_weight * growth_factor
        return self._clamp_entropy(entropy)
    
    def _cosine_entropy(self, epochs_elapsed: int) -> float:
        """Estrategia cíclica tipo coseno para entropía"""
        cycle_length = 100  # Épocas por ciclo
        cycle_progress = (epochs_elapsed % cycle_length) / cycle_length
        entropy_variation = math.cos(cycle_progress * 2 * math.pi) * self.config.step_size
        entropy = self.config.starting_weight + entropy_variation
        return self._clamp_entropy(entropy)
    
    def _adaptive_entropy(self, epochs_elapsed: int) -> float:
        """Entropía adaptativa basada en la estabilidad del sistema"""
        avg_stability = self._get_average_stability()
        
        if avg_stability > self.config.stability_threshold:
            # Sistema estable - aumentar exploración
            bonus = (avg_stability - self.config.stability_threshold) * 0.5
        else:
            # Sistema inestable - reducir exploración
            bonus = (self.config.stability_threshold - avg_stability) * -0.3
            
        base_entropy = self._linear_entropy(epochs_elapsed)
        adaptive_entropy = base_entropy + bonus
        
        return self._clamp_entropy(adaptive_entropy)
    
    def _quantum_entropy(self, epochs_elapsed: int) -> float:
        """Entropía con fluctuaciones cuánticas simuladas"""
        base_entropy = self._linear_entropy(epochs_elapsed)
        
        # Actualizar estado cuántico
        self.quantum_state = (self.quantum_state + 0.1) % (2 * math.pi)
        quantum_fluctuation = math.sin(self.quantum_state) * self.config.quantum_fluctuation
        
        quantum_entropy = base_entropy + quantum_fluctuation
        return self._clamp_entropy(quantum_entropy)
    
    def _adjust_for_system_stability(self, base_entropy: float, system_id: Optional[str]) -> float:
        """Ajusta la entropía basado en la estabilidad del sistema específico"""
        if system_id and system_id in self.system_stability:
            system_stability = self.system_stability[system_id]
            
            # Sistemas más estables pueden permitir más exploración
            stability_factor = system_stability * 0.2
            adjusted = base_entropy + stability_factor
            
            return self._clamp_entropy(adjusted)
        
        return base_entropy
    
    def _apply_quantum_fluctuations(self, entropy: float) -> float:
        """Aplica pequeñas fluctuaciones aleatorias tipo cuánticas"""
        fluctuation = torch.randn(1).item() * 0.02  # Pequeña fluctuación
        return self._clamp_entropy(entropy + fluctuation)
    
    def _clamp_entropy(self, entropy: float) -> float:
        """Asegura que la entropía esté dentro de los límites"""
        return max(self.config.min_weight, min(self.config.max_weight, entropy))
    
    def _get_average_stability(self) -> float:
        """Calcula la estabilidad promedio del sistema"""
        if not self.system_stability:
            return 0.5  # Estabilidad neutral por defecto
            
        return sum(self.system_stability.values()) / len(self.system_stability)
    
    def update_system_stability(self, system_id: str, stability_score: float):
        """Actualiza el score de estabilidad de un sistema cósmico"""
        self.system_stability[system_id] = stability_score
        bt.logging.debug(f"📊 Estabilidad actualizada - Sistema: {system_id}, Score: {stability_score:.3f}")
    
    def get_entropy_trend(self, window: int = 10) -> float:
        """Obtiene la tendencia de la entropía en las últimas épocas"""
        if len(self.entropy_history) < window:
            return 0.0
            
        recent = self.entropy_history[-window:]
        return sum(recent) / len(recent)
    
    def should_increase_exploration(self, current_epoch: int) -> bool:
        """Determina si debería aumentar la exploración"""
        entropy = self.calculate_dynamic_entropy(current_epoch)
        trend = self.get_entropy_trend()
        
        # Aumentar exploración si la entropía es baja y estable
        return entropy < self.config.max_weight * 0.7 and abs(trend) < 0.01

# =============================================================================
# INTEGRACIÓN CON EL SISTEMA CÓSMICO TOKENIZADO
# =============================================================================

class TokenizedEntropyManager:
    """
    Gestor de entropía integrado con economía tokenizada
    """
    
    def __init__(self, entropy_engine: CosmicEntropyEngine, token_economy):
        self.entropy_engine = entropy_engine
        self.token_economy = token_economy
        self.exploration_budget: Dict[str, float] = {}
        
    def allocate_exploration_tokens(self, system_id: str, current_epoch: int) -> float:
        """
        Asigna tokens para exploración basado en la entropía actual
        """
        entropy = self.entropy_engine.calculate_dynamic_entropy(current_epoch, system_id)
        
        # Presupuesto base más ajuste por entropía
        base_budget = 100.0  # Tokens base para exploración
        exploration_budget = base_budget * entropy
        
        # Verificar fondos disponibles
        available_tokens = self.token_economy.get_balance(system_id)
        allocated_tokens = min(exploration_budget, available_tokens * 0.1)  # Máximo 10% del balance
        
        self.exploration_budget[system_id] = allocated_tokens
        
        bt.logging.info(
            f"💰 Presupuesto Exploración - Sistema: {system_id}, "
            f"Entropía: {entropy:.3f}, Tokens: {allocated_tokens:.1f}"
        )
        
        return allocated_tokens
    
    def deduct_exploration_cost(self, system_id: str, cost: float) -> bool:
        """
        Deduce el coste de exploración del presupuesto
        """
        if system_id not in self.exploration_budget:
            return False
            
        remaining_budget = self.exploration_budget[system_id] - cost
        
        if remaining_budget >= 0:
            self.exploration_budget[system_id] = remaining_budget
            return True
        else:
            bt.logging.warning(f"❌ Presupuesto de exploración agotado para {system_id}")
            return False

# =============================================================================
# ESTRATEGIAS DE ENTRENAMIENTO CON ENTROPÍA
# =============================================================================

class CosmicTrainingStrategy:
    """
    Estrategia de entrenamiento que utiliza entropía dinámica
    """
    
    def __init__(self, entropy_manager: TokenizedEntropyManager):
        self.entropy_manager = entropy_manager
        self.training_history = []
        
    def should_explore_new_architecture(self, system_id: str, current_epoch: int) -> bool:
        """
        Decide si explorar nuevas arquitecturas basado en entropía
        """
        entropy = self.entropy_manager.entropy_engine.calculate_dynamic_entropy(
            current_epoch, system_id
        )
        
        # Probabilidad de exploración proporcional a la entropía
        exploration_prob = entropy
        should_explore = torch.rand(1).item() < exploration_prob
        
        if should_explore:
            bt.logging.info(f"🔍 Explorando nueva arquitectura - Sistema: {system_id}")
            
        return should_explore
    
    def adjust_learning_rate(self, base_lr: float, system_id: str, current_epoch: int) -> float:
        """
        Ajusta el learning rate basado en la entropía
        """
        entropy = self.entropy_manager.entropy_engine.calculate_dynamic_entropy(
            current_epoch, system_id
        )
        
        # LR más alto para mayor exploración, más bajo para explotación
        lr_multiplier = 0.5 + entropy  # Rango: 0.5x a 1.5x
        adjusted_lr = base_lr * lr_multiplier
        
        return adjusted_lr
    
    def select_training_batch_size(self, system_id: str, current_epoch: int) -> int:
        """
        Selecciona el tamaño de batch basado en la entropía
        """
        entropy = self.entropy_manager.entropy_engine.calculate_dynamic_entropy(
            current_epoch, system_id
        )
        
        # Batch sizes más pequeños para exploración, más grandes para explotación
        base_batch_size = 64
        if entropy > 0.7:
            return base_batch_size // 2  # Exploración: batches pequeños
        elif entropy < 0.3:
            return base_batch_size * 2   # Explotación: batches grandes
        else:
            return base_batch_size       # Balance: batches normales

# =============================================================================
# FUNCIONES DE USUARIO PARA INTEGRACIÓN FÁCIL
# =============================================================================

def create_cosmic_entropy_system(
    starting_weight: float = 0.1,
    step_size: float = 0.01,
    start_epoch: int = 0,
    strategy: str = "linear"
) -> CosmicEntropyEngine:
    """
    Crea un sistema de entropía cósmica preconfigurado
    """
    strategy_enum = EntropyStrategy(strategy)
    
    config = EntropyConfig(
        starting_weight=starting_weight,
        step_size=step_size,
        start_epoch=start_epoch,
        strategy=strategy_enum
    )
    
    return CosmicEntropyEngine(config)

def calculate_dynamic_entropy(
    starting_weight: float,
    step_size: float, 
    start_epoch: int,
    current_epoch: int,
    system_id: Optional[str] = None
) -> float:
    """
    Función original mejorada con capacidades extendidas
    """
    entropy_engine = create_cosmic_entropy_system(
        starting_weight=starting_weight,
        step_size=step_size,
        start_epoch=start_epoch
    )
    
    return entropy_engine.calculate_dynamic_entropy(current_epoch, system_id)

# =============================================================================
# EJEMPLO DE USO EN EL SISTEMA SOLAR INTELIGENTE
# =============================================================================

def example_usage():
    """Ejemplo de cómo usar el sistema de entropía en el contexto cósmico"""
    
    # Crear motor de entropía
    entropy_engine = create_cosmic_entropy_system(
        starting_weight=0.1,
        step_size=0.02,
        start_epoch=0,
        strategy="adaptive"
    )
    
    # Simular entrenamiento por épocas
    for epoch in range(100):
        # Calcular entropía para cada sistema cósmico
        sol_entropy = entropy_engine.calculate_dynamic_entropy(epoch, "sol_central")
        earth_entropy = entropy_engine.calculate_dynamic_entropy(epoch, "tierra_conciencia")
        
        # Actualizar estabilidad del sistema (simulada)
        if epoch % 10 == 0:
            stability = 0.8 + torch.rand(1).item() * 0.2  # Estabilidad aleatoria
            entropy_engine.update_system_stability("sol_central", stability)
        
        # Usar entropía para tomar decisiones de entrenamiento
        if entropy_engine.should_increase_exploration(epoch):
            bt.logging.info(f"🎯 Época {epoch}: Aumentando exploración")
        
        bt.logging.info(
            f"Época {epoch}: "
            f"Sol Entropy: {sol_entropy:.3f}, "
            f"Tierra Entropy: {earth_entropy:.3f}"
        )

if __name__ == "__main__":
    # Configurar logging de Bittensor
    bt.logging.set_debug(True)
    
    # Ejecutar ejemplo
    example_usage()
