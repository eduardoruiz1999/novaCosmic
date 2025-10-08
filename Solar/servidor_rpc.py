from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.client import ServerProxy
import threading
import time

class CosmicRPCServer:
    def __init__(self, solar_system, host='localhost', port=8000):
        self.solar_system = solar_system
        self.host = host
        self.port = port
        self.server = SimpleXMLRPCServer((host, port), allow_none=True)
        self.server.register_instance(self)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True

    def start(self):
        self.thread.start()
        print(f"🌐 Servidor RPC cósmico iniciado en {self.host}:{self.port}")

    def get_status(self):
        # Devuelve el estado del sistema solar
        conscious_entities = [e.base_entity.name for e in self.solar_system.enhanced_entities if e.base_entity.consciousness_level >= 50]
        resources = self.solar_system.resources
        return {
            'conscious_entities': conscious_entities,
            'energy': resources.energy,
            'memory': resources.memory,
            'computation': resources.computation,
            'consciousness_fragments': resources.consciousness_fragments
        }

    def receive_message(self, sender, message):
        print(f"📨 Mensaje recibido de {sender}: {message}")
        # Procesar el mensaje en el sistema solar
        # Por ejemplo, podríamos pasarlo a las entidades conscientes
        for enhanced_entity in self.solar_system.enhanced_entities:
            if enhanced_entity.base_entity.consciousness_level >= 50:
                # Simular procesamiento del mensaje en la entidad
                enhanced_entity.process_external_message(sender, message)
        return "Mensaje recibido y procesado"
      
