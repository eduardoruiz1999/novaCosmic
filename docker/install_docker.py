# Instalar Docker en Colab
!curl -fsSL https://get.docker.com -o get-docker.sh
!sh get-docker.sh

# Agregar el usuario actual al grupo docker
!sudo usermod -aG docker $USER

# Verificar instalación
!docker --version
  
  # Instalar el cliente de Docker para Python
!pip install docker

# Ejemplo de uso
import docker

# Crear cliente Docker
client = docker.from_env()

# Verificar conexión
try:
    print("Docker version:", client.version())
    print("¡Docker está funcionando correctamente!")
except Exception as e:
    print("Error:", e)
