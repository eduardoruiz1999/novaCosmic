!git clone https://github.com/NVIDIA/Megatron-LM.git
# Navigate into the cloned repository
!cd Megatron-LM

# Install required Python dependencies
!pip install -U setuptools packaging
!pip install --no-build-isolation .[dev]
# Install the core library
!pip install megatron-core
# Install with Transformer Engine for Flash Attention support
!pip install --no-build-isolation transformer-engine[pytorch]

# Add Docker's official GPG key
!sudo apt-get update
!sudo apt-get install ca-certificates curl
!sudo install -m 0755 -d /etc/apt/keyrings
!sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
!sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources
!echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
!sudo apt-get update
!sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin !docker-compose-plugin
!sudo docker run hello-world
!sudo systemctl status docker  # Check status
!sudo systemctl start docker   # Start the service
!sudo systemctl stop docker    # Stop the service
!sudo systemctl restart docker # Restart the service

