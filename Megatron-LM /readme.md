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
