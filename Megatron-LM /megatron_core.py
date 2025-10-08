import asyncio
import os
import time
from typing import Optional, Any, Union, TypedDict, Iterable

# ... your existing imports ...

# Megatron-Core imports
from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
import torch
