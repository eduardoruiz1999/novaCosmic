def initialize_megatron(tensor_model_parallel_size: int = 1, pipeline_model_parallel_size: int = 1):
    """Initialize Megatron's distributed training components."""
    # Initialize distributed process group (if not already done)
    if not torch.distributed.is_initialized():
        rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = torch.cuda.device_count()
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(
            world_size=world_size, 
            rank=rank,
            backend='nccl'
        )
    
    # Initialize model parallel
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size,
        pipeline_model_parallel_size
    )

def create_megatron_model(vocab_size: int = 10000, max_sequence_length: int = 2048):
    """Create a GPT model with Flash Attention support."""
    
    # Transformer configuration with Flash Attention enabled
    transformer_config = TransformerConfig(
        num_layers=12,
        hidden_size=768,
        num_attention_heads=12,
        seq_length=max_sequence_length,
        max_position_embeddings=max_sequence_length,
        use_cpu_initialization=False,
        pipeline_dtype=torch.bfloat16,
        # These parameters enable Flash Attention
        fp16=False,  # Use bf16 instead for modern GPUs
        bf16=True,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=False,  # Required for Flash Attention
        bias=False,  # Flash Attention typically doesn't support bias
        masked_softmax_fusion=False,  # Let Flash Attention handle this
    )
    
    # Create GPT model - this will automatically use Flash Attention when available
    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        pre_process=True,
        post_process=True
    )
    
    return gpt_model
