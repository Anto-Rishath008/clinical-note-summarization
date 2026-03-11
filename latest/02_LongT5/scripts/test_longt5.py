"""
Quick Test Script for Simplified LongT5 Model
=============================================

This script verifies that the new LongT5 model:
1. Loads correctly
2. Can perform forward pass
3. Can generate sequences
4. Works with various sequence lengths

Usage:
    python scripts/test_longt5.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.longt5_model import SimplifiedLongT5, LongT5Config, create_model


def test_model_creation():
    """Test model creation with default config"""
    print("=" * 60)
    print("Test 1: Model Creation")
    print("=" * 60)
    
    config = LongT5Config()
    model = SimplifiedLongT5(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created successfully")
    print(f"  - d_model: {config.d_model}")
    print(f"  - num_encoder_layers: {config.num_encoder_layers}")
    print(f"  - num_decoder_layers: {config.num_decoder_layers}")
    print(f"  - num_heads: {config.num_heads}")
    print(f"  - vocab_size: {config.vocab_size}")
    print(f"  - max_position_embeddings: {config.max_position_embeddings}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    return model


def test_forward_pass(model):
    """Test forward pass with dummy data"""
    print("\n" + "=" * 60)
    print("Test 2: Forward Pass")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    # Test different sequence lengths
    test_cases = [
        (2, 128, 32, "Short sequence"),
        (2, 512, 64, "Medium sequence"),
        (1, 1024, 128, "Long sequence"),
    ]
    
    for batch_size, src_len, tgt_len, desc in test_cases:
        print(f"\n  Testing: {desc} (batch={batch_size}, src={src_len}, tgt={tgt_len})")
        
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, src_len), device=device)
        labels = torch.randint(0, model.config.vocab_size, (batch_size, tgt_len), device=device)
        attention_mask = torch.ones(batch_size, src_len, device=device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        print(f"    ✓ Loss: {outputs['loss'].item():.4f}")
        print(f"    ✓ Logits shape: {outputs['logits'].shape}")
        
        # Test backward pass
        outputs['loss'].backward()
        print(f"    ✓ Backward pass successful")
        
        model.zero_grad()


def test_generation(model):
    """Test generation capabilities"""
    print("\n" + "=" * 60)
    print("Test 3: Generation")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Test greedy decoding
    print("\n  Testing: Greedy decoding")
    input_ids = torch.randint(0, model.config.vocab_size, (1, 128), device=device)
    attention_mask = torch.ones(1, 128, device=device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=32,
            min_length=5,
            num_beams=1,  # Greedy
        )
    
    print(f"    ✓ Generated sequence shape: {output_ids.shape}")
    print(f"    ✓ First 10 tokens: {output_ids[0, :10].tolist()}")
    
    # Test beam search
    print("\n  Testing: Beam search (4 beams)")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=32,
            min_length=5,
            num_beams=4,
        )
    
    print(f"    ✓ Generated sequence shape: {output_ids.shape}")
    print(f"    ✓ First 10 tokens: {output_ids[0, :10].tolist()}")


def test_config_from_dict():
    """Test creating model from dictionary config"""
    print("\n" + "=" * 60)
    print("Test 4: Config from Dictionary")
    print("=" * 60)
    
    config_dict = {
        'model': {
            'd_model': 256,
            'd_ff': 1024,
            'num_encoder_layers': 4,
            'num_decoder_layers': 4,
            'num_heads': 8,
            'dropout': 0.1,
            'max_position_embeddings': 4096,
        },
        'data': {
            'vocab_size': 16000,
            'pad_id': 3,
            'bos_id': 1,
            'eos_id': 2,
        }
    }
    
    model = create_model(config_dict)
    
    print(f"✓ Model created from dict")
    print(f"  - d_model: {model.config.d_model}")
    print(f"  - vocab_size: {model.config.vocab_size}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {total_params:,}")


def test_long_sequence():
    """Test with long sequence (stress test)"""
    print("\n" + "=" * 60)
    print("Test 5: Long Sequence (2048 tokens)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Smaller model for long sequence test
    config = LongT5Config(
        d_model=256,
        d_ff=512,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        vocab_size=1000,
    )
    
    model = SimplifiedLongT5(config).to(device)
    model.eval()
    
    seq_len = 2048
    print(f"  Testing with {seq_len} token sequence...")
    
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)
    attention_mask = torch.ones(1, seq_len, device=device)
    
    try:
        with torch.no_grad():
            encoder_output = model.get_encoder_output(input_ids, attention_mask)
        
        print(f"    ✓ Encoder output shape: {encoder_output.shape}")
        print(f"    ✓ Long sequence encoding successful!")
        
        # Memory info
        if device.type == 'cuda':
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"    ✓ GPU memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"    ⚠ OOM with {seq_len} tokens (expected on small GPUs)")
        else:
            raise


def main():
    print("\n" + "=" * 60)
    print("     Simplified LongT5 Model Test Suite")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # Run tests
        model = test_model_creation()
        test_forward_pass(model)
        test_generation(model)
        test_config_from_dict()
        test_long_sequence()
        
        print("\n" + "=" * 60)
        print("     All Tests Passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
