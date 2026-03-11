"""Debug NaN issue"""
import torch
import yaml
import sys
sys.path.insert(0, 'workspace_mamba_longt5_v1/src')

from model import MambaTransformerConfig, build_model
from data_loader import DataConfig, create_dataloaders, load_tokenizer

# Load config
with open('workspace_mamba_longt5_v1/configs/local_train.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_config = MambaTransformerConfig.from_dict(config.get('model', {}))
data_config = DataConfig.from_dict(config)

print(f'Model vocab_size: {model_config.vocab_size}')

# Create a small dataloader
train_loader, val_loader, tokenizer = create_dataloaders(config=data_config, batch_size=4)

# Get one batch
batch = next(iter(train_loader))
print(f'\nBatch src shape: {batch["src"].shape}')
print(f'Batch src min/max: {batch["src"].min().item()}, {batch["src"].max().item()}')

# Create model
device = torch.device('cuda')
model = build_model(model_config).to(device)
model.eval()

# Run forward pass step by step
with torch.no_grad():
    src = batch['src'].to(device)
    tgt_input = batch['tgt_input'].to(device)
    src_mask = batch['src_mask'].to(device)
    tgt_mask = batch['tgt_mask'].to(device)
    
    print(f'\n=== Step-by-step debug ===')
    
    # 1. Check embedding
    src_emb = model.src_embedding(src)
    print(f'1. src_embedding output: min={src_emb.min().item():.4f}, max={src_emb.max().item():.4f}, has_nan={torch.isnan(src_emb).any().item()}')
    
    # 2. Check position encoding  
    src_emb_pos = model.pos_encoding(src_emb)
    print(f'2. pos_encoding output: min={src_emb_pos.min().item():.4f}, max={src_emb_pos.max().item():.4f}, has_nan={torch.isnan(src_emb_pos).any().item()}')
    
    # 3. Check embedding layer weights
    print(f'\n3. Embedding weight stats:')
    emb_weight = model.src_embedding.weight
    print(f'   min={emb_weight.min().item():.4f}, max={emb_weight.max().item():.4f}, has_nan={torch.isnan(emb_weight).any().item()}')
    
    # 4. Try encoding just the first chunk
    print(f'\n4. Testing encoding...')
    chunks, chunk_masks = model._chunk_input(src, src_mask)
    print(f'   Number of chunks: {len(chunks)}')
    print(f'   First chunk shape: {chunks[0].shape}')
    
    # Encode first chunk manually
    chunk = chunks[0]
    mask = chunk_masks[0]
    chunk_emb = model.src_embedding(chunk)
    print(f'   Chunk embedding: has_nan={torch.isnan(chunk_emb).any().item()}')
    
    chunk_emb = model.pos_encoding(chunk_emb)
    print(f'   After pos_encoding: has_nan={torch.isnan(chunk_emb).any().item()}')
    
    # Check mamba encoder layer by layer
    print(f'\n5. Testing Mamba encoder layer by layer...')
    x = chunk_emb
    for i, layer in enumerate(model.encoder.layers):
        x = layer(x)
        print(f'   Layer {i}: has_nan={torch.isnan(x).any().item()}, min={x.min().item():.4f}, max={x.max().item():.4f}')
        if torch.isnan(x).any():
            print(f'   >>> NaN found at layer {i}!')
            break
    
    # Check final encoder norm
    x = model.encoder.final_norm(x)
    print(f'   Final norm: has_nan={torch.isnan(x).any().item()}')
    
    # 6. Test memory compression
    print(f'\n6. Testing memory compressor...')
    memory_tokens = model.memory_compressor(x, mask)
    print(f'   Memory tokens: has_nan={torch.isnan(memory_tokens).any().item()}, min={memory_tokens.min().item():.4f}, max={memory_tokens.max().item():.4f}')
    
    # 7. Test full encode - chunk by chunk
    print(f'\n7. Testing full encode (chunk by chunk)...')
    all_memory = []
    for i, (chunk, mask) in enumerate(zip(chunks, chunk_masks)):
        chunk_emb = model.src_embedding(chunk)
        chunk_emb = model.pos_encoding(chunk_emb)
        chunk_encoded = model.encoder(chunk_emb)
        memory_tokens = model.memory_compressor(chunk_encoded, mask)
        
        has_nan = torch.isnan(memory_tokens).any().item()
        print(f'   Chunk {i}: has_nan={has_nan}, min={memory_tokens.min().item():.4f}, max={memory_tokens.max().item():.4f}')
        if has_nan:
            print(f'   >>> NaN found at chunk {i}!')
            print(f'       Chunk input ids: min={chunk.min().item()}, max={chunk.max().item()}')
            print(f'       Mask True count: {mask.sum().item()}')
            break
        all_memory.append(memory_tokens)
    
    if not any(torch.isnan(m).any() for m in all_memory):
        memory = torch.cat(all_memory, dim=1)
        print(f'   Concatenated memory: has_nan={torch.isnan(memory).any().item()}')
        print(f'   Memory shape: {memory.shape}')
    
    # 8. Test decoder embedding
    print(f'\n8. Testing decoder embedding...')
    tgt_emb = model.tgt_embedding(tgt_input)
    print(f'   tgt_embedding: has_nan={torch.isnan(tgt_emb).any().item()}')
    tgt_emb = model.pos_encoding(tgt_emb)
    print(f'   After pos_encoding: has_nan={torch.isnan(tgt_emb).any().item()}')
    
    # 9. Test decode 
    print(f'\n9. Testing decode...')
    # Generate causal mask
    tgt_len = tgt_input.size(1)
    causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()
    
    logits = model.decode(tgt_input, memory, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_mask)
    print(f'   Logits: has_nan={torch.isnan(logits).any().item()}')
    
    # 10. Check decoder layer by layer
    print(f'\n10. Testing decoder layer by layer...')
    x = tgt_emb
    for i, layer in enumerate(model.decoder.layers):
        x = layer(x, memory, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_mask)
        print(f'   Layer {i}: has_nan={torch.isnan(x).any().item()}, min={x.min().item():.4f}, max={x.max().item():.4f}')
        if torch.isnan(x).any():
            print(f'   >>> NaN found at decoder layer {i}!')
            break
    
    x = model.decoder.final_norm(x)
    print(f'   Decoder final norm: has_nan={torch.isnan(x).any().item()}')
    
    # 11. Check output projection
    print(f'\n11. Testing output projection...')
    out = model.output_proj(x)
    print(f'   Output: has_nan={torch.isnan(out).any().item()}, min={out.min().item():.4f}, max={out.max().item():.4f}')
