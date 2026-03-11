import torch, csv

# Check latest checkpoint gate values
ckpt = torch.load('checkpoints/v2_run/checkpoint_step_29800.pt', map_location='cpu', weights_only=False)
sd = ckpt['model_state_dict']
gates = {k:v.item() for k,v in sd.items() if 'cross_gate' in k}
print('Step 29800 Cross-attention gates:')
for k,v in gates.items():
    print(f'  {k}: raw={v:.4f}, sigmoid={torch.sigmoid(torch.tensor(v)).item():.4f}')

# Check CSV headers and sample
with open('data/mimic-iv-bhc.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    print(f'\nCSV columns: {headers}')
    row = next(reader)
    for h in headers:
        val = row.get(h,'')
        print(f'  {h}: {val[:100]}...' if len(val)>100 else f'  {h}: {val}')

# Check memory compressor query norms
mem_q = sd.get('memory_compressor.memory_queries')
if mem_q is not None:
    print(f'\nMemory query norms: mean={mem_q.norm(dim=-1).mean():.4f}, std={mem_q.norm(dim=-1).std():.4f}')

# Check embedding norms
emb = sd.get('src_embedding.weight')
if emb is not None:
    print(f'Src embedding norms: mean={emb.norm(dim=-1).mean():.4f}, std={emb.norm(dim=-1).std():.4f}')
