# patch2.py - apply remaining V3 edits to model.py
with open('src/model.py', 'r', encoding='utf-8') as f:
    content = f.read()

edits = 0

# EDIT: encode() - source bypass
old = """        all_memory = []
        for chunk, mask in zip(chunks, chunk_masks):
            # Embed chunk
            chunk_emb = self.src_embedding(chunk)
            chunk_emb = self.pos_encoding(chunk_emb)
            
            # Encode with Mamba
            chunk_encoded = self.encoder(chunk_emb)
            
            # Compress to memory tokens
            memory_tokens = self.memory_compressor(chunk_encoded, mask)
            all_memory.append(memory_tokens)
        
        # Concatenate all memory tokens
        memory = torch.cat(all_memory, dim=1)
        
        # Cross-chunk memory interaction (V2)
        if self.cross_chunk_attn is not None:
            for layer in self.cross_chunk_attn:
                memory = layer(memory)
        
        return memory"""

new = """        all_memory = []
        all_bypass = []
        bypass_stride = self.config.source_bypass_stride

        for chunk, mask in zip(chunks, chunk_masks):
            # Embed chunk
            chunk_emb = self.src_embedding(chunk)
            chunk_emb = self.pos_encoding(chunk_emb)
            
            # Encode with Mamba
            chunk_encoded = self.encoder(chunk_emb)

            # V3: collect downsampled encoded tokens for source bypass
            if bypass_stride > 0:
                all_bypass.append(chunk_encoded[:, ::bypass_stride, :])
            
            # Compress to memory tokens
            memory_tokens = self.memory_compressor(chunk_encoded, mask)
            all_memory.append(memory_tokens)
        
        # Concatenate all memory tokens
        memory = torch.cat(all_memory, dim=1)
        
        # Cross-chunk memory interaction (V2)
        if self.cross_chunk_attn is not None:
            for layer in self.cross_chunk_attn:
                memory = layer(memory)

        # V3: Source bypass - concatenate downsampled source with memory
        if bypass_stride > 0 and all_bypass:
            source_detail = torch.cat(all_bypass, dim=1)
            self._source_detail = source_detail
            memory = torch.cat([memory, source_detail], dim=1)
        
        return memory"""

if 'all_bypass' not in content and old in content:
    content = content.replace(old, new, 1)
    edits += 1
    print('encode: PATCHED')
elif 'all_bypass' in content:
    print('encode: already patched')
else:
    print('encode: OLD TEXT NOT FOUND')
    # Debug - find all_memory
    idx = content.find('all_memory = []')
    if idx > 0:
        print('  Context:', repr(content[idx:idx+300]))

# EDIT: forward() - store source info + apply copy mechanism
old = """        # Encode source
        memory = self.encode(src, src_mask)
        
        # Generate causal mask for decoder
        causal_mask = self.generate_causal_mask(tgt.size(1), tgt.device)
        
        # Decode
        logits = self.decode(tgt, memory, causal_mask, tgt_mask)
        
        return logits"""

new = """        # Store source info for copy mechanism
        self._source_ids = src
        self._source_mask = src_mask
        
        # Encode source (sets self._source_detail if bypass enabled)
        memory = self.encode(src, src_mask)
        
        # Generate causal mask for decoder
        causal_mask = self.generate_causal_mask(tgt.size(1), tgt.device)
        
        # Decode (sets self._last_decoded)
        logits = self.decode(tgt, memory, causal_mask, tgt_mask)

        # Apply copy mechanism if enabled, otherwise plain log_softmax
        if self.copy_mechanism is not None and self._source_detail is not None:
            log_probs = self.copy_mechanism(
                self._last_decoded, logits,
                self._source_detail, src, src_mask,
            )
        else:
            log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs"""

if 'log_probs = self.copy_mechanism' not in content and old in content:
    content = content.replace(old, new, 1)
    edits += 1
    print('forward: PATCHED')
elif 'log_probs = self.copy_mechanism' in content:
    print('forward: already patched')
else:
    print('forward: OLD TEXT NOT FOUND')
    idx = content.find('# Encode source')
    if idx > 0:
        print('  Context:', repr(content[idx:idx+300]))

with open('src/model.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Edits applied:', edits)
