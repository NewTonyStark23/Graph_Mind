// Layer-by-layer parameter distribution analysis
// Shows how parameters are distributed across the model

export interface LayerDistribution {
  layerIndex: number;
  layerType: string;
  paramCount: number;
  percentage: number;
  components: {
    attention: number;
    ffn: number;
    norms: number;
    moe?: number;
  };
}

export interface ModelDistribution {
  embedding: number;
  layers: LayerDistribution[];
  lmHead: number;
  total: number;
  summary: {
    attentionPercentage: number;
    ffnPercentage: number;
    moePercentage: number;
    embeddingPercentage: number;
    lmHeadPercentage: number;
  };
}

export function analyzeLayerDistribution(
  tensors: Array<{ name: string; shape: number[] }>
): ModelDistribution {
  let embedding = 0;
  let lmHead = 0;
  const layerMap = new Map<number, {
    attention: number;
    ffn: number;
    norms: number;
    moe: number;
  }>();
  
  for (const tensor of tensors) {
    const paramCount = tensor.shape.reduce((a, b) => a * b, 1);
    
    // Embedding
    if (tensor.name.includes('embed_tokens') || tensor.name === 'wte' || tensor.name === 'transformer.wte') {
      embedding += paramCount;
      continue;
    }
    
    // LM head
    if (tensor.name.includes('lm_head') || tensor.name === 'output' || tensor.name === 'transformer.lm_head') {
      lmHead += paramCount;
      continue;
    }
    
    // Layer-specific tensors
    const layerMatch = tensor.name.match(/layers\.(\d+)\./);
    if (!layerMatch) continue;
    
    const layerIdx = parseInt(layerMatch[1]);
    if (!layerMap.has(layerIdx)) {
      layerMap.set(layerIdx, { attention: 0, ffn: 0, norms: 0, moe: 0 });
    }
    
    const layerData = layerMap.get(layerIdx)!;
    
    // Attention
    if (tensor.name.includes('self_attn') || tensor.name.includes('attn.')) {
      layerData.attention += paramCount;
    }
    // MoE
    else if (tensor.name.includes('block_sparse_moe') || tensor.name.includes('mlp.experts')) {
      layerData.moe += paramCount;
    }
    // FFN
    else if (tensor.name.includes('mlp') || tensor.name.includes('feed_forward')) {
      layerData.ffn += paramCount;
    }
    // Norms
    else if (tensor.name.includes('norm') || tensor.name.includes('layernorm')) {
      layerData.norms += paramCount;
    }
  }
  
  // Build layers array
  const layers: LayerDistribution[] = [];
  let totalAttention = 0;
  let totalFFN = 0;
  let totalMoE = 0;
  
  for (const [idx, data] of Array.from(layerMap.entries()).sort((a, b) => a[0] - b[0])) {
    const layerTotal = data.attention + data.ffn + data.norms + data.moe;
    
    layers.push({
      layerIndex: idx,
      layerType: data.moe > 0 ? 'MoE' : 'Dense',
      paramCount: layerTotal,
      percentage: 0, // Will calculate after total
      components: {
        attention: data.attention,
        ffn: data.ffn,
        norms: data.norms,
        moe: data.moe > 0 ? data.moe : undefined,
      },
    });
    
    totalAttention += data.attention;
    totalFFN += data.ffn;
    totalMoE += data.moe;
  }
  
  const total = embedding + lmHead + layers.reduce((sum, l) => sum + l.paramCount, 0);
  
  // Calculate percentages
  for (const layer of layers) {
    layer.percentage = (layer.paramCount / total) * 100;
  }
  
  return {
    embedding,
    layers,
    lmHead,
    total,
    summary: {
      attentionPercentage: (totalAttention / total) * 100,
      ffnPercentage: (totalFFN / total) * 100,
      moePercentage: (totalMoE / total) * 100,
      embeddingPercentage: (embedding / total) * 100,
      lmHeadPercentage: (lmHead / total) * 100,
    },
  };
}

export function formatParamCount(count: number): string {
  if (count < 1e6) return `${(count / 1e3).toFixed(1)}K`;
  if (count < 1e9) return `${(count / 1e6).toFixed(1)}M`;
  if (count < 1e12) return `${(count / 1e9).toFixed(2)}B`;
  return `${(count / 1e12).toFixed(2)}T`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

// Estimate memory for different precisions
export function estimateMemoryForPrecision(
  paramCount: number,
  precision: 'F32' | 'F16' | 'BF16' | 'Q8' | 'Q4'
): number {
  const bytesPerParam = {
    'F32': 4,
    'F16': 2,
    'BF16': 2,
    'Q8': 1,
    'Q4': 0.5,
  }[precision];
  
  return paramCount * bytesPerParam;
}
