// Model fingerprinting - creates unique identifiers from model structure
// Works without running the model, just from metadata

export interface ModelFingerprint {
  hash: string;
  architecture: string;
  paramCount: number;
  layerCount: number;
  hiddenSize: number;
  attentionHeads: number;
  vocabSize: number;
  tensorSignature: string;
  shapeSignature: string;
}

// Generate a simple hash from string
function simpleHash(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return Math.abs(hash).toString(16).padStart(8, '0');
}

// Create model fingerprint from tensor metadata
export function createFingerprint(
  tensors: Array<{ name: string; shape: number[]; dtype: string }>,
  config?: any
): ModelFingerprint {
  // Create tensor signature (sorted tensor names)
  const tensorNames = tensors.map(t => t.name).sort();
  const tensorSignature = simpleHash(tensorNames.join(','));
  
  // Create shape signature (tensor shapes in order)
  const shapeStrings = tensors.map(t => `${t.name}:${t.shape.join('x')}`).sort();
  const shapeSignature = simpleHash(shapeStrings.join(','));
  
  // Extract architecture info
  const architecture = config?.model_type || 
                       (tensorNames.some(n => n.includes('moe')) ? 'moe' :
                        tensorNames.some(n => n.includes('mamba')) ? 'mamba' :
                        'transformer');
  
  // Count layers
  const layerNumbers = new Set<number>();
  for (const name of tensorNames) {
    const match = name.match(/layers\.(\d+)/);
    if (match) {
      layerNumbers.add(parseInt(match[1]));
    }
  }
  const layerCount = layerNumbers.size;
  
  // Estimate hidden size from attention weights
  let hiddenSize = 0;
  for (const tensor of tensors) {
    if (tensor.name.includes('self_attn.q_proj') || tensor.name.includes('attn.q_proj')) {
      hiddenSize = tensor.shape[0];
      break;
    }
  }
  
  // Estimate attention heads
  let attentionHeads = 0;
  if (config?.num_attention_heads) {
    attentionHeads = config.num_attention_heads;
  } else if (hiddenSize > 0) {
    // Estimate: typically hidden_size / 128 or / 64
    attentionHeads = Math.round(hiddenSize / 128);
  }
  
  // Vocab size from embedding
  let vocabSize = 0;
  for (const tensor of tensors) {
    if (tensor.name.includes('embed_tokens') || tensor.name === 'wte') {
      vocabSize = tensor.shape[0];
      break;
    }
  }
  
  // Total parameter count
  const paramCount = tensors.reduce((sum, t) => {
    return sum + t.shape.reduce((a, b) => a * b, 1);
  }, 0);
  
  // Create overall hash
  const hashInput = `${architecture}:${paramCount}:${layerCount}:${hiddenSize}:${tensorSignature}:${shapeSignature}`;
  const hash = simpleHash(hashInput);
  
  return {
    hash,
    architecture,
    paramCount,
    layerCount,
    hiddenSize,
    attentionHeads,
    vocabSize,
    tensorSignature,
    shapeSignature,
  };
}

// Compare two fingerprints
export function compareFingerprints(fp1: ModelFingerprint, fp2: ModelFingerprint): {
  isIdentical: boolean;
  similarity: number;
  differences: string[];
} {
  const differences: string[] = [];
  let matchCount = 0;
  const totalChecks = 6;
  
  if (fp1.architecture === fp2.architecture) matchCount++;
  else differences.push(`Architecture: ${fp1.architecture} vs ${fp2.architecture}`);
  
  if (fp1.paramCount === fp2.paramCount) matchCount++;
  else differences.push(`Params: ${fp1.paramCount} vs ${fp2.paramCount}`);
  
  if (fp1.layerCount === fp2.layerCount) matchCount++;
  else differences.push(`Layers: ${fp1.layerCount} vs ${fp2.layerCount}`);
  
  if (fp1.hiddenSize === fp2.hiddenSize) matchCount++;
  else differences.push(`Hidden: ${fp1.hiddenSize} vs ${fp2.hiddenSize}`);
  
  if (fp1.tensorSignature === fp2.tensorSignature) matchCount++;
  else differences.push('Different tensor structure');
  
  if (fp1.shapeSignature === fp2.shapeSignature) matchCount++;
  else differences.push('Different tensor shapes');
  
  return {
    isIdentical: fp1.hash === fp2.hash,
    similarity: matchCount / totalChecks,
    differences,
  };
}
