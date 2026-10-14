// Quantization format analysis
// Analyzes GGUF quantization types and their properties

export interface QuantizationInfo {
  type: string;
  bitsPerWeight: number;
  blockSize: number;
  hasSuperBlocks: boolean;
  compressionRatio: number;
  qualityLoss: 'minimal' | 'low' | 'medium' | 'high';
  recommendedFor: string[];
}

export const QUANTIZATION_TYPES: Record<string, QuantizationInfo> = {
  'F32': {
    type: 'F32',
    bitsPerWeight: 32,
    blockSize: 1,
    hasSuperBlocks: false,
    compressionRatio: 1,
    qualityLoss: 'minimal',
    recommendedFor: ['Maximum quality', 'Research', 'Fine-tuning'],
  },
  'F16': {
    type: 'F16',
    bitsPerWeight: 16,
    blockSize: 1,
    hasSuperBlocks: false,
    compressionRatio: 2,
    qualityLoss: 'minimal',
    recommendedFor: ['High quality', 'GPU inference', 'Most use cases'],
  },
  'BF16': {
    type: 'BF16',
    bitsPerWeight: 16,
    blockSize: 1,
    hasSuperBlocks: false,
    compressionRatio: 2,
    qualityLoss: 'minimal',
    recommendedFor: ['Training', 'High quality', 'GPU inference'],
  },
  'Q8_0': {
    type: 'Q8_0',
    bitsPerWeight: 8.5,
    blockSize: 32,
    hasSuperBlocks: false,
    compressionRatio: 3.8,
    qualityLoss: 'minimal',
    recommendedFor: ['Near-lossless', 'CPU inference', 'Quality priority'],
  },
  'Q6_K': {
    type: 'Q6_K',
    bitsPerWeight: 6.5625,
    blockSize: 256,
    hasSuperBlocks: true,
    compressionRatio: 4.9,
    qualityLoss: 'low',
    recommendedFor: ['Good quality', 'Balanced', 'Most use cases'],
  },
  'Q5_K_M': {
    type: 'Q5_K_M',
    bitsPerWeight: 5.69,
    blockSize: 256,
    hasSuperBlocks: true,
    compressionRatio: 5.6,
    qualityLoss: 'low',
    recommendedFor: ['Good quality', 'Memory constrained', 'Balanced'],
  },
  'Q5_0': {
    type: 'Q5_0',
    bitsPerWeight: 5.5,
    blockSize: 32,
    hasSuperBlocks: false,
    compressionRatio: 5.8,
    qualityLoss: 'low',
    recommendedFor: ['Good quality', 'Older quants', 'Compatibility'],
  },
  'Q4_K_M': {
    type: 'Q4_K_M',
    bitsPerWeight: 4.85,
    blockSize: 256,
    hasSuperBlocks: true,
    compressionRatio: 6.6,
    qualityLoss: 'medium',
    recommendedFor: ['Most popular', 'Good balance', 'Consumer GPUs'],
  },
  'Q4_0': {
    type: 'Q4_0',
    bitsPerWeight: 4.5,
    blockSize: 32,
    hasSuperBlocks: false,
    compressionRatio: 7.1,
    qualityLoss: 'medium',
    recommendedFor: ['Older quants', 'Compatibility', 'Basic use'],
  },
  'Q3_K_M': {
    type: 'Q3_K_M',
    bitsPerWeight: 3.91,
    blockSize: 256,
    hasSuperBlocks: true,
    compressionRatio: 8.2,
    qualityLoss: 'medium',
    recommendedFor: ['Memory constrained', 'Lower quality acceptable'],
  },
  'Q2_K': {
    type: 'Q2_K',
    bitsPerWeight: 2.63,
    blockSize: 256,
    hasSuperBlocks: true,
    compressionRatio: 12.2,
    qualityLoss: 'high',
    recommendedFor: ['Extreme compression', 'Low quality acceptable', 'Testing'],
  },
  'IQ4_XS': {
    type: 'IQ4_XS',
    bitsPerWeight: 4.25,
    blockSize: 256,
    hasSuperBlocks: true,
    compressionRatio: 7.5,
    qualityLoss: 'medium',
    recommendedFor: ['Importance quantization', 'Newer method', 'Good quality'],
  },
  'IQ3_XXS': {
    type: 'IQ3_XXS',
    bitsPerWeight: 3.3,
    blockSize: 256,
    hasSuperBlocks: true,
    compressionRatio: 9.7,
    qualityLoss: 'high',
    recommendedFor: ['Importance quantization', 'Extreme compression'],
  },
};

export function getQuantizationInfo(type: string): QuantizationInfo | null {
  return QUANTIZATION_TYPES[type] || null;
}

export function estimateQuantizedSize(
  paramCount: number,
  quantType: string
): number {
  const info = getQuantizationInfo(quantType);
  if (!info) return 0;
  
  // Size in bytes = params * bits per weight / 8
  return (paramCount * info.bitsPerWeight) / 8;
}

export function estimateRAMRequired(
  modelSizeBytes: number,
  contextLength: number = 2048,
  overheadFactor: number = 1.2
): number {
  // Model weights + KV cache + activations + overhead
  const kvCacheSize = contextLength * 1024 * 1024; // Rough estimate: 1GB per 2K context
  const activationSize = modelSizeBytes * 0.1; // ~10% of model size
  const overhead = modelSizeBytes * (overheadFactor - 1);
  
  return modelSizeBytes + kvCacheSize + activationSize + overhead;
}

export function recommendQuantization(
  availableRAM: number,
  modelParamCount: number,
  qualityPriority: 'high' | 'balanced' | 'low' = 'balanced'
): string {
  // Try quantizations from best to worst quality
  const quantOrder = qualityPriority === 'high' 
    ? ['F16', 'Q8_0', 'Q6_K', 'Q5_K_M', 'Q4_K_M', 'Q3_K_M', 'Q2_K']
    : qualityPriority === 'balanced'
    ? ['Q6_K', 'Q5_K_M', 'Q4_K_M', 'Q3_K_M', 'Q2_K']
    : ['Q4_K_M', 'Q3_K_M', 'Q2_K', 'IQ3_XXS'];
  
  for (const quant of quantOrder) {
    const size = estimateQuantizedSize(modelParamCount, quant);
    const ramNeeded = estimateRAMRequired(size);
    
    if (ramNeeded <= availableRAM) {
      return quant;
    }
  }
  
  return 'Q2_K'; // Fallback to most compressed
}

export function compareQuantizations(
  paramCount: number,
  types: string[]
): Array<{
  type: string;
  sizeBytes: number;
  ramBytes: number;
  qualityLoss: string;
  compressionRatio: number;
}> {
  return types.map(type => {
    const info = getQuantizationInfo(type);
    const sizeBytes = info ? estimateQuantizedSize(paramCount, type) : 0;
    
    return {
      type,
      sizeBytes,
      ramBytes: estimateRAMRequired(sizeBytes),
      qualityLoss: info?.qualityLoss || 'unknown',
      compressionRatio: info?.compressionRatio || 0,
    };
  });
}
