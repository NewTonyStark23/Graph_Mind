// Real safetensors header parser
// The safetensors format: first 8 bytes = header size (uint64 LE), then JSON header

export interface SafetensorsTensor {
  name: string;
  dtype: string;
  shape: number[];
  dataOffsets: [number, number];
  sizeBytes: number;
}

export interface SafetensorsHeader {
  __metadata__?: Record<string, string>;
  tensors: Record<string, {
    dtype: string;
    shape: number[];
    data_offsets: [number, number];
  }>;
}

export interface ParsedSafetensors {
  header: SafetensorsHeader;
  tensors: SafetensorsTensor[];
  totalSizeBytes: number;
  headerSizeBytes: number;
  totalParameters: number;
  architecture?: string;
  modelType?: string;
}

// DType sizes in bytes
const DTYPE_SIZES: Record<string, number> = {
  'BOOL': 1,
  'U8': 1,
  'I8': 1,
  'I16': 2,
  'U16': 2,
  'I32': 4,
  'U32': 4,
  'I64': 8,
  'U64': 8,
  'F16': 2,
  'BF16': 2,
  'F32': 4,
  'F64': 8,
};

export function getDtypeSize(dtype: string): number {
  return DTYPE_SIZES[dtype] || 2; // default to 2 (F16/BF16)
}

// Parse safetensors header from a File or ArrayBuffer
export async function parseSafetensorsHeader(file: File | ArrayBuffer): Promise<ParsedSafetensors> {
  let buffer: ArrayBuffer;
  
  if (file instanceof File) {
    // Only read the first 100MB for the header (headers are usually < 1MB)
    const headerSlice = file.slice(0, 100 * 1024 * 1024);
    buffer = await headerSlice.arrayBuffer();
  } else {
    buffer = file;
  }

  const view = new DataView(buffer);
  
  // First 8 bytes = header size (uint64 LE)
  const headerSizeLow = view.getUint32(0, true);
  const headerSizeHigh = view.getUint32(4, true);
  const headerSize = headerSizeLow + headerSizeHigh * 2 ** 32;
  
  if (headerSize > buffer.byteLength - 8) {
    throw new Error(`Header size ${headerSize} exceeds buffer size ${buffer.byteLength}`);
  }
  
  // Parse JSON header
  const headerBytes = new Uint8Array(buffer, 8, headerSize);
  const headerText = new TextDecoder().decode(headerBytes);
  const rawHeader = JSON.parse(headerText);
  
  // Extract tensors
  const tensors: SafetensorsTensor[] = [];
  let totalParameters = 0;
  
  for (const [name, info] of Object.entries(rawHeader)) {
    if (name === '__metadata__') continue;
    
    const tensorInfo = info as { dtype: string; shape: number[]; data_offsets: [number, number] };
    const sizeBytes = tensorInfo.data_offsets[1] - tensorInfo.data_offsets[0];
    const paramCount = tensorInfo.shape.reduce((a, b) => a * b, 1);
    
    tensors.push({
      name,
      dtype: tensorInfo.dtype,
      shape: tensorInfo.shape,
      dataOffsets: tensorInfo.data_offsets,
      sizeBytes,
    });
    
    totalParameters += paramCount;
  }
  
  // Sort tensors by offset for sequential access
  tensors.sort((a, b) => a.dataOffsets[0] - b.dataOffsets[0]);
  
  // Try to detect architecture from tensor names
  const architecture = detectArchitecture(tensors);
  const modelType = detectModelType(tensors);
  
  const totalSizeBytes = file instanceof File ? file.size : buffer.byteLength;
  
  return {
    header: rawHeader as SafetensorsHeader,
    tensors,
    totalSizeBytes,
    headerSizeBytes: headerSize,
    totalParameters,
    architecture,
    modelType,
  };
}

// Detect architecture from tensor naming patterns
function detectArchitecture(tensors: SafetensorsTensor[]): string {
  const names = tensors.map(t => t.name);
  const nameStr = names.join(' ');
  
  const hasMoE = nameStr.includes('block_sparse_moe') || nameStr.includes('mlp.experts') || nameStr.includes('local_experts');
  const hasMamba = nameStr.includes('ssm') || nameStr.includes('mamba') || nameStr.includes('mixer');
  const hasConv = nameStr.includes('conv') || nameStr.includes('vision');
  
  if (hasMamba && hasMoE) return 'Hybrid Mamba-MoE';
  if (hasMamba) return 'Mamba (SSM)';
  if (hasMoE) return 'Mixture of Experts (MoE)';
  if (hasConv) return 'Vision Transformer / Multimodal';
  return 'Dense Transformer';
}

// Detect model type (LLM, Vision, etc.)
function detectModelType(tensors: SafetensorsTensor[]): string {
  const names = tensors.map(t => t.name);
  const nameStr = names.join(' ');
  
  if (nameStr.includes('visual') || nameStr.includes('vision') || nameStr.includes('image')) return 'Multimodal LLM';
  if (nameStr.includes('audio') || nameStr.includes('speech')) return 'Audio LLM';
  if (nameStr.includes('decoder') && !nameStr.includes('encoder')) return 'Decoder-only LLM';
  if (nameStr.includes('encoder') && nameStr.includes('decoder')) return 'Encoder-Decoder';
  return 'LLM';
}

// Compute real weight statistics by streaming through the file
export async function computeWeightStatistics(
  file: File,
  tensors: SafetensorsTensor[],
  onProgress?: (progress: number) => void
): Promise<Map<string, TensorStats>> {
  const stats = new Map<string, TensorStats>();
  const headerSize = 8 + tensors.reduce((max, t) => Math.max(max, t.dataOffsets[1]), 0);
  
  // Process tensors in chunks to avoid loading entire file
  const CHUNK_SIZE = 64 * 1024 * 1024; // 64MB chunks
  let processedBytes = 0;
  const totalDataBytes = tensors.reduce((sum, t) => sum + t.sizeBytes, 0);
  
  for (const tensor of tensors) {
    // Only compute stats for weight tensors (skip norms, biases for speed)
    if (!tensor.name.includes('weight') || tensor.shape.length < 2) {
      processedBytes += tensor.sizeBytes;
      continue;
    }
    
    // Limit to first 1000 tensors for performance
    if (stats.size >= 1000) break;
    
    const dtypeSize = getDtypeSize(tensor.dtype);
    
    // For large tensors, sample instead of reading all
    const sampleSize = Math.min(tensor.sizeBytes, 4 * 1024 * 1024); // 4MB sample
    const offset = 8 + tensor.dataOffsets[0]; // 8 bytes for header size
    
    try {
      const slice = file.slice(offset, offset + sampleSize);
      const buffer = await slice.arrayBuffer();
      const values = bufferToFloatArray(buffer, tensor.dtype);
      
      if (values.length > 0) {
        stats.set(tensor.name, computeStatsFromArray(values, tensor));
      }
    } catch (e) {
      // Skip tensors we can't read
    }
    
    processedBytes += tensor.sizeBytes;
    if (onProgress) {
      onProgress(processedBytes / totalDataBytes);
    }
  }
  
  return stats;
}

// Convert ArrayBuffer to Float array based on dtype
function bufferToFloatArray(buffer: ArrayBuffer, dtype: string): Float32Array | Float64Array {
  const view = new DataView(buffer);
  
  switch (dtype) {
    case 'F32': {
      const result = new Float32Array(buffer.byteLength / 4);
      for (let i = 0; i < result.length; i++) {
        result[i] = view.getFloat32(i * 4, true);
      }
      return result;
    }
    case 'F64': {
      const result = new Float64Array(buffer.byteLength / 8);
      for (let i = 0; i < result.length; i++) {
        result[i] = view.getFloat64(i * 8, true);
      }
      return result;
    }
    case 'F16': {
      // Convert F16 to F32
      const count = buffer.byteLength / 2;
      const result = new Float32Array(count);
      const u16 = new Uint16Array(buffer);
      for (let i = 0; i < count; i++) {
        result[i] = halfToFloat(u16[i]);
      }
      return result;
    }
    case 'BF16': {
      // BF16: upper 16 bits of F32
      const count = buffer.byteLength / 2;
      const result = new Float32Array(count);
      const u16 = new Uint16Array(buffer);
      const f32View = new Float32Array(1);
      const u32View = new Uint32Array(f32View.buffer);
      for (let i = 0; i < count; i++) {
        u32View[0] = u16[i] << 16;
        result[i] = f32View[0];
      }
      return result;
    }
    default:
      return new Float32Array(0);
  }
}

// IEEE 754 half-precision to float conversion
function halfToFloat(h: number): number {
  const sign = (h >> 15) & 0x1;
  const exp = (h >> 10) & 0x1F;
  const mant = h & 0x3FF;
  
  if (exp === 0) {
    return (sign ? -1 : 1) * Math.pow(2, -14) * (mant / 1024);
  } else if (exp === 31) {
    return mant ? NaN : (sign ? -Infinity : Infinity);
  }
  
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024);
}

export interface TensorStats {
  name: string;
  shape: number[];
  dtype: string;
  numElements: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  sparsity: number; // % near-zero
  kurtosis: number;
  skewness: number;
  l2Norm: number;
  effectiveRank: number; // estimated
}

function computeStatsFromArray(values: Float32Array | Float64Array, tensor: SafetensorsTensor): TensorStats {
  const n = values.length;
  let sum = 0, sumSq = 0, min = Infinity, max = -Infinity;
  let nearZero = 0;
  
  for (let i = 0; i < n; i++) {
    const v = values[i];
    sum += v;
    sumSq += v * v;
    if (v < min) min = v;
    if (v > max) max = v;
    if (Math.abs(v) < 1e-6) nearZero++;
  }
  
  const mean = sum / n;
  const variance = (sumSq / n) - (mean * mean);
  const std = Math.sqrt(Math.max(0, variance));
  const sparsity = nearZero / n;
  
  // L2 norm
  const l2Norm = Math.sqrt(sumSq);
  
  // Approximate kurtosis and skewness
  let m3 = 0, m4 = 0;
  for (let i = 0; i < n; i++) {
    const d = values[i] - mean;
    m3 += d * d * d;
    m4 += d * d * d * d;
  }
  m3 /= n;
  m4 /= n;
  
  const skewness = std > 0 ? m3 / (std * std * std) : 0;
  const kurtosis = std > 0 ? (m4 / (variance * variance)) - 3 : 0;
  
  // Effective rank estimate (via energy of sorted singular values approximation)
  // For a matrix, we approximate by looking at the distribution of absolute values
  const absValues = Array.from(values).map(Math.abs).sort((a, b) => b - a);
  const totalEnergy = absValues.reduce((s, v) => s + v * v, 0);
  let cumEnergy = 0;
  let effectiveRank = absValues.length;
  for (let i = 0; i < absValues.length; i++) {
    cumEnergy += absValues[i] * absValues[i];
    if (cumEnergy / totalEnergy > 0.99) {
      effectiveRank = i + 1;
      break;
    }
  }
  
  return {
    name: tensor.name,
    shape: tensor.shape,
    dtype: tensor.dtype,
    numElements: tensor.shape.reduce((a, b) => a * b, 1),
    mean,
    std,
    min,
    max,
    sparsity,
    kurtosis,
    skewness,
    l2Norm,
    effectiveRank,
  };
}

// Detect if file is safetensors by checking magic bytes
export async function isSafetensorsFile(file: File): Promise<boolean> {
  const slice = file.slice(0, 8);
  const buffer = await slice.arrayBuffer();
  const view = new DataView(buffer);
  const headerSize = view.getUint32(0, true);
  // Sanity check: header size should be reasonable (< 100MB)
  return headerSize > 0 && headerSize < 100 * 1024 * 1024;
}
