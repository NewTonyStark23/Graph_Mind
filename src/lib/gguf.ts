// Real GGUF binary format parser
// GGUF format: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

export interface GGUFMetadata {
  [key: string]: any;
}

export interface GGUFTensorInfo {
  name: string;
  nDims: number;
  shape: number[];
  type: number;
  offset: bigint;
}

export interface ParsedGGUF {
  version: number;
  tensorCount: bigint;
  metadataCount: bigint;
  metadata: GGUFMetadata;
  tensors: GGUFTensorInfo[];
  totalParameters: number;
  architecture?: string;
}

// GGUF magic bytes
const GGUF_MAGIC = 0x46554747; // "GGUF" in little-endian

// GGML tensor types
const GGML_TYPE_NAMES: Record<number, string> = {
  0: 'F32',
  1: 'F16',
  2: 'Q4_0',
  3: 'Q4_1',
  6: 'Q5_0',
  7: 'Q5_1',
  8: 'Q8_0',
  9: 'Q8_1',
  10: 'Q2_K',
  11: 'Q3_K',
  12: 'Q4_K',
  13: 'Q5_K',
  14: 'Q6_K',
  15: 'Q8_K',
  16: 'IQ2_XXS',
  17: 'IQ2_XS',
  18: 'IQ3_XXS',
  19: 'IQ1_S',
  20: 'IQ4_NL',
  21: 'IQ3_S',
  22: 'IQ2_S',
  23: 'IQ4_XS',
  24: 'I8',
  25: 'I16',
  26: 'I32',
  27: 'I64',
  28: 'F64',
  29: 'IQ1_M',
};

// Metadata value types
enum GGUFMetadataValueType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
}

export async function parseGGUF(file: File): Promise<ParsedGGUF> {
  // Read first 100MB for header parsing
  const headerSlice = file.slice(0, 100 * 1024 * 1024);
  const buffer = await headerSlice.arrayBuffer();
  const view = new DataView(buffer);
  
  let offset = 0;
  
  // Magic (4 bytes)
  const magic = view.getUint32(offset, true);
  if (magic !== GGUF_MAGIC) {
    throw new Error('Not a valid GGUF file (magic mismatch)');
  }
  offset += 4;
  
  // Version (uint32)
  const version = view.getUint32(offset, true);
  offset += 4;
  
  // Tensor count (uint64)
  const tensorCountLow = view.getUint32(offset, true);
  const tensorCountHigh = view.getUint32(offset + 4, true);
  const tensorCount = BigInt(tensorCountLow) + (BigInt(tensorCountHigh) << 32n);
  offset += 8;
  
  // Metadata KV count (uint64)
  const metadataCountLow = view.getUint32(offset, true);
  const metadataCountHigh = view.getUint32(offset + 4, true);
  const metadataCount = BigInt(metadataCountLow) + (BigInt(metadataCountHigh) << 32n);
  offset += 8;
  
  // Parse metadata
  const metadata: GGUFMetadata = {};
  for (let i = 0; i < Number(metadataCount); i++) {
    const { key, value, newOffset } = readMetadataKV(view, offset);
    metadata[key] = value;
    offset = newOffset;
    
    // Safety: don't parse too many metadata entries
    if (i > 10000) break;
  }
  
  // Parse tensor info
  const tensors: GGUFTensorInfo[] = [];
  let totalParameters = 0;
  
  for (let i = 0; i < Number(tensorCount); i++) {
    // Tensor name (string)
    const nameLength = Number(readUint64(view, offset));
    offset += 8;
    const nameBytes = new Uint8Array(buffer, offset, nameLength);
    const name = new TextDecoder().decode(nameBytes);
    offset += nameLength;
    
    // Align to 32 bytes
    offset = Math.ceil(offset / 32) * 32;
    
    // n_dims (uint32)
    const nDims = view.getUint32(offset, true);
    offset += 4;
    
    // shape (uint64[n_dims])
    const shape: number[] = [];
    for (let d = 0; d < nDims; d++) {
      const dimLow = view.getUint32(offset, true);
      const dimHigh = view.getUint32(offset + 4, true);
      shape.push(Number(BigInt(dimLow) + (BigInt(dimHigh) << 32n)));
      offset += 8;
    }
    
    // type (uint32)
    const type = view.getUint32(offset, true);
    offset += 4;
    
    // offset (uint64)
    const offsetLow = view.getUint32(offset, true);
    const offsetHigh = view.getUint32(offset + 4, true);
    const tensorOffset = BigInt(offsetLow) + (BigInt(offsetHigh) << 32n);
    offset += 8;
    
    const paramCount = shape.reduce((a, b) => a * b, 1);
    totalParameters += paramCount;
    
    tensors.push({
      name,
      nDims,
      shape,
      type,
      offset: tensorOffset,
    });
    
    // Safety: don't parse too many tensors
    if (i > 100000) break;
  }
  
  const architecture = metadata['general.architecture'] || 
                       metadata['general.name'] || 
                       'Unknown';
  
  return {
    version,
    tensorCount,
    metadataCount,
    metadata,
    tensors,
    totalParameters,
    architecture,
  };
}

function readUint64(view: DataView, offset: number): bigint {
  const low = view.getUint32(offset, true);
  const high = view.getUint32(offset + 4, true);
  return BigInt(low) + (BigInt(high) << 32n);
}

function readMetadataKV(view: DataView, offset: number): { key: string; value: any; newOffset: number } {
  // Key (string)
  const keyLength = Number(readUint64(view, offset));
  offset += 8;
  const keyBytes = new Uint8Array(view.buffer, offset, keyLength);
  const key = new TextDecoder().decode(keyBytes);
  offset += keyLength;
  
  // Value type (uint32)
  const valueType = view.getUint32(offset, true) as GGUFMetadataValueType;
  offset += 4;
  
  // Read value based on type
  const { value, newOffset } = readMetadataValue(view, offset, valueType);
  
  return { key, value, newOffset };
}

function readMetadataValue(view: DataView, offset: number, type: GGUFMetadataValueType): { value: any; newOffset: number } {
  switch (type) {
    case GGUFMetadataValueType.UINT8:
      return { value: view.getUint8(offset), newOffset: offset + 1 };
    case GGUFMetadataValueType.INT8:
      return { value: view.getInt8(offset), newOffset: offset + 1 };
    case GGUFMetadataValueType.UINT16:
      return { value: view.getUint16(offset, true), newOffset: offset + 2 };
    case GGUFMetadataValueType.INT16:
      return { value: view.getInt16(offset, true), newOffset: offset + 2 };
    case GGUFMetadataValueType.UINT32:
      return { value: view.getUint32(offset, true), newOffset: offset + 4 };
    case GGUFMetadataValueType.INT32:
      return { value: view.getInt32(offset, true), newOffset: offset + 4 };
    case GGUFMetadataValueType.FLOAT32:
      return { value: view.getFloat32(offset, true), newOffset: offset + 4 };
    case GGUFMetadataValueType.BOOL:
      return { value: view.getUint8(offset) !== 0, newOffset: offset + 1 };
    case GGUFMetadataValueType.STRING: {
      const strLen = Number(readUint64(view, offset));
      offset += 8;
      const strBytes = new Uint8Array(view.buffer, offset, strLen);
      const str = new TextDecoder().decode(strBytes);
      return { value: str, newOffset: offset + strLen };
    }
    case GGUFMetadataValueType.ARRAY: {
      const arrayType = view.getUint32(offset, true) as GGUFMetadataValueType;
      offset += 4;
      const arrayLen = Number(readUint64(view, offset));
      offset += 8;
      
      const array: any[] = [];
      for (let i = 0; i < arrayLen && i < 1000; i++) {
        const { value, newOffset } = readMetadataValue(view, offset, arrayType);
        array.push(value);
        offset = newOffset;
      }
      return { value: array, newOffset: offset };
    }
    case GGUFMetadataValueType.UINT64:
      return { value: readUint64(view, offset), newOffset: offset + 8 };
    case GGUFMetadataValueType.INT64: {
      const low = view.getUint32(offset, true);
      const high = view.getInt32(offset + 4, true);
      return { value: BigInt(low) + (BigInt(high) << 32n), newOffset: offset + 8 };
    }
    case GGUFMetadataValueType.FLOAT64:
      return { value: view.getFloat64(offset, true), newOffset: offset + 8 };
    default:
      return { value: null, newOffset: offset };
  }
}

export async function isGGUFFile(file: File): Promise<boolean> {
  const slice = file.slice(0, 4);
  const buffer = await slice.arrayBuffer();
  const view = new DataView(buffer);
  const magic = view.getUint32(0, true);
  return magic === GGUF_MAGIC;
}

export function getGGMLTypeName(typeId: number): string {
  return GGML_TYPE_NAMES[typeId] || `Unknown(${typeId})`;
}
