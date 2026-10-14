export interface AIModel {
  id: string;
  name: string;
  developer: string;
  category: ModelCategory;
  totalParams: string;
  activeParams: string;
  paramCount: number;
  architecture: string;
  contextWindow: string;
  weightsSize: string;
  ramRequired: string;
  description: string;
  capabilities: string[];
  huggingFaceUrl: string;
  githubUrl?: string;
  license: string;
  releaseDate: string;
  quantizations?: Quantization[];
  brainMap: BrainMapData;
  extractionMethods: ExtractionMethod[];
  fileStructure: FileStructureEntry[];
}

export interface ExtractionMethod {
  id: string;
  name: string;
  description: string;
  technique: string;
  dataExtracted: string[];
  status: 'available' | 'simulated' | 'requires-access';
  outputFormat: string;
  estimatedSize: string;
}

export interface FileStructureEntry {
  filename: string;
  size: string;
  type: string;
  extractable: boolean;
  description: string;
}

export interface Quantization {
  name: string;
  size: string;
  ramRequired: string;
  url: string;
  quality: string;
}

export interface BrainMapData {
  nodes: BrainNode[];
  connections: BrainConnection[];
  layers: BrainLayer[];
  tensorMap: TensorInfo[];
  weightStats: WeightStats;
}

export interface BrainNode {
  id: string;
  label: string;
  type: 'attention' | 'ffn' | 'embedding' | 'output' | 'moe-expert' | 'router' | 'mamba' | 'norm' | 'memory';
  x: number;
  y: number;
  size: number;
  weight: number;
  activity: number;
  tensorName?: string;
  shape?: string;
  dtype?: string;
}

export interface BrainConnection {
  from: string;
  to: string;
  strength: number;
  type: 'forward' | 'residual' | 'attention' | 'expert-gate';
}

export interface BrainLayer {
  id: string;
  name: string;
  type: string;
  nodeCount: number;
  depth: number;
}

export interface TensorInfo {
  name: string;
  shape: number[];
  dtype: string;
  sizeBytes: number;
  shard: number;
}

export interface WeightStats {
  meanDistribution: number;
  varianceDistribution: number;
  sparsityEstimate: number;
  topKConcentration: number;
  layerVariance: number[];
}

export type ModelCategory =
  | 'frontier-reasoning'
  | 'mixture-of-experts'
  | 'dense-llm'
  | 'multimodal'
  | 'code-specialist'
  | 'lightweight'
  | 'hybrid-architecture';

export type SortOrder = 'largest-first' | 'smallest-first' | 'name-asc' | 'name-desc';
