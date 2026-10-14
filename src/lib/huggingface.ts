// HuggingFace API client for fetching real model data
// Uses the HuggingFace Hub API: https://huggingface.co/docs/hub/api

export interface HFModelInfo {
  id: string;
  modelId: string;
  sha: string;
  lastModified: string;
  tags: string[];
  pipeline_tag?: string;
  library_name?: string;
  downloads: number;
  likes: number;
  config?: Record<string, any>;
  siblings: Array<{
    rfilename: string;
    size?: number;
  }>;
}

export interface HFConfigJson {
  architectures?: string[];
  model_type?: string;
  hidden_size?: number;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  intermediate_size?: number;
  vocab_size?: number;
  max_position_embeddings?: number;
  rms_norm_eps?: number;
  rope_theta?: number;
  num_experts?: number;
  num_experts_per_token?: number;
  [key: string]: any;
}

export interface HFTokenizerJson {
  version?: string;
  truncation?: any;
  padding?: any;
  added_tokens?: Array<{
    id: number;
    content: string;
    special?: boolean;
  }>;
  normalizer?: any;
  pre_tokenizer?: any;
  post_processor?: any;
  decoder?: any;
  model: {
    type: string;
    vocab: Record<string, number>;
    merges?: string[];
  };
}

const HF_API_BASE = 'https://huggingface.co/api';
const HF_HUB_BASE = 'https://huggingface.co';

// Fetch model info from HuggingFace API
export async function fetchModelInfo(modelId: string): Promise<HFModelInfo> {
  const response = await fetch(`${HF_API_BASE}/models/${modelId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch model info: ${response.statusText}`);
  }
  return response.json();
}

// Fetch config.json from model repository
export async function fetchModelConfig(modelId: string): Promise<HFConfigJson> {
  const response = await fetch(`${HF_HUB_BASE}/${modelId}/raw/main/config.json`);
  if (!response.ok) {
    throw new Error(`Failed to fetch config.json: ${response.statusText}`);
  }
  return response.json();
}

// Fetch tokenizer.json from model repository
export async function fetchTokenizer(modelId: string): Promise<HFTokenizerJson> {
  const response = await fetch(`${HF_HUB_BASE}/${modelId}/raw/main/tokenizer.json`);
  if (!response.ok) {
    throw new Error(`Failed to fetch tokenizer.json: ${response.statusText}`);
  }
  return response.json();
}

// List files in model repository
export async function listModelFiles(modelId: string): Promise<Array<{name: string; size: number}>> {
  const response = await fetch(`${HF_API_BASE}/models/${modelId}/tree/main`);
  if (!response.ok) {
    throw new Error(`Failed to list files: ${response.statusText}`);
  }
  const files = await response.json();
  return files.map((f: any) => ({
    name: f.path,
    size: f.size || 0,
  }));
}

// Search models on HuggingFace
export async function searchModels(query: string, limit: number = 10): Promise<HFModelInfo[]> {
  const response = await fetch(`${HF_API_BASE}/models?search=${encodeURIComponent(query)}&limit=${limit}`);
  if (!response.ok) {
    throw new Error(`Failed to search models: ${response.statusText}`);
  }
  return response.json();
}

// Get model downloads count
export async function getModelDownloads(modelId: string): Promise<number> {
  const info = await fetchModelInfo(modelId);
  return info.downloads;
}

// Check if model exists
export async function modelExists(modelId: string): Promise<boolean> {
  try {
    await fetchModelInfo(modelId);
    return true;
  } catch {
    return false;
  }
}

// Parse architecture from config.json
export function parseArchitectureFromConfig(config: HFConfigJson): {
  type: string;
  isMoE: boolean;
  isMamba: boolean;
  layerCount: number;
  hiddenSize: number;
  attentionHeads: number;
  kvHeads: number;
  experts?: number;
  expertsPerToken?: number;
} {
  const modelType = config.model_type || 'unknown';
  const isMoE = (config.num_experts || 0) > 0;
  const isMamba = modelType.includes('mamba') || modelType.includes('ssm');
  
  return {
    type: modelType,
    isMoE,
    isMamba,
    layerCount: config.num_hidden_layers || 0,
    hiddenSize: config.hidden_size || 0,
    attentionHeads: config.num_attention_heads || 0,
    kvHeads: config.num_key_value_heads || config.num_attention_heads || 0,
    experts: config.num_experts,
    expertsPerToken: config.num_experts_per_token,
  };
}

// Estimate parameter count from config
export function estimateParameterCount(config: HFConfigJson): number {
  const hiddenSize = config.hidden_size || 0;
  const layers = config.num_hidden_layers || 0;
  const vocabSize = config.vocab_size || 0;
  const intermediateSize = config.intermediate_size || hiddenSize * 4;
  const numExperts = config.num_experts || 1;
  
  // Embedding
  let params = vocabSize * hiddenSize * 2; // input + output embeddings
  
  // Per layer
  const perLayer = 
    // Attention (Q, K, V, O)
    (hiddenSize * hiddenSize * 4) +
    // FFN (gate, up, down) or MoE experts
    (hiddenSize * intermediateSize * 3 * numExperts) +
    // Layer norms
    (hiddenSize * 2);
  
  params += perLayer * layers;
  
  return params;
}

// Build real brain map from config.json
export function buildBrainMapFromConfig(config: HFConfigJson): {
  nodes: any[];
  connections: any[];
  layers: any[];
} {
  const arch = parseArchitectureFromConfig(config);
  const nodes: any[] = [];
  const connections: any[] = [];
  const layers: any[] = [];
  
  // Embedding node
  nodes.push({
    id: 'embedding',
    label: 'Token Embedding',
    type: 'embedding',
    x: 200,
    y: 50,
    size: 30,
    weight: 0.9,
    activity: 1.0,
    tensorName: 'model.embed_tokens.weight',
    shape: `[${config.vocab_size}, ${config.hidden_size}]`,
    dtype: 'BF16',
  });
  
  layers.push({
    id: 'l-embed',
    name: 'Embedding Layer',
    type: 'embedding',
    nodeCount: 1,
    depth: 0,
  });
  
  // Transformer layers
  const maxLayers = Math.min(arch.layerCount, 20); // Limit for visualization
  for (let i = 0; i < maxLayers; i++) {
    const yPos = 100 + i * 60;
    
    // Attention node
    nodes.push({
      id: `layer-${i}-attn`,
      label: `Attention (${arch.attentionHeads} heads)`,
      type: 'attention',
      x: 200,
      y: yPos,
      size: 25,
      weight: 0.8,
      activity: 0.9,
      tensorName: `model.layers.${i}.self_attn.q_proj.weight`,
      shape: `[${config.hidden_size}, ${config.hidden_size}]`,
      dtype: 'BF16',
    });
    
    // FFN or MoE node
    if (arch.isMoE && arch.experts) {
      nodes.push({
        id: `layer-${i}-moe`,
        label: `MoE (${arch.experts} experts, top-${arch.expertsPerToken})`,
        type: 'moe-expert',
        x: 500,
        y: yPos,
        size: 28,
        weight: 0.7,
        activity: 0.8,
        tensorName: `model.layers.${i}.block_sparse_moe.gate.weight`,
        shape: `[${arch.experts}, ${config.hidden_size}]`,
        dtype: 'BF16',
      });
    } else {
      nodes.push({
        id: `layer-${i}-ffn`,
        label: `FFN (${config.intermediate_size})`,
        type: 'ffn',
        x: 500,
        y: yPos,
        size: 22,
        weight: 0.7,
        activity: 0.8,
        tensorName: `model.layers.${i}.mlp.gate_proj.weight`,
        shape: `[${config.intermediate_size}, ${config.hidden_size}]`,
        dtype: 'BF16',
      });
    }
    
    // Connect attention to FFN/MoE
    connections.push({
      from: `layer-${i}-attn`,
      to: arch.isMoE ? `layer-${i}-moe` : `layer-${i}-ffn`,
      strength: 0.8,
      type: 'forward',
    });
    
    // Connect to previous layer
    if (i > 0) {
      connections.push({
        from: arch.isMoE ? `layer-${i-1}-moe` : `layer-${i-1}-ffn`,
        to: `layer-${i}-attn`,
        strength: 0.9,
        type: 'residual',
      });
    } else {
      connections.push({
        from: 'embedding',
        to: `layer-${i}-attn`,
        strength: 0.95,
        type: 'forward',
      });
    }
    
    layers.push({
      id: `layer-${i}`,
      name: `Layer ${i}`,
      type: arch.isMoE ? 'moe-transformer' : 'transformer',
      nodeCount: 2,
      depth: i + 1,
    });
  }
  
  // Output node
  const lastY = 100 + maxLayers * 60 + 50;
  nodes.push({
    id: 'output',
    label: 'LM Head',
    type: 'output',
    x: 500,
    y: lastY,
    size: 28,
    weight: 0.9,
    activity: 1.0,
    tensorName: 'lm_head.weight',
    shape: `[${config.vocab_size}, ${config.hidden_size}]`,
    dtype: 'BF16',
  });
  
  layers.push({
    id: 'l-output',
    name: 'Output Layer',
    type: 'lm-head',
    nodeCount: 1,
    depth: arch.layerCount + 1,
  });
  
  // Connect last layer to output
  if (maxLayers > 0) {
    connections.push({
      from: arch.isMoE ? `layer-${maxLayers-1}-moe` : `layer-${maxLayers-1}-ffn`,
      to: 'output',
      strength: 0.95,
      type: 'forward',
    });
  }
  
  return { nodes, connections, layers };
}
