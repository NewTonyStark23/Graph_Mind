import { AIModel, ExtractionMethod, FileStructureEntry, BrainMapData, TensorInfo, WeightStats } from '../types';

const standardExtractionMethods: ExtractionMethod[] = [
  {
    id: 'safetensors-header',
    name: 'Safetensors Header Parse',
    description: 'Extracts complete tensor metadata from .safetensors files without loading weights.',
    technique: 'Binary header parsing',
    dataExtracted: ['Tensor names', 'Shapes', 'DTypes', 'Byte offsets'],
    status: 'available',
    outputFormat: 'JSON tensor manifest',
    estimatedSize: '~50-200 KB'
  },
  {
    id: 'config-json',
    name: 'Architecture Blueprint',
    description: 'Parses config.json for complete architectural specification.',
    technique: 'JSON parsing',
    dataExtracted: ['hidden_size', 'num_layers', 'num_heads', 'vocab_size'],
    status: 'available',
    outputFormat: 'JSON architecture spec',
    estimatedSize: '~2-10 KB'
  },
  {
    id: 'weight-distribution',
    name: 'Weight Distribution Fingerprint',
    description: 'Computes statistical fingerprints using memory-mapped access.',
    technique: 'Memory-mapped tensor statistics',
    dataExtracted: ['Per-layer means', 'Variances', 'Histograms', 'Sparsity'],
    status: 'available',
    outputFormat: 'Binary fingerprint + JSON',
    estimatedSize: '~1-50 MB'
  },
  {
    id: 'graph-topology',
    name: 'Computation Graph',
    description: 'Reconstructs full computation graph from config + tensor names.',
    technique: 'Config-driven graph reconstruction',
    dataExtracted: ['Node topology', 'Edge connections', 'MoE routing'],
    status: 'available',
    outputFormat: 'Graph JSON',
    estimatedSize: '~100 KB - 2 MB'
  },
  {
    id: 'gguf-metadata',
    name: 'GGUF Metadata',
    description: 'Extracts metadata from GGUF-quantized models.',
    technique: 'GGUF binary format parsing',
    dataExtracted: ['Tokenizer vocab', 'Tensor info', 'KV metadata'],
    status: 'available',
    outputFormat: 'GGUF header dump',
    estimatedSize: '~10-100 MB'
  }
];

function generateFileStructure(paramCount: number, isMoE: boolean): FileStructureEntry[] {
  const shardCount = Math.max(1, Math.ceil(paramCount / 200));
  const files: FileStructureEntry[] = [
    { filename: 'config.json', size: '~5 KB', type: 'json', extractable: true, description: 'Architecture blueprint' },
    { filename: 'tokenizer.json', size: '~15 MB', type: 'json', extractable: true, description: 'Tokenizer vocabulary' },
  ];

  if (shardCount === 1) {
    files.push({
      filename: 'model.safetensors',
      size: `~${(paramCount * 2).toFixed(0)} GB`,
      type: 'safetensors',
      extractable: true,
      description: 'All weights'
    });
  } else {
    files.push({
      filename: 'model.safetensors.index.json',
      size: '~50-500 KB',
      type: 'json',
      extractable: true,
      description: 'Weight map'
    });
    for (let i = 1; i <= Math.min(shardCount, 3); i++) {
      files.push({
        filename: `model-${String(i).padStart(5, '0')}-of-${String(shardCount).padStart(5, '0')}.safetensors`,
        size: `~${((paramCount * 2) / shardCount).toFixed(1)} GB`,
        type: 'safetensors',
        extractable: true,
        description: `Shard ${i}/${shardCount}`
      });
    }
  }

  return files;
}

function generateTensorMap(paramCount: number, isMoE: boolean, layers: number): TensorInfo[] {
  const tensors: TensorInfo[] = [];
  tensors.push({ name: 'model.embed_tokens.weight', shape: [50000, 8192], dtype: 'BF16', sizeBytes: 50000 * 8192 * 2, shard: 0 });
  tensors.push({ name: 'lm_head.weight', shape: [50000, 8192], dtype: 'BF16', sizeBytes: 50000 * 8192 * 2, shard: 0 });

  for (let i = 0; i < Math.min(layers, 8); i++) {
    tensors.push({ name: `model.layers.${i}.self_attn.q_proj.weight`, shape: [8192, 8192], dtype: 'BF16', sizeBytes: 8192 * 8192 * 2, shard: Math.floor(i / 3) });
    tensors.push({ name: `model.layers.${i}.self_attn.k_proj.weight`, shape: [2048, 8192], dtype: 'BF16', sizeBytes: 2048 * 8192 * 2, shard: Math.floor(i / 3) });
    tensors.push({ name: `model.layers.${i}.self_attn.v_proj.weight`, shape: [2048, 8192], dtype: 'BF16', sizeBytes: 2048 * 8192 * 2, shard: Math.floor(i / 3) });
    tensors.push({ name: `model.layers.${i}.self_attn.o_proj.weight`, shape: [8192, 8192], dtype: 'BF16', sizeBytes: 8192 * 8192 * 2, shard: Math.floor(i / 3) });

    if (isMoE) {
      tensors.push({ name: `model.layers.${i}.block_sparse_moe.gate.weight`, shape: [128, 8192], dtype: 'BF16', sizeBytes: 128 * 8192 * 2, shard: Math.floor(i / 3) });
    } else {
      tensors.push({ name: `model.layers.${i}.mlp.gate_proj.weight`, shape: [28672, 8192], dtype: 'BF16', sizeBytes: 28672 * 8192 * 2, shard: Math.floor(i / 3) });
      tensors.push({ name: `model.layers.${i}.mlp.up_proj.weight`, shape: [28672, 8192], dtype: 'BF16', sizeBytes: 28672 * 8192 * 2, shard: Math.floor(i / 3) });
      tensors.push({ name: `model.layers.${i}.mlp.down_proj.weight`, shape: [8192, 28672], dtype: 'BF16', sizeBytes: 8192 * 28672 * 2, shard: Math.floor(i / 3) });
    }

    tensors.push({ name: `model.layers.${i}.input_layernorm.weight`, shape: [8192], dtype: 'BF16', sizeBytes: 8192 * 2, shard: Math.floor(i / 3) });
    tensors.push({ name: `model.layers.${i}.post_attention_layernorm.weight`, shape: [8192], dtype: 'BF16', sizeBytes: 8192 * 2, shard: Math.floor(i / 3) });
  }

  return tensors;
}

function generateBrainMap(architecture: string, layers: number, experts: number = 0, paramCount: number = 10): BrainMapData {
  const nodes: any[] = [];
  const connections: any[] = [];
  const brainLayers: any[] = [];
  const isMoE = experts > 0;

  nodes.push({
    id: 'embedding', label: 'Token Embedding', type: 'embedding',
    x: 50, y: 50, size: 30, weight: 0.8, activity: 0.9,
    tensorName: 'model.embed_tokens.weight', shape: '[50000, 8192]', dtype: 'BF16'
  });
  brainLayers.push({ id: 'l-embed', name: 'Embedding Layer', type: 'embedding', nodeCount: 1, depth: 0 });

  const layerTypes = isMoE ? ['attention', 'router', 'ffn', 'norm'] : ['attention', 'ffn', 'norm'];

  for (let i = 0; i < Math.min(layers, 12); i++) {
    const layerId = `layer-${i}`;
    const yPos = 100 + (i * 70);
    const layerNodes: string[] = [];

    layerTypes.forEach((type, j) => {
      const nodeId = `${layerId}-${type}-${j}`;
      const xPos = 50 + (j * (900 / layerTypes.length));
      nodes.push({
        id: nodeId, label: type.charAt(0).toUpperCase() + type.slice(1),
        type, x: xPos, y: yPos,
        size: type === 'attention' ? 25 : type === 'router' ? 18 : type === 'ffn' ? 22 : 12,
        weight: Math.random() * 0.5 + 0.5,
        activity: Math.random() * 0.7 + 0.3,
        tensorName: `model.layers.${i}.${type}.weight`, shape: '[8192, 8192]', dtype: 'BF16'
      });
      layerNodes.push(nodeId);

      if (j > 0) {
        connections.push({ from: layerNodes[j - 1], to: nodeId, strength: Math.random() * 0.5 + 0.5, type: 'forward' });
      }
    });

    if (i > 0) {
      connections.push({
        from: `layer-${i - 1}-${layerTypes[layerTypes.length - 1]}-${layerTypes.length - 1}`,
        to: layerNodes[0], strength: 0.9, type: 'residual'
      });
    } else {
      connections.push({ from: 'embedding', to: layerNodes[0], strength: 0.95, type: 'forward' });
    }

    brainLayers.push({
      id: layerId,
      name: `${isMoE ? 'MoE' : 'Transformer'} Block ${i + 1}`,
      type: layerTypes.join('+'),
      nodeCount: layerNodes.length,
      depth: i + 1
    });
  }

  const lastY = 100 + (Math.min(layers, 12) * 70) + 50;
  nodes.push({
    id: 'output', label: 'LM Head', type: 'output',
    x: 500, y: lastY, size: 28, weight: 0.9, activity: 1.0,
    tensorName: 'lm_head.weight', shape: '[50000, 8192]', dtype: 'BF16'
  });
  brainLayers.push({ id: 'l-output', name: 'Output Layer', type: 'lm-head', nodeCount: 1, depth: layers + 1 });

  connections.push({
    from: `layer-${Math.min(layers, 12) - 1}-${layerTypes[layerTypes.length - 1]}-${layerTypes.length - 1}`,
    to: 'output', strength: 0.95, type: 'forward'
  });

  const tensorMap = generateTensorMap(paramCount, isMoE, layers);

  const weightStats: WeightStats = {
    meanDistribution: 0.0001,
    varianceDistribution: 0.02,
    sparsityEstimate: isMoE ? 0.85 : 0.15,
    topKConcentration: 0.72,
    layerVariance: Array.from({ length: Math.min(layers, 12) }, () => Math.random() * 0.05 + 0.01)
  };

  return { nodes, connections, layers: brainLayers, tensorMap, weightStats };
}

export const models: AIModel[] = [
  {
    id: 'nemotron-3-ultra',
    name: 'NVIDIA Nemotron-3-Ultra-550B-A55B',
    developer: 'NVIDIA',
    category: 'frontier-reasoning',
    totalParams: '550B',
    activeParams: '55B',
    paramCount: 550,
    architecture: 'MoE Hybrid Mamba-Attention',
    contextWindow: '128K tokens',
    weightsSize: '~1.1 TB (BF16)',
    ramRequired: '~300 GB (NVFP4)',
    description: 'NVIDIA Nemotron 3 Ultra — 550B total / 55B active parameter MoE Hybrid Mamba-Attention model.',
    capabilities: ['Frontier Reasoning', 'Orchestration', 'Code', 'Math', 'Agent Tasks'],
    huggingFaceUrl: 'https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16',
    githubUrl: 'https://github.com/NVIDIA/NeMo',
    license: 'NVIDIA Open Model License',
    releaseDate: 'April 2026',
    quantizations: [
      { name: 'BF16', size: '~1.1 TB', ramRequired: '~1.1 TB', url: 'https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16', quality: 'Full Precision' },
      { name: 'NVFP4', size: '~275 GB', ramRequired: '~300 GB', url: 'https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4', quality: 'NVIDIA Optimized 4-bit' },
    ],
    brainMap: generateBrainMap('MoE Hybrid Mamba-Attention', 64, 64, 550),
    extractionMethods: standardExtractionMethods,
    fileStructure: generateFileStructure(550, true),
  },
  {
    id: 'qwen3-235b-a22b',
    name: 'Qwen3-235B-A22B',
    developer: 'Qwen (Alibaba)',
    category: 'mixture-of-experts',
    totalParams: '235B',
    activeParams: '22B',
    paramCount: 235,
    architecture: 'MoE Transformer',
    contextWindow: '128K tokens',
    weightsSize: '~470 GB (BF16)',
    ramRequired: '~120 GB (Q4)',
    description: 'Qwen3-235B-A22B is the flagship MoE model from Qwen3 series.',
    capabilities: ['Hybrid Thinking', 'MoE Efficiency', 'Code', 'Math', 'Reasoning'],
    huggingFaceUrl: 'https://huggingface.co/Qwen/Qwen3-235B-A22B',
    githubUrl: 'https://github.com/QwenLM/Qwen3',
    license: 'Apache 2.0',
    releaseDate: '2025',
    quantizations: [
      { name: 'BF16', size: '~470 GB', ramRequired: '~470 GB', url: 'https://huggingface.co/Qwen/Qwen3-235B-A22B', quality: 'Full Precision' },
      { name: 'Q4_K_M', size: '~120 GB', ramRequired: '~130 GB', url: 'https://huggingface.co/Qwen/Qwen3-235B-A22B-GGUF', quality: 'High' },
    ],
    brainMap: generateBrainMap('MoE Transformer', 94, 128, 235),
    extractionMethods: standardExtractionMethods,
    fileStructure: generateFileStructure(235, true),
  },
  {
    id: 'deepseek-r1',
    name: 'DeepSeek-R1',
    developer: 'DeepSeek',
    category: 'frontier-reasoning',
    totalParams: '685B',
    activeParams: '37B',
    paramCount: 685,
    architecture: 'MoE Transformer (RL-trained)',
    contextWindow: '128K tokens',
    weightsSize: '~1.37 TB (BF16)',
    ramRequired: '~350 GB (Q4)',
    description: 'DeepSeek-R1 — performance comparable to OpenAI-o1 across math, code, and reasoning.',
    capabilities: ['Chain-of-Thought Reasoning', 'Math', 'Code', 'RL-trained'],
    huggingFaceUrl: 'https://huggingface.co/deepseek-ai/DeepSeek-R1',
    githubUrl: 'https://github.com/deepseek-ai/DeepSeek-R1',
    license: 'MIT',
    releaseDate: 'January 2025',
    quantizations: [
      { name: 'BF16', size: '~1.37 TB', ramRequired: '~1.37 TB', url: 'https://huggingface.co/deepseek-ai/DeepSeek-R1', quality: 'Full Precision' },
      { name: 'Q4_K_M', size: '~350 GB', ramRequired: '~380 GB', url: 'https://huggingface.co/deepseek-ai/DeepSeek-R1', quality: 'High' },
    ],
    brainMap: generateBrainMap('MoE Transformer', 61, 256, 685),
    extractionMethods: standardExtractionMethods,
    fileStructure: generateFileStructure(685, true),
  },
  {
    id: 'llama-4-maverick',
    name: 'Llama 4 Maverick (17B-128E)',
    developer: 'Meta',
    category: 'multimodal',
    totalParams: '402B',
    activeParams: '17B',
    paramCount: 402,
    architecture: 'MoE Transformer (128 Experts)',
    contextWindow: '1M tokens',
    weightsSize: '~804 GB (BF16)',
    ramRequired: '~210 GB (Q4)',
    description: 'Llama 4 Maverick — 402B total parameters with 17B active across 128 experts.',
    capabilities: ['Multimodal', '1M Context', 'Image Understanding', 'Reasoning'],
    huggingFaceUrl: 'https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct',
    githubUrl: 'https://github.com/meta-llama/llama-models',
    license: 'Llama 4 Community License',
    releaseDate: 'April 2025',
    quantizations: [
      { name: 'BF16', size: '~804 GB', ramRequired: '~804 GB', url: 'https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E', quality: 'Full Precision' },
      { name: 'Q4_K_M', size: '~210 GB', ramRequired: '~230 GB', url: 'https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E', quality: 'High' },
    ],
    brainMap: generateBrainMap('MoE Transformer', 48, 128, 402),
    extractionMethods: standardExtractionMethods,
    fileStructure: generateFileStructure(402, true),
  },
  {
    id: 'qwen3-32b',
    name: 'Qwen3-32B',
    developer: 'Qwen (Alibaba)',
    category: 'dense-llm',
    totalParams: '32B',
    activeParams: '32B',
    paramCount: 32,
    architecture: 'Dense Transformer',
    contextWindow: '128K tokens',
    weightsSize: '~64 GB (BF16)',
    ramRequired: '~18 GB (Q4)',
    description: 'Qwen3-32B — dense 32B model with strong reasoning, coding, and multilingual capabilities.',
    capabilities: ['Hybrid Thinking', 'Code', 'Math', 'Multilingual', 'Reasoning'],
    huggingFaceUrl: 'https://huggingface.co/Qwen/Qwen3-32B',
    githubUrl: 'https://github.com/QwenLM/Qwen3',
    license: 'Apache 2.0',
    releaseDate: '2025',
    quantizations: [
      { name: 'BF16', size: '~64 GB', ramRequired: '~64 GB', url: 'https://huggingface.co/Qwen/Qwen3-32B', quality: 'Full Precision' },
      { name: 'Q4_K_M', size: '~18 GB', ramRequired: '~20 GB', url: 'https://huggingface.co/Qwen/Qwen3-32B-GGUF', quality: 'High' },
    ],
    brainMap: generateBrainMap('Dense Transformer', 64, 0, 32),
    extractionMethods: standardExtractionMethods,
    fileStructure: generateFileStructure(32, false),
  },
  {
    id: 'qwen3-8b',
    name: 'Qwen3-8B',
    developer: 'Qwen (Alibaba)',
    category: 'dense-llm',
    totalParams: '8B',
    activeParams: '8B',
    paramCount: 8,
    architecture: 'Dense Transformer',
    contextWindow: '128K tokens',
    weightsSize: '~16 GB (BF16)',
    ramRequired: '~5.5 GB (Q4)',
    description: 'Qwen3-8B — compact powerhouse for local deployment.',
    capabilities: ['Compact', 'Fast', 'Reasoning', 'Code', 'Local Deployment'],
    huggingFaceUrl: 'https://huggingface.co/Qwen/Qwen3-8B',
    githubUrl: 'https://github.com/QwenLM/Qwen3',
    license: 'Apache 2.0',
    releaseDate: '2025',
    quantizations: [
      { name: 'BF16', size: '~16 GB', ramRequired: '~16 GB', url: 'https://huggingface.co/Qwen/Qwen3-8B', quality: 'Full Precision' },
      { name: 'Q4_K_M', size: '~5 GB', ramRequired: '~6 GB', url: 'https://huggingface.co/Qwen/Qwen3-8B-GGUF', quality: 'High' },
    ],
    brainMap: generateBrainMap('Dense Transformer', 32, 0, 8),
    extractionMethods: standardExtractionMethods,
    fileStructure: generateFileStructure(8, false),
  },
];

export const categories: { id: string; label: string; color: string; icon: string }[] = [
  { id: 'frontier-reasoning', label: 'Frontier Reasoning', color: '#6366f1', icon: '🧠' },
  { id: 'mixture-of-experts', label: 'Mixture of Experts', color: '#3b82f6', icon: '⚡' },
  { id: 'dense-llm', label: 'Dense LLM', color: '#22c55e', icon: '🔷' },
  { id: 'multimodal', label: 'Multimodal', color: '#eab308', icon: '👁️' },
  { id: 'lightweight', label: 'Lightweight', color: '#a855f7', icon: '🪶' },
];
