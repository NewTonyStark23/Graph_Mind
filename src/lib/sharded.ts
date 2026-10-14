// Real sharded safetensors parser
// Parses model.safetensors.index.json to get full tensor map across shards

export interface ShardInfo {
  shardFile: string;
  tensors: string[];
  totalSizeBytes: number;
}

export interface ShardedModelInfo {
  weightMap: Record<string, string>; // tensor name -> shard file
  shards: ShardInfo[];
  totalTensors: number;
  totalSizeBytes: number;
  architecture?: string;
}

// Fetch and parse safetensors.index.json from HuggingFace
export async function fetchShardIndex(modelId: string): Promise<ShardedModelInfo> {
  // Try both common index file names
  const urls = [
    `https://huggingface.co/${modelId}/raw/main/model.safetensors.index.json`,
    `https://huggingface.co/${modelId}/resolve/main/model.safetensors.index.json`,
  ];
  
  let response: Response | null = null;
  for (const url of urls) {
    try {
      response = await fetch(url);
      if (response.ok) break;
    } catch {
      continue;
    }
  }
  
  if (!response || !response.ok) {
    throw new Error('No safetensors index found - model may not be sharded');
  }
  
  const index = await response.json();
  const weightMap = index.weight_map || {};
  
  // Group tensors by shard
  const shardMap = new Map<string, string[]>();
  for (const [tensor, shard] of Object.entries(weightMap) as [string, string][]) {
    if (!shardMap.has(shard)) {
      shardMap.set(shard, []);
    }
    shardMap.get(shard)!.push(tensor);
  }
  
  const shards: ShardInfo[] = Array.from(shardMap.entries()).map(([file, tensors]) => ({
    shardFile: file,
    tensors,
    totalSizeBytes: 0, // Would need to fetch each shard's header
  }));
  
  return {
    weightMap,
    shards,
    totalTensors: Object.keys(weightMap).length,
    totalSizeBytes: 0,
    architecture: index.metadata?.total_size ? undefined : undefined,
  };
}

// Parse a single shard's header
export async function parseShardHeader(modelId: string, shardFile: string): Promise<{
  tensors: Array<{ name: string; dtype: string; shape: number[] }>;
  totalParams: number;
}> {
  // Fetch first 10MB of shard (header is usually < 1MB)
  const url = `https://huggingface.co/${modelId}/resolve/main/${shardFile}`;
  const response = await fetch(url, {
    headers: { Range: 'bytes=0-10485760' } // First 10MB
  });
  
  if (!response.ok) {
    throw new Error(`Failed to fetch shard: ${shardFile}`);
  }
  
  const buffer = await response.arrayBuffer();
  const view = new DataView(buffer);
  
  // Parse header size (first 8 bytes)
  const headerSizeLow = view.getUint32(0, true);
  const headerSizeHigh = view.getUint32(4, true);
  const headerSize = headerSizeLow + headerSizeHigh * 2 ** 32;
  
  // Parse JSON header
  const headerBytes = new Uint8Array(buffer, 8, headerSize);
  const headerText = new TextDecoder().decode(headerBytes);
  const header = JSON.parse(headerText);
  
  const tensors: Array<{ name: string; dtype: string; shape: number[] }> = [];
  let totalParams = 0;
  
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const tensorInfo = info as any;
    const paramCount = tensorInfo.shape.reduce((a: number, b: number) => a * b, 1);
    tensors.push({
      name,
      dtype: tensorInfo.dtype,
      shape: tensorInfo.shape,
    });
    totalParams += paramCount;
  }
  
  return { tensors, totalParams };
}
