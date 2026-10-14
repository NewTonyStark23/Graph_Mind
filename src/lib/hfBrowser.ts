// Real HuggingFace model browser
// Fetches actual model data from HF API

export interface HFModel {
  id: string;
  author: string;
  lastModified: string;
  downloads: number;
  likes: number;
  pipeline_tag?: string;
  tags: string[];
  library_name?: string;
  config?: {
    model_type?: string;
    hidden_size?: number;
    num_hidden_layers?: number;
    num_attention_heads?: number;
    vocab_size?: number;
    architectures?: string[];
  };
}

export async function searchHFModels(
  query: string = '',
  options: {
    sort?: 'downloads' | 'likes' | 'lastModified';
    direction?: 'asc' | 'desc';
    limit?: number;
    filter?: string[];
  } = {}
): Promise<HFModel[]> {
  const { sort = 'downloads', direction = 'desc', limit = 20, filter = [] } = options;
  
  const params = new URLSearchParams();
  if (query) params.append('search', query);
  params.append('sort', sort);
  params.append('direction', direction);
  params.append('limit', limit.toString());
  
  for (const f of filter) {
    params.append('filter', f);
  }
  
  const url = `https://huggingface.co/api/models?${params.toString()}`;
  
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Failed to fetch models:', error);
    return [];
  }
}

export async function getHFModel(modelId: string): Promise<HFModel | null> {
  try {
    const response = await fetch(`https://huggingface.co/api/models/${modelId}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Failed to fetch model ${modelId}:`, error);
    return null;
  }
}

export async function getModelFiles(modelId: string): Promise<Array<{
  name: string;
  size: number;
  type: string;
}>> {
  try {
    const response = await fetch(`https://huggingface.co/api/models/${modelId}/tree/main`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const files = await response.json();
    return files.map((f: any) => ({
      name: f.path || f.rfilename,
      size: f.size || 0,
      type: f.type || 'file',
    }));
  } catch (error) {
    console.error(`Failed to fetch files for ${modelId}:`, error);
    return [];
  }
}

export async function getModelConfig(modelId: string): Promise<any | null> {
  try {
    const response = await fetch(`https://huggingface.co/${modelId}/raw/main/config.json`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Failed to fetch config for ${modelId}:`, error);
    return null;
  }
}

export async function getModelCard(modelId: string): Promise<string | null> {
  try {
    const response = await fetch(`https://huggingface.co/${modelId}/raw/main/README.md`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.text();
  } catch (error) {
    console.error(`Failed to fetch model card for ${modelId}:`, error);
    return null;
  }
}

// Get popular LLM models
export async function getPopularLLMs(limit: number = 20): Promise<HFModel[]> {
  return searchHFModels('', {
    sort: 'downloads',
    direction: 'desc',
    limit,
    filter: ['text-generation'],
  });
}

// Get trending models
export async function getTrendingModels(limit: number = 20): Promise<HFModel[]> {
  return searchHFModels('', {
    sort: 'lastModified',
    direction: 'desc',
    limit,
  });
}

// Search by architecture
export async function getModelsByArchitecture(
  architecture: string,
  limit: number = 20
): Promise<HFModel[]> {
  return searchHFModels(architecture, {
    sort: 'downloads',
    direction: 'desc',
    limit,
  });
}

// Format model size for display
export function formatModelSize(sizeBytes: number): string {
  if (sizeBytes < 1024) return `${sizeBytes} B`;
  if (sizeBytes < 1024 * 1024) return `${(sizeBytes / 1024).toFixed(2)} KB`;
  if (sizeBytes < 1024 * 1024 * 1024) return `${(sizeBytes / (1024 * 1024)).toFixed(2)} MB`;
  return `${(sizeBytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

// Format parameter count
export function formatParamCount(count: number): string {
  if (count < 1e6) return `${(count / 1e3).toFixed(1)}K`;
  if (count < 1e9) return `${(count / 1e6).toFixed(1)}M`;
  if (count < 1e12) return `${(count / 1e9).toFixed(1)}B`;
  return `${(count / 1e12).toFixed(1)}T`;
}
