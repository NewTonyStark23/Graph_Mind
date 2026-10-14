import { useState, useMemo } from 'react';
import { AIModel, SortOrder } from '../types';
import { models, categories } from '../data/models';
import { deepMethods, mergingStrategies } from '../data/deepMethods';
import ModelDetail from './ModelDetail';
import GraphifyView from './GraphifyView';
import FileAnalyzer from './FileAnalyzer';

export default function ModelPanel() {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [sortOrder, setSortOrder] = useState<SortOrder>('largest-first');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedModel, setSelectedModel] = useState<AIModel | null>(null);
  const [graphifyPreview, setGraphifyPreview] = useState<AIModel | null>(null);
  const [viewMode, setViewMode] = useState<'list' | 'grid'>('list');
  const [showDeepMethods, setShowDeepMethods] = useState(false);
  const [showExtractionGuide, setShowExtractionGuide] = useState(false);
  const [showDownloads, setShowDownloads] = useState(false);
  const [showFileAnalyzer, setShowFileAnalyzer] = useState(false);
  const [downloadedModels, setDownloadedModels] = useState<Set<string>>(new Set());
  const [savedGraphifies, setSavedGraphifies] = useState<Set<string>>(new Set());

  const filteredModels = useMemo(() => {
    let result = [...models];
    if (selectedCategory !== 'all') {
      result = result.filter(m => m.category === selectedCategory);
    }
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      result = result.filter(m =>
        m.name.toLowerCase().includes(q) ||
        m.developer.toLowerCase().includes(q) ||
        m.architecture.toLowerCase().includes(q) ||
        m.capabilities.some(c => c.toLowerCase().includes(q))
      );
    }
    switch (sortOrder) {
      case 'largest-first': result.sort((a, b) => b.paramCount - a.paramCount); break;
      case 'smallest-first': result.sort((a, b) => a.paramCount - b.paramCount); break;
      case 'name-asc': result.sort((a, b) => a.name.localeCompare(b.name)); break;
      case 'name-desc': result.sort((a, b) => b.name.localeCompare(a.name)); break;
    }
    return result;
  }, [selectedCategory, sortOrder, searchQuery]);

  const getCategoryInfo = (catId: string) => categories.find(c => c.id === catId);

  const handleExportGraphify = (model: AIModel) => {
    const graphifyData = {
      version: '2.0',
      format: 'graphify',
      model: model.name,
      developer: model.developer,
      architecture: model.architecture,
      totalParams: model.totalParams,
      activeParams: model.activeParams,
      extraction: {
        methods: model.extractionMethods,
        fileStructure: model.fileStructure,
      },
      brainMap: {
        nodes: model.brainMap.nodes,
        connections: model.brainMap.connections,
        layers: model.brainMap.layers,
        tensorMap: model.brainMap.tensorMap,
        weightStats: model.brainMap.weightStats,
      },
      meta: {
        exported: new Date().toISOString(),
        nodes: model.brainMap.nodes.length,
        connections: model.brainMap.connections.length,
        layers: model.brainMap.layers.length,
        tensors: model.brainMap.tensorMap.length,
        estimatedSize: `${Math.max(1, Math.round(model.paramCount * 0.002))} MB`,
        source: model.huggingFaceUrl,
      }
    };
    const blob = new Blob([JSON.stringify(graphifyData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${model.name.replace(/[^a-zA-Z0-9]/g, '_')}_graphify.json`;
    a.click();
    URL.revokeObjectURL(url);
    setSavedGraphifies(prev => new Set(prev).add(model.id));
  };

  const handleDownloadModel = (model: AIModel) => {
    window.open(model.huggingFaceUrl, '_blank');
    setDownloadedModels(prev => new Set(prev).add(model.id));
  };

  const handleDeleteModel = (modelId: string) => {
    setDownloadedModels(prev => {
      const next = new Set(prev);
      next.delete(modelId);
      return next;
    });
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <header className="sticky top-0 z-40 bg-slate-900/95 backdrop-blur-sm border-b border-slate-700">
        <div className="max-w-[1800px] mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center text-lg font-bold shadow-lg">
                  G
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              </div>
              <div>
                <h1 className="text-lg font-bold text-slate-100">GRAPHIFY PANEL</h1>
                <p className="text-xs text-slate-400">Real AI Model Extractor • Safetensors & GGUF Parser • Weight Statistics • Architecture Analysis</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowExtractionGuide(!showExtractionGuide)}
                className="px-3 py-1.5 bg-indigo-600/20 border border-indigo-500/30 rounded-lg text-xs text-indigo-300 hover:bg-indigo-600/40 transition-colors"
              >
                🔬 Extraction Guide
              </button>
              <button
                onClick={() => setShowDeepMethods(!showDeepMethods)}
                className="px-3 py-1.5 bg-violet-600/20 border border-violet-500/30 rounded-lg text-xs text-violet-300 hover:bg-violet-600/40 transition-colors"
              >
                ⚛️ Deep Analysis
              </button>
              <button
                onClick={() => setShowDownloads(!showDownloads)}
                className="px-3 py-1.5 bg-green-600/20 border border-green-500/30 rounded-lg text-xs text-green-300 hover:bg-green-600/40 transition-colors"
              >
                📥 Downloads ({downloadedModels.size})
              </button>
              <button
                onClick={() => setShowFileAnalyzer(true)}
                className="px-3 py-1.5 bg-blue-600/20 border border-blue-500/30 rounded-lg text-xs text-blue-300 hover:bg-blue-600/40 transition-colors"
              >
                🔬 Analyze File
              </button>
              <div className="text-xs text-slate-400 hidden md:block">
                <span className="text-blue-400 font-mono">{models.length}</span> models indexed
              </div>
              <div className="flex bg-slate-800 rounded-lg border border-slate-700 overflow-hidden">
                <button
                  onClick={() => setViewMode('list')}
                  className={`px-3 py-1.5 text-xs transition-colors ${viewMode === 'list' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}
                >
                  ☰ List
                </button>
                <button
                  onClick={() => setViewMode('grid')}
                  className={`px-3 py-1.5 text-xs transition-colors ${viewMode === 'grid' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}
                >
                  ⊞ Grid
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {showExtractionGuide && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-4xl max-h-[85vh] bg-slate-900 border border-slate-700 rounded-xl overflow-hidden flex flex-col">
            <div className="flex items-center justify-between p-4 border-b border-slate-700">
              <h2 className="text-lg font-bold text-white">🔬 5 Extraction Methods</h2>
              <button onClick={() => setShowExtractionGuide(false)} className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white text-sm rounded-lg">✕ Close</button>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              <div className="bg-gradient-to-r from-indigo-900/30 to-blue-900/30 border border-indigo-500/30 rounded-lg p-4">
                <p className="text-sm text-slate-300 leading-relaxed">
                  These are <span className="text-blue-400 font-semibold">5 real, technically valid methods</span> to extract model architecture data from HuggingFace repositories — <span className="text-green-400">without needing to load the full model into RAM</span>.
                </p>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                  <h3 className="text-sm font-bold text-blue-400 mb-2">① Safetensors Header Parse</h3>
                  <p className="text-xs text-slate-400 mb-2">Parse JSON header from .safetensors files containing all tensor names, shapes, dtypes, and byte offsets.</p>
                  <div className="text-[10px] text-slate-500">Tools: Python struct, safetensors library</div>
                </div>
                <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                  <h3 className="text-sm font-bold text-green-400 mb-2">② config.json Blueprint</h3>
                  <p className="text-xs text-slate-400 mb-2">Parse config.json for complete architectural specification — every hyperparameter, layer count, hidden dimensions.</p>
                  <div className="text-[10px] text-slate-500">Extracts: Full reconstruction blueprint</div>
                </div>
                <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                  <h3 className="text-sm font-bold text-purple-400 mb-2">③ Weight Distribution Fingerprint</h3>
                  <p className="text-xs text-slate-400 mb-2">Compute statistical fingerprints using memory-mapped file access without loading full weights.</p>
                  <div className="text-[10px] text-slate-500">Tools: numpy.memmap, safetensors lazy loading</div>
                </div>
                <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                  <h3 className="text-sm font-bold text-amber-400 mb-2">④ Computation Graph</h3>
                  <p className="text-xs text-slate-400 mb-2">Reconstruct full computation graph from config + tensor naming conventions.</p>
                  <div className="text-[10px] text-slate-500">Extracts: Full topology, connections, MoE routing</div>
                </div>
                <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4 md:col-span-2">
                  <h3 className="text-sm font-bold text-red-400 mb-2">⑤ GGUF Metadata Extraction</h3>
                  <p className="text-xs text-slate-400 mb-2">For GGUF-quantized models, parse binary header containing tokenizer vocabulary, tensor info, all hyperparameters.</p>
                  <div className="text-[10px] text-slate-500">Tools: llama.cpp gguf-dump, gguf-py library</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {showDeepMethods && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-6xl max-h-[90vh] bg-slate-900 border border-slate-700 rounded-xl overflow-hidden flex flex-col">
            <div className="flex items-center justify-between p-4 border-b border-slate-700">
              <h2 className="text-lg font-bold text-white">⚛️ Deep Analysis Methods</h2>
              <button onClick={() => setShowDeepMethods(false)} className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white text-sm rounded-lg">✕ Close</button>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              <div className="bg-gradient-to-r from-indigo-900/30 to-violet-900/30 border border-indigo-500/30 rounded-lg p-4">
                <p className="text-sm text-slate-300 leading-relaxed">
                  These are <span className="text-indigo-400 font-semibold">10 advanced extraction methods</span> for atomic-level model analysis essential for safe merging.
                </p>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {deepMethods.map((method, idx) => (
                  <div key={method.id} className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm font-bold text-indigo-400">{idx + 1}. {method.name}</h3>
                      <span className={`text-[10px] px-2 py-0.5 rounded ${
                        method.difficulty === 'beginner' ? 'bg-green-500/20 text-green-400' :
                        method.difficulty === 'intermediate' ? 'bg-yellow-500/20 text-yellow-400' :
                        method.difficulty === 'advanced' ? 'bg-orange-500/20 text-orange-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        {method.difficulty}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 mb-2">{method.description}</p>
                    <div className="text-[10px] space-y-1">
                      <div className="text-slate-500">Output: <span className="text-amber-400">{method.outputData}</span></div>
                      <div className="text-slate-500">Why: <span className="text-violet-400">{method.relevance}</span></div>
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-6">
                <h3 className="text-lg font-bold text-white mb-3">🔀 Model Merging Strategies</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {mergingStrategies.map((strategy) => (
                    <div key={strategy.name} className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                      <h4 className="text-sm font-bold text-blue-400 mb-2">{strategy.name}</h4>
                      <p className="text-xs text-slate-400 mb-2">{strategy.description}</p>
                      <div className="text-[10px] space-y-1">
                        <div className="text-slate-500">When: <span className="text-green-400">{strategy.when}</span></div>
                        <div className="text-slate-500">Risk: <span className={
                          strategy.risk.includes('Low') ? 'text-green-400' :
                          strategy.risk.includes('Medium') ? 'text-yellow-400' : 'text-red-400'
                        }>{strategy.risk}</span></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {showDownloads && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-4xl max-h-[85vh] bg-slate-900 border border-slate-700 rounded-xl overflow-hidden flex flex-col">
            <div className="flex items-center justify-between p-4 border-b border-slate-700">
              <h2 className="text-lg font-bold text-white">📥 Download Manager</h2>
              <button onClick={() => setShowDownloads(false)} className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white text-sm rounded-lg">✕ Close</button>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {downloadedModels.size === 0 ? (
                <div className="text-center py-12 text-slate-400">
                  <div className="text-4xl mb-3">📭</div>
                  <p>No models downloaded yet</p>
                  <p className="text-xs mt-2">Click "🤗 DL" on any model to download it</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {models.filter(m => downloadedModels.has(m.id)).map(model => (
                    <div key={model.id} className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <h3 className="font-semibold text-white text-sm">{model.name}</h3>
                            {savedGraphifies.has(model.id) && (
                              <span className="px-2 py-0.5 bg-green-500/20 border border-green-500/30 rounded text-[10px] text-green-400">
                                ✓ Graphify Saved
                              </span>
                            )}
                          </div>
                          <div className="text-xs text-slate-400 mt-1">
                            {model.developer} · {model.totalParams} params · {model.weightsSize}
                          </div>
                        </div>
                        <div className="flex gap-2">
                          {!savedGraphifies.has(model.id) && (
                            <button
                              onClick={() => handleExportGraphify(model)}
                              className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs rounded-lg transition-colors"
                            >
                              💾 Save Graphify
                            </button>
                          )}
                          <button
                            onClick={() => {
                              if (confirm(`Delete ${model.name}?`)) {
                                handleDeleteModel(model.id);
                              }
                            }}
                            className="px-3 py-1.5 bg-red-600/20 border border-red-500/30 text-red-400 hover:bg-red-600/40 text-xs rounded-lg transition-colors"
                          >
                            🗑️ Delete
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="max-w-[1800px] mx-auto px-4 py-4">
        <div className="flex flex-col md:flex-row gap-3 mb-4">
          <div className="flex-1 relative">
            <input
              type="text"
              placeholder="Search models, developers, architectures..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/30"
            />
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500">🔍</span>
          </div>
          <select
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value as SortOrder)}
            className="px-4 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:border-blue-500 cursor-pointer"
          >
            <option value="largest-first">Largest First ↓</option>
            <option value="smallest-first">Smallest First ↑</option>
            <option value="name-asc">Name A-Z</option>
            <option value="name-desc">Name Z-A</option>
          </select>
        </div>

        <div className="flex flex-wrap gap-2 mb-4 pb-3 border-b border-slate-800">
          <button
            onClick={() => setSelectedCategory('all')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              selectedCategory === 'all' ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20' : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white'
            }`}
          >
            🌐 All ({models.length})
          </button>
          {categories.map((cat) => {
            const count = models.filter(m => m.category === cat.id).length;
            if (count === 0) return null;
            return (
              <button
                key={cat.id}
                onClick={() => setSelectedCategory(cat.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  selectedCategory === cat.id ? 'text-white shadow-lg' : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-white'
                }`}
                style={selectedCategory === cat.id ? { backgroundColor: cat.color + '33', border: `1px solid ${cat.color}50` } : {}}
              >
                {cat.icon} {cat.label} ({count})
              </button>
            );
          })}
        </div>

        <div className={viewMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3' : 'space-y-2'}>
          {filteredModels.map((model) => {
            const catInfo = getCategoryInfo(model.category);
            return (
              <div
                key={model.id}
                className={`group bg-slate-800/50 border border-slate-700/50 rounded-lg hover:border-slate-600 transition-all hover:bg-slate-800/80 ${viewMode === 'grid' ? 'p-4' : 'p-3'}`}
              >
                <div className={`flex ${viewMode === 'grid' ? 'flex-col' : 'items-center gap-4'}`}>
                  <div className={`${viewMode === 'grid' ? '' : 'flex-1 min-w-0'}`}>
                    <div className="flex items-center gap-2 flex-wrap">
                      <h3 className="font-semibold text-white text-sm truncate">{model.name}</h3>
                      {catInfo && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-medium"
                          style={{ backgroundColor: catInfo.color + '20', color: catInfo.color }}>
                          {catInfo.icon} {catInfo.label}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-3 mt-1 text-xs text-slate-400">
                      <span>{model.developer}</span>
                      <span className="text-slate-600">|</span>
                      <span className="font-mono text-blue-400">{model.totalParams}</span>
                      <span className="text-slate-600">total</span>
                      <span className="text-slate-600">|</span>
                      <span className="font-mono text-green-400">{model.activeParams}</span>
                      <span className="text-slate-600">active</span>
                    </div>
                    {viewMode === 'list' && (
                      <div className="flex items-center gap-3 mt-1 text-xs text-slate-500">
                        <span>📦 {model.weightsSize}</span>
                        <span>💾 RAM: {model.ramRequired}</span>
                      </div>
                    )}
                  </div>
                  <div className={`flex items-center gap-2 ${viewMode === 'grid' ? 'mt-3' : ''}`}>
                    <button
                      onClick={() => setGraphifyPreview(model)}
                      className="px-2.5 py-1.5 bg-indigo-600/20 border border-indigo-500/30 rounded text-xs text-indigo-300 hover:bg-indigo-600/40 transition-colors"
                      title="Preview Brain Map"
                    >
                      🧠 Graphify
                    </button>
                    <button
                      onClick={() => setSelectedModel(model)}
                      className="px-2.5 py-1.5 bg-blue-600/20 border border-blue-500/30 rounded text-xs text-blue-300 hover:bg-blue-600/40 transition-colors"
                      title="Full Details"
                    >
                      🔬 Extract
                    </button>
                    <button
                      onClick={() => handleDownloadModel(model)}
                      className={`px-2.5 py-1.5 rounded text-xs transition-colors ${
                        downloadedModels.has(model.id)
                          ? 'bg-green-600/20 border border-green-500/30 text-green-300 hover:bg-green-600/40'
                          : 'bg-amber-600/20 border border-amber-500/30 text-amber-300 hover:bg-amber-600/40'
                      }`}
                      title={downloadedModels.has(model.id) ? 'Downloaded' : 'Download'}
                    >
                      {downloadedModels.has(model.id) ? '✓ Downloaded' : '🤗 DL'}
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {filteredModels.length === 0 && (
          <div className="text-center py-12 text-slate-500">
            <div className="text-4xl mb-3">🔍</div>
            <p>No models match your search criteria</p>
          </div>
        )}

        <div className="mt-6 pt-4 border-t border-slate-800 grid grid-cols-2 md:grid-cols-6 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-blue-400 font-mono">{models.length}</div>
            <div className="text-xs text-slate-500">Total Models</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-indigo-400 font-mono">
              {(models.reduce((a, m) => a + m.paramCount, 0) / 1000).toFixed(1)}T
            </div>
            <div className="text-xs text-slate-500">Combined Parameters</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-green-400 font-mono">15</div>
            <div className="text-xs text-slate-500">Extraction Methods</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-amber-400 font-mono">
              {models.reduce((a, m) => a + (m.quantizations?.length || 0), 0)}
            </div>
            <div className="text-xs text-slate-500">Quant Variants</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-red-400 font-mono">
              {models.reduce((a, m) => a + m.brainMap.tensorMap.length, 0)}
            </div>
            <div className="text-xs text-slate-500">Tensors Mapped</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-violet-400 font-mono">5</div>
            <div className="text-xs text-slate-500">Merge Strategies</div>
          </div>
        </div>
      </div>

      {graphifyPreview && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-4xl bg-slate-900 border border-slate-700 rounded-xl overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-slate-700">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-indigo-400 animate-pulse"></div>
                <h2 className="text-sm font-bold text-white">Graphify: {graphifyPreview.name}</h2>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => handleExportGraphify(graphifyPreview)}
                  className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-xs rounded-lg transition-colors"
                >
                  💾 Export .graphify
                </button>
                <button
                  onClick={() => setGraphifyPreview(null)}
                  className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white text-xs rounded-lg transition-colors"
                >
                  ✕ Close
                </button>
              </div>
            </div>
            <div className="h-[500px]">
              <GraphifyView brainData={graphifyPreview.brainMap} modelName={graphifyPreview.name} />
            </div>
          </div>
        </div>
      )}

      {selectedModel && (
        <ModelDetail
          model={selectedModel}
          onClose={() => setSelectedModel(null)}
          onExportGraphify={handleExportGraphify}
        />
      )}

      {showFileAnalyzer && (
        <FileAnalyzer onClose={() => setShowFileAnalyzer(false)} />
      )}
    </div>
  );
}
