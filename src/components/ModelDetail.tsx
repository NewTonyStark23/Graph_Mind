import { AIModel } from '../types';
import GraphifyView from './GraphifyView';

interface ModelDetailProps {
  model: AIModel;
  onClose: () => void;
  onExportGraphify: (model: AIModel) => void;
}

export default function ModelDetail({ model, onClose, onExportGraphify }: ModelDetailProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
      <div className="w-full max-w-7xl max-h-[92vh] bg-slate-900 border border-slate-700 rounded-xl overflow-hidden flex flex-col">
        <div className="flex items-center justify-between p-4 border-b border-slate-700 bg-slate-800/50">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 rounded-full bg-blue-400 animate-pulse"></div>
            <h2 className="text-lg font-bold text-white">{model.name}</h2>
            <span className="text-xs px-2 py-0.5 bg-slate-700 rounded text-slate-400">{model.developer}</span>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => onExportGraphify(model)}
              className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded-lg transition-colors"
            >
              📤 Export Graphify
            </button>
            <button
              onClick={onClose}
              className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white text-sm rounded-lg transition-colors"
            >
              ✕ Close
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="space-y-4">
              <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                <h3 className="text-sm font-semibold text-blue-400 mb-3">📊 Model Specifications</h3>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <span className="text-slate-500 text-xs">Total Parameters</span>
                    <div className="text-white font-mono">{model.totalParams}</div>
                  </div>
                  <div>
                    <span className="text-slate-500 text-xs">Active Parameters</span>
                    <div className="text-white font-mono">{model.activeParams}</div>
                  </div>
                  <div>
                    <span className="text-slate-500 text-xs">Architecture</span>
                    <div className="text-white text-xs">{model.architecture}</div>
                  </div>
                  <div>
                    <span className="text-slate-500 text-xs">Context Window</span>
                    <div className="text-white font-mono">{model.contextWindow}</div>
                  </div>
                  <div>
                    <span className="text-slate-500 text-xs">Weights Size</span>
                    <div className="text-white font-mono">{model.weightsSize}</div>
                  </div>
                  <div>
                    <span className="text-slate-500 text-xs">RAM Required</span>
                    <div className="text-white font-mono">{model.ramRequired}</div>
                  </div>
                </div>
              </div>

              <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                <h3 className="text-sm font-semibold text-slate-300 mb-2">📝 Description</h3>
                <p className="text-sm text-slate-300 leading-relaxed">{model.description}</p>
              </div>

              <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                <h3 className="text-sm font-semibold text-slate-300 mb-2">⚡ Capabilities</h3>
                <div className="flex flex-wrap gap-2">
                  {model.capabilities.map((cap) => (
                    <span key={cap} className="px-2 py-1 bg-slate-700/50 border border-slate-600 rounded text-xs text-slate-300">
                      {cap}
                    </span>
                  ))}
                </div>
              </div>

              <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                <h3 className="text-sm font-semibold text-slate-300 mb-3">🔗 Download Sources</h3>
                <div className="space-y-2">
                  <a
                    href={model.huggingFaceUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 px-3 py-2 bg-yellow-500/10 border border-yellow-500/30 rounded-lg text-sm text-yellow-400 hover:bg-yellow-500/20 transition-colors"
                  >
                    <span>🤗</span>
                    <span className="truncate">HuggingFace</span>
                    <span className="ml-auto text-xs">↗</span>
                  </a>
                  {model.githubUrl && (
                    <a
                      href={model.githubUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 px-3 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-sm text-slate-300 hover:bg-slate-700 transition-colors"
                    >
                      <span>🐙</span>
                      <span className="truncate">GitHub</span>
                      <span className="ml-auto text-xs">↗</span>
                    </a>
                  )}
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-slate-800/50 border border-slate-700 rounded-lg overflow-hidden">
                <div className="p-3 border-b border-slate-700 flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-300">🧠 Graphify Brain Map</h3>
                  <span className="text-xs text-slate-500">1:1 Architecture Render</span>
                </div>
                <div className="h-[450px]">
                  <GraphifyView brainData={model.brainMap} modelName={model.name} />
                </div>
              </div>

              <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                <h3 className="text-sm font-semibold text-slate-300 mb-3">📈 Weight Distribution Stats</h3>
                <div className="space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Mean Distribution</span>
                    <span className="text-blue-400 font-mono">{model.brainMap.weightStats.meanDistribution.toExponential(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Variance Distribution</span>
                    <span className="text-blue-400 font-mono">{model.brainMap.weightStats.varianceDistribution.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Sparsity Estimate</span>
                    <span className="text-amber-400 font-mono">{(model.brainMap.weightStats.sparsityEstimate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Top-K Concentration</span>
                    <span className="text-green-400 font-mono">{(model.brainMap.weightStats.topKConcentration * 100).toFixed(1)}%</span>
                  </div>
                </div>
                <div className="mt-3 flex items-end gap-0.5 h-12">
                  {model.brainMap.weightStats.layerVariance.map((v, i) => (
                    <div
                      key={i}
                      className="flex-1 bg-gradient-to-t from-indigo-600 to-blue-400 rounded-t opacity-80"
                      style={{ height: `${(v / 0.06) * 100}%` }}
                      title={`Layer ${i}: variance ${v.toFixed(4)}`}
                    />
                  ))}
                </div>
                <div className="text-[10px] text-slate-500 mt-1">Layer-wise weight variance distribution</div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                <h3 className="text-sm font-semibold text-slate-300 mb-3">🔬 Extraction Methods</h3>
                <div className="space-y-3">
                  {model.extractionMethods.map((method) => (
                    <div key={method.id} className="border border-slate-700/50 rounded-lg p-3 bg-slate-800/30">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-semibold text-white">{method.name}</span>
                        <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                          method.status === 'available' ? 'bg-green-500/20 text-green-400' :
                          method.status === 'simulated' ? 'bg-yellow-500/20 text-yellow-400' :
                          'bg-red-500/20 text-red-400'
                        }`}>
                          {method.status}
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400 mb-2 leading-relaxed">{method.description}</p>
                      <div className="text-[10px] text-slate-500">
                        <div>Output: <span className="text-amber-400">{method.outputFormat}</span> (~{method.estimatedSize})</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                <h3 className="text-sm font-semibold text-slate-300 mb-3">📁 Repository File Structure</h3>
                <div className="space-y-1 max-h-[250px] overflow-y-auto">
                  {model.fileStructure.map((file) => (
                    <div key={file.filename} className="flex items-center gap-2 text-xs py-1.5 border-b border-slate-700/30">
                      <span className={`w-2 h-2 rounded-full ${file.extractable ? 'bg-green-400' : 'bg-slate-600'}`}></span>
                      <span className="font-mono text-slate-300 truncate flex-1">{file.filename}</span>
                      <span className="text-slate-500 text-[10px]">{file.size}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                <h3 className="text-sm font-semibold text-slate-300 mb-3">🧬 Tensor Map</h3>
                <div className="space-y-1 max-h-[200px] overflow-y-auto text-[10px]">
                  {model.brainMap.tensorMap.slice(0, 15).map((tensor, i) => (
                    <div key={i} className="flex items-center gap-2 py-0.5 border-b border-slate-700/20">
                      <span className="font-mono text-purple-300 truncate flex-1">{tensor.name}</span>
                      <span className="text-blue-400 font-mono">{JSON.stringify(tensor.shape)}</span>
                      <span className="text-amber-400">{tensor.dtype}</span>
                    </div>
                  ))}
                  {model.brainMap.tensorMap.length > 15 && (
                    <div className="text-slate-500 py-1">... and {model.brainMap.tensorMap.length - 15} more tensors</div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
