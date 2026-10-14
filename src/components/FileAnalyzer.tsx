import { useState, useRef, useCallback, useEffect } from 'react';
import { parseSafetensorsHeader, computeWeightStatistics, isSafetensorsFile, ParsedSafetensors, TensorStats } from '../lib/safetensors';
import { parseGGUF, isGGUFFile, ParsedGGUF, getGGMLTypeName } from '../lib/gguf';
import { fetchModelConfig, fetchTokenizer, parseArchitectureFromConfig, buildBrainMapFromConfig, estimateParameterCount } from '../lib/huggingface';
import { cacheModelExtraction, getCachedByModel, getCacheStats } from '../lib/cache';
import { parseTokenizer, TokenizerAnalysis } from '../lib/tokenizer';
import { computeSVD, SVDResult } from '../lib/svd';
import {
  analyzeAttentionPatterns,
  analyzeMoE,
  analyzeWeightMagnitudes,
  scoreLayerImportance,
  estimateActivationPatterns,
  analyzeGradientFlow,
  calculateMemoryFootprint,
  estimateInferenceSpeed,
  analyzeCompression,
} from '../lib/advancedAnalysis';
import GraphifyView from './GraphifyView';
import { BrainMapData } from '../types';

interface FileAnalyzerProps {
  onClose: () => void;
}

export default function FileAnalyzer({ onClose }: FileAnalyzerProps) {
  const [file, setFile] = useState<File | null>(null);
  const [fileType, setFileType] = useState<'safetensors' | 'gguf' | null>(null);
  const [safetensorsData, setSafetensorsData] = useState<ParsedSafetensors | null>(null);
  const [ggufData, setGgufData] = useState<ParsedGGUF | null>(null);
  const [weightStats, setWeightStats] = useState<Map<string, TensorStats> | null>(null);
  const [svdResults, setSvdResults] = useState<Map<string, SVDResult> | null>(null);
  const [tokenizerData, setTokenizerData] = useState<TokenizerAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'tensors' | 'stats' | 'svd' | 'tokenizer' | 'advanced' | 'brainmap'>('overview');
  const [cacheStats, setCacheStats] = useState<{ count: number; sizeMB: number } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getCacheStats().then(setCacheStats);
  }, []);

  const handleFileSelect = useCallback(async (selectedFile: File) => {
    setError(null);
    setLoading(true);
    setProgress(0);
    setFile(selectedFile);
    
    try {
      // Check file type
      const isSt = await isSafetensorsFile(selectedFile);
      const isGguf = await isGGUFFile(selectedFile);
      
      if (isSt) {
        setFileType('safetensors');
        setProgressMessage('Parsing safetensors header...');
        setProgress(10);
        const parsed = await parseSafetensorsHeader(selectedFile);
        setSafetensorsData(parsed);
        
        // Compute weight statistics
        setProgressMessage('Computing weight statistics...');
        setProgress(30);
        const stats = await computeWeightStatistics(
          selectedFile,
          parsed.tensors,
          (p) => {
            setProgress(30 + p * 30);
          }
        );
        setWeightStats(stats);
        
        // Compute SVD for key tensors
        setProgressMessage('Computing SVD analysis...');
        setProgress(60);
        const svdMap = new Map<string, SVDResult>();
        let processed = 0;
        const tensorsToAnalyze = parsed.tensors.filter(t => t.name.includes('weight') && t.shape.length === 2).slice(0, 20);
        
        for (const tensor of tensorsToAnalyze) {
          try {
            const offset = 8 + tensor.dataOffsets[0];
            const sampleSize = Math.min(tensor.sizeBytes, 2 * 1024 * 1024); // 2MB sample
            const slice = selectedFile.slice(offset, offset + sampleSize);
            const buffer = await slice.arrayBuffer();
            
            // Convert to Float32Array for SVD
            const values = new Float32Array(buffer.byteLength / 4);
            const view = new DataView(buffer);
            for (let i = 0; i < values.length; i++) {
              values[i] = view.getFloat32(i * 4, true);
            }
            
            const svd = computeSVD(values, tensor.shape[0], tensor.shape[1], 50);
            svdMap.set(tensor.name, svd);
          } catch (e) {
            // Skip tensors we can't process
          }
          
          processed++;
          setProgress(60 + (processed / tensorsToAnalyze.length) * 20);
        }
        setSvdResults(svdMap);
        
        // Cache the extraction
        setProgressMessage('Caching results...');
        setProgress(90);
        await cacheModelExtraction(
          selectedFile.name,
          selectedFile.name,
          selectedFile.size,
          {
            tensors: parsed.tensors,
            statistics: Object.fromEntries(stats),
          }
        );
        
        setProgress(100);
        setProgressMessage('Complete!');
      } else if (isGguf) {
        setFileType('gguf');
        setProgressMessage('Parsing GGUF file...');
        setProgress(20);
        const parsed = await parseGGUF(selectedFile);
        setGgufData(parsed);
        setProgress(100);
        setProgressMessage('Complete!');
      } else {
        setError('File format not recognized. Please upload a .safetensors or .gguf file.');
      }
    } catch (err) {
      setError(`Error parsing file: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      handleFileSelect(droppedFile);
    }
  }, [handleFileSelect]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      handleFileSelect(selectedFile);
    }
  }, [handleFileSelect]);

  const exportGraphify = useCallback(() => {
    if (!file || (!safetensorsData && !ggufData)) return;
    
    const graphifyData = {
      version: '4.0',
      format: 'graphify-real-advanced',
      sourceFile: file.name,
      fileSize: file.size,
      extractedAt: new Date().toISOString(),
      safetensors: safetensorsData ? {
        headerSize: safetensorsData.headerSizeBytes,
        totalParameters: safetensorsData.totalParameters,
        architecture: safetensorsData.architecture,
        modelType: safetensorsData.modelType,
        tensors: safetensorsData.tensors.map(t => ({
          name: t.name,
          dtype: t.dtype,
          shape: t.shape,
          sizeBytes: t.sizeBytes,
        })),
      } : null,
      gguf: ggufData ? {
        version: ggufData.version,
        tensorCount: Number(ggufData.tensorCount),
        totalParameters: ggufData.totalParameters,
        architecture: ggufData.architecture,
        metadata: ggufData.metadata,
        tensors: ggufData.tensors.map(t => ({
          name: t.name,
          nDims: t.nDims,
          shape: t.shape,
          type: getGGMLTypeName(t.type),
        })),
      } : null,
      weightStats: weightStats ? Object.fromEntries(weightStats) : null,
      svdResults: svdResults ? Object.fromEntries(svdResults) : null,
    };
    
    const blob = new Blob([JSON.stringify(graphifyData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${file.name.replace(/\.[^.]+$/, '')}_graphify_advanced.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [file, safetensorsData, ggufData, weightStats, svdResults]);

  // Generate real brain map from extracted data
  const generateRealBrainMap = useCallback((): BrainMapData | null => {
    if (!safetensorsData && !ggufData) return null;
    
    const tensors = safetensorsData?.tensors || ggufData?.tensors.map(t => ({
      name: t.name,
      dtype: getGGMLTypeName(t.type),
      shape: t.shape,
      dataOffsets: [0, 0] as [number, number],
      sizeBytes: 0,
    })) || [];
    
    const nodes: any[] = [];
    const connections: any[] = [];
    const layers: any[] = [];
    
    // Extract layer structure from tensor names
    const layerMap = new Map<number, any[]>();
    
    tensors.forEach((tensor) => {
      const match = tensor.name.match(/layers\.(\d+)\./);
      if (match) {
        const layerNum = parseInt(match[1]);
        if (!layerMap.has(layerNum)) {
          layerMap.set(layerNum, []);
        }
        layerMap.get(layerNum)!.push(tensor);
      }
    });
    
    // Create nodes for each layer
    const sortedLayers = Array.from(layerMap.entries()).sort((a, b) => a[0] - b[0]);
    
    sortedLayers.forEach(([layerNum, layerTensors], layerIdx) => {
      const yPos = 100 + layerIdx * 70;
      
      // Attention node
      const attnTensor = layerTensors.find(t => t.name.includes('self_attn') || t.name.includes('attn'));
      if (attnTensor) {
        nodes.push({
          id: `layer-${layerNum}-attn`,
          label: 'Attention',
          type: 'attention',
          x: 200,
          y: yPos,
          size: 25,
          weight: 0.8,
          activity: 0.9,
          tensorName: attnTensor.name,
          shape: JSON.stringify(attnTensor.shape),
          dtype: attnTensor.dtype,
        });
      }
      
      // FFN/MoE node
      const ffnTensor = layerTensors.find(t => t.name.includes('mlp') || t.name.includes('feed_forward'));
      if (ffnTensor) {
        const isMoE = ffnTensor.name.includes('experts') || ffnTensor.name.includes('block_sparse_moe');
        nodes.push({
          id: `layer-${layerNum}-ffn`,
          label: isMoE ? 'MoE' : 'FFN',
          type: isMoE ? 'moe-expert' : 'ffn',
          x: 500,
          y: yPos,
          size: 22,
          weight: 0.7,
          activity: 0.8,
          tensorName: ffnTensor.name,
          shape: JSON.stringify(ffnTensor.shape),
          dtype: ffnTensor.dtype,
        });
        
        if (attnTensor) {
          connections.push({
            from: `layer-${layerNum}-attn`,
            to: `layer-${layerNum}-ffn`,
            strength: 0.8,
            type: 'forward',
          });
        }
      }
      
      if (layerIdx > 0) {
        const prevLayerNum = sortedLayers[layerIdx - 1][0];
        connections.push({
          from: `layer-${prevLayerNum}-ffn`,
          to: `layer-${layerNum}-attn`,
          strength: 0.9,
          type: 'residual',
        });
      }
      
      layers.push({
        id: `layer-${layerNum}`,
        name: `Layer ${layerNum}`,
        type: 'transformer',
        nodeCount: layerTensors.length,
        depth: layerNum,
      });
    });
    
    // Add embedding node
    const embedTensor = tensors.find(t => t.name.includes('embed_tokens') || t.name.includes('wte'));
    if (embedTensor) {
      nodes.unshift({
        id: 'embedding',
        label: 'Embedding',
        type: 'embedding',
        x: 200,
        y: 50,
        size: 30,
        weight: 0.9,
        activity: 1.0,
        tensorName: embedTensor.name,
        shape: JSON.stringify(embedTensor.shape),
        dtype: embedTensor.dtype,
      });
      
      if (sortedLayers.length > 0) {
        connections.push({
          from: 'embedding',
          to: `layer-${sortedLayers[0][0]}-attn`,
          strength: 0.95,
          type: 'forward',
        });
      }
    }
    
    // Add output node
    const outputTensor = tensors.find(t => t.name.includes('lm_head') || t.name === 'output.weight');
    if (outputTensor) {
      nodes.push({
        id: 'output',
        label: 'LM Head',
        type: 'output',
        x: 500,
        y: 100 + sortedLayers.length * 70 + 50,
        size: 28,
        weight: 0.9,
        activity: 1.0,
        tensorName: outputTensor.name,
        shape: JSON.stringify(outputTensor.shape),
        dtype: outputTensor.dtype,
      });
      
      if (sortedLayers.length > 0) {
        const lastLayerNum = sortedLayers[sortedLayers.length - 1][0];
        connections.push({
          from: `layer-${lastLayerNum}-ffn`,
          to: 'output',
          strength: 0.95,
          type: 'forward',
        });
      }
    }
    
    return {
      nodes,
      connections,
      layers,
      tensorMap: tensors.map(t => ({
        name: t.name,
        shape: t.shape,
        dtype: t.dtype,
        sizeBytes: t.sizeBytes || 0,
        shard: 0,
      })),
      weightStats: {
        meanDistribution: 0,
        varianceDistribution: 0,
        sparsityEstimate: 0,
        topKConcentration: 0,
        layerVariance: [],
      },
    };
  }, [safetensorsData, ggufData]);

  const formatBytes = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };

  const formatParams = (params: number): string => {
    if (params < 1e9) return `${(params / 1e6).toFixed(2)}M`;
    if (params < 1e12) return `${(params / 1e9).toFixed(2)}B`;
    return `${(params / 1e12).toFixed(2)}T`;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
      <div className="w-full max-w-7xl max-h-[95vh] bg-slate-900 border border-slate-700 rounded-xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-700 bg-slate-800/50">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 rounded-full bg-blue-400 animate-pulse"></div>
            <h2 className="text-lg font-bold text-white">Advanced Model Extractor</h2>
            {file && <span className="text-xs text-slate-400">{file.name}</span>}
            {cacheStats && (
              <span className="text-xs text-slate-500">Cache: {cacheStats.count} models, {cacheStats.sizeMB.toFixed(1)} MB</span>
            )}
          </div>
          <div className="flex gap-2">
            {file && (safetensorsData || ggufData) && (
              <button
                onClick={exportGraphify}
                className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded-lg transition-colors"
              >
                💾 Export Advanced Graphify
              </button>
            )}
            <button
              onClick={onClose}
              className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white text-sm rounded-lg transition-colors"
            >
              ✕ Close
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {!file ? (
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              className="border-2 border-dashed border-slate-600 rounded-lg p-12 text-center hover:border-blue-500 transition-colors cursor-pointer"
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="text-6xl mb-4">📁</div>
              <h3 className="text-xl font-bold text-white mb-2">Drop Model File Here</h3>
              <p className="text-slate-400 mb-4">Supports .safetensors and .gguf files</p>
              <p className="text-xs text-slate-500">
                Real extraction: parses binary headers, computes SVD, analyzes weights, caches results
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept=".safetensors,.gguf"
                onChange={handleFileInput}
                className="hidden"
              />
            </div>
          ) : loading ? (
            <div className="text-center py-12">
              <div className="text-4xl mb-4 animate-pulse">⚙️</div>
              <h3 className="text-xl font-bold text-white mb-2">{progressMessage}</h3>
              <div className="w-96 mx-auto bg-slate-800 rounded-full h-4 overflow-hidden">
                <div
                  className="bg-blue-500 h-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
              <p className="text-slate-400 mt-2">{progress.toFixed(1)}% complete</p>
            </div>
          ) : error ? (
            <div className="text-center py-12">
              <div className="text-4xl mb-4">❌</div>
              <h3 className="text-xl font-bold text-red-400 mb-2">Extraction Failed</h3>
              <p className="text-slate-400">{error}</p>
              <button
                onClick={() => { setFile(null); setError(null); }}
                className="mt-4 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg"
              >
                Try Again
              </button>
            </div>
          ) : (
            <div>
              {/* Tabs */}
              <div className="flex gap-2 mb-4 border-b border-slate-700 overflow-x-auto">
                {(['overview', 'tensors', 'stats', 'svd', 'advanced', 'brainmap'] as const).map(tab => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`px-4 py-2 text-sm font-medium transition-colors whitespace-nowrap ${
                      activeTab === tab
                        ? 'text-blue-400 border-b-2 border-blue-400'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    {tab === 'overview' && '📊 Overview'}
                    {tab === 'tensors' && '🧬 Tensors'}
                    {tab === 'stats' && '📈 Statistics'}
                    {tab === 'svd' && '🔬 SVD'}
                    {tab === 'advanced' && '⚡ Advanced'}
                    {tab === 'brainmap' && '🧠 Brain Map'}
                  </button>
                ))}
              </div>

              {/* Overview Tab */}
              {activeTab === 'overview' && safetensorsData && (
                <div className="space-y-4">
                  <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                    <h3 className="text-sm font-semibold text-blue-400 mb-3">File Information</h3>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-slate-500">Filename</span>
                        <div className="text-white font-mono">{file?.name}</div>
                      </div>
                      <div>
                        <span className="text-slate-500">File Size</span>
                        <div className="text-white font-mono">{file && formatBytes(file.size)}</div>
                      </div>
                      <div>
                        <span className="text-slate-500">Format</span>
                        <div className="text-white font-mono">{fileType?.toUpperCase()}</div>
                      </div>
                      <div>
                        <span className="text-slate-500">Total Parameters</span>
                        <div className="text-white font-mono">{formatParams(safetensorsData.totalParameters)}</div>
                      </div>
                      <div>
                        <span className="text-slate-500">Architecture</span>
                        <div className="text-white">{safetensorsData.architecture}</div>
                      </div>
                      <div>
                        <span className="text-slate-500">Model Type</span>
                        <div className="text-white">{safetensorsData.modelType}</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Tensors Tab */}
              {activeTab === 'tensors' && safetensorsData && (
                <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                  <h3 className="text-sm font-semibold text-blue-400 mb-3">
                    Tensor Map ({safetensorsData.tensors.length} tensors)
                  </h3>
                  <div className="space-y-1 max-h-[600px] overflow-y-auto text-xs">
                    {safetensorsData.tensors.map((tensor, idx) => (
                      <div key={idx} className="flex items-center gap-2 py-1 border-b border-slate-700/20">
                        <span className="text-purple-300 font-mono flex-1 truncate">{tensor.name}</span>
                        <span className="text-blue-400 font-mono">{JSON.stringify(tensor.shape)}</span>
                        <span className="text-amber-400 font-mono">{tensor.dtype}</span>
                        <span className="text-slate-500 font-mono">{formatBytes(tensor.sizeBytes)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Statistics Tab */}
              {activeTab === 'stats' && weightStats && (
                <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                  <h3 className="text-sm font-semibold text-blue-400 mb-3">
                    Weight Statistics ({weightStats.size} tensors analyzed)
                  </h3>
                  <div className="space-y-1 max-h-[600px] overflow-y-auto text-xs">
                    {Array.from(weightStats.entries()).map(([name, stats]) => (
                      <div key={name} className="border border-slate-700/30 rounded p-2 mb-2">
                        <div className="text-purple-300 font-mono mb-1">{name}</div>
                        <div className="grid grid-cols-4 gap-2 text-[10px]">
                          <div>
                            <span className="text-slate-500">Mean:</span>{' '}
                            <span className="text-white font-mono">{stats.mean.toExponential(2)}</span>
                          </div>
                          <div>
                            <span className="text-slate-500">Std:</span>{' '}
                            <span className="text-white font-mono">{stats.std.toExponential(2)}</span>
                          </div>
                          <div>
                            <span className="text-slate-500">Sparsity:</span>{' '}
                            <span className="text-amber-400 font-mono">{(stats.sparsity * 100).toFixed(2)}%</span>
                          </div>
                          <div>
                            <span className="text-slate-500">Rank:</span>{' '}
                            <span className="text-green-400 font-mono">{stats.effectiveRank}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* SVD Tab */}
              {activeTab === 'svd' && svdResults && (
                <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                  <h3 className="text-sm font-semibold text-blue-400 mb-3">
                    SVD Analysis ({svdResults.size} tensors)
                  </h3>
                  <div className="space-y-2 max-h-[600px] overflow-y-auto text-xs">
                    {Array.from(svdResults.entries()).map(([name, svd]) => (
                      <div key={name} className="border border-slate-700/30 rounded p-2 mb-2">
                        <div className="text-purple-300 font-mono mb-1">{name}</div>
                        <div className="grid grid-cols-3 gap-2 text-[10px]">
                          <div>
                            <span className="text-slate-500">Effective Rank:</span>{' '}
                            <span className="text-green-400 font-mono">{svd.effectiveRank}</span>
                          </div>
                          <div>
                            <span className="text-slate-500">Top-10 Energy:</span>{' '}
                            <span className="text-blue-400 font-mono">{((svd.topKEnergy.get(10) || 0) * 100).toFixed(1)}%</span>
                          </div>
                          <div>
                            <span className="text-slate-500">Top-100 Energy:</span>{' '}
                            <span className="text-blue-400 font-mono">{((svd.topKEnergy.get(100) || 0) * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Advanced Tab */}
              {activeTab === 'advanced' && safetensorsData && weightStats && (
                <div className="space-y-4">
                  <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                    <h3 className="text-sm font-semibold text-blue-400 mb-3">Advanced Analysis</h3>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      {(() => {
                        const magnitude = analyzeWeightMagnitudes(weightStats);
                        return (
                          <>
                            <div>
                              <span className="text-slate-500">Outlier %:</span>{' '}
                              <span className="text-white font-mono">{(magnitude.outlierPercentage * 100).toFixed(2)}%</span>
                            </div>
                            <div>
                              <span className="text-slate-500">Dead Neurons:</span>{' '}
                              <span className="text-white font-mono">{(magnitude.deadNeuronPercentage * 100).toFixed(2)}%</span>
                            </div>
                          </>
                        );
                      })()}
                    </div>
                  </div>
                </div>
              )}

              {/* Brain Map Tab */}
              {activeTab === 'brainmap' && (
                <div className="bg-slate-800/50 border border-slate-700 rounded-lg overflow-hidden">
                  <div className="p-3 border-b border-slate-700">
                    <h3 className="text-sm font-semibold text-blue-400">Real Architecture Graph</h3>
                    <p className="text-xs text-slate-500">Generated from actual tensor structure</p>
                  </div>
                  <div className="h-[600px]">
                    {(() => {
                      const brainMap = generateRealBrainMap();
                      return brainMap ? (
                        <GraphifyView brainData={brainMap} modelName={file?.name || 'Model'} />
                      ) : (
                        <div className="flex items-center justify-center h-full text-slate-500">
                          No brain map data available
                        </div>
                      );
                    })()}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
