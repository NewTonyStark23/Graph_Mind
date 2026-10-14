// Cross-model comparison tool
// Compares two models' extracted data without running them

import { TensorStats } from './safetensors';
import { SVDResult } from './svd';
import { ModelFingerprint, compareFingerprints } from './fingerprint';

export interface ModelComparison {
  overallCompatibility: number;
  architectureMatch: boolean;
  paramCountDiff: number;
  layerCountDiff: number;
  hiddenSizeMatch: boolean;
  vocabSizeMatch: boolean;
  tensorOverlap: number;
  weightDistributionSimilarity: number;
  svdCorrelation: number;
  mergeRecommendation: 'SAFE' | 'CAUTION' | 'RISKY' | 'IMPOSSIBLE';
  mergeStrategy: string;
  warnings: string[];
}

// Compare two models
export function compareModels(
  fp1: ModelFingerprint,
  fp2: ModelFingerprint,
  stats1: Map<string, TensorStats>,
  stats2: Map<string, TensorStats>,
  svd1: Map<string, SVDResult>,
  svd2: Map<string, SVDResult>
): ModelComparison {
  const warnings: string[] = [];
  
  // Basic architecture checks
  const architectureMatch = fp1.architecture === fp2.architecture;
  const hiddenSizeMatch = fp1.hiddenSize === fp2.hiddenSize;
  const vocabSizeMatch = fp1.vocabSize === fp2.vocabSize;
  
  if (!architectureMatch) {
    warnings.push(`Different architectures: ${fp1.architecture} vs ${fp2.architecture}`);
  }
  if (!hiddenSizeMatch) {
    warnings.push(`Different hidden sizes: ${fp1.hiddenSize} vs ${fp2.hiddenSize}`);
  }
  if (!vocabSizeMatch) {
    warnings.push(`Different vocab sizes: ${fp1.vocabSize} vs ${fp2.vocabSize}`);
  }
  
  // Parameter count difference
  const paramCountDiff = Math.abs(fp1.paramCount - fp2.paramCount) / Math.max(fp1.paramCount, fp2.paramCount);
  
  // Layer count difference
  const layerCountDiff = Math.abs(fp1.layerCount - fp2.layerCount);
  
  // Tensor overlap (how many tensor names match)
  const tensors1 = new Set(stats1.keys());
  const tensors2 = new Set(stats2.keys());
  const overlap = new Set([...tensors1].filter(x => tensors2.has(x)));
  const tensorOverlap = overlap.size / Math.max(tensors1.size, tensors2.size);
  
  if (tensorOverlap < 0.5) {
    warnings.push(`Low tensor overlap: ${(tensorOverlap * 100).toFixed(1)}%`);
  }
  
  // Weight distribution similarity
  let weightSimilarity = 0;
  let comparisonCount = 0;
  
  for (const tensorName of overlap) {
    const stat1 = stats1.get(tensorName);
    const stat2 = stats2.get(tensorName);
    
    if (stat1 && stat2) {
      // Compare mean and std
      const meanDiff = Math.abs(stat1.mean - stat2.mean) / (Math.abs(stat1.mean) + 1e-6);
      const stdDiff = Math.abs(stat1.std - stat2.std) / (stat1.std + 1e-6);
      
      const similarity = 1 - Math.min(1, (meanDiff + stdDiff) / 2);
      weightSimilarity += similarity;
      comparisonCount++;
    }
  }
  
  const weightDistributionSimilarity = comparisonCount > 0 ? weightSimilarity / comparisonCount : 0;
  
  // SVD correlation
  let svdCorrelation = 0;
  let svdComparisonCount = 0;
  
  for (const tensorName of overlap) {
    const svdResult1 = svd1.get(tensorName);
    const svdResult2 = svd2.get(tensorName);
    
    if (svdResult1 && svdResult2) {
      // Compare singular value distributions
      const minLen = Math.min(svdResult1.singularValues.length, svdResult2.singularValues.length);
      if (minLen > 0) {
        const sv1 = svdResult1.singularValues.slice(0, minLen);
        const sv2 = svdResult2.singularValues.slice(0, minLen);
        
        // Normalize
        const sum1 = sv1.reduce((a, b) => a + b, 0);
        const sum2 = sv2.reduce((a, b) => a + b, 0);
        
        if (sum1 > 0 && sum2 > 0) {
          const norm1 = sv1.map(s => s / sum1);
          const norm2 = sv2.map(s => s / sum2);
          
          // Cosine similarity
          let dotProduct = 0;
          let norm1Sq = 0;
          let norm2Sq = 0;
          
          for (let i = 0; i < norm1.length; i++) {
            dotProduct += norm1[i] * norm2[i];
            norm1Sq += norm1[i] * norm1[i];
            norm2Sq += norm2[i] * norm2[i];
          }
          
          const correlation = dotProduct / (Math.sqrt(norm1Sq) * Math.sqrt(norm2Sq));
          svdCorrelation += correlation;
          svdComparisonCount++;
        }
      }
    }
  }
  
  const avgSvdCorrelation = svdComparisonCount > 0 ? svdCorrelation / svdComparisonCount : 0;
  
  // Overall compatibility score
  let overallCompatibility = 0;
  let factors = 0;
  
  if (architectureMatch) { overallCompatibility += 1; factors++; }
  if (hiddenSizeMatch) { overallCompatibility += 1; factors++; }
  if (vocabSizeMatch) { overallCompatibility += 1; factors++; }
  overallCompatibility += (1 - paramCountDiff) * 0.5; factors += 0.5;
  overallCompatibility += tensorOverlap * 0.5; factors += 0.5;
  overallCompatibility += weightDistributionSimilarity; factors++;
  overallCompatibility += avgSvdCorrelation; factors++;
  
  overallCompatibility = factors > 0 ? overallCompatibility / factors : 0;
  
  // Merge recommendation
  let mergeRecommendation: 'SAFE' | 'CAUTION' | 'RISKY' | 'IMPOSSIBLE';
  let mergeStrategy: string;
  
  if (!architectureMatch || !hiddenSizeMatch) {
    mergeRecommendation = 'IMPOSSIBLE';
    mergeStrategy = 'Cannot merge - incompatible architectures';
  } else if (overallCompatibility > 0.85) {
    mergeRecommendation = 'SAFE';
    mergeStrategy = 'SLERP or linear interpolation recommended';
  } else if (overallCompatibility > 0.7) {
    mergeRecommendation = 'CAUTION';
    mergeStrategy = 'TIES-Merging or DARE recommended';
  } else if (overallCompatibility > 0.5) {
    mergeRecommendation = 'RISKY';
    mergeStrategy = 'Frankenmerge (layer-level) only';
  } else {
    mergeRecommendation = 'IMPOSSIBLE';
    mergeStrategy = 'Too different - merge will likely fail';
  }
  
  return {
    overallCompatibility,
    architectureMatch,
    paramCountDiff,
    layerCountDiff,
    hiddenSizeMatch,
    vocabSizeMatch,
    tensorOverlap,
    weightDistributionSimilarity,
    svdCorrelation: avgSvdCorrelation,
    mergeRecommendation,
    mergeStrategy,
    warnings,
  };
}
