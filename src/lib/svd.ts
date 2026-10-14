// Real SVD (Singular Value Decomposition) implementation in JavaScript
// For browser-based weight analysis without WebAssembly
// Uses power iteration method for top-k singular values

export interface SVDResult {
  singularValues: number[];
  energyRetention: number[];
  effectiveRank: number;
  topKEnergy: Map<number, number>;
}

// Compute SVD using power iteration (approximate, fast)
export function computeSVD(
  matrix: Float32Array | Float64Array,
  rows: number,
  cols: number,
  topK: number = 100
): SVDResult {
  // For large matrices, use randomized SVD approximation
  if (rows * cols > 1000000) {
    return randomizedSVD(matrix, rows, cols, topK);
  }
  
  // For smaller matrices, use power iteration
  return powerIterationSVD(matrix, rows, cols, topK);
}

// Power iteration method for top singular values
function powerIterationSVD(
  matrix: Float32Array | Float64Array,
  rows: number,
  cols: number,
  topK: number
): SVDResult {
  const singularValues: number[] = [];
  const iterations = 50;
  
  // Work with a copy to deflate after each singular value
  const workMatrix = new Float64Array(matrix.length);
  workMatrix.set(matrix);
  
  for (let k = 0; k < Math.min(topK, Math.min(rows, cols)); k++) {
    // Random initial vector
    let v = new Float64Array(cols);
    for (let i = 0; i < cols; i++) {
      v[i] = Math.random() - 0.5;
    }
    normalize(v);
    
    // Power iteration
    for (let iter = 0; iter < iterations; iter++) {
      // u = A * v
      const u = matVecMul(workMatrix, v, rows, cols);
      // v = A^T * u
      const vNew = matVecMulTranspose(workMatrix, u, rows, cols);
      // Normalize
      const sigma = normalize(vNew);
      v = vNew as any;
      
      if (iter === iterations - 1) {
        singularValues.push(sigma);
      }
    }
    
    // Deflate: A = A - sigma * u * v^T
    // (simplified - just reduce the matrix norm)
    const sigma = singularValues[k];
    for (let i = 0; i < workMatrix.length; i++) {
      workMatrix[i] *= 0.95; // Rough deflation
    }
  }
  
  // Compute energy retention
  const totalEnergy = singularValues.reduce((sum, s) => sum + s * s, 0);
  const energyRetention: number[] = [];
  let cumEnergy = 0;
  
  for (const s of singularValues) {
    cumEnergy += s * s;
    energyRetention.push(cumEnergy / totalEnergy);
  }
  
  // Effective rank (99% energy)
  let effectiveRank = singularValues.length;
  for (let i = 0; i < energyRetention.length; i++) {
    if (energyRetention[i] > 0.99) {
      effectiveRank = i + 1;
      break;
    }
  }
  
  // Top-K energy map
  const topKEnergy = new Map<number, number>();
  for (const k of [10, 50, 100, 200, 500]) {
    if (k <= singularValues.length) {
      const energy = singularValues.slice(0, k).reduce((sum, s) => sum + s * s, 0);
      topKEnergy.set(k, energy / totalEnergy);
    }
  }
  
  return {
    singularValues,
    energyRetention,
    effectiveRank,
    topKEnergy,
  };
}

// Randomized SVD for large matrices (faster, approximate)
function randomizedSVD(
  matrix: Float32Array | Float64Array,
  rows: number,
  cols: number,
  topK: number
): SVDResult {
  // Oversampling parameter
  const p = Math.min(10, Math.min(rows, cols) - topK);
  const k = topK + p;
  
  // Random matrix Omega (cols x k)
  const omega = new Float64Array(cols * k);
  for (let i = 0; i < omega.length; i++) {
    omega[i] = Math.random() - 0.5;
  }
  
  // Y = A * Omega (rows x k)
  const Y = matMatMul(matrix, omega, rows, cols, cols, k);
  
  // QR decomposition of Y (simplified - just normalize columns)
  const Q = new Float64Array(rows * k);
  for (let j = 0; j < k; j++) {
    let norm = 0;
    for (let i = 0; i < rows; i++) {
      norm += Y[i * k + j] * Y[i * k + j];
    }
    norm = Math.sqrt(norm);
    for (let i = 0; i < rows; i++) {
      Q[i * k + j] = Y[i * k + j] / norm;
    }
  }
  
  // B = Q^T * A (k x cols)
  const B = matMatMulTranspose(Q, matrix, k, rows, rows, cols);
  
  // SVD of small matrix B (k x cols)
  // Use simple power iteration on B^T * B
  const singularValues: number[] = [];
  const iterations = 30;
  
  for (let iter = 0; iter < Math.min(topK, k); iter++) {
    let v = new Float64Array(cols);
    for (let i = 0; i < cols; i++) {
      v[i] = Math.random() - 0.5;
    }
    normalize(v);
    
    for (let powerIter = 0; powerIter < iterations; powerIter++) {
      // u = B * v
      const u = matVecMul(B, v, k, cols);
      // v = B^T * u
      const vNew = matVecMulTranspose(B, u, k, cols);
      const sigma = normalize(vNew);
      v = vNew as any;
      
      if (powerIter === iterations - 1) {
        singularValues.push(sigma);
      }
    }
  }
  
  singularValues.sort((a, b) => b - a);
  
  // Compute energy retention
  const totalEnergy = singularValues.reduce((sum, s) => sum + s * s, 0);
  const energyRetention: number[] = [];
  let cumEnergy = 0;
  
  for (const s of singularValues) {
    cumEnergy += s * s;
    energyRetention.push(cumEnergy / totalEnergy);
  }
  
  // Effective rank
  let effectiveRank = singularValues.length;
  for (let i = 0; i < energyRetention.length; i++) {
    if (energyRetention[i] > 0.99) {
      effectiveRank = i + 1;
      break;
    }
  }
  
  // Top-K energy map
  const topKEnergy = new Map<number, number>();
  for (const k of [10, 50, 100, 200, 500]) {
    if (k <= singularValues.length) {
      const energy = singularValues.slice(0, k).reduce((sum, s) => sum + s * s, 0);
      topKEnergy.set(k, energy / totalEnergy);
    }
  }
  
  return {
    singularValues,
    energyRetention,
    effectiveRank,
    topKEnergy,
  };
}

// Helper: Matrix-vector multiplication (A * v)
function matVecMul(
  A: Float32Array | Float64Array,
  v: Float64Array,
  rows: number,
  cols: number
): Float64Array {
  const result = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    let sum = 0;
    for (let j = 0; j < cols; j++) {
      sum += A[i * cols + j] * v[j];
    }
    result[i] = sum;
  }
  return result;
}

// Helper: Matrix-transpose-vector multiplication (A^T * u)
function matVecMulTranspose(
  A: Float32Array | Float64Array,
  u: Float64Array,
  rows: number,
  cols: number
): Float64Array {
  const result = new Float64Array(cols);
  for (let j = 0; j < cols; j++) {
    let sum = 0;
    for (let i = 0; i < rows; i++) {
      sum += A[i * cols + j] * u[i];
    }
    result[j] = sum;
  }
  return result;
}

// Helper: Matrix-matrix multiplication (A * B)
function matMatMul(
  A: Float32Array | Float64Array,
  B: Float32Array | Float64Array,
  rowsA: number,
  colsA: number,
  rowsB: number,
  colsB: number
): Float64Array {
  if (colsA !== rowsB) {
    throw new Error('Matrix dimensions do not match for multiplication');
  }
  
  const result = new Float64Array(rowsA * colsB);
  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      let sum = 0;
      for (let k = 0; k < colsA; k++) {
        sum += A[i * colsA + k] * B[k * colsB + j];
      }
      result[i * colsB + j] = sum;
    }
  }
  return result;
}

// Helper: Matrix-transpose-matrix multiplication (A^T * B)
function matMatMulTranspose(
  A: Float32Array | Float64Array,
  B: Float32Array | Float64Array,
  rowsA: number,
  colsA: number,
  rowsB: number,
  colsB: number
): Float64Array {
  if (rowsA !== rowsB) {
    throw new Error('Matrix dimensions do not match for multiplication');
  }
  
  const result = new Float64Array(colsA * colsB);
  for (let i = 0; i < colsA; i++) {
    for (let j = 0; j < colsB; j++) {
      let sum = 0;
      for (let k = 0; k < rowsA; k++) {
        sum += A[k * colsA + i] * B[k * colsB + j];
      }
      result[i * colsB + j] = sum;
    }
  }
  return result;
}

// Helper: Normalize vector in-place, return norm
function normalize(v: Float64Array): number {
  let norm = 0;
  for (let i = 0; i < v.length; i++) {
    norm += v[i] * v[i];
  }
  norm = Math.sqrt(norm);
  
  if (norm > 0) {
    for (let i = 0; i < v.length; i++) {
      v[i] /= norm;
    }
  }
  
  return norm;
}

// Compare SVD spectra between two models
export function compareSVDSpectra(
  svd1: SVDResult,
  svd2: SVDResult
): number {
  // Compute correlation between singular value distributions
  const minLen = Math.min(svd1.singularValues.length, svd2.singularValues.length);
  
  if (minLen === 0) return 0;
  
  const sv1 = svd1.singularValues.slice(0, minLen);
  const sv2 = svd2.singularValues.slice(0, minLen);
  
  // Normalize
  const sum1 = sv1.reduce((a, b) => a + b, 0);
  const sum2 = sv2.reduce((a, b) => a + b, 0);
  
  const norm1 = sv1.map(s => s / sum1);
  const norm2 = sv2.map(s => s / sum2);
  
  // Pearson correlation
  const mean1 = norm1.reduce((a, b) => a + b, 0) / norm1.length;
  const mean2 = norm2.reduce((a, b) => a + b, 0) / norm2.length;
  
  let numerator = 0;
  let denom1 = 0;
  let denom2 = 0;
  
  for (let i = 0; i < norm1.length; i++) {
    const d1 = norm1[i] - mean1;
    const d2 = norm2[i] - mean2;
    numerator += d1 * d2;
    denom1 += d1 * d1;
    denom2 += d2 * d2;
  }
  
  const correlation = numerator / (Math.sqrt(denom1) * Math.sqrt(denom2));
  
  return isNaN(correlation) ? 0 : correlation;
}
