// Real tokenizer parser for tokenizer.json files
// Supports BPE, WordPiece, Unigram, and SentencePiece tokenizers

export interface TokenizerInfo {
  type: string;
  vocabSize: number;
  specialTokens: string[];
  addedTokens: string[];
  sampleVocab: Array<{ token: string; id: number }>;
  mergesCount?: number;
  maxTokenLength: number;
  avgTokenLength: number;
}

export interface TokenizerAnalysis {
  info: TokenizerInfo;
  tokenLengthDistribution: Map<number, number>;
  unicodeCoverage: number;
  hasSubwordTokenization: boolean;
  commonPrefixes: string[];
  commonSuffixes: string[];
}

// Parse tokenizer.json file
export async function parseTokenizer(file: File | string): Promise<TokenizerAnalysis> {
  let content: string;
  
  if (typeof file === 'string') {
    content = file;
  } else {
    content = await file.text();
  }
  
  const tokenizer = JSON.parse(content);
  
  const vocab = tokenizer.model.vocab || {};
  const vocabSize = Object.keys(vocab).length;
  const tokenizerType = tokenizer.model.type || 'unknown';
  
  // Extract special tokens
  const specialTokens: string[] = [];
  const addedTokens: string[] = [];
  
  if (tokenizer.added_tokens) {
    for (const token of tokenizer.added_tokens) {
      if (token.special) {
        specialTokens.push(token.content);
      } else {
        addedTokens.push(token.content);
      }
    }
  }
  
  // Sample vocabulary (first 100 tokens)
  const sampleVocab = Object.entries(vocab)
    .slice(0, 100)
    .map(([token, id]) => ({ token, id: id as number }));
  
  // Analyze token lengths
  const tokenLengths = Object.keys(vocab).map(t => t.length);
  const maxTokenLength = Math.max(...tokenLengths);
  const avgTokenLength = tokenLengths.reduce((a, b) => a + b, 0) / tokenLengths.length;
  
  // Token length distribution
  const tokenLengthDistribution = new Map<number, number>();
  for (const len of tokenLengths) {
    tokenLengthDistribution.set(len, (tokenLengthDistribution.get(len) || 0) + 1);
  }
  
  // Check for subword tokenization
  const hasSubwordTokenization = 
    tokenizerType === 'BPE' || 
    tokenizerType === 'WordPiece' ||
    tokenizerType === 'Unigram' ||
    Object.keys(vocab).some(t => t.startsWith('▁') || t.startsWith('##'));
  
  // Find common prefixes and suffixes
  const prefixCounts = new Map<string, number>();
  const suffixCounts = new Map<string, number>();
  
  for (const token of Object.keys(vocab)) {
    if (token.length >= 3) {
      const prefix = token.slice(0, 3);
      const suffix = token.slice(-3);
      prefixCounts.set(prefix, (prefixCounts.get(prefix) || 0) + 1);
      suffixCounts.set(suffix, (suffixCounts.get(suffix) || 0) + 1);
    }
  }
  
  const commonPrefixes = Array.from(prefixCounts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([prefix]) => prefix);
  
  const commonSuffixes = Array.from(suffixCounts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([suffix]) => suffix);
  
  // Unicode coverage estimation
  const unicodeChars = new Set<string>();
  for (const token of Object.keys(vocab)) {
    for (const char of token) {
      unicodeChars.add(char);
    }
  }
  const unicodeCoverage = unicodeChars.size / 1000; // Rough estimate
  
  const info: TokenizerInfo = {
    type: tokenizerType,
    vocabSize,
    specialTokens,
    addedTokens,
    sampleVocab,
    mergesCount: tokenizer.model.merges?.length,
    maxTokenLength,
    avgTokenLength,
  };
  
  return {
    info,
    tokenLengthDistribution,
    unicodeCoverage,
    hasSubwordTokenization,
    commonPrefixes,
    commonSuffixes,
  };
}

// Compare two tokenizers for compatibility
export function compareTokenizers(
  tokenizer1: TokenizerAnalysis,
  tokenizer2: TokenizerAnalysis
): {
  vocabOverlap: number;
  specialTokenMatch: boolean;
  typeMatch: boolean;
  compatibilityScore: number;
} {
  // This would require actual vocab data, simplified here
  const typeMatch = tokenizer1.info.type === tokenizer2.info.type;
  const specialTokenMatch = 
    tokenizer1.info.specialTokens.length === tokenizer2.info.specialTokens.length;
  
  // Rough compatibility estimation
  let compatibilityScore = 0;
  if (typeMatch) compatibilityScore += 0.4;
  if (specialTokenMatch) compatibilityScore += 0.3;
  if (Math.abs(tokenizer1.info.vocabSize - tokenizer2.info.vocabSize) < 1000) {
    compatibilityScore += 0.3;
  }
  
  return {
    vocabOverlap: 0, // Would need actual vocab comparison
    specialTokenMatch,
    typeMatch,
    compatibilityScore,
  };
}

// Estimate tokens per word for a given text
export function estimateTokenization(
  text: string,
  tokenizer: TokenizerAnalysis
): {
  estimatedTokens: number;
  charsPerToken: number;
} {
  const words = text.split(/\s+/);
  const avgWordLength = words.reduce((sum, w) => sum + w.length, 0) / words.length;
  
  // Rough estimation based on average token length
  const charsPerToken = tokenizer.info.avgTokenLength;
  const estimatedTokens = Math.ceil(text.length / charsPerToken);
  
  return {
    estimatedTokens,
    charsPerToken,
  };
}
