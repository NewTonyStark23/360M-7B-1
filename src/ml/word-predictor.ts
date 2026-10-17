// ============================================================
// WORD PREDICTOR WITH SELF-CORRECTION
// Automated word prediction that saves tokens and self-learns
// AI approves or swaps predicted words to optimize speed
// ============================================================

export interface Prediction {
  word: string;
  confidence: number;      // 0-1 confidence score
  tokenCount: number;      // Tokens this word uses
  alternatives: string[];  // Alternative predictions
}

export interface PredictionResult {
  predicted: string;
  approved: boolean;       // Did AI approve this prediction?
  swapped: string | null;  // If swapped, what was the replacement?
  tokensSaved: number;     // How many tokens were saved
  correction?: string;     // If self-corrected, what was the correction
}

export class WordPredictor {
  private wordFrequency: Map<string, number>;
  private contextPatterns: Map<string, Map<string, number>>;
  private correctionHistory: Array<{ wrong: string; right: string; context: string }>;
  private vocabulary: Set<string>;
  
  constructor() {
    this.wordFrequency = new Map();
    this.contextPatterns = new Map();
    this.correctionHistory = [];
    this.vocabulary = new Set();
    this.initializeVocabulary();
  }

  private initializeVocabulary(): void {
    // Pre-load common English words
    const commonWords = [
      'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I',
      'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
      'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
      'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
      'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
      'function', 'const', 'let', 'var', 'return', 'if', 'else', 'for', 'while',
      'class', 'import', 'export', 'async', 'await', 'try', 'catch', 'throw',
      'true', 'false', 'null', 'undefined', 'number', 'string', 'boolean',
      'algorithm', 'data', 'structure', 'variable', 'loop', 'condition',
      'neural', 'network', 'machine', 'learning', 'artificial', 'intelligence',
    ];
    
    for (const word of commonWords) {
      this.vocabulary.add(word);
      this.wordFrequency.set(word, Math.random() * 100);
    }
  }

  // Predict next word based on context
  predict(context: string, topK = 5): Prediction[] {
    const words = context.toLowerCase().split(/\s+/);
    const lastWords = words.slice(-3).join(' ');
    
    const predictions: Prediction[] = [];
    
    // Check context patterns
    const patternMap = this.contextPatterns.get(lastWords);
    if (patternMap) {
      const sorted = Array.from(patternMap.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, topK);
      
      for (const [word, count] of sorted) {
        predictions.push({
          word,
          confidence: count / 100,
          tokenCount: this.estimateTokenCount(word),
          alternatives: [],
        });
      }
    }
    
    // Fallback to frequency-based prediction
    if (predictions.length === 0) {
      const sorted = Array.from(this.wordFrequency.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, topK);
      
      for (const [word, freq] of sorted) {
        predictions.push({
          word,
          confidence: freq / 1000,
          tokenCount: this.estimateTokenCount(word),
          alternatives: [],
        });
      }
    }
    
    // Fill alternatives
    for (let i = 0; i < predictions.length; i++) {
      predictions[i].alternatives = predictions
        .filter((_, idx) => idx !== i)
        .map(p => p.word)
        .slice(0, 3);
    }
    
    return predictions;
  }

  // AI approves or swaps prediction
  approveOrSwap(
    prediction: Prediction,
    actualNext: string,
    context: string
  ): PredictionResult {
    const approved = prediction.word === actualNext;
    let swapped: string | null = null;
    let correction: string | undefined;
    
    if (!approved) {
      // Check if prediction is semantically similar
      const similarity = this.calculateSimilarity(prediction.word, actualNext);
      
      if (similarity > 0.7) {
        // Close enough, approve with note
        swapped = actualNext;
      } else {
        // Wrong prediction, swap and self-learn
        swapped = actualNext;
        correction = actualNext;
        
        // Self-learning: record correction
        this.correctionHistory.push({
          wrong: prediction.word,
          right: actualNext,
          context: context.slice(-50),
        });
        
        // Update patterns
        this.learnFromCorrection(context, prediction.word, actualNext);
      }
    }
    
    const tokensSaved = approved ? prediction.tokenCount : 0;
    
    return {
      predicted: prediction.word,
      approved,
      swapped,
      tokensSaved,
      correction,
    };
  }

  // Learn from correction
  private learnFromCorrection(context: string, wrong: string, right: string): void {
    const words = context.toLowerCase().split(/\s+/);
    const lastWords = words.slice(-3).join(' ');
    
    // Decrease confidence in wrong prediction
    const wrongFreq = this.wordFrequency.get(wrong) || 0;
    this.wordFrequency.set(wrong, Math.max(0, wrongFreq - 10));
    
    // Increase confidence in correct word
    const rightFreq = this.wordFrequency.get(right) || 0;
    this.wordFrequency.set(right, rightFreq + 20);
    
    // Update context pattern
    if (!this.contextPatterns.has(lastWords)) {
      this.contextPatterns.set(lastWords, new Map());
    }
    const patternMap = this.contextPatterns.get(lastWords)!;
    
    // Decrease wrong pattern
    const wrongCount = patternMap.get(wrong) || 0;
    patternMap.set(wrong, Math.max(0, wrongCount - 5));
    
    // Increase right pattern
    const rightCount = patternMap.get(right) || 0;
    patternMap.set(right, rightCount + 10);
  }

  // Calculate semantic similarity (simplified)
  private calculateSimilarity(word1: string, word2: string): number {
    // Simple character-level similarity
    const len1 = word1.length;
    const len2 = word2.length;
    const maxLen = Math.max(len1, len2);
    
    if (maxLen === 0) return 1;
    
    let matches = 0;
    for (let i = 0; i < Math.min(len1, len2); i++) {
      if (word1[i] === word2[i]) matches++;
    }
    
    return matches / maxLen;
  }

  // Estimate token count for a word
  private estimateTokenCount(word: string): number {
    // Rough estimation: ~4 characters per token
    return Math.ceil(word.length / 4);
  }

  // Get correction statistics
  getStats(): {
    totalCorrections: number;
    accuracy: number;
    topCorrections: Array<{ wrong: string; right: string; count: number }>;
  } {
    const correctionCounts = new Map<string, number>();
    
    for (const corr of this.correctionHistory) {
      const key = `${corr.wrong}→${corr.right}`;
      correctionCounts.set(key, (correctionCounts.get(key) || 0) + 1);
    }
    
    const topCorrections = Array.from(correctionCounts.entries())
      .map(([key, count]) => {
        const [wrong, right] = key.split('→');
        return { wrong, right, count };
      })
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);
    
    const totalPredictions = this.correctionHistory.length;
    const accuracy = totalPredictions > 0
      ? 1 - (this.correctionHistory.length / (totalPredictions + 100))
      : 0.5;
    
    return {
      totalCorrections: this.correctionHistory.length,
      accuracy,
      topCorrections,
    };
  }

  // Add word to vocabulary
  addWord(word: string, frequency = 10): void {
    this.vocabulary.add(word);
    this.wordFrequency.set(word, frequency);
  }

  // Get vocabulary size
  getVocabularySize(): number {
    return this.vocabulary.size;
  }
}
