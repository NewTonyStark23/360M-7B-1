// ============================================================
// ELASTIC TRANSFORMER - SINGLE MODEL WITH DYNAMIC SIZING
// ONE model that self-adjusts based on task complexity
// Like a brain: uses more resources when needed, fewer when not
// NOT two models - ONE unified model with elastic internals
// ============================================================

import { Tensor } from './tensor';

export interface ElasticConfig {
  baseDModel: number;      // Minimum dimensions (360M equivalent)
  maxDModel: number;       // Maximum dimensions (7B equivalent)
  baseLayers: number;      // Minimum layers
  maxLayers: number;       // Maximum layers
  baseHeads: number;       // Minimum attention heads
  maxHeads: number;        // Maximum attention heads
  vocabSize: number;
  maxSeqLen: number;
}

export const ELASTIC_CONFIG: ElasticConfig = {
  baseDModel: 1024,        // 360M baseline
  maxDModel: 4096,         // 7B maximum
  baseLayers: 12,          // 12 layers minimum
  maxLayers: 32,           // 32 layers maximum
  baseHeads: 8,            // 8 heads minimum
  maxHeads: 32,            // 32 heads maximum
  vocabSize: 32000,
  maxSeqLen: 8192,
};

export class ElasticTransformer {
  private config: ElasticConfig;
  private weights: Map<string, Tensor>;
  private currentComplexity: number;  // 0.0 (simple) to 1.0 (complex)
  private tokenizer: { encode: (text: string) => number[]; decode: (tokens: number[]) => string };
  
  // Dynamic sizing state
  private activeDModel: number;
  private activeLayers: number;
  private activeHeads: number;

  constructor(config: ElasticConfig = ELASTIC_CONFIG) {
    this.config = config;
    this.weights = new Map();
    this.tokenizer = this.createTokenizer();
    this.currentComplexity = 0.5; // Start in middle
    this.activeDModel = config.baseDModel;
    this.activeLayers = config.baseLayers;
    this.activeHeads = config.baseHeads;
    this.initializeWeights();
  }

  private createTokenizer() {
    return {
      encode: (text: string): number[] => {
        const encoder = new TextEncoder();
        return Array.from(encoder.encode(text));
      },
      decode: (tokens: number[]): string => {
        const decoder = new TextDecoder();
        return decoder.decode(new Uint8Array(tokens));
      },
    };
  }

  private initializeWeights(): void {
    // Initialize weights for MAXIMUM size (7B equivalent)
    // The model will activate subsets based on complexity
    const maxDModel = this.config.maxDModel;
    const maxLayers = this.config.maxLayers;

    // Shared embeddings (always fully active)
    this.weights.set('token_embedding', Tensor.random([this.config.vocabSize, maxDModel], 0.02));
    this.weights.set('position_embedding', Tensor.random([this.config.maxSeqLen, maxDModel], 0.02));

    // ALL layers initialized (model will activate as needed)
    for (let l = 0; l < maxLayers; l++) {
      const prefix = `layer_${l}`;
      
      // Attention weights (full size)
      this.weights.set(`${prefix}_Wq`, Tensor.xavier(maxDModel, maxDModel, [maxDModel, maxDModel]));
      this.weights.set(`${prefix}_Wk`, Tensor.xavier(maxDModel, maxDModel, [maxDModel, maxDModel]));
      this.weights.set(`${prefix}_Wv`, Tensor.xavier(maxDModel, maxDModel, [maxDModel, maxDModel]));
      this.weights.set(`${prefix}_Wo`, Tensor.xavier(maxDModel, maxDModel, [maxDModel, maxDModel]));

      // FFN weights (full size)
      const dFF = maxDModel * 4; // Standard FFN expansion
      this.weights.set(`${prefix}_W1`, Tensor.xavier(maxDModel, dFF, [maxDModel, dFF]));
      this.weights.set(`${prefix}_W2`, Tensor.xavier(dFF, maxDModel, [dFF, maxDModel]));

      // Layer norms
      this.weights.set(`${prefix}_ln1_gamma`, Tensor.ones([maxDModel]));
      this.weights.set(`${prefix}_ln1_beta`, Tensor.zeros([maxDModel]));
      this.weights.set(`${prefix}_ln2_gamma`, Tensor.ones([maxDModel]));
      this.weights.set(`${prefix}_ln2_beta`, Tensor.zeros([maxDModel]));
    }

    // Final layer norm
    this.weights.set('final_ln_gamma', Tensor.ones([maxDModel]));
    this.weights.set('final_ln_beta', Tensor.zeros([maxDModel]));
  }

  // Analyze input complexity and self-adjust
  private analyzeComplexity(inputTokens: number[]): void {
    const contextLength = inputTokens.length;
    const contextText = this.tokenizer.decode(inputTokens);
    
    // Complexity heuristics
    let complexity = 0.0;
    
    // Length factor (longer = more complex)
    complexity += Math.min(contextLength / 512, 0.3);
    
    // Code detection (code = more complex)
    const codeIndicators = ['function', 'const ', 'let ', 'var ', 'class ', 'def ', 'import ', 'export '];
    const hasCode = codeIndicators.some(ind => contextText.includes(ind));
    if (hasCode) complexity += 0.3;
    
    // Reasoning detection (Q&A, logic = more complex)
    const reasoningIndicators = ['Question:', 'Answer:', 'therefore', 'because', 'if ', 'then '];
    const hasReasoning = reasoningIndicators.some(ind => contextText.includes(ind));
    if (hasReasoning) complexity += 0.2;
    
    // Technical terms (technical = more complex)
    const technicalTerms = ['algorithm', 'neural', 'transformer', 'gradient', 'optimization', 'recursion'];
    const hasTechnical = technicalTerms.some(term => contextText.toLowerCase().includes(term));
    if (hasTechnical) complexity += 0.2;
    
    // Clamp to [0, 1]
    this.currentComplexity = Math.max(0, Math.min(1, complexity));
    
    // Self-adjust model size based on complexity
    this.adjustSize(this.currentComplexity);
  }

  // Dynamically adjust model size
  private adjustSize(complexity: number): void {
    const { baseDModel, maxDModel, baseLayers, maxLayers, baseHeads, maxHeads } = this.config;
    
    // Linear interpolation based on complexity
    // complexity 0.0 = base size (360M equivalent)
    // complexity 1.0 = max size (7B equivalent)
    
    this.activeDModel = Math.round(baseDModel + (maxDModel - baseDModel) * complexity);
    this.activeLayers = Math.round(baseLayers + (maxLayers - baseLayers) * complexity);
    this.activeHeads = Math.round(baseHeads + (maxHeads - baseHeads) * complexity);
  }

  // Forward pass with dynamic sizing
  forward(inputTokens: number[]): { logits: Tensor; activeSize: any } {
    // Analyze complexity and self-adjust
    this.analyzeComplexity(inputTokens);
    
    const seqLen = inputTokens.length;
    const activeDModel = this.activeDModel;
    const activeLayers = this.activeLayers;
    const activeHeads = this.activeHeads;
    const dHead = activeDModel / activeHeads;

    // Token embedding (use active dimensions)
    const tokenEmb = this.embedTokens(inputTokens, activeDModel);
    const posEmb = this.getPositionEmbedding(seqLen, activeDModel);
    let x = Tensor.add(tokenEmb, posEmb);

    // Process through ACTIVE layers only
    for (let l = 0; l < activeLayers; l++) {
      // Layer norm 1
      const xNorm1 = this.layerNorm(x, `layer_${l}_ln1`, activeDModel);

      // Multi-head attention (with active dimensions)
      const attnOut = this.selfAttention(
        xNorm1,
        `layer_${l}`,
        activeDModel,
        activeHeads,
        seqLen
      );

      // Residual connection
      x = Tensor.add(x, attnOut);

      // Layer norm 2
      const xNorm2 = this.layerNorm(x, `layer_${l}_ln2`, activeDModel);

      // Feed-forward network (with active dimensions)
      const ffnOut = this.feedForward(xNorm2, `layer_${l}`, activeDModel, seqLen);

      // Residual connection
      x = Tensor.add(x, ffnOut);
    }

    // Final layer norm
    x = this.layerNorm(x, 'final_ln', activeDModel);

    // Output projection (tied with token embedding)
    const tokenEmbWeight = this.weights.get('token_embedding')!;
    const slicedEmbWeight = this.sliceWeight(tokenEmbWeight, [this.config.vocabSize, activeDModel]);
    const logits = Tensor.matmul(x, Tensor.transpose(slicedEmbWeight));

    return {
      logits,
      activeSize: {
        dModel: activeDModel,
        layers: activeLayers,
        heads: activeHeads,
        complexity: this.currentComplexity,
        estimatedParams: this.estimateActiveParams(),
      },
    };
  }

  private embedTokens(tokens: number[], activeDModel: number): Tensor {
    const weight = this.weights.get('token_embedding')!;
    const seqLen = tokens.length;
    const data = new Float32Array(seqLen * activeDModel);

    for (let i = 0; i < seqLen; i++) {
      const tokenIdx = tokens[i] % weight.shape[0];
      for (let j = 0; j < activeDModel; j++) {
        // Use first activeDModel dimensions
        data[i * activeDModel + j] = weight.data[tokenIdx * weight.shape[1] + j];
      }
    }

    return new Tensor(data, [seqLen, activeDModel]);
  }

  private getPositionEmbedding(seqLen: number, activeDModel: number): Tensor {
    const weight = this.weights.get('position_embedding')!;
    const data = new Float32Array(seqLen * activeDModel);

    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < activeDModel; j++) {
        data[i * activeDModel + j] = weight.data[i * weight.shape[1] + j];
      }
    }

    return new Tensor(data, [seqLen, activeDModel]);
  }

  private layerNorm(x: Tensor, prefix: string, activeDModel: number): Tensor {
    const gamma = this.weights.get(`${prefix}_gamma`)!;
    const beta = this.weights.get(`${prefix}_beta`)!;
    
    // Slice to active dimensions
    const slicedGamma = this.sliceWeight(gamma, [activeDModel]);
    const slicedBeta = this.sliceWeight(beta, [activeDModel]);
    
    return Tensor.layerNorm(x, slicedGamma, slicedBeta);
  }

  private selfAttention(
    x: Tensor,
    layerPrefix: string,
    activeDModel: number,
    activeHeads: number,
    seqLen: number
  ): Tensor {
    const dHead = activeDModel / activeHeads;

    // Get weight matrices and slice to active dimensions
    const Wq = this.sliceWeight(this.weights.get(`${layerPrefix}_Wq`)!, [activeDModel, activeDModel]);
    const Wk = this.sliceWeight(this.weights.get(`${layerPrefix}_Wk`)!, [activeDModel, activeDModel]);
    const Wv = this.sliceWeight(this.weights.get(`${layerPrefix}_Wv`)!, [activeDModel, activeDModel]);
    const Wo = this.sliceWeight(this.weights.get(`${layerPrefix}_Wo`)!, [activeDModel, activeDModel]);

    // Project to Q, K, V
    const Q = Tensor.matmul(x, Wq);
    const K = Tensor.matmul(x, Wk);
    const V = Tensor.matmul(x, Wv);

    // Reshape for multi-head
    const Qh = this.reshapeForAttention(Q, activeHeads, dHead, seqLen);
    const Kh = this.reshapeForAttention(K, activeHeads, dHead, seqLen);
    const Vh = this.reshapeForAttention(V, activeHeads, dHead, seqLen);

    // Scaled dot-product attention
    const scale = 1.0 / Math.sqrt(dHead);
    const scores = Tensor.bmm(Qh, Tensor.transpose2d(Kh));
    const scaledScores = Tensor.scale(scores, scale);
    const maskedScores = this.applyCausalMask(scaledScores, seqLen);
    const attn = Tensor.softmax(maskedScores, -1);
    const attnOut = Tensor.bmm(attn, Vh);

    // Reshape back
    const attnReshaped = this.reshapeFromAttention(attnOut, activeHeads, dHead, seqLen);

    // Output projection
    return Tensor.matmul(attnReshaped, Wo);
  }

  private feedForward(x: Tensor, layerPrefix: string, activeDModel: number, seqLen: number): Tensor {
    const dFF = activeDModel * 4;
    
    // Get weights and slice
    const W1 = this.sliceWeight(this.weights.get(`${layerPrefix}_W1`)!, [activeDModel, dFF]);
    const W2 = this.sliceWeight(this.weights.get(`${layerPrefix}_W2`)!, [dFF, activeDModel]);

    const hidden = Tensor.matmul(x, W1);
    const activated = Tensor.gelu(hidden);
    return Tensor.matmul(activated, W2);
  }

  private sliceWeight(weight: Tensor, targetShape: number[]): Tensor {
    // Slice weight tensor to target shape (use first N dimensions)
    if (weight.shape.length !== targetShape.length) {
      throw new Error('Shape mismatch');
    }

    const data = new Float32Array(targetShape.reduce((a, b) => a * b, 1));
    
    if (targetShape.length === 1) {
      // 1D: just copy first N elements
      for (let i = 0; i < targetShape[0]; i++) {
        data[i] = weight.data[i];
      }
    } else if (targetShape.length === 2) {
      // 2D: copy submatrix
      const [rows, cols] = targetShape;
      const srcCols = weight.shape[1];
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          data[i * cols + j] = weight.data[i * srcCols + j];
        }
      }
    }

    return new Tensor(data, targetShape);
  }

  private reshapeForAttention(x: Tensor, nHeads: number, dHead: number, seqLen: number): Tensor {
    const data = new Float32Array(nHeads * seqLen * dHead);
    for (let h = 0; h < nHeads; h++) {
      for (let s = 0; s < seqLen; s++) {
        for (let d = 0; d < dHead; d++) {
          data[h * seqLen * dHead + s * dHead + d] = x.data[s * (nHeads * dHead) + h * dHead + d];
        }
      }
    }
    return new Tensor(data, [nHeads, seqLen, dHead]);
  }

  private reshapeFromAttention(x: Tensor, nHeads: number, dHead: number, seqLen: number): Tensor {
    const dModel = nHeads * dHead;
    const data = new Float32Array(seqLen * dModel);
    for (let h = 0; h < nHeads; h++) {
      for (let s = 0; s < seqLen; s++) {
        for (let d = 0; d < dHead; d++) {
          data[s * dModel + h * dHead + d] = x.data[h * seqLen * dHead + s * dHead + d];
        }
      }
    }
    return new Tensor(data, [seqLen, dModel]);
  }

  private applyCausalMask(scores: Tensor, seqLen: number): Tensor {
    const data = new Float32Array(scores.data.length);
    const nHeads = scores.shape[0];
    for (let h = 0; h < nHeads; h++) {
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
          const idx = h * seqLen * seqLen + i * seqLen + j;
          data[idx] = j > i ? -1e9 : scores.data[idx];
        }
      }
    }
    return new Tensor(data, [...scores.shape]);
  }

  // Estimate active parameter count
  private estimateActiveParams(): number {
    const d = this.activeDModel;
    const l = this.activeLayers;
    const h = this.activeHeads;
    const dFF = d * 4;
    const vocab = this.config.vocabSize;
    
    // Embeddings
    let params = vocab * d + this.config.maxSeqLen * d;
    
    // Per layer
    params += l * (
      d * d * 4 +  // Q, K, V, O projections
      d * dFF * 2 + // FFN up and down
      d * 4          // Layer norms (gamma + beta for 2 norms)
    );
    
    // Final norm
    params += d * 2;
    
    return params;
  }

  // Generate text with automatic complexity adjustment
  generate(prompt: string, maxTokens: number = 50, temperature: number = 0.8): { text: string; stats: any } {
    let tokens = this.tokenizer.encode(prompt);
    const generated: number[] = [];
    const sizeHistory: any[] = [];

    for (let i = 0; i < maxTokens; i++) {
      const { logits, activeSize } = this.forward(tokens);
      sizeHistory.push(activeSize);
      
      const lastLogits = this.getLastRow(logits);
      const nextToken = this.sampleToken(lastLogits, temperature);
      generated.push(nextToken);
      tokens.push(nextToken);

      if (tokens.length > this.config.maxSeqLen) {
        tokens = tokens.slice(-this.config.maxSeqLen);
      }
    }

    const text = this.tokenizer.decode(generated);
    
    // Compute average active size
    const avgDModel = Math.round(sizeHistory.reduce((sum, s) => sum + s.dModel, 0) / sizeHistory.length);
    const avgLayers = Math.round(sizeHistory.reduce((sum, s) => sum + s.layers, 0) / sizeHistory.length);
    const avgHeads = Math.round(sizeHistory.reduce((sum, s) => sum + s.heads, 0) / sizeHistory.length);
    const avgComplexity = sizeHistory.reduce((sum, s) => sum + s.complexity, 0) / sizeHistory.length;
    const avgParams = Math.round(sizeHistory.reduce((sum, s) => sum + s.estimatedParams, 0) / sizeHistory.length);

    return {
      text,
      stats: {
        avgDModel,
        avgLayers,
        avgHeads,
        avgComplexity,
        avgParams,
        sizeRange: {
          min: sizeHistory[0],
          max: sizeHistory[sizeHistory.length - 1],
        },
      },
    };
  }

  private getLastRow(tensor: Tensor): number[] {
    const rows = tensor.shape[0];
    const cols = tensor.shape[1];
    const result: number[] = [];
    for (let j = 0; j < cols; j++) {
      result.push(tensor.data[(rows - 1) * cols + j]);
    }
    return result;
  }

  private sampleToken(logits: number[], temperature: number): number {
    const scaled = logits.map(l => l / temperature);
    const maxVal = Math.max(...scaled);
    const exps = scaled.map(v => Math.exp(v - maxVal));
    const sum = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(e => e / sum);

    const r = Math.random();
    let cum = 0;
    for (let i = 0; i < probs.length; i++) {
      cum += probs[i];
      if (r <= cum) return i;
    }
    return probs.length - 1;
  }

  // Get current model state
  getState(): any {
    return {
      activeDModel: this.activeDModel,
      activeLayers: this.activeLayers,
      activeHeads: this.activeHeads,
      currentComplexity: this.currentComplexity,
      estimatedParams: this.estimateActiveParams(),
      config: this.config,
    };
  }

  // Export weights (full model)
  exportWeights(): ArrayBuffer {
    const tensors: Tensor[] = [];
    
    // Embeddings
    tensors.push(this.weights.get('token_embedding')!);
    tensors.push(this.weights.get('position_embedding')!);
    
    // All layers
    for (let l = 0; l < this.config.maxLayers; l++) {
      const prefix = `layer_${l}`;
      tensors.push(this.weights.get(`${prefix}_Wq`)!);
      tensors.push(this.weights.get(`${prefix}_Wk`)!);
      tensors.push(this.weights.get(`${prefix}_Wv`)!);
      tensors.push(this.weights.get(`${prefix}_Wo`)!);
      tensors.push(this.weights.get(`${prefix}_W1`)!);
      tensors.push(this.weights.get(`${prefix}_W2`)!);
      tensors.push(this.weights.get(`${prefix}_ln1_gamma`)!);
      tensors.push(this.weights.get(`${prefix}_ln1_beta`)!);
      tensors.push(this.weights.get(`${prefix}_ln2_gamma`)!);
      tensors.push(this.weights.get(`${prefix}_ln2_beta`)!);
    }
    
    // Final norm
    tensors.push(this.weights.get('final_ln_gamma')!);
    tensors.push(this.weights.get('final_ln_beta')!);

    // Serialize
    let totalSize = 12;
    for (const t of tensors) {
      totalSize += 4 + t.shape.length * 4 + t.data.length * 4;
    }

    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);
    let offset = 0;

    view.setUint32(offset, tensors.length); offset += 4;
    view.setUint32(offset, this.config.maxDModel); offset += 4;
    view.setUint32(offset, this.config.maxLayers); offset += 4;

    for (const t of tensors) {
      view.setUint32(offset, t.shape.length); offset += 4;
      for (const s of t.shape) {
        view.setUint32(offset, s); offset += 4;
      }
      const dataView = new Float32Array(buffer, offset, t.data.length);
      dataView.set(t.data);
      offset += t.data.length * 4;
    }

    return buffer;
  }

  downloadWeights(filename?: string): void {
    const buffer = this.exportWeights();
    const blob = new Blob([buffer], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || `elastic-transformer-${Date.now()}.bin`;
    a.click();
    URL.revokeObjectURL(url);
  }
}
