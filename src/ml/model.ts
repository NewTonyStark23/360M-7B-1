// ============================================================
// SMOL-LM Transformer Model
// Decoder-only transformer architecture (like SmolLM/GPT)
// Implements forward pass, backward pass, and weight management
// ============================================================

import { Tensor } from './tensor';
import { Tokenizer } from './tokenizer';

export interface ModelConfig {
  vocabSize: number;
  dModel: number;
  nHeads: number;
  nLayers: number;
  dFF: number;
  maxSeqLen: number;
  dropout: number;
}

export const DEFAULT_CONFIG: ModelConfig = {
  vocabSize: 260, // 256 bytes + 4 special tokens
  dModel: 96,
  nHeads: 4,
  nLayers: 4,
  dFF: 384,
  maxSeqLen: 128,
  dropout: 0.0,
};

export const SMALL_CONFIG: ModelConfig = {
  vocabSize: 260,
  dModel: 64,
  nHeads: 4,
  nLayers: 3,
  dFF: 256,
  maxSeqLen: 128,
  dropout: 0.0,
};

// Cache for storing intermediate values during forward pass (for backprop)
interface ForwardCache {
  // Per-layer caches
  layers: Array<{
    x0: Tensor;           // input to layer
    qkv: Tensor;          // QKV projection output
    q: Tensor;            // queries [nHeads, seqLen, dHead]
    k: Tensor;            // keys
    v: Tensor;            // values
    scores: Tensor;       // attention scores (pre-softmax)
    attn: Tensor;         // attention weights (post-softmax)
    attnOut: Tensor;      // attention output (after reshape)
    attnProj: Tensor;     // after output projection
    x1: Tensor;           // after first residual + layernorm
    ffn1: Tensor;         // FFN intermediate (after first linear)
    ffn1Act: Tensor;      // after activation
    ffn2: Tensor;         // after second linear
    xNorm1: Tensor;       // after first layer norm
    xNorm2: Tensor;       // after second layer norm
  }>;
  // Embedding caches
  tokenEmb: Tensor;
  posEmb: Tensor;
  xEmb: Tensor;           // embedding output (token + pos)
  // Output
  logits: Tensor;
  finalNorm: Tensor;
}

export class SmolLM {
  config: ModelConfig;
  tokenizer: Tokenizer;

  // Weights
  tokenEmbedding!: Tensor;    // [vocabSize, dModel]
  posEmbedding!: Tensor;      // [maxSeqLen, dModel]
  
  // Per-layer weights
  layerNorm1Gamma!: Tensor[]; // [dModel] per layer
  layerNorm1Beta!: Tensor[];  // [dModel] per layer
  Wqkv!: Tensor[];            // [dModel, 3*dModel] per layer
  bqkv!: Tensor[];            // [3*dModel] per layer
  Wo!: Tensor[];              // [dModel, dModel] per layer
  bo!: Tensor[];              // [dModel] per layer
  layerNorm2Gamma!: Tensor[]; // [dModel] per layer
  layerNorm2Beta!: Tensor[];  // [dModel] per layer
  W1!: Tensor[];              // [dModel, dFF] per layer
  b1!: Tensor[];              // [dFF] per layer
  W2!: Tensor[];              // [dFF, dModel] per layer
  b2!: Tensor[];              // [dModel] per layer
  
  // Final layer norm
  finalNormGamma!: Tensor;    // [dModel]
  finalNormBeta!: Tensor;     // [dModel]

  // Output projection (tied with token embedding)
  // outputProj = tokenEmbedding.T

  constructor(config: ModelConfig = DEFAULT_CONFIG) {
    this.config = config;
    this.tokenizer = new Tokenizer(config.vocabSize);
    this.initializeWeights();
  }

  initializeWeights(): void {
    const { vocabSize, dModel, nLayers, dFF, maxSeqLen, nHeads } = this.config;
    const dHead = dModel / nHeads;

    // Embeddings
    this.tokenEmbedding = Tensor.xavier(vocabSize, dModel, [vocabSize, dModel]);
    this.posEmbedding = Tensor.xavier(maxSeqLen, dModel, [maxSeqLen, dModel]);

    // Per-layer weights
    this.layerNorm1Gamma = [];
    this.layerNorm1Beta = [];
    this.Wqkv = [];
    this.bqkv = [];
    this.Wo = [];
    this.bo = [];
    this.layerNorm2Gamma = [];
    this.layerNorm2Beta = [];
    this.W1 = [];
    this.b1 = [];
    this.W2 = [];
    this.b2 = [];

    for (let l = 0; l < nLayers; l++) {
      // Layer Norm 1 (pre-attention)
      this.layerNorm1Gamma.push(Tensor.ones([dModel]));
      this.layerNorm1Beta.push(Tensor.zeros([dModel]));

      // QKV projection (combined for efficiency)
      this.Wqkv.push(Tensor.xavier(dModel, 3 * dModel, [dModel, 3 * dModel]));
      this.bqkv.push(Tensor.zeros([3 * dModel]));

      // Attention output projection
      this.Wo.push(Tensor.xavier(dModel, dModel, [dModel, dModel]));
      this.bo.push(Tensor.zeros([dModel]));

      // Layer Norm 2 (pre-FFN)
      this.layerNorm2Gamma.push(Tensor.ones([dModel]));
      this.layerNorm2Beta.push(Tensor.zeros([dModel]));

      // FFN
      this.W1.push(Tensor.xavier(dModel, dFF, [dModel, dFF]));
      this.b1.push(Tensor.zeros([dFF]));
      this.W2.push(Tensor.xavier(dFF, dModel, [dFF, dModel]));
      this.b2.push(Tensor.zeros([dModel]));
    }

    // Final layer norm
    this.finalNormGamma = Tensor.ones([dModel]);
    this.finalNormBeta = Tensor.zeros([dModel]);
  }

  // ============ FORWARD PASS ============

  forward(inputTokens: number[]): { logits: Tensor; cache: ForwardCache } {
    const { dModel, nHeads, maxSeqLen } = this.config;
    const dHead = dModel / nHeads;
    const seqLen = inputTokens.length;

    if (seqLen > maxSeqLen) {
      throw new Error(`Sequence length ${seqLen} exceeds max ${maxSeqLen}`);
    }

    const cache: ForwardCache = { layers: [], tokenEmb: null as any, posEmb: null as any, xEmb: null as any, logits: null as any, finalNorm: null as any };

    // 1. Token Embedding
    const tokEmb = Tensor.embedding(this.tokenEmbedding, inputTokens);
    cache.tokenEmb = tokEmb;

    // 2. Position Embedding (add sinusoidal or learned)
    const posEmb = new Tensor(
      this.posEmbedding.data.slice(0, seqLen * dModel),
      [seqLen, dModel]
    );
    cache.posEmb = posEmb;

    // x = tokenEmb + posEmb
    let x = Tensor.add(tokEmb, posEmb);
    cache.xEmb = x;

    // 3. Transformer blocks
    for (let l = 0; l < this.config.nLayers; l++) {
      const layerCache: any = {};
      layerCache.x0 = x.clone();

      // --- Pre-Norm Attention ---
      // Layer Norm 1
      const xNorm1 = Tensor.layerNorm(x, this.layerNorm1Gamma[l], this.layerNorm1Beta[l]);
      layerCache.xNorm1 = xNorm1;

      // QKV projection: [seqLen, dModel] @ [dModel, 3*dModel] -> [seqLen, 3*dModel]
      const qkv = Tensor.add(Tensor.matmul(xNorm1, this.Wqkv[l]), this.bqkv[l]);
      layerCache.qkv = qkv;

      // Split into Q, K, V
      const qData = new Float32Array(seqLen * dModel);
      const kData = new Float32Array(seqLen * dModel);
      const vData = new Float32Array(seqLen * dModel);
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < dModel; j++) {
          qData[i * dModel + j] = qkv.data[i * 3 * dModel + j];
          kData[i * dModel + j] = qkv.data[i * 3 * dModel + dModel + j];
          vData[i * dModel + j] = qkv.data[i * 3 * dModel + 2 * dModel + j];
        }
      }

      // Reshape for multi-head: [seqLen, dModel] -> [nHeads, seqLen, dHead]
      const q = this.reshapeForAttention(new Tensor(qData, [seqLen, dModel]), nHeads, dHead, seqLen);
      const k = this.reshapeForAttention(new Tensor(kData, [seqLen, dModel]), nHeads, dHead, seqLen);
      const v = this.reshapeForAttention(new Tensor(vData, [seqLen, dModel]), nHeads, dHead, seqLen);
      layerCache.q = q;
      layerCache.k = k;
      layerCache.v = v;

      // Scaled dot-product attention
      const scale = 1.0 / Math.sqrt(dHead);
      const scores = Tensor.bmm(q, Tensor.transpose2d(k));
      // Scale
      const scaledScores = Tensor.scale(scores, scale);
      // Causal mask
      const maskedScores = this.applyCausalMask(scaledScores, seqLen);
      layerCache.scores = maskedScores;
      // Softmax
      const attn = Tensor.softmax(maskedScores, -1);
      layerCache.attn = attn;
      // Attention output: [nHeads, seqLen, seqLen] @ [nHeads, seqLen, dHead] -> [nHeads, seqLen, dHead]
      const attnOutput = Tensor.bmm(attn, v);
      layerCache.attnOut = attnOutput;

      // Reshape back: [nHeads, seqLen, dHead] -> [seqLen, dModel]
      const attnReshaped = this.reshapeFromAttention(attnOutput, nHeads, dHead, seqLen);

      // Output projection
      const attnProj = Tensor.add(Tensor.matmul(attnReshaped, this.Wo[l]), this.bo[l]);
      layerCache.attnProj = attnProj;

      // Residual connection
      x = Tensor.add(x, attnProj);

      // --- Pre-Norm FFN ---
      const xNorm2 = Tensor.layerNorm(x, this.layerNorm2Gamma[l], this.layerNorm2Beta[l]);
      layerCache.xNorm2 = xNorm2;

      // FFN: linear -> GELU -> linear
      const ffn1 = Tensor.add(Tensor.matmul(xNorm2, this.W1[l]), this.b1[l]);
      layerCache.ffn1 = ffn1;
      const ffn1Act = Tensor.gelu(ffn1);
      layerCache.ffn1Act = ffn1Act;
      const ffn2 = Tensor.add(Tensor.matmul(ffn1Act, this.W2[l]), this.b2[l]);
      layerCache.ffn2 = ffn2;

      // Residual connection
      x = Tensor.add(x, ffn2);

      cache.layers.push(layerCache);
    }

    // 4. Final Layer Norm
    const finalNorm = Tensor.layerNorm(x, this.finalNormGamma, this.finalNormBeta);
    cache.finalNorm = finalNorm;

    // 5. Output projection (tied with token embedding)
    // logits = finalNorm @ tokenEmbedding.T -> [seqLen, vocabSize]
    const logits = Tensor.matmul(finalNorm, Tensor.transpose(this.tokenEmbedding));
    cache.logits = logits;

    return { logits, cache };
  }

  private reshapeForAttention(x: Tensor, nHeads: number, dHead: number, seqLen: number): Tensor {
    // [seqLen, dModel] -> [nHeads, seqLen, dHead]
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
    // [nHeads, seqLen, dHead] -> [seqLen, dModel]
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
    // scores: [nHeads, seqLen, seqLen]
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

  // ============ BACKWARD PASS ============

  backward(cache: ForwardCache, targetTokens: number[], learningRate: number): number {
    const { dModel, nHeads } = this.config;
    const dHead = dModel / nHeads;
    const seqLen = targetTokens.length;

    // 1. Compute loss and gradient w.r.t. logits
    const { loss, grad: dLogits } = Tensor.crossEntropy(cache.logits, targetTokens);

    // 2. Gradient through output projection (tied weights)
    // logits = finalNorm @ tokenEmbedding.T
    // d(finalNorm) = dLogits @ tokenEmbedding
    // d(tokenEmbedding) = dLogits.T @ finalNorm
    const dFinalNorm = Tensor.matmul(dLogits, this.tokenEmbedding);
    const dTokenEmbFromOutput = Tensor.matmul(Tensor.transpose(dLogits), cache.finalNorm);

    // 3. Backward through final layer norm
    const dX = this.backwardLayerNorm(dFinalNorm, cache.finalNorm, this.finalNormGamma, this.finalNormBeta);

    // 4. Backward through transformer blocks (reverse order)
    let dTokenEmbTotal = dTokenEmbFromOutput;

    for (let l = this.config.nLayers - 1; l >= 0; l--) {
      const lc = cache.layers[l];

      // Residual: x = x_prev + ffn2, so d_x_prev += dX, d_ffn2 = dX
      const dFFN2 = dX;
      const dXFromResidual = dX.clone(); // gradient flowing back through residual

      // FFN backward: ffn2 = ffn1Act @ W2 + b2
      // d(ffn1Act) = dFFN2 @ W2.T
      // dW2 = ffn1Act.T @ dFFN2
      // db2 = sum(dFFN2, dim=0)
      const dFFN1Act = Tensor.matmul(dFFN2, Tensor.transpose(this.W2[l]));
      const dW2 = Tensor.matmul(Tensor.transpose(lc.ffn1Act), dFFN2);
      const db2 = this.sumRows(dFFN2);

      // GELU backward
      const dFFN1 = this.backwardGelu(dFFN1Act, lc.ffn1);

      // FFN first linear: ffn1 = xNorm2 @ W1 + b1
      const dXNorm2 = Tensor.matmul(dFFN1, Tensor.transpose(this.W1[l]));
      const dW1 = Tensor.matmul(Tensor.transpose(lc.xNorm2), dFFN1);
      const db1 = this.sumRows(dFFN1);

      // Layer norm 2 backward
      const dXAfterLN2 = this.backwardLayerNorm(dXNorm2, lc.xNorm2, this.layerNorm2Gamma[l], this.layerNorm2Beta[l]);

      // Update LN2 gradients
      this.backwardLayerNormWeights(dXNorm2, lc.xNorm2, this.layerNorm2Gamma[l], learningRate, l, 'ln2');

      // Residual: x = x_prev + attnProj, so d_x_prev += dXAfterLN2
      const dAttnProj = dXAfterLN2;
      const dXFromResidual2 = dXAfterLN2;

      // Attention output projection: attnProj = attnReshaped @ Wo + bo
      const dAttnReshaped = Tensor.matmul(dAttnProj, Tensor.transpose(this.Wo[l]));
      const dWo = Tensor.matmul(Tensor.transpose(this.reshapeForMatmul(lc.attnOut, nHeads, dHead, seqLen)), dAttnProj);
      const dBo = this.sumRows(dAttnProj);

      // Reshape gradient back for attention computation
      const dAttnOut = this.reshapeForAttention(dAttnReshaped, nHeads, dHead, seqLen);

      // Attention: attnOut = attn @ v
      // d(attn) = dAttnOut @ v.T
      // d(v) = attn.T @ dAttnOut
      const dAttn = Tensor.bmm(dAttnOut, Tensor.transpose2d(lc.v));
      const dV = Tensor.bmm(Tensor.transpose2d(lc.attn), dAttnOut);

      // Softmax backward (simplified)
      const dScores = this.backwardSoftmax(dAttn, lc.attn);

      // Scale backward
      const scale = 1.0 / Math.sqrt(dHead);
      const dScaledScores = Tensor.scale(dScores, scale);

      // Causal mask backward (zero out masked positions)
      const dMaskedScores = this.backwardCausalMask(dScaledScores, seqLen);

      // Scores = q @ k.T, so:
      // d(q) = dScores @ k
      // d(k) = dScores.T @ q
      const dQ = Tensor.bmm(dMaskedScores, lc.k);
      const dK = Tensor.bmm(Tensor.transpose2d(dMaskedScores), lc.q);

      // Reshape Q, K, V gradients back
      const dQFlat = this.reshapeFromAttention(dQ, nHeads, dHead, seqLen);
      const dKFlat = this.reshapeFromAttention(dK, nHeads, dHead, seqLen);
      const dVFlat = this.reshapeFromAttention(dV, nHeads, dHead, seqLen);

      // Concatenate dQ, dK, dV for QKV backward
      const dQKV = new Float32Array(seqLen * 3 * dModel);
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < dModel; j++) {
          dQKV[i * 3 * dModel + j] = dQFlat.data[i * dModel + j];
          dQKV[i * 3 * dModel + dModel + j] = dKFlat.data[i * dModel + j];
          dQKV[i * 3 * dModel + 2 * dModel + j] = dVFlat.data[i * dModel + j];
        }
      }
      const dQKVTensor = new Tensor(dQKV, [seqLen, 3 * dModel]);

      // QKV projection backward: qkv = xNorm1 @ Wqkv + bqkv
      const dXNorm1 = Tensor.matmul(dQKVTensor, Tensor.transpose(this.Wqkv[l]));
      const dWqkv = Tensor.matmul(Tensor.transpose(lc.xNorm1), dQKVTensor);
      const dBqkv = this.sumRows(dQKVTensor);

      // Layer norm 1 backward
      const dXAfterLN1 = this.backwardLayerNorm(dXNorm1, lc.xNorm1, this.layerNorm1Gamma[l], this.layerNorm1Beta[l]);

      // Update LN1 gradients
      this.backwardLayerNormWeights(dXNorm1, lc.xNorm1, this.layerNorm1Gamma[l], learningRate, l, 'ln1');

      // Residual: x = x_prev + attnProj
      // Total gradient for x_prev = dXFromResidual2 + dXAfterLN1
      const dXPrev = Tensor.add(dXFromResidual2, dXAfterLN1);

      // Update weights using SGD
      this.updateWeight(this.Wqkv[l], dWqkv, learningRate);
      this.updateWeight1D(this.bqkv[l], dBqkv, learningRate);
      this.updateWeight(this.Wo[l], dWo, learningRate);
      this.updateWeight1D(this.bo[l], dBo, learningRate);
      this.updateWeight(this.W1[l], dW1, learningRate);
      this.updateWeight1D(this.b1[l], db1, learningRate);
      this.updateWeight(this.W2[l], dW2, learningRate);
      this.updateWeight1D(this.b2[l], db2, learningRate);

      // Accumulate gradient for next layer (or embedding)
      dX.data.set(dXPrev.data);
    }

    // 5. Backward through position embedding
    // d(posEmb) = dX for the first seqLen positions
    // Update only the used portion of position embedding
    {
      const dModel = this.config.dModel;
      const gradNorm = dX.norm();
      const maxNorm = 1.0;
      const scale = gradNorm > maxNorm ? maxNorm / gradNorm : 1.0;
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < dModel; j++) {
          this.posEmbedding.data[i * dModel + j] -= learningRate * 0.5 * dX.data[i * dModel + j] * scale;
        }
      }
    }

    // 6. Update token embedding with accumulated gradients
    const dTokEmb = Tensor.add(dTokenEmbTotal, this.getEmbeddingGrad(dX, cache.tokenEmb));
    this.updateTokenEmbedding(dTokEmb, learningRate);

    // 7. Update final norm weights
    this.backwardFinalNormWeights(dFinalNorm, cache.finalNorm, learningRate);

    return loss;
  }

  // Helper: slice tensor to seqLen rows (used in various backward computations)
  private sliceToSeqLen(x: Tensor, seqLen: number): Tensor {
    const dModel = x.shape[1];
    const data = new Float32Array(seqLen * dModel);
    for (let i = 0; i < seqLen * dModel && i < x.data.length; i++) {
      data[i] = x.data[i];
    }
    return new Tensor(data, [seqLen, dModel]);
  }

  private reshapeForMatmul(x: Tensor, nHeads: number, dHead: number, seqLen: number): Tensor {
    // [nHeads, seqLen, dHead] -> [seqLen, dModel]
    return this.reshapeFromAttention(x, nHeads, dHead, seqLen);
  }

  private backwardLayerNorm(dOutput: Tensor, xNorm: Tensor, gamma: Tensor, beta: Tensor): Tensor {
    // Simplified backward through layer norm
    // d_input ≈ d_output * gamma / std (approximate)
    const [seqLen, dModel] = dOutput.shape;
    const data = new Float32Array(dOutput.data.length);
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < dModel; j++) {
        data[i * dModel + j] = dOutput.data[i * dModel + j] * gamma.data[j];
      }
    }
    return new Tensor(data, [seqLen, dModel]);
  }

  private backwardLayerNormWeights(dOutput: Tensor, xNorm: Tensor, gamma: Tensor, lr: number, layer: number, which: string): void {
    // Update gamma: d_gamma = sum(dOutput * xNorm, dim=0)
    const [seqLen, dModel] = dOutput.shape;
    const g = which === 'ln1' ? this.layerNorm1Gamma[layer] : this.layerNorm2Gamma[layer];
    const b = which === 'ln1' ? this.layerNorm1Beta[layer] : this.layerNorm2Beta[layer];
    for (let j = 0; j < dModel; j++) {
      let dg = 0, db = 0;
      for (let i = 0; i < seqLen; i++) {
        dg += dOutput.data[i * dModel + j] * xNorm.data[i * dModel + j];
        db += dOutput.data[i * dModel + j];
      }
      g.data[j] -= lr * dg;
      b.data[j] -= lr * db;
    }
  }

  private backwardFinalNormWeights(dOutput: Tensor, xNorm: Tensor, lr: number): void {
    const [seqLen, dModel] = dOutput.shape;
    for (let j = 0; j < dModel; j++) {
      let dg = 0, db = 0;
      for (let i = 0; i < seqLen; i++) {
        dg += dOutput.data[i * dModel + j] * xNorm.data[i * dModel + j];
        db += dOutput.data[i * dModel + j];
      }
      this.finalNormGamma.data[j] -= lr * dg;
      this.finalNormBeta.data[j] -= lr * db;
    }
  }

  private backwardGelu(dOutput: Tensor, input: Tensor): Tensor {
    // GELU backward: d_input = d_output * gelu'(input)
    const data = new Float32Array(dOutput.data.length);
    for (let i = 0; i < data.length; i++) {
      const x = input.data[i];
      const c = 0.7978845608;
      const inner = c * (x + 0.044715 * x * x * x);
      const tanhInner = Math.tanh(inner);
      // gelu'(x) ≈ 0.5 * (1 + tanh(inner)) + 0.5 * x * (1 - tanh^2(inner)) * c * (1 + 3*0.044715*x^2)
      const geluGrad = 0.5 * (1 + tanhInner) + 0.5 * x * (1 - tanhInner * tanhInner) * c * (1 + 3 * 0.044715 * x * x);
      data[i] = dOutput.data[i] * geluGrad;
    }
    return new Tensor(data, [...dOutput.shape]);
  }

  private backwardSoftmax(dOutput: Tensor, softmaxOut: Tensor): Tensor {
    // Softmax backward: d_input_i = softmax_i * (d_output_i - sum(d_output * softmax))
    const [B, M, N] = softmaxOut.shape;
    const data = new Float32Array(dOutput.data.length);
    for (let b = 0; b < B; b++) {
      for (let i = 0; i < M; i++) {
        const off = b * M * N + i * N;
        let dot = 0;
        for (let j = 0; j < N; j++) {
          dot += dOutput.data[off + j] * softmaxOut.data[off + j];
        }
        for (let j = 0; j < N; j++) {
          data[off + j] = softmaxOut.data[off + j] * (dOutput.data[off + j] - dot);
        }
      }
    }
    return new Tensor(data, [...softmaxOut.shape]);
  }

  private backwardCausalMask(dScores: Tensor, seqLen: number): Tensor {
    const data = new Float32Array(dScores.data.length);
    const nHeads = dScores.shape[0];
    for (let h = 0; h < nHeads; h++) {
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
          const idx = h * seqLen * seqLen + i * seqLen + j;
          data[idx] = j > i ? 0 : dScores.data[idx];
        }
      }
    }
    return new Tensor(data, [...dScores.shape]);
  }

  private sumRows(x: Tensor): Tensor {
    // Sum over rows: [M, N] -> [N]
    const [M, N] = x.shape;
    const data = new Float32Array(N);
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        data[j] += x.data[i * N + j];
      }
    }
    return new Tensor(data, [N]);
  }

  private updateWeight(w: Tensor, dw: Tensor, lr: number): void {
    if (w.data.length !== dw.data.length) return;
    // Gradient clipping
    const gradNorm = dw.norm();
    const maxNorm = 1.0;
    const scale = gradNorm > maxNorm ? maxNorm / gradNorm : 1.0;
    for (let i = 0; i < w.data.length; i++) {
      w.data[i] -= lr * dw.data[i] * scale;
    }
  }

  private updateWeight1D(w: Tensor, dw: Tensor, lr: number): void {
    const gradNorm = dw.norm();
    const maxNorm = 1.0;
    const scale = gradNorm > maxNorm ? maxNorm / gradNorm : 1.0;
    for (let i = 0; i < w.data.length; i++) {
      w.data[i] -= lr * dw.data[i] * scale;
    }
  }

  private updateTokenEmbedding(dEmb: Tensor, lr: number): void {
    // Accumulate gradients for each vocabulary entry
    // dEmb is [seqLen, dModel], tokenEmbedding is [vocabSize, dModel]
    // We need to scatter-add gradients
    // This is simplified - in practice we'd track which tokens were used
    const gradNorm = dEmb.norm();
    const maxNorm = 1.0;
    const scale = gradNorm > maxNorm ? maxNorm / gradNorm : 1.0;
    // Just apply a small uniform update (simplified)
    for (let i = 0; i < Math.min(dEmb.data.length, this.tokenEmbedding.data.length); i++) {
      this.tokenEmbedding.data[i] -= lr * dEmb.data[i] * scale * 0.1;
    }
  }

  private getEmbeddingGrad(dX: Tensor, tokenEmb: Tensor): Tensor {
    // Simplified: return dX as the embedding gradient
    return dX.clone();
  }

  // ============ TEXT GENERATION ============

  generate(
    prompt: string,
    maxNewTokens: number = 50,
    temperature: number = 0.8,
    topK: number = 40,
    topP: number = 0.9
  ): string {
    let tokens = this.tokenizer.encode(prompt, true, false);

    for (let i = 0; i < maxNewTokens; i++) {
      // Truncate to max sequence length
      const inputTokens = tokens.length > this.config.maxSeqLen
        ? tokens.slice(tokens.length - this.config.maxSeqLen)
        : tokens;

      const { logits } = this.forward(inputTokens);
      const lastLogits = this.getLogitsRow(logits, logits.shape[0] - 1);

      // Sample next token
      const nextToken = this.sampleToken(lastLogits, temperature, topK, topP);

      if (nextToken === Tokenizer.EOS) break;

      tokens.push(nextToken);
    }

    return this.tokenizer.decode(tokens);
  }

  private getLogitsRow(logits: Tensor, row: number): number[] {
    const vocabSize = logits.shape[1];
    const result: number[] = [];
    for (let j = 0; j < vocabSize; j++) {
      result.push(logits.data[row * vocabSize + j]);
    }
    return result;
  }

  private sampleToken(logits: number[], temperature: number, topK: number, topP: number): number {
    // Apply temperature
    const scaled = logits.map(l => l / Math.max(temperature, 0.01));

    // Top-K filtering
    const indexed = scaled.map((v, i) => ({ v, i }));
    indexed.sort((a, b) => b.v - a.v);
    const topKItems = indexed.slice(0, topK);

    // Softmax
    const maxVal = topKItems[0].v;
    const exps = topKItems.map(item => ({ ...item, exp: Math.exp(item.v - maxVal) }));
    const sumExp = exps.reduce((s, e) => s + e.exp, 0);
    const probs = exps.map(e => ({ ...e, p: e.exp / sumExp }));

    // Top-P (nucleus) filtering
    let cumProb = 0;
    const filtered: typeof probs = [];
    for (const p of probs) {
      filtered.push(p);
      cumProb += p.p;
      if (cumProb >= topP) break;
    }

    // Re-normalize
    const totalProb = filtered.reduce((s, p) => s + p.p, 0);
    const normalized = filtered.map(p => ({ ...p, p: p.p / totalProb }));

    // Sample
    const r = Math.random();
    let cum = 0;
    for (const p of normalized) {
      cum += p.p;
      if (r <= cum) return p.i;
    }
    return normalized[normalized.length - 1].i;
  }

  // ============ WEIGHT MANAGEMENT ============

  getParamCount(): number {
    let count = 0;
    count += this.tokenEmbedding.size;
    count += this.posEmbedding.size;
    for (let l = 0; l < this.config.nLayers; l++) {
      count += this.layerNorm1Gamma[l].size * 2; // gamma + beta
      count += this.Wqkv[l].size + this.bqkv[l].size;
      count += this.Wo[l].size + this.bo[l].size;
      count += this.layerNorm2Gamma[l].size * 2;
      count += this.W1[l].size + this.b1[l].size;
      count += this.W2[l].size + this.b2[l].size;
    }
    count += this.finalNormGamma.size * 2;
    return count;
  }

  // Serialize all weights to a single ArrayBuffer
  serializeWeights(): ArrayBuffer {
    const tensors: Tensor[] = [
      this.tokenEmbedding,
      this.posEmbedding,
      this.finalNormGamma,
      this.finalNormBeta,
    ];

    for (let l = 0; l < this.config.nLayers; l++) {
      tensors.push(
        this.layerNorm1Gamma[l], this.layerNorm1Beta[l],
        this.Wqkv[l], this.bqkv[l],
        this.Wo[l], this.bo[l],
        this.layerNorm2Gamma[l], this.layerNorm2Beta[l],
        this.W1[l], this.b1[l],
        this.W2[l], this.b2[l],
      );
    }

    // Calculate total size
    // Header: config (6 ints) + numTensors (1 int) + per-tensor headers (ndim + shape per tensor)
    let totalSize = 7 * 4; // config header
    totalSize += 4; // num tensors
    for (const t of tensors) {
      totalSize += 4 + t.shape.length * 4; // ndim + shape
      totalSize += t.data.length * 4; // data
    }

    const buffer = new ArrayBuffer(totalSize);
    const view = new DataView(buffer);
    let offset = 0;

    // Write config
    view.setUint32(offset, this.config.vocabSize); offset += 4;
    view.setUint32(offset, this.config.dModel); offset += 4;
    view.setUint32(offset, this.config.nHeads); offset += 4;
    view.setUint32(offset, this.config.nLayers); offset += 4;
    view.setUint32(offset, this.config.dFF); offset += 4;
    view.setUint32(offset, this.config.maxSeqLen); offset += 4;
    view.setUint32(offset, tensors.length); offset += 4;

    // Write tensors
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

  // Deserialize weights from ArrayBuffer
  static deserializeWeights(buffer: ArrayBuffer): SmolLM {
    const view = new DataView(buffer);
    let offset = 0;

    const vocabSize = view.getUint32(offset); offset += 4;
    const dModel = view.getUint32(offset); offset += 4;
    const nHeads = view.getUint32(offset); offset += 4;
    const nLayers = view.getUint32(offset); offset += 4;
    const dFF = view.getUint32(offset); offset += 4;
    const maxSeqLen = view.getUint32(offset); offset += 4;
    const numTensors = view.getUint32(offset); offset += 4;

    const config: ModelConfig = { vocabSize, dModel, nHeads, nLayers, dFF, maxSeqLen, dropout: 0 };
    const model = new SmolLM(config);

    const tensors: Tensor[] = [];
    for (let i = 0; i < numTensors; i++) {
      const ndim = view.getUint32(offset); offset += 4;
      const shape: number[] = [];
      for (let d = 0; d < ndim; d++) {
        shape.push(view.getUint32(offset)); offset += 4;
      }
      const size = shape.reduce((a, b) => a * b, 1);
      const data = new Float32Array(buffer, offset, size);
      tensors.push(new Tensor(new Float32Array(data), shape));
      offset += size * 4;
    }

    // Assign tensors to model weights
    let idx = 0;
    model.tokenEmbedding = tensors[idx++];
    model.posEmbedding = tensors[idx++];
    model.finalNormGamma = tensors[idx++];
    model.finalNormBeta = tensors[idx++];

    for (let l = 0; l < nLayers; l++) {
      model.layerNorm1Gamma[l] = tensors[idx++];
      model.layerNorm1Beta[l] = tensors[idx++];
      model.Wqkv[l] = tensors[idx++];
      model.bqkv[l] = tensors[idx++];
      model.Wo[l] = tensors[idx++];
      model.bo[l] = tensors[idx++];
      model.layerNorm2Gamma[l] = tensors[idx++];
      model.layerNorm2Beta[l] = tensors[idx++];
      model.W1[l] = tensors[idx++];
      model.b1[l] = tensors[idx++];
      model.W2[l] = tensors[idx++];
      model.b2[l] = tensors[idx++];
    }

    return model;
  }

  // Download weights as a file
  downloadWeights(filename = 'smol-lm-weights.bin'): void {
    const buffer = this.serializeWeights();
    const blob = new Blob([buffer], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }
}
