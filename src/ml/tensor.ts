// ============================================================
// SMOL-LM Tensor Engine - Core mathematical operations
// Implements all tensor ops needed for transformer inference & training
// ============================================================

export class Tensor {
  data: Float32Array;
  shape: number[];
  strides: number[];
  grad: Float32Array | null;
  requiresGrad: boolean;

  constructor(data: Float32Array | number[], shape: number[], requiresGrad = false) {
    this.data = data instanceof Float32Array ? data : new Float32Array(data);
    this.shape = shape;
    this.requiresGrad = requiresGrad;
    this.grad = requiresGrad ? new Float32Array(this.data.length) : null;
    this.strides = Tensor.computeStrides(shape);
  }

  static computeStrides(shape: number[]): number[] {
    const strides = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  get size(): number {
    return this.data.length;
  }

  get ndim(): number {
    return this.shape.length;
  }

  static zeros(shape: number[], requiresGrad = false): Tensor {
    return new Tensor(new Float32Array(shape.reduce((a, b) => a * b, 1)), shape, requiresGrad);
  }

  static ones(shape: number[]): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    data.fill(1);
    return new Tensor(data, shape);
  }

  static random(shape: number[], scale = 1.0, requiresGrad = false): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      // Box-Muller transform for normal distribution
      const u1 = Math.random();
      const u2 = Math.random();
      data[i] = scale * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    return new Tensor(data, shape, requiresGrad);
  }

  static xavier(fanIn: number, fanOut: number, shape: number[]): Tensor {
    const scale = Math.sqrt(2.0 / (fanIn + fanOut));
    return Tensor.random(shape, scale, true);
  }

  clone(): Tensor {
    const t = new Tensor(new Float32Array(this.data), [...this.shape], this.requiresGrad);
    if (this.grad) t.grad = new Float32Array(this.grad);
    return t;
  }

  reshape(newShape: number[]): Tensor {
    const newSize = newShape.reduce((a, b) => a * b, 1);
    if (newSize !== this.data.length) throw new Error(`Cannot reshape [${this.shape}] to [${newShape}]`);
    const t = new Tensor(new Float32Array(this.data), newShape, this.requiresGrad);
    return t;
  }

  // Get element at flat index
  get(idx: number): number {
    return this.data[idx];
  }

  set(idx: number, val: number): void {
    this.data[idx] = val;
  }

  // ============ BASIC OPERATIONS ============

  static add(a: Tensor, b: Tensor): Tensor {
    // Broadcasting add
    if (a.shape.length === b.shape.length && a.shape.every((s, i) => s === b.shape[i])) {
      const data = new Float32Array(a.data.length);
      for (let i = 0; i < data.length; i++) data[i] = a.data[i] + b.data[i];
      return new Tensor(data, [...a.shape]);
    }
    // Broadcast b over a (b is smaller, e.g. bias)
    if (b.data.length === a.shape[a.shape.length - 1]) {
      const data = new Float32Array(a.data.length);
      const lastDim = a.shape[a.shape.length - 1];
      for (let i = 0; i < data.length; i++) {
        data[i] = a.data[i] + b.data[i % lastDim];
      }
      return new Tensor(data, [...a.shape]);
    }
    throw new Error(`Incompatible shapes for add: [${a.shape}] + [${b.shape}]`);
  }

  static sub(a: Tensor, b: Tensor): Tensor {
    const data = new Float32Array(a.data.length);
    for (let i = 0; i < data.length; i++) data[i] = a.data[i] - b.data[i];
    return new Tensor(data, [...a.shape]);
  }

  static mul(a: Tensor, b: Tensor): Tensor {
    const data = new Float32Array(a.data.length);
    for (let i = 0; i < data.length; i++) data[i] = a.data[i] * b.data[i];
    return new Tensor(data, [...a.shape]);
  }

  static scale(a: Tensor, s: number): Tensor {
    const data = new Float32Array(a.data.length);
    for (let i = 0; i < data.length; i++) data[i] = a.data[i] * s;
    return new Tensor(data, [...a.shape]);
  }

  // ============ MATRIX OPERATIONS ============

  // Matrix multiply: [M, K] @ [K, N] -> [M, N]
  static matmul(a: Tensor, b: Tensor): Tensor {
    if (a.ndim !== 2 || b.ndim !== 2) throw new Error('matmul requires 2D tensors');
    const M = a.shape[0], K = a.shape[1], N = b.shape[1];
    if (K !== b.shape[0]) throw new Error(`matmul shape mismatch: [${a.shape}] @ [${b.shape}]`);

    const data = new Float32Array(M * N);
    for (let i = 0; i < M; i++) {
      for (let k = 0; k < K; k++) {
        const aVal = a.data[i * K + k];
        if (aVal === 0) continue;
        for (let j = 0; j < N; j++) {
          data[i * N + j] += aVal * b.data[k * N + j];
        }
      }
    }
    return new Tensor(data, [M, N]);
  }

  // Batched matmul: [B, M, K] @ [B, K, N] -> [B, M, N]
  static bmm(a: Tensor, b: Tensor): Tensor {
    const B = a.shape[0], M = a.shape[1], K = a.shape[2], N = b.shape[2];
    const data = new Float32Array(B * M * N);
    for (let batch = 0; batch < B; batch++) {
      const aOff = batch * M * K;
      const bOff = batch * K * N;
      const cOff = batch * M * N;
      for (let i = 0; i < M; i++) {
        for (let k = 0; k < K; k++) {
          const aVal = a.data[aOff + i * K + k];
          if (aVal === 0) continue;
          for (let j = 0; j < N; j++) {
            data[cOff + i * N + j] += aVal * b.data[bOff + k * N + j];
          }
        }
      }
    }
    return new Tensor(data, [B, M, N]);
  }

  static transpose(a: Tensor): Tensor {
    if (a.ndim !== 2) throw new Error('transpose requires 2D');
    const [M, N] = a.shape;
    const data = new Float32Array(M * N);
    for (let i = 0; i < M; i++)
      for (let j = 0; j < N; j++)
        data[j * M + i] = a.data[i * N + j];
    return new Tensor(data, [N, M]);
  }

  // Transpose last two dims for 3D tensor [B, M, N] -> [B, N, M]
  static transpose2d(a: Tensor): Tensor {
    const [B, M, N] = a.shape;
    const data = new Float32Array(B * M * N);
    for (let b = 0; b < B; b++)
      for (let i = 0; i < M; i++)
        for (let j = 0; j < N; j++)
          data[b * N * M + j * M + i] = a.data[b * M * N + i * N + j];
    return new Tensor(data, [B, N, M]);
  }

  // ============ ACTIVATION FUNCTIONS ============

  static relu(a: Tensor): Tensor {
    const data = new Float32Array(a.data.length);
    for (let i = 0; i < data.length; i++) data[i] = Math.max(0, a.data[i]);
    return new Tensor(data, [...a.shape]);
  }

  static gelu(a: Tensor): Tensor {
    const data = new Float32Array(a.data.length);
    for (let i = 0; i < data.length; i++) {
      const x = a.data[i];
      // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
      const c = 0.7978845608; // sqrt(2/pi)
      const inner = c * (x + 0.044715 * x * x * x);
      data[i] = 0.5 * x * (1 + Math.tanh(inner));
    }
    return new Tensor(data, [...a.shape]);
  }

  static sigmoid(a: Tensor): Tensor {
    const data = new Float32Array(a.data.length);
    for (let i = 0; i < data.length; i++) {
      data[i] = 1 / (1 + Math.exp(-a.data[i]));
    }
    return new Tensor(data, [...a.shape]);
  }

  // ============ NORMALIZATION ============

  static layerNorm(a: Tensor, gamma: Tensor, beta: Tensor, eps = 1e-5): Tensor {
    const [seqLen, dModel] = a.shape;
    const data = new Float32Array(a.data.length);

    for (let i = 0; i < seqLen; i++) {
      const offset = i * dModel;
      // Compute mean
      let mean = 0;
      for (let j = 0; j < dModel; j++) mean += a.data[offset + j];
      mean /= dModel;
      // Compute variance
      let var_ = 0;
      for (let j = 0; j < dModel; j++) {
        const diff = a.data[offset + j] - mean;
        var_ += diff * diff;
      }
      var_ /= dModel;
      // Normalize
      const invStd = 1 / Math.sqrt(var_ + eps);
      for (let j = 0; j < dModel; j++) {
        const normalized = (a.data[offset + j] - mean) * invStd;
        data[offset + j] = normalized * gamma.data[j] + beta.data[j];
      }
    }
    return new Tensor(data, [...a.shape]);
  }

  // ============ SOFTMAX ============

  static softmax(a: Tensor, dim = -1): Tensor {
    const data = new Float32Array(a.data.length);
    const shape = a.shape;

    if (a.ndim === 2) {
      const rows = shape[0], cols = shape[1];
      for (let i = 0; i < rows; i++) {
        let maxVal = -Infinity;
        for (let j = 0; j < cols; j++) maxVal = Math.max(maxVal, a.data[i * cols + j]);
        let sum = 0;
        for (let j = 0; j < cols; j++) {
          data[i * cols + j] = Math.exp(a.data[i * cols + j] - maxVal);
          sum += data[i * cols + j];
        }
        for (let j = 0; j < cols; j++) data[i * cols + j] /= sum;
      }
    } else if (a.ndim === 3) {
      const [B, M, N] = shape;
      for (let b = 0; b < B; b++) {
        for (let i = 0; i < M; i++) {
          const off = b * M * N + i * N;
          let maxVal = -Infinity;
          for (let j = 0; j < N; j++) maxVal = Math.max(maxVal, a.data[off + j]);
          let sum = 0;
          for (let j = 0; j < N; j++) {
            data[off + j] = Math.exp(a.data[off + j] - maxVal);
            sum += data[off + j];
          }
          for (let j = 0; j < N; j++) data[off + j] /= sum;
        }
      }
    }
    return new Tensor(data, [...shape]);
  }

  // ============ EMBEDDING ============

  static embedding(weight: Tensor, indices: number[]): Tensor {
    const seqLen = indices.length;
    const dModel = weight.shape[1];
    const data = new Float32Array(seqLen * dModel);
    for (let i = 0; i < seqLen; i++) {
      const idx = indices[i];
      const srcOff = idx * dModel;
      const dstOff = i * dModel;
      for (let j = 0; j < dModel; j++) {
        data[dstOff + j] = weight.data[srcOff + j];
      }
    }
    return new Tensor(data, [seqLen, dModel]);
  }

  // ============ POSITIONAL ENCODING (sinusoidal) ============

  static sinusoidalPE(seqLen: number, dModel: number): Tensor {
    const data = new Float32Array(seqLen * dModel);
    for (let pos = 0; pos < seqLen; pos++) {
      for (let i = 0; i < dModel; i++) {
        const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / dModel);
        data[pos * dModel + i] = i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
      }
    }
    return new Tensor(data, [seqLen, dModel]);
  }

  // ============ CROSS ENTROPY LOSS ============

  static crossEntropy(logits: Tensor, targets: number[]): { loss: number; grad: Tensor } {
    const [seqLen, vocabSize] = logits.shape;
    const probs = Tensor.softmax(logits, -1);
    let loss = 0;
    const grad = new Float32Array(seqLen * vocabSize);

    for (let i = 0; i < seqLen; i++) {
      const target = targets[i];
      const prob = Math.max(probs.data[i * vocabSize + target], 1e-10);
      loss -= Math.log(prob);
      // Gradient of cross-entropy w.r.t. logits = probs - one_hot(target)
      for (let j = 0; j < vocabSize; j++) {
        grad[i * vocabSize + j] = probs.data[i * vocabSize + j];
      }
      grad[i * vocabSize + target] -= 1;
    }

    return { loss: loss / seqLen, grad: new Tensor(grad, [seqLen, vocabSize]) };
  }

  // ============ UTILITIES ============

  norm(): number {
    let sum = 0;
    for (let i = 0; i < this.data.length; i++) sum += this.data[i] * this.data[i];
    return Math.sqrt(sum);
  }

  mean(): number {
    let sum = 0;
    for (let i = 0; i < this.data.length; i++) sum += this.data[i];
    return sum / this.data.length;
  }

  max(): number {
    let m = -Infinity;
    for (let i = 0; i < this.data.length; i++) if (this.data[i] > m) m = this.data[i];
    return m;
  }

  min(): number {
    let m = Infinity;
    for (let i = 0; i < this.data.length; i++) if (this.data[i] < m) m = this.data[i];
    return m;
  }

  // Serialize to ArrayBuffer for download
  toArrayBuffer(): ArrayBuffer {
    const headerSize = 64; // bytes for metadata
    const dataSize = this.data.length * 4;
    const buffer = new ArrayBuffer(headerSize + dataSize);
    const view = new DataView(buffer);
    // Write shape
    view.setUint32(0, this.shape.length);
    for (let i = 0; i < this.shape.length; i++) {
      view.setUint32(4 + i * 4, this.shape[i]);
    }
    // Write data
    const dataView = new Float32Array(buffer, headerSize);
    dataView.set(this.data);
    return buffer;
  }

  static fromArrayBuffer(buffer: ArrayBuffer): Tensor {
    const view = new DataView(buffer);
    const ndim = view.getUint32(0);
    const shape: number[] = [];
    for (let i = 0; i < ndim; i++) {
      shape.push(view.getUint32(4 + i * 4));
    }
    const data = new Float32Array(buffer, 64);
    return new Tensor(data, shape);
  }
}
