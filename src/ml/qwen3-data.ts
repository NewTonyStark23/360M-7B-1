// ============================================================
// QWEN3-STYLE TRAINING DATA
// Based on Qwen3's training methodology
// Multi-stage training: pretrain → SFT → RLHF
// ============================================================

export const QWEN3_PRETRAIN_DATA = {
  // Stage 1: Pre-training corpus (simplified for demo)
  corpus: {
    english: {
      name: 'English Web Text',
      samples: [
        'The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.',
        'Machine learning is a subset of artificial intelligence that enables systems to learn from data.',
        'Natural language processing combines linguistics with computer science to understand human language.',
        'Transformers revolutionized NLP with their self-attention mechanism and parallel processing capability.',
        'Large language models are trained on vast corpora of text to predict the next token in a sequence.',
        'Knowledge distillation transfers knowledge from a large teacher model to a smaller student model.',
        'Mixture of experts architectures activate only a subset of parameters for each input token.',
        'Progressive training starts with large context and gradually compresses knowledge into smaller models.',
        'The attention mechanism allows models to focus on relevant parts of the input when generating output.',
        'Layer normalization stabilizes training by normalizing activations within each layer of the network.',
        'Residual connections help gradients flow through deep networks by adding skip connections between layers.',
        'Feed-forward networks process each position independently after attention has gathered context.',
        'Positional encodings inject sequence order information since transformers process tokens in parallel.',
        'Tokenization converts raw text into discrete tokens that the model can process efficiently.',
        'Byte-level tokenization works on any language without requiring language-specific vocabulary.',
        'Training large models requires careful attention to learning rate schedules and batch sizes.',
        'Gradient clipping prevents exploding gradients by capping the maximum gradient norm during training.',
        'Weight decay regularizes the model by penalizing large weights to prevent overfitting.',
        'Warmup gradually increases the learning rate at the start of training for stability.',
        'Cosine annealing decreases the learning rate following a cosine curve for smooth convergence.',
      ],
    },
    code: {
      name: 'Code Patterns',
      samples: [
        'function fibonacci(n) { if (n <= 1) return n; return fibonacci(n-1) + fibonacci(n-2); }',
        'const quicksort = (arr) => arr.length <= 1 ? arr : quicksort(arr.slice(1).filter(x => x < arr[0])).concat(arr[0], quicksort(arr.slice(1).filter(x => x >= arr[0])));',
        'class NeuralNetwork { constructor(layers) { this.layers = layers; this.weights = []; } }',
        'def train(model, data, epochs, lr): for epoch in range(epochs): loss = forward(model, data); backward(model, loss); update(model, lr)',
        'import torch; model = torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6)',
        'const attention = (Q, K, V) => { const scores = matmul(Q, transpose(K)); return softmax(scores); }',
        'for (let i = 0; i < tokens.length; i++) { const embedding = embed(tokens[i]); const output = forward(embedding); }',
        'const loss = crossEntropy(logits, targets); loss.backward(); optimizer.step();',
      ],
    },
    reasoning: {
      name: 'Reasoning Patterns',
      samples: [
        'Question: What is 2+2? Answer: Let me think step by step. 2+2 equals 4.',
        'Question: If it rains, the ground gets wet. It is raining. What can we conclude? Answer: The ground is wet.',
        'Question: All humans are mortal. Socrates is human. What can we conclude? Answer: Socrates is mortal.',
        'Question: How do you solve x+5=10? Answer: Subtract 5 from both sides. x=10-5=5.',
        'Question: What comes next in the sequence 2,4,6,8? Answer: Each number increases by 2, so next is 10.',
        'Question: If A>B and B>C, what is the relationship between A and C? Answer: A>C by transitivity.',
      ],
    },
  },

  // Stage 2: Supervised Fine-Tuning (SFT) data
  sft_data: {
    conversations: [
      {
        prompt: 'Explain what a transformer is.',
        response: 'A transformer is a neural network architecture that uses self-attention mechanisms to process sequences. It was introduced in the "Attention Is All You Need" paper and has become the foundation for modern language models.',
      },
      {
        prompt: 'What is knowledge distillation?',
        response: 'Knowledge distillation is a technique where a smaller model (student) learns from a larger model (teacher). The student tries to match the teacher\'s output distribution, allowing it to capture knowledge in a compact form.',
      },
      {
        prompt: 'How does dynamic model resizing work?',
        response: 'Dynamic resizing allows a model to operate at different scales by activating different subsets of parameters. A 7B model can compress to 360M by using fewer layers, smaller dimensions, or mixture of experts, while preserving knowledge through distillation.',
      },
    ],
  },
};

// Generate training chunks from corpus
export function generateTrainingChunks(corpus: string, chunkSize = 128, stride = 64): string[] {
  const chunks: string[] = [];
  for (let i = 0; i < corpus.length - chunkSize; i += stride) {
    chunks.push(corpus.slice(i, i + chunkSize));
  }
  return chunks;
}

// Combine all pre-training data
export function getAllPretrainData(): string {
  const allText: string[] = [];
  for (const category of Object.values(QWEN3_PRETRAIN_DATA.corpus)) {
    allText.push(...category.samples);
  }
  return allText.join('\n\n');
}

// Get SFT conversations
export function getSFTConversations(): Array<{ prompt: string; response: string }> {
  return QWEN3_PRETRAIN_DATA.sft_data.conversations;
}
