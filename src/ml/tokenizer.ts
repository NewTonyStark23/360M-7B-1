// ============================================================
// SMOL-LM Byte-Level Tokenizer
// Encodes text as byte sequences (vocab=256)
// This is the simplest effective tokenization approach
// ============================================================

export class Tokenizer {
  vocabSize: number;
  // Special tokens
  static readonly PAD = 0;
  static readonly BOS = 1;
  static readonly EOS = 2;
  static readonly UNK = 3;

  constructor(vocabSize = 256) {
    this.vocabSize = vocabSize;
  }

  // Encode text to byte sequence with special tokens
  encode(text: string, addBOS = true, addEOS = true): number[] {
    const encoder = new TextEncoder();
    const bytes = Array.from(encoder.encode(text));
    const tokens: number[] = [];
    if (addBOS) tokens.push(Tokenizer.BOS);
    tokens.push(...bytes.map(b => b + 4)); // offset by special tokens
    if (addEOS) tokens.push(Tokenizer.EOS);
    return tokens;
  }

  // Decode byte sequence back to text
  decode(tokens: number[]): string {
    const bytes: number[] = [];
    for (const t of tokens) {
      if (t <= 3) continue; // skip special tokens
      if (t >= 4 && t < 260) bytes.push(t - 4);
    }
    const decoder = new TextDecoder('utf-8', { fatal: false });
    return decoder.decode(new Uint8Array(bytes));
  }

  // Encode without special tokens (for training targets)
  encodeRaw(text: string): number[] {
    const encoder = new TextEncoder();
    return Array.from(encoder.encode(text)).map(b => b + 4);
  }

  // Get token string representation
  tokenToString(token: number): string {
    if (token === Tokenizer.PAD) return '<pad>';
    if (token === Tokenizer.BOS) return '<bos>';
    if (token === Tokenizer.EOS) return '<eos>';
    if (token === Tokenizer.UNK) return '<unk>';
    if (token >= 4 && token < 260) {
      try {
        return new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array([token - 4]));
      } catch {
        return `0x${(token - 4).toString(16)}`;
      }
    }
    return `<${token}>`;
  }
}
