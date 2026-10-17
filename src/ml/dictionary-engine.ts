// ============================================================
// DICTIONARY ENGINE
// Pre-loaded multi-word phrases for fast sentence/code generation
// Enables rapid composition without token-by-token generation
// ============================================================

export interface DictionaryEntry {
  phrase: string;
  category: 'common' | 'code' | 'technical' | 'creative' | 'formal';
  tokenCount: number;
  usageCount: number;
  context: string[];  // Typical contexts where this phrase is used
}

export class DictionaryEngine {
  private phrases: Map<string, DictionaryEntry>;
  private categories: Map<string, string[]>;
  
  constructor() {
    this.phrases = new Map();
    this.categories = new Map();
    this.initializeDictionary();
  }

  private initializeDictionary(): void {
    // Common phrases
    const commonPhrases = [
      'the quick brown fox',
      'in order to',
      'as a result',
      'on the other hand',
      'for example',
      'in conclusion',
      'it is important to note',
      'based on the above',
      'therefore',
      'however',
      'moreover',
      'furthermore',
      'in addition',
      'as mentioned',
      'according to',
    ];

    // Code patterns
    const codePhrases = [
      'function() {',
      'const x =',
      'if (condition) {',
      'for (let i = 0;',
      'return result;',
      'console.log(',
      'import { useState }',
      'export default',
      'async function',
      'await promise',
      'try {',
      'catch (error) {',
      'class Component',
      'render() {',
      'useState(0)',
    ];

    // Technical phrases
    const technicalPhrases = [
      'machine learning algorithm',
      'neural network architecture',
      'data structure',
      'time complexity',
      'space complexity',
      'artificial intelligence',
      'deep learning model',
      'natural language processing',
      'computer vision',
      'reinforcement learning',
      'supervised learning',
      'unsupervised learning',
      'gradient descent',
      'backpropagation',
      'attention mechanism',
    ];

    // Creative phrases
    const creativePhrases = [
      'once upon a time',
      'in a world where',
      'the hero embarked',
      'a journey of discovery',
      'through the looking glass',
      'beyond the horizon',
      'in the depths of',
      'with great power',
      'comes great responsibility',
      'the story unfolds',
    ];

    // Formal phrases
    const formalPhrases = [
      'Dear Sir/Madam',
      'I am writing to',
      'Please find attached',
      'Thank you for your',
      'Best regards',
      'Sincerely',
      'Yours faithfully',
      'To whom it may concern',
      'Re: Subject',
      'As per our discussion',
    ];

    // Add all phrases
    this.addPhrases(commonPhrases, 'common');
    this.addPhrases(codePhrases, 'code');
    this.addPhrases(technicalPhrases, 'technical');
    this.addPhrases(creativePhrases, 'creative');
    this.addPhrases(formalPhrases, 'formal');
  }

  private addPhrases(phrases: string[], category: string): void {
    const categoryList = this.categories.get(category) || [];
    
    for (const phrase of phrases) {
      const entry: DictionaryEntry = {
        phrase,
        category: category as any,
        tokenCount: this.estimateTokenCount(phrase),
        usageCount: 0,
        context: [],
      };
      
      this.phrases.set(phrase, entry);
      categoryList.push(phrase);
    }
    
    this.categories.set(category, categoryList);
  }

  // Find matching phrases for given context
  findMatches(context: string, category?: string, maxResults = 10): DictionaryEntry[] {
    const matches: DictionaryEntry[] = [];
    const lowerContext = context.toLowerCase();
    
    for (const entry of this.phrases.values()) {
      // Filter by category if specified
      if (category && entry.category !== category) continue;
      
      // Check if phrase is relevant to context
      const words = entry.phrase.toLowerCase().split(/\s+/);
      const contextWords = lowerContext.split(/\s+/);
      
      // Count matching words
      let matchScore = 0;
      for (const word of words) {
        if (contextWords.some(cw => cw.includes(word) || word.includes(cw))) {
          matchScore++;
        }
      }
      
      if (matchScore > 0) {
        matches.push({
          ...entry,
          usageCount: entry.usageCount + 1,
        });
      }
    }
    
    // Sort by match score and usage count
    matches.sort((a, b) => {
      const aWords = a.phrase.split(/\s+/).length;
      const bWords = b.phrase.split(/\s+/).length;
      return (b.usageCount * bWords) - (a.usageCount * aWords);
    });
    
    return matches.slice(0, maxResults);
  }

  // Get phrases by category
  getByCategory(category: string): DictionaryEntry[] {
    const phraseIds = this.categories.get(category) || [];
    return phraseIds.map(id => this.phrases.get(id)!).filter(e => e);
  }

  // Generate sentence using dictionary phrases
  generateFromDictionary(prompt: string, category?: string): string {
    const matches = this.findMatches(prompt, category, 5);
    
    if (matches.length === 0) {
      return prompt;
    }
    
    // Use the best matching phrase
    const bestMatch = matches[0];
    return prompt + ' ' + bestMatch.phrase;
  }

  // Estimate token count
  private estimateTokenCount(phrase: string): number {
    return Math.ceil(phrase.length / 4);
  }

  // Get dictionary statistics
  getStats(): {
    totalPhrases: number;
    categories: Record<string, number>;
    totalTokens: number;
  } {
    const categories: Record<string, number> = {};
    let totalTokens = 0;
    
    for (const entry of this.phrases.values()) {
      categories[entry.category] = (categories[entry.category] || 0) + 1;
      totalTokens += entry.tokenCount;
    }
    
    return {
      totalPhrases: this.phrases.size,
      categories,
      totalTokens,
    };
  }

  // Add custom phrase
  addPhrase(phrase: string, category: string, contexts: string[] = []): void {
    const entry: DictionaryEntry = {
      phrase,
      category: category as any,
      tokenCount: this.estimateTokenCount(phrase),
      usageCount: 0,
      context: contexts,
    };
    
    this.phrases.set(phrase, entry);
    
    const categoryList = this.categories.get(category) || [];
    categoryList.push(phrase);
    this.categories.set(category, categoryList);
  }

  // Export dictionary
  exportDictionary(): DictionaryEntry[] {
    return Array.from(this.phrases.values());
  }
}
