// ============================================================
// SELF-VETTER
// Validates output correctness and triggers self-learning
// Ensures message accuracy and improves over time
// ============================================================

export interface VetResult {
  isValid: boolean;
  confidence: number;      // 0-1 confidence in correctness
  issues: string[];        // Detected issues
  corrections: string[];   // Suggested corrections
  learned: boolean;        // Did the model learn from this?
}

export interface ValidationRule {
  id: string;
  name: string;
  check: (text: string) => { valid: boolean; issue?: string };
}

export class SelfVetter {
  private rules: ValidationRule[];
  private correctionLog: Array<{ input: string; output: string; correction: string }>;
  private accuracyHistory: number[];
  
  constructor() {
    this.rules = [];
    this.correctionLog = [];
    this.accuracyHistory = [];
    this.initializeRules();
  }

  private initializeRules(): void {
    // Grammar rules
    this.rules.push({
      id: 'capitalization',
      name: 'Sentence Capitalization',
      check: (text: string) => {
        const sentences = text.split(/[.!?]+/).filter(s => s.trim());
        for (const sentence of sentences) {
          const trimmed = sentence.trim();
          if (trimmed && trimmed[0] !== trimmed[0].toUpperCase()) {
            return { valid: false, issue: `Sentence should start with capital: "${trimmed.slice(0, 20)}..."` };
          }
        }
        return { valid: true };
      },
    });

    this.rules.push({
      id: 'punctuation',
      name: 'Proper Punctuation',
      check: (text: string) => {
        if (text.length > 10 && !/[.!?]$/.test(text.trim())) {
          return { valid: false, issue: 'Text should end with proper punctuation' };
        }
        return { valid: true };
      },
    });

    // Code validation rules
    this.rules.push({
      id: 'code_braces',
      name: 'Balanced Braces',
      check: (text: string) => {
        if (!text.includes('{')) return { valid: true };
        
        let braceCount = 0;
        for (const char of text) {
          if (char === '{') braceCount++;
          if (char === '}') braceCount--;
          if (braceCount < 0) {
            return { valid: false, issue: 'Unbalanced braces: extra closing brace' };
          }
        }
        if (braceCount !== 0) {
          return { valid: false, issue: `Unbalanced braces: ${braceCount} unclosed` };
        }
        return { valid: true };
      },
    });

    this.rules.push({
      id: 'code_parentheses',
      name: 'Balanced Parentheses',
      check: (text: string) => {
        if (!text.includes('(')) return { valid: true };
        
        let parenCount = 0;
        for (const char of text) {
          if (char === '(') parenCount++;
          if (char === ')') parenCount--;
          if (parenCount < 0) {
            return { valid: false, issue: 'Unbalanced parentheses: extra closing' };
          }
        }
        if (parenCount !== 0) {
          return { valid: false, issue: `Unbalanced parentheses: ${parenCount} unclosed` };
        }
        return { valid: true };
      },
    });

    // Semantic validation
    this.rules.push({
      id: 'coherence',
      name: 'Text Coherence',
      check: (text: string) => {
        const words = text.toLowerCase().split(/\s+/);
        const uniqueWords = new Set(words);
        const diversity = uniqueWords.size / words.length;
        
        if (diversity < 0.3 && words.length > 20) {
          return { valid: false, issue: 'Low lexical diversity - text may be repetitive' };
        }
        return { valid: true };
      },
    });

    this.rules.push({
      id: 'length',
      name: 'Appropriate Length',
      check: (text: string) => {
        const words = text.split(/\s+/).length;
        if (words < 3) {
          return { valid: false, issue: 'Text too short' };
        }
        if (words > 1000) {
          return { valid: false, issue: 'Text too long - consider breaking into paragraphs' };
        }
        return { valid: true };
      },
    });
  }

  // Vet output text
  vet(text: string, context?: string): VetResult {
    const issues: string[] = [];
    const corrections: string[] = [];
    
    // Run all validation rules
    for (const rule of this.rules) {
      const result = rule.check(text);
      if (!result.valid && result.issue) {
        issues.push(`[${rule.name}] ${result.issue}`);
        
        // Generate correction suggestion
        const correction = this.suggestCorrection(text, rule.id);
        if (correction) {
          corrections.push(correction);
        }
      }
    }
    
    const isValid = issues.length === 0;
    const confidence = isValid ? 1.0 : Math.max(0, 1 - (issues.length * 0.2));
    
    // Self-learning: if issues found, log for improvement
    let learned = false;
    if (!isValid && corrections.length > 0) {
      this.logCorrection(context || '', text, corrections[0]);
      learned = true;
    }
    
    // Update accuracy history
    this.accuracyHistory.push(confidence);
    
    return {
      isValid,
      confidence,
      issues,
      corrections,
      learned,
    };
  }

  // Suggest correction for an issue
  private suggestCorrection(text: string, ruleId: string): string | null {
    switch (ruleId) {
      case 'capitalization':
        // Capitalize first letter of each sentence
        return text.replace(/(^\s*|[.!?]\s+)([a-z])/g, (_, prefix, letter) => prefix + letter.toUpperCase());
      
      case 'punctuation':
        // Add period if missing
        return text.trim() + '.';
      
      case 'code_braces':
      case 'code_parentheses':
        // Can't auto-fix balanced issues without more context
        return null;
      
      default:
        return null;
    }
  }

  // Log correction for self-learning
  private logCorrection(input: string, output: string, correction: string): void {
    this.correctionLog.push({
      input: input.slice(-100),
      output: output.slice(0, 200),
      correction: correction.slice(0, 200),
    });
    
    // Keep only last 1000 corrections
    if (this.correctionLog.length > 1000) {
      this.correctionLog = this.correctionLog.slice(-1000);
    }
  }

  // Get vetting statistics
  getStats(): {
    totalVetted: number;
    averageConfidence: number;
    totalCorrections: number;
    commonIssues: Array<{ issue: string; count: number }>;
  } {
    const totalVetted = this.accuracyHistory.length;
    const averageConfidence = totalVetted > 0
      ? this.accuracyHistory.reduce((a, b) => a + b, 0) / totalVetted
      : 0;
    
    // Count common issues
    const issueCounts = new Map<string, number>();
    for (const log of this.correctionLog) {
      const issue = log.correction.split(']')[0].replace('[', '');
      issueCounts.set(issue, (issueCounts.get(issue) || 0) + 1);
    }
    
    const commonIssues = Array.from(issueCounts.entries())
      .map(([issue, count]) => ({ issue, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);
    
    return {
      totalVetted,
      averageConfidence,
      totalCorrections: this.correctionLog.length,
      commonIssues,
    };
  }

  // Add custom validation rule
  addRule(rule: ValidationRule): void {
    this.rules.push(rule);
  }

  // Get recent corrections
  getRecentCorrections(limit = 10): Array<{ input: string; output: string; correction: string }> {
    return this.correctionLog.slice(-limit).reverse();
  }
}
