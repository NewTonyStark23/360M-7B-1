// ============================================================
// ATOMIC MIND GRAPH
// Graph-based knowledge structure with zoomable detail levels
// Enables unlimited context through hierarchical knowledge mapping
// ============================================================

export interface AtomicNode {
  id: string;
  concept: string;
  summary: string;           // High-level summary
  details: string[];         // Detailed atomic notes
  connections: string[];     // Connected node IDs
  importance: number;        // 0-1 relevance score
  lastAccessed: number;      // Timestamp
  depth: number;             // Zoom level (0 = overview, 1+ = detail)
  embedding?: number[];      // Semantic embedding for similarity search
}

export interface AtomicEdge {
  from: string;
  to: string;
  relationship: string;      // "related_to", "causes", "part_of", etc.
  strength: number;          // 0-1 connection strength
}

export class AtomicMindGraph {
  private nodes: Map<string, AtomicNode>;
  private edges: AtomicEdge[];
  private rootNodes: string[];
  private currentFocus: string | null;
  private zoomLevel: number;
  
  constructor() {
    this.nodes = new Map();
    this.edges = [];
    this.rootNodes = [];
    this.currentFocus = null;
    this.zoomLevel = 0;
    this.initializeBaseKnowledge();
  }

  private initializeBaseKnowledge(): void {
    // Pre-load foundational knowledge atoms
    const baseConcepts = [
      {
        id: 'programming',
        concept: 'Programming',
        summary: 'Writing instructions for computers',
        details: [
          'Programming is the process of creating software',
          'Uses languages like Python, JavaScript, Rust',
          'Involves logic, algorithms, and data structures',
          'Can be functional, object-oriented, or procedural',
        ],
        connections: ['algorithms', 'data_structures', 'languages'],
        importance: 0.95,
      },
      {
        id: 'algorithms',
        concept: 'Algorithms',
        summary: 'Step-by-step problem solving procedures',
        details: [
          'Algorithms are finite sequences of well-defined instructions',
          'Common types: sorting, searching, graph traversal',
          'Measured by time and space complexity',
          'Examples: quicksort, binary search, Dijkstra',
        ],
        connections: ['programming', 'data_structures', 'complexity'],
        importance: 0.9,
      },
      {
        id: 'ai',
        concept: 'Artificial Intelligence',
        summary: 'Systems that mimic human intelligence',
        details: [
          'AI enables machines to learn from data',
          'Includes machine learning, deep learning, NLP',
          'Transformers revolutionized language understanding',
          'Applications: chatbots, image recognition, autonomous systems',
        ],
        connections: ['machine_learning', 'neural_networks', 'nlp'],
        importance: 0.95,
      },
      {
        id: 'machine_learning',
        concept: 'Machine Learning',
        summary: 'AI that learns from data patterns',
        details: [
          'ML algorithms improve through experience',
          'Types: supervised, unsupervised, reinforcement',
          'Requires training data and feature engineering',
          'Models: decision trees, neural networks, SVMs',
        ],
        connections: ['ai', 'neural_networks', 'data'],
        importance: 0.9,
      },
      {
        id: 'neural_networks',
        concept: 'Neural Networks',
        summary: 'Computing systems inspired by biological brains',
        details: [
          'Composed of layers of interconnected nodes',
          'Learn by adjusting connection weights',
          'Types: feedforward, CNN, RNN, Transformer',
          'Enable deep learning with multiple layers',
        ],
        connections: ['machine_learning', 'ai', 'transformers'],
        importance: 0.9,
      },
      {
        id: 'transformers',
        concept: 'Transformers',
        summary: 'Attention-based neural architecture',
        details: [
          'Use self-attention mechanism',
          'Process sequences in parallel',
          'Foundation for modern LLMs',
          'Key components: encoder, decoder, attention heads',
        ],
        connections: ['neural_networks', 'nlp', 'llm'],
        importance: 0.95,
      },
    ];

    for (const concept of baseConcepts) {
      this.addNode({
        id: concept.id,
        concept: concept.concept,
        summary: concept.summary,
        details: concept.details,
        connections: concept.connections,
        importance: concept.importance,
        lastAccessed: Date.now(),
        depth: 0,
      });
      this.rootNodes.push(concept.id);
    }

    // Create edges
    for (const node of baseConcepts) {
      for (const conn of node.connections) {
        if (!this.edges.find(e => e.from === node.id && e.to === conn)) {
          this.edges.push({
            from: node.id,
            to: conn,
            relationship: 'related_to',
            strength: 0.8,
          });
        }
      }
    }
  }

  addNode(node: AtomicNode): void {
    this.nodes.set(node.id, node);
  }

  addEdge(edge: AtomicEdge): void {
    this.edges.push(edge);
  }

  // Zoom into a concept for more detail
  zoomIn(nodeId: string): AtomicNode | null {
    const node = this.nodes.get(nodeId);
    if (!node) return null;
    
    node.depth++;
    node.lastAccessed = Date.now();
    this.currentFocus = nodeId;
    this.zoomLevel = node.depth;
    
    return node;
  }

  // Zoom out to see broader context
  zoomOut(nodeId: string): AtomicNode | null {
    const node = this.nodes.get(nodeId);
    if (!node) return null;
    
    node.depth = Math.max(0, node.depth - 1);
    this.zoomLevel = node.depth;
    
    return node;
  }

  // Get current view based on zoom level
  getView(nodeId: string, zoomLevel?: number): { summary: string; details: string[] } {
    const node = this.nodes.get(nodeId);
    if (!node) return { summary: '', details: [] };
    
    const level = zoomLevel ?? node.depth;
    
    if (level === 0) {
      // Overview: just summary
      return { summary: node.summary, details: [] };
    } else {
      // Detailed: include atomic notes
      const detailCount = Math.min(node.details.length, level * 2);
      return {
        summary: node.summary,
        details: node.details.slice(0, detailCount),
      };
    }
  }

  // Find related concepts
  getRelatedConcepts(nodeId: string, maxResults = 5): AtomicNode[] {
    const node = this.nodes.get(nodeId);
    if (!node) return [];
    
    const related: AtomicNode[] = [];
    for (const connId of node.connections) {
      const connNode = this.nodes.get(connId);
      if (connNode) {
        related.push(connNode);
      }
    }
    
    // Sort by importance
    related.sort((a, b) => b.importance - a.importance);
    return related.slice(0, maxResults);
  }

  // Search for concepts by keyword
  search(query: string): AtomicNode[] {
    const results: AtomicNode[] = [];
    const lowerQuery = query.toLowerCase();
    
    for (const node of this.nodes.values()) {
      if (
        node.concept.toLowerCase().includes(lowerQuery) ||
        node.summary.toLowerCase().includes(lowerQuery) ||
        node.details.some(d => d.toLowerCase().includes(lowerQuery))
      ) {
        results.push(node);
      }
    }
    
    results.sort((a, b) => b.importance - a.importance);
    return results;
  }

  // Add new knowledge atom (self-learning)
  learnNewConcept(concept: string, details: string[], relatedTo?: string): void {
    const id = concept.toLowerCase().replace(/\s+/g, '_');
    
    if (this.nodes.has(id)) {
      // Update existing node
      const node = this.nodes.get(id)!;
      node.details.push(...details);
      node.lastAccessed = Date.now();
    } else {
      // Create new node
      this.addNode({
        id,
        concept,
        summary: details[0] || concept,
        details,
        connections: relatedTo ? [relatedTo] : [],
        importance: 0.5,
        lastAccessed: Date.now(),
        depth: 0,
      });
      
      if (relatedTo) {
        this.addEdge({
          from: id,
          to: relatedTo,
          relationship: 'related_to',
          strength: 0.7,
        });
      }
    }
  }

  // Get graph statistics
  getStats(): { nodes: number; edges: number; rootNodes: number } {
    return {
      nodes: this.nodes.size,
      edges: this.edges.length,
      rootNodes: this.rootNodes.length,
    };
  }

  // Get all root concepts
  getRootConcepts(): AtomicNode[] {
    return this.rootNodes.map(id => this.nodes.get(id)!).filter(n => n);
  }

  // Export graph for visualization
  exportGraph(): { nodes: AtomicNode[]; edges: AtomicEdge[] } {
    return {
      nodes: Array.from(this.nodes.values()),
      edges: this.edges,
    };
  }
}
