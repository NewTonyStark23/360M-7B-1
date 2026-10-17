import { useState } from 'react';

const features = [
  {
    id: 1,
    name: 'Context Bleed Prevention',
    category: 'Memory',
    severity: 'critical',
    description: 'Prevents context from previous conversations leaking into unrelated sessions, maintaining strict isolation between interaction threads.',
    status: 'active',
    effectiveness: 97,
  },
  {
    id: 2,
    name: 'Intent Drift Detection',
    category: 'Alignment',
    severity: 'high',
    description: 'Monitors and detects when AI responses gradually drift away from the user\'s original intent, triggering realignment protocols.',
    status: 'active',
    effectiveness: 94,
  },
  {
    id: 3,
    name: 'Hallucination Guard',
    category: 'Accuracy',
    severity: 'critical',
    description: 'Multi-layer verification system that cross-references generated content against known facts, flagging potential fabrications before output.',
    status: 'active',
    effectiveness: 91,
  },
  {
    id: 4,
    name: 'Token Budget Manager',
    category: 'Efficiency',
    severity: 'medium',
    description: 'Intelligent token allocation system that prioritizes critical information and compresses less important context to stay within limits.',
    status: 'active',
    effectiveness: 89,
  },
  {
    id: 5,
    name: 'Persona Consistency Lock',
    category: 'Identity',
    severity: 'high',
    description: 'Ensures the AI maintains consistent personality traits, knowledge boundaries, and behavioral patterns throughout extended interactions.',
    status: 'active',
    effectiveness: 96,
  },
  {
    id: 6,
    name: 'Recursive Self-Correction',
    category: 'Quality',
    severity: 'high',
    description: 'Enables the AI to review its own outputs, identify errors or inconsistencies, and self-correct before finalizing responses.',
    status: 'active',
    effectiveness: 88,
  },
  {
    id: 7,
    name: 'Ambiguity Resolution Engine',
    category: 'Understanding',
    severity: 'medium',
    description: 'Detects ambiguous user inputs and proactively seeks clarification rather than making assumptions that lead to incorrect outputs.',
    status: 'active',
    effectiveness: 92,
  },
  {
    id: 8,
    name: 'Knowledge Boundary Awareness',
    category: 'Safety',
    severity: 'critical',
    description: 'Clearly defines and communicates the boundaries of the AI\'s knowledge, preventing overconfident assertions beyond its training data.',
    status: 'active',
    effectiveness: 95,
  },
  {
    id: 9,
    name: 'Emotional State Tracker',
    category: 'Empathy',
    severity: 'medium',
    description: 'Monitors user emotional cues and adjusts response tone accordingly, providing appropriate empathy or directness as needed.',
    status: 'active',
    effectiveness: 87,
  },
  {
    id: 10,
    name: 'Multi-Step Reasoning Validator',
    category: 'Logic',
    severity: 'high',
    description: 'Validates each step in multi-step reasoning chains, catching logical fallacies or incorrect intermediate conclusions early.',
    status: 'active',
    effectiveness: 90,
  },
  {
    id: 11,
    name: 'Source Attribution System',
    category: 'Transparency',
    severity: 'medium',
    description: 'Automatically attributes claims to their source material, allowing users to verify information and understand confidence levels.',
    status: 'active',
    effectiveness: 85,
  },
  {
    id: 12,
    name: 'Conversation State Snapshot',
    category: 'Memory',
    severity: 'high',
    description: 'Creates periodic snapshots of conversation state, enabling recovery from context loss and maintaining continuity across sessions.',
    status: 'active',
    effectiveness: 93,
  },
  {
    id: 13,
    name: 'Bias Detection Filter',
    category: 'Fairness',
    severity: 'critical',
    description: 'Real-time monitoring for cognitive biases in AI outputs, including confirmation bias, anchoring effects, and cultural assumptions.',
    status: 'active',
    effectiveness: 86,
  },
  {
    id: 14,
    name: 'Task Decomposition Engine',
    category: 'Planning',
    severity: 'medium',
    description: 'Breaks complex tasks into manageable sub-tasks with clear dependencies, preventing the AI from becoming overwhelmed by scope.',
    status: 'active',
    effectiveness: 91,
  },
  {
    id: 15,
    name: 'Feedback Integration Loop',
    category: 'Learning',
    severity: 'medium',
    description: 'Systematically incorporates user feedback within the current session, adjusting behavior in real-time without requiring model retraining.',
    status: 'active',
    effectiveness: 88,
  },
  {
    id: 16,
    name: 'Priority Conflict Resolver',
    category: 'Alignment',
    severity: 'high',
    description: 'When multiple instructions conflict, this protocol determines priority hierarchy and resolves contradictions transparently.',
    status: 'active',
    effectiveness: 89,
  },
  {
    id: 17,
    name: 'Output Format Adapter',
    category: 'Flexibility',
    severity: 'low',
    description: 'Dynamically adapts output format based on context—code blocks for technical queries, prose for explanations, tables for comparisons.',
    status: 'active',
    effectiveness: 94,
  },
  {
    id: 18,
    name: 'Confidence Calibration',
    category: 'Accuracy',
    severity: 'high',
    description: 'Ensures the AI\'s expressed confidence matches its actual accuracy, preventing both overconfidence and unnecessary hedging.',
    status: 'active',
    effectiveness: 87,
  },
  {
    id: 19,
    name: 'Safety Boundary Monitor',
    category: 'Safety',
    severity: 'critical',
    description: 'Continuous monitoring for outputs that might cross ethical, legal, or safety boundaries, with multiple fallback layers of protection.',
    status: 'active',
    effectiveness: 98,
  },
  {
    id: 20,
    name: 'Session Continuity Bridge',
    category: 'Memory',
    severity: 'high',
    description: 'Maintains thread continuity across session boundaries, ensuring that context from previous interactions is properly bridged.',
    status: 'active',
    effectiveness: 92,
  },
];

const categories = ['All', ...Array.from(new Set(features.map(f => f.category)))];

const severityColors: Record<string, string> = {
  critical: 'bg-red-500/10 text-red-400 border-red-500/30',
  high: 'bg-orange-500/10 text-orange-400 border-orange-500/30',
  medium: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30',
  low: 'bg-green-500/10 text-green-400 border-green-500/30',
};

export default function FeatureDashboard() {
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [expandedFeature, setExpandedFeature] = useState<number | null>(null);

  const filteredFeatures = selectedCategory === 'All'
    ? features
    : features.filter(f => f.category === selectedCategory);

  return (
    <section className="py-24 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-red-500/10 border border-red-500/20 mb-4">
            <span className="text-xs text-red-400 font-medium">PROTOCOL DASHBOARD</span>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="bg-gradient-to-r from-red-300 to-orange-400 bg-clip-text text-transparent">
              20 Feature Protocols
            </span>
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Each protocol addresses a specific AI weakness, forged through rigorous testing and iteration.
          </p>
        </div>

        {/* Category filter */}
        <div className="flex flex-wrap justify-center gap-2 mb-10">
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                selectedCategory === cat
                  ? 'bg-orange-500/20 text-orange-300 border border-orange-500/30'
                  : 'bg-gray-800/50 text-gray-400 border border-gray-700 hover:border-gray-600'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>

        {/* Feature grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {filteredFeatures.map((feature) => (
            <div
              key={feature.id}
              className={`rounded-xl border transition-all duration-300 cursor-pointer ${
                expandedFeature === feature.id
                  ? 'border-orange-500/40 bg-gray-900/80 shadow-lg shadow-orange-500/5'
                  : 'border-gray-800 bg-gray-900/30 hover:border-gray-700'
              }`}
              onClick={() => setExpandedFeature(expandedFeature === feature.id ? null : feature.id)}
            >
              <div className="p-5">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span className="flex items-center justify-center w-8 h-8 rounded-lg bg-orange-500/10 text-orange-400 text-sm font-bold">
                      {feature.id}
                    </span>
                    <div>
                      <h3 className="font-semibold text-white text-sm">{feature.name}</h3>
                      <span className="text-xs text-gray-500">{feature.category}</span>
                    </div>
                  </div>
                  <span className={`px-2 py-0.5 rounded-full text-xs border ${severityColors[feature.severity]}`}>
                    {feature.severity}
                  </span>
                </div>

                {expandedFeature === feature.id && (
                  <div className="mt-4 pt-4 border-t border-gray-800">
                    <p className="text-gray-400 text-sm leading-relaxed mb-4">
                      {feature.description}
                    </p>
                    <div className="flex items-center gap-4">
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs text-gray-500">Effectiveness</span>
                          <span className="text-xs text-orange-400 font-medium">{feature.effectiveness}%</span>
                        </div>
                        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-orange-500 to-red-500 rounded-full transition-all duration-1000"
                            style={{ width: `${feature.effectiveness}%` }}
                          />
                        </div>
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                        <span className="text-xs text-green-400">{feature.status}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Summary stats */}
        <div className="mt-12 grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: 'Critical Protocols', value: features.filter(f => f.severity === 'critical').length, color: 'text-red-400' },
            { label: 'High Priority', value: features.filter(f => f.severity === 'high').length, color: 'text-orange-400' },
            { label: 'Avg Effectiveness', value: `${Math.round(features.reduce((a, f) => a + f.effectiveness, 0) / features.length)}%`, color: 'text-amber-400' },
            { label: 'Active Protocols', value: features.filter(f => f.status === 'active').length, color: 'text-green-400' },
          ].map((stat) => (
            <div key={stat.label} className="p-4 rounded-xl bg-gray-900/50 border border-gray-800 text-center">
              <div className={`text-2xl font-bold ${stat.color}`}>{stat.value}</div>
              <div className="text-xs text-gray-500 mt-1">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
