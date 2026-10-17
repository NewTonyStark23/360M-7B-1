export default function AdapterPhilosophy() {
  const principles = [
    {
      title: 'Resource Reuse',
      icon: '♻️',
      description: 'Never discard what can be repurposed. Like using adapters instead of replacing cables, extend existing AI capabilities rather than rebuilding from scratch.',
      metric: '73% less rebuild',
    },
    {
      title: 'Token Economy',
      icon: '🪙',
      description: 'Every token is precious. The Adapter Philosophy ensures maximum information density per token, reducing waste and maximizing output quality.',
      metric: '40% token savings',
    },
    {
      title: 'Extension Over Replacement',
      icon: '🔌',
      description: 'Build adapters and extensions that enhance existing systems rather than wholesale replacements. Modular upgrades beat full rewrites.',
      metric: '5x faster iteration',
    },
    {
      title: 'Context Preservation',
      icon: '🧠',
      description: 'Maintain the full context of previous interactions. Like an adapter that preserves signal integrity, never lose critical information in transitions.',
      metric: '99.7% retention',
    },
  ];

  return (
    <section className="py-24 px-6 relative">
      {/* Background pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0" style={{
          backgroundImage: `repeating-linear-gradient(45deg, transparent, transparent 35px, rgba(251, 146, 60, 0.1) 35px, rgba(251, 146, 60, 0.1) 36px)`,
        }} />
      </div>

      <div className="max-w-7xl mx-auto relative z-10">
        {/* Section header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/20 mb-4">
            <span className="text-xs text-amber-400 font-medium">ADAPTER PHILOSOPHY</span>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="bg-gradient-to-r from-amber-300 to-orange-400 bg-clip-text text-transparent">
              Extend, Don't Replace
            </span>
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            The core philosophy behind CRUCIBLE: use adapters and extensions to maximize resource efficiency,
            just as you'd use an adapter instead of replacing an entire cable system.
          </p>
        </div>

        {/* Visual metaphor */}
        <div className="mb-16 p-8 rounded-2xl bg-gradient-to-r from-amber-900/20 to-orange-900/20 border border-amber-500/20">
          <div className="flex flex-col md:flex-row items-center justify-center gap-8">
            <div className="text-center">
              <div className="text-4xl mb-2">🔌</div>
              <div className="text-sm text-gray-400">Old Cable</div>
              <div className="text-red-400 text-xs line-through">Replace entirely</div>
            </div>
            <div className="flex items-center gap-4">
              <div className="w-16 h-1 bg-gradient-to-r from-red-500 to-orange-500 rounded" />
              <svg className="w-8 h-8 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
              <div className="w-16 h-1 bg-gradient-to-r from-orange-500 to-green-500 rounded" />
            </div>
            <div className="text-center">
              <div className="text-4xl mb-2">🔧</div>
              <div className="text-sm text-gray-400">Adapter</div>
              <div className="text-green-400 text-xs font-medium">Extend & reuse</div>
            </div>
          </div>
        </div>

        {/* Principles grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {principles.map((principle) => (
            <div
              key={principle.title}
              className="group p-6 rounded-2xl bg-gray-900/50 border border-gray-800 hover:border-amber-500/30 transition-all duration-300 hover:shadow-lg hover:shadow-amber-500/5"
            >
              <div className="flex items-start gap-4">
                <div className="text-3xl">{principle.icon}</div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-lg font-bold text-white group-hover:text-amber-300 transition-colors">
                      {principle.title}
                    </h3>
                    <span className="px-2 py-0.5 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-xs">
                      {principle.metric}
                    </span>
                  </div>
                  <p className="text-gray-400 text-sm leading-relaxed">
                    {principle.description}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
