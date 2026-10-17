import { useState, useEffect } from 'react';

function AnimatedCounter({ target, duration = 2000 }: { target: number; duration?: number }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let start = 0;
    const increment = target / (duration / 16);
    const timer = setInterval(() => {
      start += increment;
      if (start >= target) {
        setCount(target);
        clearInterval(timer);
      } else {
        setCount(Math.floor(start));
      }
    }, 16);
    return () => clearInterval(timer);
  }, [target, duration]);

  return <span>{count}</span>;
}

export default function ProtocolStats() {
  const stats = [
    {
      icon: '🔥',
      value: 20,
      suffix: '',
      label: 'Active Protocols',
      description: 'Battle-tested features addressing core AI weaknesses',
    },
    {
      icon: '⚡',
      value: 99,
      suffix: '%',
      label: 'Uptime Guarantee',
      description: 'Protocol systems maintain continuous operation',
    },
    {
      icon: '🛡️',
      value: 5,
      suffix: '',
      label: 'Safety Layers',
      description: 'Multiple redundant safety verification systems',
    },
    {
      icon: '🧬',
      value: 100,
      suffix: 'K+',
      label: 'Test Iterations',
      description: 'Rigorous testing cycles before deployment',
    },
  ];

  return (
    <section className="py-16 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="rounded-3xl bg-gradient-to-br from-gray-900 to-gray-950 border border-gray-800 p-8 md:p-12">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            {stats.map((stat) => (
              <div key={stat.label} className="text-center group">
                <div className="text-4xl mb-4 group-hover:scale-110 transition-transform">
                  {stat.icon}
                </div>
                <div className="text-3xl md:text-4xl font-black bg-gradient-to-r from-orange-300 to-red-400 bg-clip-text text-transparent mb-2">
                  <AnimatedCounter target={stat.value} />
                  {stat.suffix}
                </div>
                <div className="text-white font-medium text-sm mb-1">{stat.label}</div>
                <div className="text-gray-500 text-xs">{stat.description}</div>
              </div>
            ))}
          </div>

          {/* Divider with forge line */}
          <div className="mt-10 pt-10 border-t border-gray-800">
            <div className="flex flex-col md:flex-row items-center justify-between gap-6">
              <div>
                <h3 className="text-xl font-bold text-white mb-1">Protocol Health Monitor</h3>
                <p className="text-gray-400 text-sm">Real-time status of all CRUCIBLE systems</p>
              </div>
              <div className="flex items-center gap-6">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-green-400 animate-pulse" />
                  <span className="text-sm text-gray-300">All Systems Operational</span>
                </div>
                <div className="px-3 py-1 rounded-full bg-orange-500/10 border border-orange-500/20">
                  <span className="text-xs text-orange-400 font-mono">v2.0.1</span>
                </div>
              </div>
            </div>

            {/* Mini status bars */}
            <div className="mt-8 grid grid-cols-2 md:grid-cols-5 gap-4">
              {[
                { name: 'Memory', status: 98 },
                { name: 'Safety', status: 100 },
                { name: 'Accuracy', status: 95 },
                { name: 'Speed', status: 92 },
                { name: 'Alignment', status: 97 },
              ].map((system) => (
                <div key={system.name} className="p-3 rounded-lg bg-gray-800/50">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-gray-400">{system.name}</span>
                    <span className="text-xs text-green-400">{system.status}%</span>
                  </div>
                  <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-green-500 to-emerald-400 rounded-full"
                      style={{ width: `${system.status}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
