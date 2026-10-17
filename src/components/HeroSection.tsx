import { useState, useEffect } from 'react';

export default function HeroSection() {
  const [particles, setParticles] = useState<Array<{ id: number; x: number; y: number; size: number; speed: number; opacity: number }>>([]);

  useEffect(() => {
    const p = Array.from({ length: 30 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 4 + 1,
      speed: Math.random() * 2 + 0.5,
      opacity: Math.random() * 0.5 + 0.2,
    }));
    setParticles(p);
  }, []);

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20">
      {/* Animated fire particles */}
      <div className="absolute inset-0 overflow-hidden">
        {particles.map((p) => (
          <div
            key={p.id}
            className="absolute rounded-full bg-orange-400"
            style={{
              left: `${p.x}%`,
              top: `${p.y}%`,
              width: `${p.size}px`,
              height: `${p.size}px`,
              opacity: p.opacity,
              animation: `float ${p.speed + 3}s ease-in-out infinite alternate`,
              animationDelay: `${p.id * 0.1}s`,
            }}
          />
        ))}
      </div>

      {/* Central forge glow */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="w-[600px] h-[600px] rounded-full bg-gradient-radial from-orange-600/10 via-red-900/5 to-transparent blur-2xl" />
      </div>

      <div className="relative z-10 text-center px-6 max-w-5xl mx-auto">
        {/* Protocol badge */}
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-orange-500/10 to-red-500/10 border border-orange-500/20 mb-8">
          <svg className="w-4 h-4 text-orange-400" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M12.395 2.553a1 1 0 00-1.45-.385c-.345.23-.614.558-.822.88-.214.33-.403.713-.57 1.116-.334.804-.614 1.768-.84 2.734a31.365 31.365 0 00-.613 3.58 2.64 2.64 0 01-.945-1.067c-.328-.68-.398-1.534-.398-2.654A1 1 0 005.05 6.05 6.981 6.981 0 003 11a7 7 0 1011.95-4.95c-.592-.591-.98-.985-1.348-1.467-.363-.476-.724-1.063-1.207-2.03zM12.12 15.12A3 3 0 017 13s.879.5 2.5.5c0-1 .5-4 1.25-4.5.5 1 .786 1.293 1.371 1.879A2.99 2.99 0 0113 13a2.99 2.99 0 01-.879 2.121z" clipRule="evenodd" />
          </svg>
          <span className="text-sm text-orange-300 font-medium">AI System Protocol v2.0</span>
        </div>

        {/* Main title */}
        <h1 className="text-6xl md:text-8xl font-black mb-6 leading-tight">
          <span className="bg-gradient-to-r from-orange-300 via-amber-400 to-red-500 bg-clip-text text-transparent">
            CRUCIBLE
          </span>
          <br />
          <span className="text-3xl md:text-5xl text-gray-300 font-light">Protocol</span>
        </h1>

        {/* Subtitle */}
        <p className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto mb-10 leading-relaxed">
          A system that <span className="text-orange-400 font-medium">forges solutions</span> through rigorous testing.
          Designed to address the fundamental weaknesses of AI systems with 20 battle-tested protocols.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <button className="px-8 py-4 bg-gradient-to-r from-orange-500 to-red-600 rounded-xl font-semibold text-white shadow-lg shadow-orange-500/25 hover:shadow-orange-500/40 transition-all hover:scale-105">
            Explore Protocol
          </button>
          <button className="px-8 py-4 border border-orange-500/30 rounded-xl font-semibold text-orange-300 hover:bg-orange-500/10 transition-all">
            View Dashboard
          </button>
        </div>

        {/* Stats row */}
        <div className="mt-16 grid grid-cols-3 gap-8 max-w-lg mx-auto">
          {[
            { value: '20', label: 'Protocols' },
            { value: '5', label: 'Dream Team' },
            { value: '∞', label: 'Iterations' },
          ].map((stat) => (
            <div key={stat.label} className="text-center">
              <div className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent">
                {stat.value}
              </div>
              <div className="text-xs text-gray-500 mt-1">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
        <svg className="w-6 h-6 text-orange-400/60" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
        </svg>
      </div>
    </section>
  );
}
