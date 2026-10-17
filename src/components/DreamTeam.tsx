import { useState } from 'react';

const teamMembers = [
  {
    name: 'Apple',
    role: 'Design & UX Excellence',
    icon: '🍎',
    color: 'from-gray-700 to-gray-900',
    borderColor: 'border-gray-500/30',
    description: 'Seamless user experience design, privacy-first architecture, and elegant system integration.',
    strengths: ['Human Interface Design', 'Privacy Architecture', 'Ecosystem Integration'],
  },
  {
    name: 'NVIDIA',
    role: 'Compute & AI Infrastructure',
    icon: '🟢',
    color: 'from-green-700 to-green-900',
    borderColor: 'border-green-500/30',
    description: 'Unmatched GPU compute power, CUDA acceleration, and cutting-edge AI model training infrastructure.',
    strengths: ['GPU Computing', 'CUDA Ecosystem', 'Model Training'],
  },
  {
    name: 'Google',
    role: 'Research & Scale',
    icon: '🔍',
    color: 'from-blue-700 to-blue-900',
    borderColor: 'border-blue-500/30',
    description: 'World-class AI research, massive-scale distributed systems, and breakthrough transformer architectures.',
    strengths: ['AI Research', 'Distributed Systems', 'Transformer Models'],
  },
  {
    name: 'Microsoft',
    role: 'Enterprise & Cloud',
    icon: '🪟',
    color: 'from-cyan-700 to-cyan-900',
    borderColor: 'border-cyan-500/30',
    description: 'Enterprise-grade cloud infrastructure, developer tools, and responsible AI frameworks.',
    strengths: ['Cloud Infrastructure', 'Developer Tools', 'Enterprise AI'],
  },
  {
    name: 'Tony Stark',
    role: 'Innovation & Vision',
    icon: '⚡',
    color: 'from-red-700 to-orange-900',
    borderColor: 'border-red-500/30',
    description: 'Visionary innovation, rapid prototyping, and the audacity to push boundaries beyond what\'s possible.',
    strengths: ['Rapid Innovation', 'System Architecture', 'Visionary Design'],
  },
];

export default function DreamTeam() {
  const [activeMember, setActiveMember] = useState(0);

  return (
    <section className="py-24 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-orange-500/10 border border-orange-500/20 mb-4">
            <span className="text-xs text-orange-400 font-medium">DREAM TEAM</span>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="bg-gradient-to-r from-orange-300 to-red-400 bg-clip-text text-transparent">
              Combined Strengths
            </span>
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            The CRUCIBLE Protocol is forged by the combined expertise of the world's leading technology powers.
          </p>
        </div>

        {/* Team grid */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-12">
          {teamMembers.map((member, index) => (
            <button
              key={member.name}
              onClick={() => setActiveMember(index)}
              className={`relative p-6 rounded-2xl border transition-all duration-300 text-left ${
                activeMember === index
                  ? `${member.borderColor} bg-gradient-to-br ${member.color} shadow-lg scale-105`
                  : 'border-gray-800 bg-gray-900/50 hover:border-gray-700'
              }`}
            >
              <div className="text-3xl mb-3">{member.icon}</div>
              <h3 className={`font-bold text-lg ${activeMember === index ? 'text-white' : 'text-gray-300'}`}>
                {member.name}
              </h3>
              <p className={`text-xs mt-1 ${activeMember === index ? 'text-gray-300' : 'text-gray-500'}`}>
                {member.role}
              </p>
              {activeMember === index && (
                <div className="absolute top-2 right-2">
                  <span className="w-2 h-2 rounded-full bg-orange-400 animate-pulse" />
                </div>
              )}
            </button>
          ))}
        </div>

        {/* Active member detail */}
        <div className={`rounded-2xl border ${teamMembers[activeMember].borderColor} bg-gradient-to-br ${teamMembers[activeMember].color} p-8 transition-all duration-500`}>
          <div className="flex flex-col md:flex-row items-start gap-8">
            <div className="text-6xl">{teamMembers[activeMember].icon}</div>
            <div className="flex-1">
              <h3 className="text-2xl font-bold text-white mb-2">
                {teamMembers[activeMember].name}
              </h3>
              <p className="text-orange-300 text-sm font-medium mb-3">
                {teamMembers[activeMember].role}
              </p>
              <p className="text-gray-300 mb-6 leading-relaxed">
                {teamMembers[activeMember].description}
              </p>
              <div className="flex flex-wrap gap-2">
                {teamMembers[activeMember].strengths.map((strength) => (
                  <span
                    key={strength}
                    className="px-3 py-1 rounded-full bg-white/10 text-xs text-gray-200 border border-white/10"
                  >
                    {strength}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
