import { useState, useEffect } from 'react';
import HeroSection from './components/HeroSection';
import DreamTeam from './components/DreamTeam';
import AdapterPhilosophy from './components/AdapterPhilosophy';
import FeatureDashboard from './components/FeatureDashboard';
import ProtocolStats from './components/ProtocolStats';
import Footer from './components/Footer';

function App() {
  const [activeSection, setActiveSection] = useState('home');
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white font-sans overflow-x-hidden">
      {/* Ambient fire glow background */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-orange-600/5 rounded-full blur-3xl animate-pulse" />
        <div className="absolute top-1/3 right-1/4 w-80 h-80 bg-red-600/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
        <div className="absolute bottom-1/4 left-1/3 w-72 h-72 bg-amber-500/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />
      </div>

      {/* Navigation */}
      <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${scrollY > 50 ? 'bg-[#0a0a0f]/95 backdrop-blur-md border-b border-orange-900/30' : 'bg-transparent'}`}>
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center shadow-lg shadow-orange-500/20">
              <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z" />
              </svg>
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-orange-400 to-red-500 bg-clip-text text-transparent">
              CRUCIBLE
            </span>
          </div>
          <div className="hidden md:flex items-center gap-6">
            {['home', 'team', 'philosophy', 'dashboard'].map((section) => (
              <button
                key={section}
                onClick={() => {
                  setActiveSection(section);
                  document.getElementById(section)?.scrollIntoView({ behavior: 'smooth' });
                }}
                className={`text-sm capitalize transition-colors ${activeSection === section ? 'text-orange-400' : 'text-gray-400 hover:text-orange-300'}`}
              >
                {section}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full bg-green-500/10 border border-green-500/30 text-green-400 text-xs">
              <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              Protocol Active
            </span>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="relative z-10">
        <div id="home">
          <HeroSection />
        </div>
        <div id="team">
          <DreamTeam />
        </div>
        <div id="philosophy">
          <AdapterPhilosophy />
        </div>
        <ProtocolStats />
        <div id="dashboard">
          <FeatureDashboard />
        </div>
      </main>

      <Footer />
    </div>
  );
}

export default App;
