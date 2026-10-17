export default function Footer() {
  return (
    <footer className="relative py-16 px-6 border-t border-gray-800/50">
      {/* Top glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1/2 h-px bg-gradient-to-r from-transparent via-orange-500/50 to-transparent" />

      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-10 mb-12">
          {/* Brand */}
          <div className="md:col-span-2">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z" />
                </svg>
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-orange-400 to-red-500 bg-clip-text text-transparent">
                CRUCIBLE Protocol
              </span>
            </div>
            <p className="text-gray-400 text-sm leading-relaxed max-w-md mb-4">
              Forging the future of AI systems through rigorous testing, adapter philosophy,
              and the combined strength of the world's leading technology teams.
            </p>
            <div className="flex items-center gap-4">
              <a href="https://github.com/NewTonyStark23/CRUCIBLE_Protocol" target="_blank" rel="noopener noreferrer" className="text-gray-500 hover:text-orange-400 transition-colors">
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                </svg>
              </a>
            </div>
          </div>

          {/* Links */}
          <div>
            <h4 className="text-white font-semibold text-sm mb-4">Protocol</h4>
            <ul className="space-y-2">
              {['Overview', 'Features', 'Dashboard', 'Documentation'].map((link) => (
                <li key={link}>
                  <a href="#" className="text-gray-500 text-sm hover:text-orange-400 transition-colors">{link}</a>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="text-white font-semibold text-sm mb-4">Team</h4>
            <ul className="space-y-2">
              {['Apple', 'NVIDIA', 'Google', 'Microsoft', 'Tony Stark'].map((link) => (
                <li key={link}>
                  <a href="#" className="text-gray-500 text-sm hover:text-orange-400 transition-colors">{link}</a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="pt-8 border-t border-gray-800/50 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-gray-600 text-xs">
            © 2026 CRUCIBLE Protocol. Forged with 🔥 by the Dream Team.
          </p>
          <div className="flex items-center gap-4">
            <span className="inline-flex items-center gap-1 text-xs text-gray-500">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
              All systems operational
            </span>
            <span className="text-xs text-gray-600">v2.0.1</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
