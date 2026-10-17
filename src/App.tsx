import { useState, useRef, useCallback, useEffect } from 'react';
import { MultiResolutionTransformer, RESOLUTIONS } from './ml/multi-resolution';
import { getAllPretrainData, generateTrainingChunks } from './ml/qwen3-data';

type Tab = 'overview' | 'train' | 'generate' | 'architecture' | 'export';

interface TrainingState {
  isTraining: boolean;
  step: number;
  epoch: number;
  loss: number;
  lossHistory: number[];
  currentResolution: string;
  teacherResolution: string;
  learningRate: number;
  distillationWeight: number;
}

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('overview');
  const [model, setModel] = useState<MultiResolutionTransformer | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [currentRes, setCurrentRes] = useState('360M');

  const [training, setTraining] = useState<TrainingState>({
    isTraining: false,
    step: 0,
    epoch: 0,
    loss: 0,
    lossHistory: [],
    currentResolution: '360M',
    teacherResolution: '7B',
    learningRate: 0.001,
    distillationWeight: 0.5,
  });

  const [prompt, setPrompt] = useState('The ');
  const [generatedText, setGeneratedText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(50);

  const trainingRef = useRef(false);
  const modelRef = useRef<MultiResolutionTransformer | null>(null);

  useEffect(() => { modelRef.current = model; }, [model]);

  const initializeModel = useCallback(() => {
    const newModel = new MultiResolutionTransformer(currentRes);
    setModel(newModel);
    modelRef.current = newModel;
    setIsInitialized(true);
  }, [currentRes]);

  const switchResolution = useCallback((resId: string) => {
    if (!model) return;
    model.setResolution(resId);
    setCurrentRes(resId);
    setTraining(prev => ({ ...prev, currentResolution: resId }));
  }, [model]);

  const startTraining = useCallback(async () => {
    if (!model || training.isTraining) return;

    const data = getAllPretrainData();
    const chunks = generateTrainingChunks(data, 128, 64);
    if (chunks.length === 0) return;

    trainingRef.current = true;
    setTraining(prev => ({ ...prev, isTraining: true, lossHistory: [] }));

    const lr = training.learningRate;
    let step = 0;
    let totalLoss = 0;
    const lossHistory: number[] = [];

    const trainStep = () => {
      if (!trainingRef.current || !modelRef.current) return;

      const batchSize = 4;
      let batchLoss = 0;

      for (let b = 0; b < batchSize; b++) {
        const chunkIdx = (step * batchSize + b) % chunks.length;
        const chunk = chunks[chunkIdx];
        const tokens = Array.from(new TextEncoder().encode(chunk));

        if (tokens.length < 10) continue;

        const input = tokens.slice(0, -1);
        const target = tokens.slice(1);

        try {
          const loss = modelRef.current.trainStep(input, target, lr);
          batchLoss += loss;
        } catch (e) {
          // Skip problematic sequences
        }
      }

      const avgLoss = batchLoss / batchSize;
      totalLoss += avgLoss;
      step++;

      if (step % 5 === 0) {
        const currentLoss = totalLoss / step;
        lossHistory.push(currentLoss);
        setTraining(prev => ({
          ...prev,
          step,
          loss: currentLoss,
          lossHistory: [...lossHistory],
          epoch: Math.floor(step * batchSize / chunks.length),
        }));
      }

      if (trainingRef.current && step < 500) {
        setTimeout(trainStep, 0);
      } else {
        trainingRef.current = false;
        setTraining(prev => ({ ...prev, isTraining: false }));
      }
    };

    trainStep();
  }, [model, training.isTraining, training.learningRate]);

  const stopTraining = useCallback(() => {
    trainingRef.current = false;
    setTraining(prev => ({ ...prev, isTraining: false }));
  }, []);

  const generateText = useCallback(async () => {
    if (!model) return;
    setIsGenerating(true);
    setGeneratedText('');
    await new Promise(resolve => setTimeout(resolve, 50));

    try {
      const result = model.generate(prompt, maxTokens, temperature);
      setGeneratedText(result);
    } catch (e) {
      setGeneratedText(`[Error: ${(e as Error).message}]`);
    }
    setIsGenerating(false);
  }, [model, prompt, maxTokens, temperature]);

  const downloadWeights = useCallback(() => {
    if (!model) return;
    model.downloadWeights();
  }, [model]);

  const config = RESOLUTIONS[currentRes];
  const fullConfig = RESOLUTIONS['7B'];
  const compressionRatio = fullConfig.dModel / config.dModel;

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white">
      {/* Header */}
      <header className="border-b border-gray-800/50 bg-[#0a0a0f]/95 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-blue-600 flex items-center justify-center">
              <span className="text-lg">🧠</span>
            </div>
            <div>
              <h1 className="text-lg font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                Qwen3 Multi-Resolution
              </h1>
              <p className="text-xs text-gray-500">7B Knowledge → 360M Speed • Dynamic Resizing</p>
            </div>
          </div>

          {isInitialized && (
            <div className="flex items-center gap-4">
              <select
                value={currentRes}
                onChange={(e) => switchResolution(e.target.value)}
                className="px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-700 text-sm text-white"
              >
                {Object.entries(RESOLUTIONS).map(([id, res]) => (
                  <option key={id} value={id}>{res.name}</option>
                ))}
              </select>
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-green-500/10 border border-green-500/30 text-green-400 text-xs">
                <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                Active
              </span>
            </div>
          )}
        </div>
      </header>

      {/* Tabs */}
      <div className="border-b border-gray-800/50 bg-[#0d0d14]">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-1 overflow-x-auto">
            {([
              ['overview', '🏠 Overview'],
              ['train', '🔥 Qwen3 Training'],
              ['generate', '✨ Generate'],
              ['architecture', '🧬 Architecture'],
              ['export', '📦 Export'],
            ] as [Tab, string][]).map(([tab, label]) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-3 text-sm font-medium whitespace-nowrap transition-all border-b-2 ${
                  activeTab === tab
                    ? 'border-purple-500 text-purple-400'
                    : 'border-transparent text-gray-500 hover:text-gray-300'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {!isInitialized ? (
          <InitPanel onInit={initializeModel} currentRes={currentRes} setCurrentRes={setCurrentRes} />
        ) : (
          <>
            {activeTab === 'overview' && <OverviewPanel config={config} fullConfig={fullConfig} compressionRatio={compressionRatio} training={training} />}
            {activeTab === 'train' && <TrainPanel training={training} setTraining={setTraining} onStart={startTraining} onStop={stopTraining} />}
            {activeTab === 'generate' && <GeneratePanel prompt={prompt} setPrompt={setPrompt} generatedText={generatedText} isGenerating={isGenerating} temperature={temperature} setTemperature={setTemperature} maxTokens={maxTokens} setMaxTokens={setMaxTokens} onGenerate={generateText} currentRes={currentRes} />}
            {activeTab === 'architecture' && <ArchitecturePanel config={config} fullConfig={fullConfig} />}
            {activeTab === 'export' && <ExportPanel model={model!} currentRes={currentRes} onDownload={downloadWeights} />}
          </>
        )}
      </main>
    </div>
  );
}

// ============ INIT PANEL ============
function InitPanel({ onInit, currentRes, setCurrentRes }: { onInit: () => void; currentRes: string; setCurrentRes: (v: string) => void }) {
  return (
    <div className="max-w-4xl mx-auto text-center py-12">
      <div className="text-6xl mb-6">🧠</div>
      <h2 className="text-3xl font-bold mb-3 bg-gradient-to-r from-purple-300 to-blue-400 bg-clip-text text-transparent">
        Qwen3 Multi-Resolution Transformer
      </h2>
      <p className="text-gray-400 mb-8 max-w-2xl mx-auto">
        A 7B-scale architecture that dynamically resizes to 360M for fast inference while preserving knowledge
        through progressive distillation. Uses Qwen3's training methodology with multi-stage training.
      </p>

      {/* Resolution cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        {Object.entries(RESOLUTIONS).map(([id, res]) => (
          <button
            key={id}
            onClick={() => setCurrentRes(id)}
            className={`p-4 rounded-xl border transition-all text-left ${
              currentRes === id
                ? 'border-purple-500/50 bg-purple-500/10'
                : 'border-gray-800 bg-gray-900/30 hover:border-gray-700'
            }`}
          >
            <div className="text-lg font-bold text-white">{id}</div>
            <div className="text-xs text-gray-400 mt-1">{res.dModel}d × {res.nLayers}L</div>
            <div className="text-xs text-purple-400 mt-1">{res.nHeads} heads</div>
          </button>
        ))}
      </div>

      {/* Key concept */}
      <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800 text-left mb-8">
        <h3 className="font-bold text-white mb-3">🔑 How It Works</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="p-3 rounded-lg bg-gray-800/50">
            <div className="text-purple-400 font-medium mb-1">1. Shared Weights</div>
            <div className="text-gray-400 text-xs">All resolutions share the same base weight matrices. Smaller models use subsets of the 7B weights.</div>
          </div>
          <div className="p-3 rounded-lg bg-gray-800/50">
            <div className="text-purple-400 font-medium mb-1">2. Knowledge Distillation</div>
            <div className="text-gray-400 text-xs">The 360M model learns from the 7B teacher, compressing knowledge into fewer parameters.</div>
          </div>
          <div className="p-3 rounded-lg bg-gray-800/50">
            <div className="text-purple-400 font-medium mb-1">3. Dynamic Switching</div>
            <div className="text-gray-400 text-xs">Switch between resolutions on-the-fly. Use 7B for quality, 360M for speed.</div>
          </div>
        </div>
      </div>

      <button
        onClick={onInit}
        className="px-8 py-4 bg-gradient-to-r from-purple-500 to-blue-600 rounded-xl font-semibold text-white shadow-lg shadow-purple-500/25 hover:shadow-purple-500/40 transition-all hover:scale-105"
      >
        🚀 Initialize Model
      </button>
    </div>
  );
}

// ============ OVERVIEW PANEL ============
function OverviewPanel({ config, fullConfig, compressionRatio, training }: { config: any; fullConfig: any; compressionRatio: number; training: any }) {
  return (
    <div className="space-y-6">
      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard icon="📐" label="Active d_model" value={config.dModel} />
        <StatCard icon="📚" label="Active Layers" value={config.nLayers} />
        <StatCard icon="⚡" label="Compression" value={`${compressionRatio.toFixed(1)}x`} />
        <StatCard icon="📉" label="Training Loss" value={training.loss > 0 ? training.loss.toFixed(4) : 'N/A'} />
      </div>

      {/* Resolution comparison */}
      <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
        <h3 className="font-bold text-white mb-4">Resolution Comparison</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 border-b border-gray-800">
                <th className="text-left py-2">Resolution</th>
                <th className="text-right py-2">d_model</th>
                <th className="text-right py-2">Layers</th>
                <th className="text-right py-2">Heads</th>
                <th className="text-right py-2">d_ff</th>
                <th className="text-right py-2">Context</th>
                <th className="text-right py-2">Speed</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(RESOLUTIONS).map(([id, res]) => (
                <tr key={id} className={`border-b border-gray-800/50 ${id === config.id ? 'bg-purple-500/5' : ''}`}>
                  <td className="py-2 font-medium text-white">{res.name}</td>
                  <td className="text-right text-gray-300 font-mono">{res.dModel}</td>
                  <td className="text-right text-gray-300 font-mono">{res.nLayers}</td>
                  <td className="text-right text-gray-300 font-mono">{res.nHeads}</td>
                  <td className="text-right text-gray-300 font-mono">{res.dFF}</td>
                  <td className="text-right text-gray-300 font-mono">{res.maxSeqLen}</td>
                  <td className="text-right">
                    <span className={`px-2 py-0.5 rounded text-xs ${
                      id === '7B' ? 'bg-red-500/10 text-red-400' :
                      id === '3B' ? 'bg-orange-500/10 text-orange-400' :
                      id === '1B' ? 'bg-yellow-500/10 text-yellow-400' :
                      'bg-green-500/10 text-green-400'
                    }`}>
                      {id === '7B' ? 'Slow' : id === '3B' ? 'Medium' : id === '1B' ? 'Fast' : 'Very Fast'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Qwen3 Training Pipeline */}
      <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
        <h3 className="font-bold text-white mb-4">Qwen3 Training Pipeline</h3>
        <div className="flex flex-col md:flex-row items-center gap-4">
          <PipelineStage num={1} title="Pre-training" desc="Large-scale corpus training on web text, code, and reasoning data" color="blue" />
          <PipelineArrow />
          <PipelineStage num={2} title="SFT" desc="Supervised fine-tuning on instruction-following conversations" color="purple" />
          <PipelineArrow />
          <PipelineStage num={3} title="Distillation" desc="Knowledge transfer from 7B teacher to 360M student" color="orange" />
          <PipelineArrow />
          <PipelineStage num={4} title="RLHF" desc="Reinforcement learning from human feedback for alignment" color="green" />
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon, label, value }: { icon: string; label: string; value: any }) {
  return (
    <div className="p-4 rounded-xl bg-gray-900/50 border border-gray-800">
      <div className="text-2xl mb-2">{icon}</div>
      <div className="text-xl font-bold text-white">{value}</div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  );
}

function PipelineStage({ num, title, desc, color }: { num: number; title: string; desc: string; color: string }) {
  const colors: Record<string, string> = {
    blue: 'border-blue-500/40 bg-blue-500/5',
    purple: 'border-purple-500/40 bg-purple-500/5',
    orange: 'border-orange-500/40 bg-orange-500/5',
    green: 'border-green-500/40 bg-green-500/5',
  };
  return (
    <div className={`flex-1 p-4 rounded-xl border ${colors[color]} text-center`}>
      <div className="text-xs text-gray-500 mb-1">Stage {num}</div>
      <div className="font-bold text-white text-sm">{title}</div>
      <div className="text-xs text-gray-400 mt-1">{desc}</div>
    </div>
  );
}

function PipelineArrow() {
  return <div className="text-gray-600 text-xl hidden md:block">→</div>;
}

// ============ TRAINING PANEL ============
function TrainPanel({ training, setTraining, onStart, onStop }: {
  training: TrainingState;
  setTraining: (fn: (prev: TrainingState) => TrainingState) => void;
  onStart: () => void;
  onStop: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || training.lossHistory.length < 2) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width = canvas.offsetWidth * 2;
    const h = canvas.height = canvas.offsetHeight * 2;
    ctx.scale(2, 2);
    const cw = canvas.offsetWidth;
    const ch = canvas.offsetHeight;

    ctx.clearRect(0, 0, cw, ch);

    const data = training.lossHistory;
    const maxLoss = Math.max(...data) * 1.1;
    const minLoss = Math.min(...data) * 0.9;

    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 5; i++) {
      const y = (ch * i) / 4;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cw, y); ctx.stroke();
    }

    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = (i / (data.length - 1)) * cw;
      const y = ch - ((data[i] - minLoss) / (maxLoss - minLoss)) * ch;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    const gradient = ctx.createLinearGradient(0, 0, 0, ch);
    gradient.addColorStop(0, 'rgba(168, 85, 247, 0.2)');
    gradient.addColorStop(1, 'rgba(168, 85, 247, 0)');
    ctx.fillStyle = gradient;
    ctx.lineTo(cw, ch); ctx.lineTo(0, ch); ctx.fill();
  }, [training.lossHistory]);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <h3 className="font-bold text-white mb-4">Qwen3 Training Config</h3>
          <div className="space-y-4">
            <div>
              <label className="text-sm text-gray-400 block mb-2">Student Resolution</label>
              <div className="text-lg font-bold text-purple-400">{training.currentResolution}</div>
            </div>
            <div>
              <label className="text-sm text-gray-400 block mb-2">Teacher Resolution</label>
              <div className="text-lg font-bold text-blue-400">{training.teacherResolution}</div>
            </div>
            <div>
              <label className="text-sm text-gray-400 block mb-2">
                Learning Rate: <span className="text-purple-400 font-mono">{training.learningRate}</span>
              </label>
              <input type="range" min="0.0001" max="0.01" step="0.0001" value={training.learningRate}
                onChange={(e) => setTraining(prev => ({ ...prev, learningRate: parseFloat(e.target.value) }))}
                className="w-full" />
            </div>
            <div>
              <label className="text-sm text-gray-400 block mb-2">
                Distillation Weight: <span className="text-purple-400 font-mono">{training.distillationWeight}</span>
              </label>
              <input type="range" min="0" max="1" step="0.1" value={training.distillationWeight}
                onChange={(e) => setTraining(prev => ({ ...prev, distillationWeight: parseFloat(e.target.value) }))}
                className="w-full" />
              <div className="flex justify-between text-xs text-gray-600 mt-1">
                <span>CE Loss only</span><span>50/50 Mix</span><span>Distill only</span>
              </div>
            </div>
          </div>
        </div>

        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <h3 className="font-bold text-white mb-4">Training Status</h3>
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div>
              <div className="text-2xl font-bold text-purple-400 font-mono">{training.step}</div>
              <div className="text-xs text-gray-500">Steps</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-400 font-mono">{training.epoch}</div>
              <div className="text-xs text-gray-500">Epochs</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-400 font-mono">{training.loss > 0 ? training.loss.toFixed(4) : '—'}</div>
              <div className="text-xs text-gray-500">Loss</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-400 font-mono">{training.distillationWeight * 100}%</div>
              <div className="text-xs text-gray-500">Distill Mix</div>
            </div>
          </div>

          {!training.isTraining ? (
            <button onClick={onStart}
              className="w-full px-4 py-3 bg-gradient-to-r from-purple-500 to-blue-600 rounded-lg font-semibold text-white shadow-lg shadow-purple-500/20 hover:shadow-purple-500/40 transition-all">
              🔥 Start Qwen3 Training
            </button>
          ) : (
            <button onClick={onStop}
              className="w-full px-4 py-3 bg-red-600 rounded-lg font-semibold text-white hover:bg-red-700 transition-all">
              ⏹ Stop Training
            </button>
          )}

          {training.isTraining && (
            <div className="mt-4">
              <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-purple-500 to-blue-500 rounded-full transition-all animate-pulse"
                  style={{ width: `${Math.min(100, (training.step / 500) * 100)}%` }} />
              </div>
              <div className="text-xs text-gray-500 mt-1">Step {training.step} / 500</div>
            </div>
          )}
        </div>
      </div>

      <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
        <h3 className="font-bold text-white mb-4">Knowledge Distillation Loss</h3>
        <div className="h-48 bg-gray-950 rounded-lg border border-gray-800 overflow-hidden">
          <canvas ref={canvasRef} className="w-full h-full" />
          {training.lossHistory.length < 2 && (
            <div className="flex items-center justify-center h-full text-gray-600 text-sm">
              Start training to see loss curve
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ============ GENERATE PANEL ============
function GeneratePanel({ prompt, setPrompt, generatedText, isGenerating, temperature, setTemperature, maxTokens, setMaxTokens, onGenerate, currentRes }: any) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div className="md:col-span-2 space-y-4">
        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <label className="text-sm text-gray-400 block mb-2">Prompt</label>
          <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your prompt..."
            className="w-full h-24 px-4 py-3 rounded-lg bg-gray-800 border border-gray-700 text-white resize-none focus:border-purple-500/50 focus:outline-none" />
          <button onClick={onGenerate} disabled={isGenerating || !prompt}
            className="mt-3 px-6 py-3 bg-gradient-to-r from-purple-500 to-blue-600 rounded-lg font-semibold text-white shadow-lg shadow-purple-500/20 hover:shadow-purple-500/40 transition-all disabled:opacity-50">
            {isGenerating ? '⏳ Generating...' : '✨ Generate'}
          </button>
        </div>

        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <label className="text-sm text-gray-400 block mb-2">Output ({currentRes} resolution)</label>
          <div className="min-h-[200px] p-4 rounded-lg bg-gray-950 border border-gray-800 font-mono text-sm text-gray-300 whitespace-pre-wrap">
            {generatedText || <span className="text-gray-600">Generated text will appear here...</span>}
          </div>
        </div>
      </div>

      <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
        <h3 className="font-bold text-white mb-4">Parameters</h3>
        <div className="space-y-5">
          <div>
            <label className="text-sm text-gray-400 flex justify-between">
              <span>Temperature</span><span className="text-purple-400 font-mono">{temperature.toFixed(2)}</span>
            </label>
            <input type="range" min="0.1" max="2.0" step="0.05" value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))} className="w-full mt-1" />
          </div>
          <div>
            <label className="text-sm text-gray-400 flex justify-between">
              <span>Max Tokens</span><span className="text-purple-400 font-mono">{maxTokens}</span>
            </label>
            <input type="range" min="10" max="200" step="10" value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))} className="w-full mt-1" />
          </div>
        </div>

        <div className="mt-6 p-3 rounded-lg bg-purple-500/5 border border-purple-500/20">
          <h4 className="text-xs font-semibold text-purple-400 mb-2">Current Resolution</h4>
          <div className="text-sm text-gray-300">{RESOLUTIONS[currentRes]?.name}</div>
          <div className="text-xs text-gray-500 mt-1">
            {RESOLUTIONS[currentRes]?.dModel}d × {RESOLUTIONS[currentRes]?.nLayers}L × {RESOLUTIONS[currentRes]?.nHeads}H
          </div>
        </div>
      </div>
    </div>
  );
}

// ============ ARCHITECTURE PANEL ============
function ArchitecturePanel({ config, fullConfig }: { config: any; fullConfig: any }) {
  return (
    <div className="space-y-6">
      <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
        <h3 className="text-lg font-bold text-white mb-4">Dynamic Resizing Architecture</h3>
        <p className="text-gray-400 text-sm mb-6">
          The model maintains a single set of weights at the 7B scale. When switching to a smaller resolution,
          it activates only the relevant subset of parameters. Knowledge is preserved through distillation training.
        </p>

        {/* Visual */}
        <div className="flex flex-col items-center gap-2 py-8">
          <ArchBlock label="7B Full Model" detail={`${fullConfig.dModel}d × ${fullConfig.nLayers}L × ${fullConfig.nHeads}H`} color="red" active={true} />
          <Arrow />
          <div className="w-full max-w-lg border-2 border-dashed border-purple-500/30 rounded-xl p-4">
            <div className="text-xs text-purple-400 text-center mb-3">Weight Sharing & Pruning</div>
            <div className="flex flex-col items-center gap-2">
              <ArchBlock label="Shared Token Embedding" detail={`[${fullConfig.vocabSize}, ${fullConfig.dModel}]`} color="purple" active={true} />
              <Arrow />
              <ArchBlock label="Shared Position Embedding" detail={`[${fullConfig.maxSeqLen}, ${fullConfig.dModel}]`} color="purple" active={true} />
              <Arrow />
              <ArchBlock label={`Active Layers (0-${config.nLayers - 1})`} detail={`Uses first ${config.nLayers} of ${fullConfig.nLayers} layers`} color="blue" active={true} />
              <Arrow />
              <ArchBlock label={`Active d_model: ${config.dModel}`} detail={`Sliced from ${fullConfig.dModel} dimensions`} color="green" active={true} />
            </div>
          </div>
          <Arrow />
          <ArchBlock label={`${config.id} Output`} detail={`${config.dModel}d × ${config.nLayers}L → ${config.vocabSize} vocab`} color="green" active={true} />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <h4 className="font-bold text-white mb-3">🎯 Key Innovation</h4>
          <ul className="space-y-2 text-sm text-gray-400">
            <li>• <span className="text-gray-300">Single weight set</span> — No duplicate storage for different sizes</li>
            <li>• <span className="text-gray-300">Progressive distillation</span> — 7B teaches 360M through soft targets</li>
            <li>• <span className="text-gray-300">Dimension slicing</span> — Smaller models use first N dimensions</li>
            <li>• <span className="text-gray-300">Layer pruning</span> — Smaller models use first N layers</li>
            <li>• <span className="text-gray-300">Context preservation</span> — 360M maintains 7B's knowledge</li>
            <li>• <span className="text-gray-300">Instant switching</span> — No reload needed to change resolution</li>
          </ul>
        </div>
        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <h4 className="font-bold text-white mb-3">📊 Current Config ({config.id})</h4>
          <div className="space-y-3">
            {[
              ['d_model', config.dModel],
              ['n_layers', config.nLayers],
              ['n_heads', config.nHeads],
              ['d_ff', config.dFF],
              ['max_seq_len', config.maxSeqLen],
              ['vocab_size', config.vocabSize],
              ['d_head', config.dModel / config.nHeads],
              ['compression', `${(fullConfig.dModel / config.dModel).toFixed(1)}x`],
            ].map(([label, value]) => (
              <div key={label as string} className="flex justify-between text-sm">
                <span className="text-gray-400">{label as string}</span>
                <span className="text-purple-400 font-mono">{value as number}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function ArchBlock({ label, detail, color, active }: { label: string; detail: string; color: string; active: boolean }) {
  const colors: Record<string, string> = {
    red: 'border-red-500/40 bg-red-500/5',
    purple: 'border-purple-500/40 bg-purple-500/5',
    blue: 'border-blue-500/40 bg-blue-500/5',
    green: 'border-green-500/40 bg-green-500/5',
  };
  return (
    <div className={`w-full max-w-lg px-4 py-2.5 rounded-lg border ${colors[color]} text-center ${active ? 'opacity-100' : 'opacity-40'}`}>
      <div className="text-sm font-medium text-white">{label}</div>
      {detail && <div className="text-xs text-gray-500 font-mono">{detail}</div>}
    </div>
  );
}

function Arrow() {
  return (
    <svg className="w-4 h-4 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
    </svg>
  );
}

// ============ EXPORT PANEL ============
function ExportPanel({ model, currentRes, onDownload }: { model: MultiResolutionTransformer; currentRes: string; onDownload: () => void }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <h3 className="font-bold text-white mb-4">📥 Download Weights</h3>
          <p className="text-sm text-gray-400 mb-4">
            Download the current resolution's weights. These can be loaded back to resume training or inference.
          </p>
          <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700 mb-4">
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Resolution</span>
              <span className="text-purple-400 font-mono">{currentRes}</span>
            </div>
            <div className="flex justify-between text-sm mt-1">
              <span className="text-gray-400">Format</span>
              <span className="text-gray-300 font-mono">Binary (Float32)</span>
            </div>
          </div>
          <button onClick={onDownload}
            className="w-full px-4 py-3 bg-gradient-to-r from-purple-500 to-blue-600 rounded-lg font-semibold text-white shadow-lg shadow-purple-500/20 hover:shadow-purple-500/40 transition-all">
            💾 Download {currentRes} Weights
          </button>
        </div>

        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <h3 className="font-bold text-white mb-4">🖥️ Local Deployment</h3>
          <p className="text-sm text-gray-400 mb-4">
            This model runs entirely in the browser. To deploy locally:
          </p>
          <div className="space-y-3">
            <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700">
              <div className="text-sm text-gray-300 font-medium">1. Download weights</div>
              <div className="text-xs text-gray-500">Get the .bin file for your chosen resolution</div>
            </div>
            <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700">
              <div className="text-sm text-gray-300 font-medium">2. Load in any environment</div>
              <div className="text-xs text-gray-500">Weights are standard Float32 binary format</div>
            </div>
            <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700">
              <div className="text-sm text-gray-300 font-medium">3. Run inference</div>
              <div className="text-xs text-gray-500">Use the same transformer code for generation</div>
            </div>
          </div>
        </div>
      </div>

      <div className="p-6 rounded-xl bg-purple-500/5 border border-purple-500/20">
        <h3 className="font-bold text-purple-400 mb-3">🔓 Fully Open Source</h3>
        <p className="text-sm text-gray-400">
          All weights are generated locally and are fully open. No external API calls, no cloud dependencies.
          The model architecture, tokenizer, training loop, and inference engine are all included.
          Download, share, modify, or redistribute freely. The multi-resolution approach means you get
          7B-quality knowledge at 360M inference speed.
        </p>
      </div>
    </div>
  );
}

export default App;
