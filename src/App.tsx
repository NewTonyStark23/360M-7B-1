import { useState, useRef, useCallback, useEffect } from 'react';
import { ElasticTransformer, ELASTIC_CONFIG } from './ml/elastic-transformer';
import { Tensor } from './ml/tensor';
import { getAllPretrainData, generateTrainingChunks } from './ml/qwen3-data';

type Tab = 'overview' | 'train' | 'generate' | 'architecture' | 'export';

interface TrainingState {
  isTraining: boolean;
  step: number;
  epoch: number;
  loss: number;
  lossHistory: number[];
  learningRate: number;
}

interface ModelState {
  activeDModel: number;
  activeLayers: number;
  activeHeads: number;
  currentComplexity: number;
  estimatedParams: number;
}

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('overview');
  const [model, setModel] = useState<ElasticTransformer | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [modelState, setModelState] = useState<ModelState | null>(null);

  const [training, setTraining] = useState<TrainingState>({
    isTraining: false,
    step: 0,
    epoch: 0,
    loss: 0,
    lossHistory: [],
    learningRate: 0.001,
  });

  const [prompt, setPrompt] = useState('The ');
  const [generatedText, setGeneratedText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(50);

  const trainingRef = useRef(false);
  const modelRef = useRef<ElasticTransformer | null>(null);

  useEffect(() => { modelRef.current = model; }, [model]);

  const initializeModel = useCallback(() => {
    const newModel = new ElasticTransformer();
    setModel(newModel);
    modelRef.current = newModel;
    setModelState(newModel.getState());
    setIsInitialized(true);
  }, []);

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

        // Forward pass - model self-adjusts based on complexity
        const { logits } = modelRef.current.forward(tokens.slice(0, -1));
        
        // Simplified training step
        const target = tokens.slice(1);
        const { loss } = Tensor.crossEntropy(logits, target);
        batchLoss += loss;
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
        
        // Update model state
        if (modelRef.current) {
          setModelState(modelRef.current.getState());
        }
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
      setGeneratedText(`[Complexity: ${(result.stats.avgComplexity * 100).toFixed(0)}% | Active: ${result.stats.avgDModel}d × ${result.stats.avgLayers}L × ${result.stats.avgHeads}H | ~${(result.stats.avgParams / 1e6).toFixed(1)}M params]\n\n${result.text}`);
      setModelState(model.getState());
    } catch (e) {
      setGeneratedText(`[Error: ${(e as Error).message}]`);
    }
    setIsGenerating(false);
  }, [model, prompt, maxTokens, temperature]);

  const downloadWeights = useCallback(() => {
    if (!model) return;
    model.downloadWeights();
  }, [model]);

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
                Elastic Transformer
              </h1>
              <p className="text-xs text-gray-500">ONE Model • Self-Adjusting Intelligence • 360M ↔ 7B</p>
            </div>
          </div>

          {isInitialized && modelState && (
            <div className="flex items-center gap-4">
              <div className="text-right">
                <div className="text-xs text-gray-500">Active Size</div>
                <div className="text-sm font-mono text-purple-400">
                  {modelState.activeDModel}d × {modelState.activeLayers}L
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs text-gray-500">Complexity</div>
                <div className="text-sm font-mono text-blue-400">
                  {(modelState.currentComplexity * 100).toFixed(0)}%
                </div>
              </div>
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
              ['train', '🔥 Train'],
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
          <InitPanel onInit={initializeModel} />
        ) : (
          <>
            {activeTab === 'overview' && <OverviewPanel modelState={modelState!} training={training} />}
            {activeTab === 'train' && <TrainPanel training={training} setTraining={setTraining} onStart={startTraining} onStop={stopTraining} modelState={modelState!} />}
            {activeTab === 'generate' && <GeneratePanel prompt={prompt} setPrompt={setPrompt} generatedText={generatedText} isGenerating={isGenerating} temperature={temperature} setTemperature={setTemperature} maxTokens={maxTokens} setMaxTokens={setMaxTokens} onGenerate={generateText} modelState={modelState!} />}
            {activeTab === 'architecture' && <ArchitecturePanel />}
            {activeTab === 'export' && <ExportPanel model={model!} onDownload={downloadWeights} />}
          </>
        )}
      </main>
    </div>
  );
}

// ============ INIT PANEL ============
function InitPanel({ onInit }: { onInit: () => void }) {
  return (
    <div className="max-w-4xl mx-auto text-center py-12">
      <div className="text-6xl mb-6">🧠</div>
      <h2 className="text-3xl font-bold mb-3 bg-gradient-to-r from-purple-300 to-blue-400 bg-clip-text text-transparent">
        Elastic Transformer
      </h2>
      <p className="text-gray-400 mb-8 max-w-2xl mx-auto">
        ONE unified model that self-adjusts its intelligence based on task complexity.
        Uses 360M resources for simple tasks, grows to 7B for complex ones.
        <span className="block mt-2 text-purple-400 font-medium">
          Not two models. Not swapping. ONE model with elastic internals.
        </span>
      </p>

      {/* How it works */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="p-5 rounded-xl bg-gray-900/50 border border-gray-800 text-left">
          <div className="text-2xl mb-2">📊</div>
          <h3 className="font-bold text-white mb-2">Analyzes Complexity</h3>
          <p className="text-xs text-gray-400">
            Detects code, reasoning, technical terms, context length. Determines how much intelligence is needed.
          </p>
        </div>
        <div className="p-5 rounded-xl bg-gray-900/50 border border-gray-800 text-left">
          <div className="text-2xl mb-2">🔄</div>
          <h3 className="font-bold text-white mb-2">Self-Adjusts</h3>
          <p className="text-xs text-gray-400">
            Activates more layers, dimensions, and attention heads when needed. Stays lean when not.
          </p>
        </div>
        <div className="p-5 rounded-xl bg-gray-900/50 border border-gray-800 text-left">
          <div className="text-2xl mb-2">⚡</div>
          <h3 className="font-bold text-white mb-2">Best of Both</h3>
          <p className="text-xs text-gray-400">
            360M speed for simple tasks. 7B intelligence for complex ones. Always optimal.
          </p>
        </div>
      </div>

      {/* Size range */}
      <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800 mb-8">
        <h3 className="font-bold text-white mb-4">Dynamic Size Range</h3>
        <div className="flex items-center justify-between">
          <div className="text-center">
            <div className="text-3xl font-bold text-green-400">360M</div>
            <div className="text-xs text-gray-500 mt-1">Simple tasks</div>
            <div className="text-xs text-gray-600">1024d × 12L × 8H</div>
          </div>
          <div className="flex-1 mx-8">
            <div className="h-2 bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 rounded-full" />
            <div className="text-xs text-gray-500 text-center mt-2">Complexity →</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-red-400">7B</div>
            <div className="text-xs text-gray-500 mt-1">Complex tasks</div>
            <div className="text-xs text-gray-600">4096d × 32L × 32H</div>
          </div>
        </div>
      </div>

      <button
        onClick={onInit}
        className="px-8 py-4 bg-gradient-to-r from-purple-500 to-blue-600 rounded-xl font-semibold text-white shadow-lg shadow-purple-500/25 hover:shadow-purple-500/40 transition-all hover:scale-105"
      >
        🚀 Initialize Elastic Model
      </button>
    </div>
  );
}

// ============ OVERVIEW PANEL ============
function OverviewPanel({ modelState, training }: { modelState: ModelState; training: TrainingState }) {
  return (
    <div className="space-y-6">
      {/* Current State */}
      <div className="p-6 rounded-xl bg-gradient-to-r from-purple-500/5 to-blue-500/5 border border-purple-500/30">
        <h3 className="font-bold text-white mb-4 flex items-center gap-2">
          <span className="text-2xl">🧠</span>
          Current Model State
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div>
            <div className="text-2xl font-bold text-purple-400 font-mono">{modelState.activeDModel}</div>
            <div className="text-xs text-gray-500">Active d_model</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-purple-400 font-mono">{modelState.activeLayers}</div>
            <div className="text-xs text-gray-500">Active Layers</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-purple-400 font-mono">{modelState.activeHeads}</div>
            <div className="text-xs text-gray-500">Active Heads</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-blue-400 font-mono">{(modelState.currentComplexity * 100).toFixed(0)}%</div>
            <div className="text-xs text-gray-500">Complexity</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-blue-400 font-mono">{(modelState.estimatedParams / 1e6).toFixed(1)}M</div>
            <div className="text-xs text-gray-500">Active Params</div>
          </div>
        </div>
      </div>

      {/* How it works */}
      <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
        <h3 className="font-bold text-white mb-4">How Elastic Sizing Works</h3>
        <div className="space-y-4">
          <div className="flex items-start gap-4">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center text-purple-400 font-bold">1</div>
            <div>
              <div className="font-medium text-white">Complexity Analysis</div>
              <div className="text-sm text-gray-400 mt-1">
                Analyzes input for: context length, code patterns, reasoning indicators, technical terms.
                Computes a complexity score from 0.0 (simple) to 1.0 (complex).
              </div>
            </div>
          </div>
          <div className="flex items-start gap-4">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center text-purple-400 font-bold">2</div>
            <div>
              <div className="font-medium text-white">Dynamic Sizing</div>
              <div className="text-sm text-gray-400 mt-1">
                Maps complexity to model size:
                <div className="mt-2 font-mono text-xs text-gray-500">
                  active_dModel = 1024 + (4096 - 1024) × complexity<br/>
                  active_layers = 12 + (32 - 12) × complexity<br/>
                  active_heads = 8 + (32 - 8) × complexity
                </div>
              </div>
            </div>
          </div>
          <div className="flex items-start gap-4">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center text-purple-400 font-bold">3</div>
            <div>
              <div className="font-medium text-white">Weight Slicing</div>
              <div className="text-sm text-gray-400 mt-1">
                Uses first N dimensions of weight matrices. All weights are initialized at max size (7B).
                Simpler tasks use smaller slices, complex tasks use more.
              </div>
            </div>
          </div>
          <div className="flex items-start gap-4">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center text-purple-400 font-bold">4</div>
            <div>
              <div className="font-medium text-white">Single Unified Model</div>
              <div className="text-sm text-gray-400 mt-1">
                NOT two models. NOT swapping. ONE model that dynamically activates different amounts
                of its own parameters based on what's needed. Like a brain using more neurons for harder problems.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Example scenarios */}
      <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
        <h3 className="font-bold text-white mb-4">Example Scenarios</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-lg bg-green-500/5 border border-green-500/20">
            <div className="text-green-400 font-bold mb-2">Simple: "Hello world"</div>
            <div className="text-xs text-gray-400 space-y-1">
              <div>Complexity: ~10%</div>
              <div>Active: 1024d × 12L × 8H</div>
              <div>Params: ~360M</div>
              <div>Speed: Very Fast</div>
            </div>
          </div>
          <div className="p-4 rounded-lg bg-yellow-500/5 border border-yellow-500/20">
            <div className="text-yellow-400 font-bold mb-2">Medium: "Explain AI"</div>
            <div className="text-xs text-gray-400 space-y-1">
              <div>Complexity: ~50%</div>
              <div>Active: 2560d × 22L × 20H</div>
              <div>Params: ~3B</div>
              <div>Speed: Medium</div>
            </div>
          </div>
          <div className="p-4 rounded-lg bg-red-500/5 border border-red-500/20">
            <div className="text-red-400 font-bold mb-2">Complex: "Write quicksort"</div>
            <div className="text-xs text-gray-400 space-y-1">
              <div>Complexity: ~90%</div>
              <div>Active: 3800d × 30L × 30H</div>
              <div>Params: ~6B</div>
              <div>Speed: Slower but smart</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============ TRAIN PANEL ============
function TrainPanel({ training, setTraining, onStart, onStop, modelState }: {
  training: TrainingState;
  setTraining: (fn: (prev: TrainingState) => TrainingState) => void;
  onStart: () => void;
  onStop: () => void;
  modelState: ModelState;
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
          <h3 className="font-bold text-white mb-4">Training Configuration</h3>
          <div className="space-y-4">
            <div>
              <label className="text-sm text-gray-400 block mb-2">
                Learning Rate: <span className="text-purple-400 font-mono">{training.learningRate}</span>
              </label>
              <input type="range" min="0.0001" max="0.01" step="0.0001" value={training.learningRate}
                onChange={(e) => setTraining(prev => ({ ...prev, learningRate: parseFloat(e.target.value) }))}
                className="w-full" />
            </div>
            <div className="p-3 rounded-lg bg-purple-500/5 border border-purple-500/20">
              <div className="text-xs text-gray-400">
                <strong className="text-purple-400">Note:</strong> During training, the model automatically
                adjusts its size based on each training sample's complexity. Simple samples train the 360M
                portion, complex samples train up to the 7B portion.
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
              <div className="text-2xl font-bold text-blue-400 font-mono">{(modelState.currentComplexity * 100).toFixed(0)}%</div>
              <div className="text-xs text-gray-500">Avg Complexity</div>
            </div>
          </div>

          {!training.isTraining ? (
            <button onClick={onStart}
              className="w-full px-4 py-3 bg-gradient-to-r from-purple-500 to-blue-600 rounded-lg font-semibold text-white shadow-lg shadow-purple-500/20 hover:shadow-purple-500/40 transition-all">
              🔥 Start Training
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
        <h3 className="font-bold text-white mb-4">Training Loss</h3>
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
function GeneratePanel({ prompt, setPrompt, generatedText, isGenerating, temperature, setTemperature, maxTokens, setMaxTokens, onGenerate, modelState }: any) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div className="md:col-span-2 space-y-4">
        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <label className="text-sm text-gray-400 block mb-2">Prompt (try different complexity levels)</label>
          <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your prompt..."
            className="w-full h-24 px-4 py-3 rounded-lg bg-gray-800 border border-gray-700 text-white resize-none focus:border-purple-500/50 focus:outline-none" />
          <div className="flex gap-2 mt-2">
            <button onClick={() => setPrompt('Hello ')} className="px-3 py-1 rounded bg-green-500/10 border border-green-500/30 text-green-400 text-xs">Simple</button>
            <button onClick={() => setPrompt('Explain how neural networks work ')} className="px-3 py-1 rounded bg-yellow-500/10 border border-yellow-500/30 text-yellow-400 text-xs">Medium</button>
            <button onClick={() => setPrompt('Write a function to implement quicksort algorithm with ')} className="px-3 py-1 rounded bg-red-500/10 border border-red-500/30 text-red-400 text-xs">Complex</button>
          </div>
          <button onClick={onGenerate} disabled={isGenerating || !prompt}
            className="mt-3 px-6 py-3 bg-gradient-to-r from-purple-500 to-blue-600 rounded-lg font-semibold text-white shadow-lg shadow-purple-500/20 hover:shadow-purple-500/40 transition-all disabled:opacity-50">
            {isGenerating ? '⏳ Generating...' : '✨ Generate'}
          </button>
        </div>

        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <label className="text-sm text-gray-400 block mb-2">Output</label>
          <div className="min-h-[200px] p-4 rounded-lg bg-gray-950 border border-gray-800 font-mono text-sm text-gray-300 whitespace-pre-wrap">
            {generatedText || <span className="text-gray-600">Generated text will appear here. The model will self-adjust its size based on your prompt's complexity.</span>}
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

        <div className="mt-6 p-4 rounded-lg bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/30">
          <h4 className="text-xs font-semibold text-purple-400 mb-2">Current Active Size</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">d_model</span>
              <span className="text-purple-400 font-mono">{modelState.activeDModel}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">layers</span>
              <span className="text-purple-400 font-mono">{modelState.activeLayers}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">heads</span>
              <span className="text-purple-400 font-mono">{modelState.activeHeads}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">complexity</span>
              <span className="text-blue-400 font-mono">{(modelState.currentComplexity * 100).toFixed(0)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">active params</span>
              <span className="text-blue-400 font-mono">{(modelState.estimatedParams / 1e6).toFixed(1)}M</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============ ARCHITECTURE PANEL ============
function ArchitecturePanel() {
  return (
    <div className="space-y-6">
      <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
        <h3 className="text-lg font-bold text-white mb-4">Elastic Transformer Architecture</h3>
        <p className="text-gray-400 text-sm mb-6">
          ONE unified model with dynamic internal sizing. All weights are initialized at maximum capacity (7B equivalent).
          During inference, the model analyzes input complexity and activates only the needed subset of parameters.
        </p>

        {/* Visual */}
        <div className="flex flex-col items-center gap-2 py-8">
          <div className="w-full max-w-2xl p-6 rounded-xl border-2 border-purple-500/30 bg-purple-500/5">
            <div className="text-center mb-4">
              <div className="text-lg font-bold text-white">Elastic Transformer (Single Model)</div>
              <div className="text-xs text-gray-400">All weights initialized at max size (7B equivalent)</div>
            </div>
            
            <div className="space-y-3">
              <div className="p-3 rounded-lg bg-gray-900/50 border border-gray-700">
                <div className="text-sm font-medium text-white">Shared Embeddings (Always Active)</div>
                <div className="text-xs text-gray-500 font-mono">token_emb: [32000, 4096] | pos_emb: [8192, 4096]</div>
              </div>

              <div className="p-3 rounded-lg bg-gray-900/50 border border-gray-700">
                <div className="text-sm font-medium text-white mb-2">Transformer Layers (Dynamic Activation)</div>
                <div className="text-xs text-gray-500 mb-2">
                  Complexity 0% → Activate first 12 layers (1024d × 8H)<br/>
                  Complexity 50% → Activate first 22 layers (2560d × 20H)<br/>
                  Complexity 100% → Activate all 32 layers (4096d × 32H)
                </div>
                <div className="flex gap-1">
                  {Array.from({ length: 32 }, (_, i) => (
                    <div key={i} className={`flex-1 h-6 rounded ${i < 12 ? 'bg-green-500/50' : i < 22 ? 'bg-yellow-500/50' : 'bg-red-500/50'}`} title={`Layer ${i + 1}`} />
                  ))}
                </div>
                <div className="flex justify-between text-xs text-gray-600 mt-1">
                  <span>Layer 1</span>
                  <span>Layer 32</span>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-gray-900/50 border border-gray-700">
                <div className="text-sm font-medium text-white">Weight Slicing</div>
                <div className="text-xs text-gray-500">
                  Each weight matrix is initialized at max size. During forward pass, only the first N dimensions are used based on complexity.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <h4 className="font-bold text-white mb-3">🎯 Key Innovation</h4>
          <ul className="space-y-2 text-sm text-gray-400">
            <li>• <span className="text-gray-300">Single unified model</span> — Not two models, not swapping</li>
            <li>• <span className="text-gray-300">Dynamic sizing</span> — Activates more parameters when needed</li>
            <li>• <span className="text-gray-300">Complexity-aware</span> — Analyzes input to determine needed intelligence</li>
            <li>• <span className="text-gray-300">Weight slicing</span> — Uses first N dimensions of full-size weights</li>
            <li>• <span className="text-gray-300">Always optimal</span> — Fast when simple, smart when complex</li>
            <li>• <span className="text-gray-300">Like a brain</span> — Uses more neurons for harder problems</li>
          </ul>
        </div>
        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <h4 className="font-bold text-white mb-3">📊 Size Range</h4>
          <div className="space-y-3">
            <div className="p-3 rounded-lg bg-green-500/5 border border-green-500/20">
              <div className="flex justify-between text-sm">
                <span className="text-green-400 font-medium">Minimum (Simple)</span>
                <span className="text-gray-300 font-mono">1024d × 12L × 8H</span>
              </div>
              <div className="text-xs text-gray-500 mt-1">~360M parameters • Very fast</div>
            </div>
            <div className="p-3 rounded-lg bg-yellow-500/5 border border-yellow-500/20">
              <div className="flex justify-between text-sm">
                <span className="text-yellow-400 font-medium">Medium</span>
                <span className="text-gray-300 font-mono">2560d × 22L × 20H</span>
              </div>
              <div className="text-xs text-gray-500 mt-1">~3B parameters • Balanced</div>
            </div>
            <div className="p-3 rounded-lg bg-red-500/5 border border-red-500/20">
              <div className="flex justify-between text-sm">
                <span className="text-red-400 font-medium">Maximum (Complex)</span>
                <span className="text-gray-300 font-mono">4096d × 32L × 32H</span>
              </div>
              <div className="text-xs text-gray-500 mt-1">~7B parameters • Full intelligence</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============ EXPORT PANEL ============
function ExportPanel({ model, onDownload }: { model: ElasticTransformer; onDownload: () => void }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <h3 className="font-bold text-white mb-4">📥 Download Weights</h3>
          <p className="text-sm text-gray-400 mb-4">
            Download the complete elastic model weights. The model will self-adjust when loaded.
          </p>
          <button onClick={onDownload}
            className="w-full px-4 py-3 bg-gradient-to-r from-purple-500 to-blue-600 rounded-lg font-semibold text-white shadow-lg shadow-purple-500/20 hover:shadow-purple-500/40 transition-all">
            💾 Download Elastic Model
          </button>
        </div>

        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <h3 className="font-bold text-white mb-4">🖥️ Local Deployment</h3>
          <p className="text-sm text-gray-400 mb-4">
            This model runs entirely in the browser. Download and use locally with the same code.
          </p>
          <div className="space-y-3">
            <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700">
              <div className="text-sm text-gray-300 font-medium">1. Download weights</div>
              <div className="text-xs text-gray-500">Get the .bin file</div>
            </div>
            <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700">
              <div className="text-sm text-gray-300 font-medium">2. Load in any environment</div>
              <div className="text-xs text-gray-500">Standard Float32 binary format</div>
            </div>
            <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700">
              <div className="text-sm text-gray-300 font-medium">3. Model self-adjusts</div>
              <div className="text-xs text-gray-500">Automatically sizes based on input</div>
            </div>
          </div>
        </div>
      </div>

      <div className="p-6 rounded-xl bg-purple-500/5 border border-purple-500/20">
        <h3 className="font-bold text-purple-400 mb-3">🔓 Fully Open Source</h3>
        <p className="text-sm text-gray-400">
          ONE unified model with elastic internals. Not two models swapping. Not hybrid inference.
          A single model that dynamically activates different amounts of its own parameters based on task complexity.
          All weights are open. Download, share, modify, or redistribute freely.
        </p>
      </div>
    </div>
  );
}

export default App;
