import { useState, useRef, useCallback, useEffect } from 'react';
import { SmolLM, DEFAULT_CONFIG, SMALL_CONFIG, ModelConfig } from './ml/model';
import { TRAINING_CORPORA, getTrainingChunks } from './ml/training-data';
import { Tokenizer } from './ml/tokenizer';

type Tab = 'overview' | 'train' | 'generate' | 'architecture' | 'export';
type ModelSize = 'small' | 'default';

interface TrainingState {
  isTraining: boolean;
  epoch: number;
  step: number;
  loss: number;
  lossHistory: number[];
  currentCorpus: string;
  learningRate: number;
  samplesTrained: number;
}

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('overview');
  const [model, setModel] = useState<SmolLM | null>(null);
  const [modelSize, setModelSize] = useState<ModelSize>('small');
  const [isInitialized, setIsInitialized] = useState(false);

  // Training state
  const [training, setTraining] = useState<TrainingState>({
    isTraining: false,
    epoch: 0,
    step: 0,
    loss: 0,
    lossHistory: [],
    currentCorpus: 'simple_english',
    learningRate: 0.01,
    samplesTrained: 0,
  });

  // Generation state
  const [prompt, setPrompt] = useState('the ');
  const [generatedText, setGeneratedText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(100);
  const [topK, setTopK] = useState(40);
  const [topP, setTopP] = useState(0.9);

  // Custom training text
  const [customText, setCustomText] = useState('');
  const [useCustom, setUseCustom] = useState(false);

  const trainingRef = useRef(false);
  const modelRef = useRef<SmolLM | null>(null);

  // Keep ref in sync
  useEffect(() => {
    modelRef.current = model;
  }, [model]);

  const initializeModel = useCallback(() => {
    const config = modelSize === 'small' ? SMALL_CONFIG : DEFAULT_CONFIG;
    const newModel = new SmolLM(config);
    setModel(newModel);
    modelRef.current = newModel;
    setIsInitialized(true);
  }, [modelSize]);

  const loadWeights = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const buffer = ev.target?.result as ArrayBuffer;
      const loadedModel = SmolLM.deserializeWeights(buffer);
      setModel(loadedModel);
      modelRef.current = loadedModel;
      setIsInitialized(true);
    };
    reader.readAsArrayBuffer(file);
  }, []);

  const startTraining = useCallback(async () => {
    if (!model || training.isTraining) return;

    const text = useCustom && customText.length > 50
      ? customText
      : TRAINING_CORPORA[training.currentCorpus].text;

    const chunks = getTrainingChunks(text, 64, 32);
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
        const tokens = modelRef.current.tokenizer.encode(chunk, true, true);

        if (tokens.length < 4) continue;

        // Input: all but last, Target: all but first (next token prediction)
        const input = tokens.slice(0, -1);
        const target = tokens.slice(1);

        try {
          const { cache } = modelRef.current.forward(input);
          const loss = modelRef.current.backward(cache, target, lr);
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
          samplesTrained: step * batchSize,
          epoch: Math.floor(step * batchSize / chunks.length),
        }));
      }

      if (trainingRef.current && step < 2000) {
        // Use setTimeout to yield to UI
        setTimeout(trainStep, 0);
      } else {
        trainingRef.current = false;
        setTraining(prev => ({ ...prev, isTraining: false }));
      }
    };

    trainStep();
  }, [model, training.isTraining, training.currentCorpus, training.learningRate, useCustom, customText]);

  const stopTraining = useCallback(() => {
    trainingRef.current = false;
    setTraining(prev => ({ ...prev, isTraining: false }));
  }, []);

  const generateText = useCallback(async () => {
    if (!model) return;
    setIsGenerating(true);
    setGeneratedText('');

    // Generate in chunks to show progress
    await new Promise(resolve => setTimeout(resolve, 50));

    try {
      const result = model.generate(prompt, maxTokens, temperature, topK, topP);
      setGeneratedText(result);
    } catch (e) {
      setGeneratedText(`[Generation error: ${(e as Error).message}]`);
    }

    setIsGenerating(false);
  }, [model, prompt, temperature, maxTokens, topK, topP]);

  const downloadWeights = useCallback(() => {
    if (!model) return;
    model.downloadWeights(`smol-lm-${modelSize}-${Date.now()}.bin`);
  }, [model, modelSize]);

  const config = modelSize === 'small' ? SMALL_CONFIG : DEFAULT_CONFIG;

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white font-sans">
      {/* Header */}
      <header className="border-b border-gray-800/50 bg-[#0a0a0f]/95 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center shadow-lg shadow-orange-500/20">
              <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-bold bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent">
                SMOL-LM
              </h1>
              <p className="text-xs text-gray-500">Custom Transformer • Open Weights • Runs Anywhere</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {isInitialized && (
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-green-500/10 border border-green-500/30 text-green-400 text-xs">
                <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                Model Loaded
              </span>
            )}
            <span className="text-xs text-gray-600 font-mono">
              {model ? `${(model.getParamCount() / 1000).toFixed(0)}K params` : 'No model'}
            </span>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
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
                    ? 'border-orange-500 text-orange-400'
                    : 'border-transparent text-gray-500 hover:text-gray-300'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {!isInitialized ? (
          <InitPanel
            modelSize={modelSize}
            setModelSize={setModelSize}
            onInit={initializeModel}
            onLoadWeights={loadWeights}
            config={config}
          />
        ) : (
          <>
            {activeTab === 'overview' && (
              <OverviewPanel model={model!} training={training} />
            )}
            {activeTab === 'train' && (
              <TrainPanel
                training={training}
                setTraining={setTraining}
                onStart={startTraining}
                onStop={stopTraining}
                useCustom={useCustom}
                setUseCustom={setUseCustom}
                customText={customText}
                setCustomText={setCustomText}
              />
            )}
            {activeTab === 'generate' && (
              <GeneratePanel
                prompt={prompt}
                setPrompt={setPrompt}
                generatedText={generatedText}
                isGenerating={isGenerating}
                temperature={temperature}
                setTemperature={setTemperature}
                maxTokens={maxTokens}
                setMaxTokens={setMaxTokens}
                topK={topK}
                setTopK={setTopK}
                topP={topP}
                setTopP={setTopP}
                onGenerate={generateText}
              />
            )}
            {activeTab === 'architecture' && (
              <ArchitecturePanel config={model!.config} />
            )}
            {activeTab === 'export' && (
              <ExportPanel
                model={model!}
                modelSize={modelSize}
                onDownload={downloadWeights}
                onLoadWeights={loadWeights}
              />
            )}
          </>
        )}
      </main>
    </div>
  );
}

// ============ INITIALIZATION PANEL ============
function InitPanel({ modelSize, setModelSize, onInit, onLoadWeights, config }: {
  modelSize: ModelSize;
  setModelSize: (s: ModelSize) => void;
  onInit: () => void;
  onLoadWeights: (e: React.ChangeEvent<HTMLInputElement>) => void;
  config: ModelConfig;
}) {
  return (
    <div className="max-w-3xl mx-auto text-center py-12">
      <div className="text-6xl mb-6">🧠</div>
      <h2 className="text-3xl font-bold mb-3 bg-gradient-to-r from-orange-300 to-red-400 bg-clip-text text-transparent">
        Initialize Your Language Model
      </h2>
      <p className="text-gray-400 mb-10 max-w-xl mx-auto">
        A fully functional transformer language model running entirely in your browser.
        Custom weights, open architecture, ready to train and generate.
      </p>

      {/* Model size selection */}
      <div className="grid grid-cols-2 gap-4 mb-8">
        <button
          onClick={() => setModelSize('small')}
          className={`p-6 rounded-xl border transition-all text-left ${
            modelSize === 'small'
              ? 'border-orange-500/50 bg-orange-500/10'
              : 'border-gray-800 bg-gray-900/30 hover:border-gray-700'
          }`}
        >
          <div className="text-2xl mb-2">⚡</div>
          <h3 className="font-bold text-white mb-1">Small</h3>
          <p className="text-xs text-gray-400 mb-3">Fast training, quick iteration</p>
          <div className="text-xs text-gray-500 space-y-1">
            <div>64 dims • 3 layers • 4 heads</div>
            <div className="text-orange-400 font-mono">~250K parameters</div>
          </div>
        </button>
        <button
          onClick={() => setModelSize('default')}
          className={`p-6 rounded-xl border transition-all text-left ${
            modelSize === 'default'
              ? 'border-orange-500/50 bg-orange-500/10'
              : 'border-gray-800 bg-gray-900/30 hover:border-gray-700'
          }`}
        >
          <div className="text-2xl mb-2">🔥</div>
          <h3 className="font-bold text-white mb-1">Default</h3>
          <p className="text-xs text-gray-400 mb-3">Better quality, slower training</p>
          <div className="text-xs text-gray-500 space-y-1">
            <div>96 dims • 4 layers • 4 heads</div>
            <div className="text-orange-400 font-mono">~500K parameters</div>
          </div>
        </button>
      </div>

      {/* Config details */}
      <div className="mb-8 p-4 rounded-xl bg-gray-900/50 border border-gray-800 text-left">
        <h4 className="text-sm font-semibold text-gray-300 mb-3">Configuration</h4>
        <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
          {[
            ['Vocab', config.vocabSize],
            ['d_model', config.dModel],
            ['n_heads', config.nHeads],
            ['n_layers', config.nLayers],
            ['d_ff', config.dFF],
            ['max_len', config.maxSeqLen],
          ].map(([label, value]) => (
            <div key={label as string} className="text-center">
              <div className="text-lg font-bold text-orange-400 font-mono">{value as number}</div>
              <div className="text-xs text-gray-500">{label as string}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
        <button
          onClick={onInit}
          className="px-8 py-4 bg-gradient-to-r from-orange-500 to-red-600 rounded-xl font-semibold text-white shadow-lg shadow-orange-500/25 hover:shadow-orange-500/40 transition-all hover:scale-105"
        >
          🔥 Initialize New Model
        </button>
        <label className="px-8 py-4 border border-gray-700 rounded-xl font-semibold text-gray-300 hover:border-orange-500/50 hover:text-orange-300 transition-all cursor-pointer">
          📂 Load Weights
          <input type="file" accept=".bin" onChange={onLoadWeights} className="hidden" />
        </label>
      </div>

      <p className="mt-8 text-xs text-gray-600">
        Inspired by SmolLM architecture • Byte-level tokenization • Decoder-only transformer
      </p>
    </div>
  );
}

// ============ OVERVIEW PANEL ============
function OverviewPanel({ model, training }: { model: SmolLM; training: TrainingState }) {
  const config = model.config;
  return (
    <div className="space-y-8">
      {/* Model Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Parameters', value: `${(model.getParamCount() / 1000).toFixed(0)}K`, icon: '🧮' },
          { label: 'Layers', value: config.nLayers, icon: '📚' },
          { label: 'Training Steps', value: training.step, icon: '🔄' },
          { label: 'Current Loss', value: training.loss > 0 ? training.loss.toFixed(3) : 'N/A', icon: '📉' },
        ].map((stat) => (
          <div key={stat.label} className="p-4 rounded-xl bg-gray-900/50 border border-gray-800">
            <div className="text-2xl mb-2">{stat.icon}</div>
            <div className="text-xl font-bold text-white">{stat.value}</div>
            <div className="text-xs text-gray-500">{stat.label}</div>
          </div>
        ))}
      </div>

      {/* Architecture Summary */}
      <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
        <h3 className="text-lg font-bold text-white mb-4">Architecture Summary</h3>
        <div className="space-y-3">
          <ArchRow label="Token Embedding" detail={`${config.vocabSize} × ${config.dModel}`} params={config.vocabSize * config.dModel} />
          <ArchRow label="Position Embedding" detail={`${config.maxSeqLen} × ${config.dModel}`} params={config.maxSeqLen * config.dModel} />
          {Array.from({ length: config.nLayers }, (_, i) => (
            <div key={i} className="pl-4 border-l-2 border-orange-500/30 space-y-2">
              <div className="text-xs text-orange-400 font-medium">Layer {i}</div>
              <ArchRow label="  LayerNorm1" detail={`γ, β [${config.dModel}]`} params={config.dModel * 2} />
              <ArchRow label="  QKV Projection" detail={`[${config.dModel} × ${config.dModel * 3}]`} params={config.dModel * config.dModel * 3 + config.dModel * 3} />
              <ArchRow label="  Multi-Head Attention" detail={`${config.nHeads} heads, d_head=${config.dModel / config.nHeads}`} params={0} />
              <ArchRow label="  Output Projection" detail={`[${config.dModel} × ${config.dModel}]`} params={config.dModel * config.dModel + config.dModel} />
              <ArchRow label="  LayerNorm2" detail={`γ, β [${config.dModel}]`} params={config.dModel * 2} />
              <ArchRow label="  FFN Up" detail={`[${config.dModel} × ${config.dFF}] + GELU`} params={config.dModel * config.dFF + config.dFF} />
              <ArchRow label="  FFN Down" detail={`[${config.dFF} × ${config.dModel}]`} params={config.dFF * config.dModel + config.dModel} />
            </div>
          ))}
          <ArchRow label="Final LayerNorm" detail={`γ, β [${config.dModel}]`} params={config.dModel * 2} />
          <ArchRow label="Output (tied)" detail="Shares token embedding weights" params={0} />
        </div>
      </div>

      {/* Training Data Info */}
      <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
        <h3 className="text-lg font-bold text-white mb-4">Available Training Corpora</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(TRAINING_CORPORA).map(([key, corpus]) => (
            <div key={key} className="p-4 rounded-lg bg-gray-800/50 border border-gray-700">
              <h4 className="font-semibold text-white text-sm">{corpus.name}</h4>
              <p className="text-xs text-gray-400 mt-1">{corpus.description}</p>
              <div className="mt-2 text-xs text-gray-500 font-mono">
                {corpus.text.length.toLocaleString()} chars
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ArchRow({ label, detail, params }: { label: string; detail: string; params: number }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-gray-300">{label}</span>
      <div className="flex items-center gap-4">
        <span className="text-gray-500 text-xs">{detail}</span>
        {params > 0 && <span className="text-orange-400/70 text-xs font-mono">{params.toLocaleString()}</span>}
      </div>
    </div>
  );
}

// ============ TRAINING PANEL ============
function TrainPanel({ training, setTraining, onStart, onStop, useCustom, setUseCustom, customText, setCustomText }: {
  training: TrainingState;
  setTraining: (fn: (prev: TrainingState) => TrainingState) => void;
  onStart: () => void;
  onStop: () => void;
  useCustom: boolean;
  setUseCustom: (v: boolean) => void;
  customText: string;
  setCustomText: (v: string) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Draw loss chart
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

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 5; i++) {
      const y = (ch * i) / 4;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(cw, y);
      ctx.stroke();
    }

    // Loss line
    ctx.strokeStyle = '#f97316';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = (i / (data.length - 1)) * cw;
      const y = ch - ((data[i] - minLoss) / (maxLoss - minLoss)) * ch;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Gradient fill
    const gradient = ctx.createLinearGradient(0, 0, 0, ch);
    gradient.addColorStop(0, 'rgba(249, 115, 22, 0.2)');
    gradient.addColorStop(1, 'rgba(249, 115, 22, 0)');
    ctx.fillStyle = gradient;
    ctx.lineTo(cw, ch);
    ctx.lineTo(0, ch);
    ctx.fill();
  }, [training.lossHistory]);

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <h3 className="font-bold text-white mb-4">Training Configuration</h3>

          <div className="space-y-4">
            <div>
              <label className="text-sm text-gray-400 block mb-2">Training Corpus</label>
              <div className="flex items-center gap-2 mb-3">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={useCustom}
                    onChange={(e) => setUseCustom(e.target.checked)}
                    className="rounded border-gray-600 bg-gray-800 text-orange-500"
                  />
                  <span className="text-sm text-gray-300">Use custom text</span>
                </label>
              </div>
              {!useCustom ? (
                <select
                  value={training.currentCorpus}
                  onChange={(e) => setTraining(prev => ({ ...prev, currentCorpus: e.target.value }))}
                  className="w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white text-sm"
                >
                  {Object.entries(TRAINING_CORPORA).map(([key, c]) => (
                    <option key={key} value={key}>{c.name} ({c.text.length} chars)</option>
                  ))}
                </select>
              ) : (
                <textarea
                  value={customText}
                  onChange={(e) => setCustomText(e.target.value)}
                  placeholder="Paste your training text here (min 50 chars)..."
                  className="w-full h-32 px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white text-sm resize-none"
                />
              )}
            </div>

            <div>
              <label className="text-sm text-gray-400 block mb-2">
                Learning Rate: <span className="text-orange-400 font-mono">{training.learningRate}</span>
              </label>
              <input
                type="range"
                min="0.001"
                max="0.1"
                step="0.001"
                value={training.learningRate}
                onChange={(e) => setTraining(prev => ({ ...prev, learningRate: parseFloat(e.target.value) }))}
                className="w-full accent-orange-500"
              />
            </div>
          </div>
        </div>

        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <h3 className="font-bold text-white mb-4">Training Status</h3>

          <div className="grid grid-cols-2 gap-4 mb-6">
            <div>
              <div className="text-2xl font-bold text-orange-400 font-mono">{training.step}</div>
              <div className="text-xs text-gray-500">Steps</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-orange-400 font-mono">{training.epoch}</div>
              <div className="text-xs text-gray-500">Epochs</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-orange-400 font-mono">
                {training.loss > 0 ? training.loss.toFixed(4) : '—'}
              </div>
              <div className="text-xs text-gray-500">Loss</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-orange-400 font-mono">{training.samplesTrained}</div>
              <div className="text-xs text-gray-500">Samples</div>
            </div>
          </div>

          <div className="flex gap-3">
            {!training.isTraining ? (
              <button
                onClick={onStart}
                className="flex-1 px-4 py-3 bg-gradient-to-r from-orange-500 to-red-600 rounded-lg font-semibold text-white shadow-lg shadow-orange-500/20 hover:shadow-orange-500/40 transition-all"
              >
                🔥 Start Training
              </button>
            ) : (
              <button
                onClick={onStop}
                className="flex-1 px-4 py-3 bg-red-600 rounded-lg font-semibold text-white hover:bg-red-700 transition-all"
              >
                ⏹ Stop Training
              </button>
            )}
          </div>

          {training.isTraining && (
            <div className="mt-4">
              <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-orange-500 to-red-500 rounded-full transition-all animate-pulse"
                  style={{ width: `${Math.min(100, (training.step / 2000) * 100)}%` }}
                />
              </div>
              <div className="text-xs text-gray-500 mt-1">Step {training.step} / 2000</div>
            </div>
          )}
        </div>
      </div>

      {/* Loss Chart */}
      <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
        <h3 className="font-bold text-white mb-4">Training Loss</h3>
        <div className="h-48 bg-gray-950 rounded-lg border border-gray-800 overflow-hidden">
          <canvas ref={canvasRef} className="w-full h-full" />
          {training.lossHistory.length < 2 && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-600 text-sm">
              Start training to see loss curve
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ============ GENERATION PANEL ============
function GeneratePanel({ prompt, setPrompt, generatedText, isGenerating, temperature, setTemperature, maxTokens, setMaxTokens, topK, setTopK, topP, setTopP, onGenerate }: {
  prompt: string;
  setPrompt: (v: string) => void;
  generatedText: string;
  isGenerating: boolean;
  temperature: number;
  setTemperature: (v: number) => void;
  maxTokens: number;
  setMaxTokens: (v: number) => void;
  topK: number;
  setTopK: (v: number) => void;
  topP: number;
  setTopP: (v: number) => void;
  onGenerate: () => void;
}) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Prompt & Output */}
        <div className="md:col-span-2 space-y-4">
          <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
            <label className="text-sm text-gray-400 block mb-2">Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter your prompt..."
              className="w-full h-24 px-4 py-3 rounded-lg bg-gray-800 border border-gray-700 text-white resize-none focus:border-orange-500/50 focus:outline-none"
            />
            <button
              onClick={onGenerate}
              disabled={isGenerating || !prompt}
              className="mt-3 px-6 py-3 bg-gradient-to-r from-orange-500 to-red-600 rounded-lg font-semibold text-white shadow-lg shadow-orange-500/20 hover:shadow-orange-500/40 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isGenerating ? '⏳ Generating...' : '✨ Generate'}
            </button>
          </div>

          <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
            <label className="text-sm text-gray-400 block mb-2">Generated Output</label>
            <div className="min-h-[200px] p-4 rounded-lg bg-gray-950 border border-gray-800 font-mono text-sm text-gray-300 whitespace-pre-wrap">
              {generatedText || <span className="text-gray-600">Generated text will appear here...</span>}
            </div>
          </div>
        </div>

        {/* Parameters */}
        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <h3 className="font-bold text-white mb-4">Generation Parameters</h3>
          <div className="space-y-5">
            <div>
              <label className="text-sm text-gray-400 flex justify-between">
                <span>Temperature</span>
                <span className="text-orange-400 font-mono">{temperature.toFixed(2)}</span>
              </label>
              <input type="range" min="0.1" max="2.0" step="0.05" value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))} className="w-full accent-orange-500 mt-1" />
              <div className="flex justify-between text-xs text-gray-600 mt-1">
                <span>Deterministic</span><span>Creative</span>
              </div>
            </div>

            <div>
              <label className="text-sm text-gray-400 flex justify-between">
                <span>Max Tokens</span>
                <span className="text-orange-400 font-mono">{maxTokens}</span>
              </label>
              <input type="range" min="10" max="200" step="10" value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value))} className="w-full accent-orange-500 mt-1" />
            </div>

            <div>
              <label className="text-sm text-gray-400 flex justify-between">
                <span>Top-K</span>
                <span className="text-orange-400 font-mono">{topK}</span>
              </label>
              <input type="range" min="1" max="100" step="1" value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))} className="w-full accent-orange-500 mt-1" />
            </div>

            <div>
              <label className="text-sm text-gray-400 flex justify-between">
                <span>Top-P (Nucleus)</span>
                <span className="text-orange-400 font-mono">{topP.toFixed(2)}</span>
              </label>
              <input type="range" min="0.1" max="1.0" step="0.05" value={topP}
                onChange={(e) => setTopP(parseFloat(e.target.value))} className="w-full accent-orange-500 mt-1" />
            </div>
          </div>

          <div className="mt-6 p-3 rounded-lg bg-gray-800/50 border border-gray-700">
            <h4 className="text-xs font-semibold text-gray-400 mb-2">Quick Prompts</h4>
            <div className="flex flex-wrap gap-2">
              {['the ', 'hello ', 'function ', 'if ', 'roses are '].map((p) => (
                <button key={p} onClick={() => setPrompt(p)}
                  className="px-2 py-1 rounded bg-gray-700 text-xs text-gray-300 hover:bg-orange-500/20 hover:text-orange-300 transition-colors">
                  "{p}"
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============ ARCHITECTURE PANEL ============
function ArchitecturePanel({ config }: { config: ModelConfig }) {
  const dHead = config.dModel / config.nHeads;

  return (
    <div className="space-y-6">
      <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
        <h3 className="text-lg font-bold text-white mb-6">Decoder-Only Transformer Architecture</h3>
        <p className="text-gray-400 text-sm mb-6">
          This model implements the same architecture as SmolLM, GPT-2, and other decoder-only transformers.
          Each layer consists of causal self-attention followed by a feed-forward network, with residual connections and layer normalization.
        </p>

        {/* Visual Architecture */}
        <div className="flex flex-col items-center gap-2 py-8">
          {/* Input */}
          <ArchBlock label="Input Tokens" detail={`[batch, seq_len]`} color="blue" />
          <Arrow />
          <ArchBlock label="Token Embedding" detail={`[${config.vocabSize}, ${config.dModel}]`} color="purple" />
          <Arrow />
          <ArchBlock label="+ Position Embedding" detail={`[${config.maxSeqLen}, ${config.dModel}]`} color="purple" />
          <Arrow />

          {/* Transformer blocks */}
          <div className="w-full max-w-md border-2 border-dashed border-orange-500/30 rounded-xl p-4">
            <div className="text-xs text-orange-400 text-center mb-3">× {config.nLayers} Transformer Blocks</div>
            <div className="flex flex-col items-center gap-2">
              <ArchBlock label="Layer Norm" detail={`[${config.dModel}]`} color="green" />
              <Arrow />
              <ArchBlock label="Multi-Head Causal Attention" detail={`${config.nHeads} heads, d_head=${dHead}`} color="orange" />
              <Arrow />
              <ArchBlock label="+ Residual" detail="" color="gray" />
              <Arrow />
              <ArchBlock label="Layer Norm" detail={`[${config.dModel}]`} color="green" />
              <Arrow />
              <ArchBlock label="Feed-Forward (GELU)" detail={`${config.dModel} → ${config.dFF} → ${config.dModel}`} color="red" />
              <Arrow />
              <ArchBlock label="+ Residual" detail="" color="gray" />
            </div>
          </div>

          <Arrow />
          <ArchBlock label="Final Layer Norm" detail={`[${config.dModel}]`} color="green" />
          <Arrow />
          <ArchBlock label="Output Projection (tied)" detail={`[${config.dModel}, ${config.vocabSize}]`} color="purple" />
          <Arrow />
          <ArchBlock label="Softmax → Sample" detail="Temperature + Top-K + Top-P" color="blue" />
        </div>
      </div>

      {/* Key Design Decisions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <h4 className="font-bold text-white mb-3">🔑 Key Design Decisions</h4>
          <ul className="space-y-2 text-sm text-gray-400">
            <li>• <span className="text-gray-300">Byte-level tokenization</span> — No BPE needed, works on any UTF-8 text</li>
            <li>• <span className="text-gray-300">Pre-norm architecture</span> — LayerNorm before attention/FFN (more stable training)</li>
            <li>• <span className="text-gray-300">Tied embeddings</span> — Input and output share weights (fewer params)</li>
            <li>• <span className="text-gray-300">Causal masking</span> — Can only attend to past tokens (autoregressive)</li>
            <li>• <span className="text-gray-300">GELU activation</span> — Smooth approximation used in modern LLMs</li>
            <li>• <span className="text-gray-300">Combined QKV</span> — Single matrix for Q, K, V (faster inference)</li>
          </ul>
        </div>
        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <h4 className="font-bold text-white mb-3">📊 Model Statistics</h4>
          <div className="space-y-3">
            {[
              ['Total Parameters', `${(config.vocabSize * config.dModel * 2 + config.maxSeqLen * config.dModel + config.nLayers * (config.dModel * config.dModel * 4 + config.dModel * config.dFF * 2 + config.dModel * 8)).toLocaleString()}`],
              ['Attention Params/Layer', `${(config.dModel * config.dModel * 4 + config.dModel * 4).toLocaleString()}`],
              ['FFN Params/Layer', `${(config.dModel * config.dFF * 2 + config.dFF + config.dModel).toLocaleString()}`],
              ['Vocab Size', `${config.vocabSize} (256 bytes + 4 special)`],
              ['Context Window', `${config.maxSeqLen} tokens`],
              ['Inference Memory', `~${(config.dModel * config.dModel * config.nLayers * 4 / 1024 / 1024).toFixed(1)} MB (FP32)`],
            ].map(([label, value]) => (
              <div key={label} className="flex justify-between text-sm">
                <span className="text-gray-400">{label}</span>
                <span className="text-orange-400 font-mono">{value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function ArchBlock({ label, detail, color }: { label: string; detail: string; color: string }) {
  const colors: Record<string, string> = {
    blue: 'border-blue-500/40 bg-blue-500/5',
    purple: 'border-purple-500/40 bg-purple-500/5',
    green: 'border-green-500/40 bg-green-500/5',
    orange: 'border-orange-500/40 bg-orange-500/10',
    red: 'border-red-500/40 bg-red-500/5',
    gray: 'border-gray-600/40 bg-gray-800/30',
  };
  return (
    <div className={`w-full max-w-md px-4 py-2.5 rounded-lg border ${colors[color]} text-center`}>
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
function ExportPanel({ model, modelSize, onDownload, onLoadWeights }: {
  model: SmolLM;
  modelSize: ModelSize;
  onDownload: () => void;
  onLoadWeights: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  const weightSize = model.serializeWeights().byteLength;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Download */}
        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <h3 className="font-bold text-white mb-4">📥 Download Weights</h3>
          <p className="text-sm text-gray-400 mb-4">
            Download the complete model weights as a binary file. Load them later to resume exactly where you left off.
          </p>
          <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700 mb-4">
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">File size</span>
              <span className="text-orange-400 font-mono">{(weightSize / 1024).toFixed(1)} KB</span>
            </div>
            <div className="flex justify-between text-sm mt-1">
              <span className="text-gray-400">Format</span>
              <span className="text-gray-300 font-mono">Binary (Float32)</span>
            </div>
            <div className="flex justify-between text-sm mt-1">
              <span className="text-gray-400">Parameters</span>
              <span className="text-gray-300 font-mono">{model.getParamCount().toLocaleString()}</span>
            </div>
          </div>
          <button
            onClick={onDownload}
            className="w-full px-4 py-3 bg-gradient-to-r from-orange-500 to-red-600 rounded-lg font-semibold text-white shadow-lg shadow-orange-500/20 hover:shadow-orange-500/40 transition-all"
          >
            💾 Download smol-lm-{modelSize}.bin
          </button>
        </div>

        {/* Upload */}
        <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
          <h3 className="font-bold text-white mb-4">📤 Load Weights</h3>
          <p className="text-sm text-gray-400 mb-4">
            Load previously saved weights to continue from a trained checkpoint. Compatible with any SMOL-LM binary.
          </p>
          <label className="block w-full px-4 py-8 border-2 border-dashed border-gray-700 rounded-lg text-center cursor-pointer hover:border-orange-500/50 transition-colors">
            <div className="text-3xl mb-2">📂</div>
            <div className="text-sm text-gray-300">Click to load .bin file</div>
            <div className="text-xs text-gray-500 mt-1">or drag and drop</div>
            <input type="file" accept=".bin" onChange={onLoadWeights} className="hidden" />
          </label>
        </div>
      </div>

      {/* Standalone runner info */}
      <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
        <h3 className="font-bold text-white mb-4">🖥️ Running Locally</h3>
        <p className="text-sm text-gray-400 mb-4">
          This model runs entirely in the browser using JavaScript. To run it locally on your PC:
        </p>
        <div className="space-y-3">
          <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700">
            <div className="text-sm text-gray-300 font-medium mb-1">1. Download this web app</div>
            <div className="text-xs text-gray-500">Save the page or clone the repository. The model runs in any modern browser.</div>
          </div>
          <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700">
            <div className="text-sm text-gray-300 font-medium mb-1">2. Train or load weights</div>
            <div className="text-xs text-gray-500">Train in-browser or load pre-trained weights from a .bin file.</div>
          </div>
          <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700">
            <div className="text-sm text-gray-300 font-medium mb-1">3. Generate text</div>
            <div className="text-xs text-gray-500">Use the generation panel with your preferred parameters.</div>
          </div>
          <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700">
            <div className="text-sm text-gray-300 font-medium mb-1">4. Export & share</div>
            <div className="text-xs text-gray-500">Download weights to share your trained model with others.</div>
          </div>
        </div>

        <div className="mt-6 p-4 rounded-lg bg-orange-500/5 border border-orange-500/20">
          <h4 className="text-sm font-semibold text-orange-400 mb-2">🔓 Fully Open Source</h4>
          <p className="text-xs text-gray-400">
            All weights are generated locally and are fully open. No external API calls, no cloud dependencies.
            The model architecture, tokenizer, training loop, and inference engine are all included.
            You own your weights completely — download, share, modify, or redistribute them freely.
          </p>
        </div>
      </div>

      {/* Technical details */}
      <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
        <h3 className="font-bold text-white mb-4">🔧 Technical Details</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            ['Architecture', 'Decoder-only Transformer'],
            ['Tokenization', 'Byte-level (UTF-8)'],
            ['Precision', 'Float32'],
            ['Attention', 'Causal Multi-Head'],
            ['Activation', 'GELU'],
            ['Normalization', 'Pre-Norm (RMS-style)'],
            ['Training', 'SGD + Gradient Clipping'],
            ['Sampling', 'Temp + Top-K + Top-P'],
          ].map(([label, value]) => (
            <div key={label}>
              <div className="text-xs text-gray-500">{label}</div>
              <div className="text-sm text-gray-300 font-medium">{value}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
