import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, MotionConfig, motion } from 'framer-motion';
import { Download, Play, SlidersHorizontal, Square } from 'lucide-react';
import {
  EngineState,
  POLICY_META,
  PipelineConfig,
  PipelineResults,
  SimSettings,
  SimulationResults,
  ViewId,
} from './types';
import { ParetoChart, TrajectoryChart } from './components/Charts';
import { LandingPage } from './components/LandingPage';
import { PipelineStudio } from './components/PipelineStudio';
import { TopologyReplay } from './components/TopologyReplay';

const FULL: SimSettings = {
  rounds: 20,
  episodes: 130,
  repeats: 5,
  target: 80,
  sensitivity: 70,
  local_epochs: 2,
  learning_rate: 0.23,
  seed: 42,
  latency_weight: 0.018,
  carbon_weight: 0.32,
};
const QUICK: SimSettings = { ...FULL, rounds: 12, episodes: 35, repeats: 3, sensitivity: 12 };

const CALIBRATED_NODE_CI: Record<string, number> = {
  PC1: 0.675,
  PC2: 0.52,
  PC3: 0.82,
  PC4: 0.42,
};

const PNG_FIGURES = [
  { value: 'accuracy_by_round.png', label: 'Learning progress (Accuracy by round)' },
  { value: 'tradeoff.png', label: 'Accuracy–carbon trade-off' },
  { value: 'cumulative_latency.png', label: 'Cumulative synchronous latency' },
  { value: 'cumulative_carbon.png', label: 'Cumulative carbon emissions' },
  { value: 'node_energy_scenarios.png', label: 'Client energy benchmark profiles' },
];

async function api<T>(path: string, body?: unknown): Promise<T> {
  const opts: RequestInit =
    body === undefined
      ? {}
      : {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        };
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'Request failed');
  return data as T;
}

const PATH_TO_VIEW: Record<string, ViewId> = {
  '/': 'landing',
  '/index.html': 'landing',
  '/overview': 'overview',
  '/pipeline': 'pipeline',
  '/replay': 'replay',
  '/specs': 'specs',
};
const VIEW_TO_PATH: Record<ViewId, string> = {
  landing: '/',
  overview: '/overview',
  pipeline: '/pipeline',
  replay: '/replay',
  specs: '/specs',
};

export default function App() {
  const [view, setView] = useState<ViewId>(() => PATH_TO_VIEW[window.location.pathname] ?? 'landing');
  const [preset, setPreset] = useState<'quick' | 'full'>('full');
  const [settings, setSettings] = useState<SimSettings>(FULL);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const [offline, setOffline] = useState(false);
  const [engine, setEngine] = useState<EngineState>({
    status: 'idle',
    progress: 0,
    message: 'Connecting…',
    run_id: null,
    result_id: '',
    log: [],
  });
  const [results, setResults] = useState<SimulationResults | null>(null);
  const [pipeline, setPipeline] = useState<PipelineResults | null>(null);
  const [error, setError] = useState('');

  const [chartTab, setChartTab] = useState<'tradeoff' | 'accuracy' | 'latency' | 'carbon' | 'png'>('tradeoff');
  const [pngFile, setPngFile] = useState('accuracy_by_round.png');
  const [hiddenPolicies, setHiddenPolicies] = useState<Set<string>>(new Set());

  const resultIdRef = useRef<string>('');
  const runInFlightRef = useRef<boolean>(false);
  const logRef = useRef<HTMLPreElement>(null);

  const syncSettingsFromResults = useCallback((res: SimulationResults) => {
    const run = res.parameters?.run;
    if (!run) return;
    const nextSettings: SimSettings = {
      rounds: Number(run.rounds ?? 20),
      episodes: Number(run.episodes ?? 130),
      repeats: Number(run.repeats ?? 5),
      target: Number(run.target ?? 0.8) <= 1 ? Math.round(Number(run.target ?? 0.8) * 1000) / 10 : Number(run.target),
      sensitivity: Number(run.sensitivity ?? 70),
      local_epochs: Number(run.local_epochs ?? 2),
      learning_rate: Number(run.learning_rate ?? 0.23),
      seed: Number(run.seed ?? 42),
      latency_weight: Number(run.latency_weight ?? 0.018),
      carbon_weight: Number(run.carbon_weight ?? 0.32),
    };
    setSettings(nextSettings);
    setPreset(nextSettings.rounds === 12 && nextSettings.episodes === 35 ? 'quick' : 'full');
  }, []);

  const poll = useCallback(async (forceRefresh = false) => {
    if (runInFlightRef.current) return;
    try {
      const state = await api<EngineState>('/api/state');
      if (runInFlightRef.current) return;
      setEngine(state);
      if (state.status === 'failed') setError(state.message);
      const shouldLoad = !resultIdRef.current || forceRefresh;
      if (shouldLoad && state.result_id && state.result_id !== resultIdRef.current) {
        const res = await api<SimulationResults>('/api/results');
        resultIdRef.current = res.result_id || state.result_id;
        setResults(res);
        syncSettingsFromResults(res);
      }
    } catch (e) {
      setEngine((prev) => ({ ...prev, status: 'failed', message: 'Engine unavailable' }));
      setError(e instanceof Error ? e.message : 'Unable to reach engine');
    }
  }, [syncSettingsFromResults]);

  const navigate = useCallback(
    (next: ViewId) => {
      setView(next);
      window.scrollTo({ top: 0, behavior: 'smooth' });
      if (!offline && window.location.protocol.startsWith('http')) {
        const targetPath = VIEW_TO_PATH[next] ?? '/';
        if (window.location.pathname !== targetPath) {
          window.history.pushState({ view: next }, '', targetPath);
        }
      }
    },
    [offline],
  );

  useEffect(() => {
    const onPop = () => {
      const mapped = PATH_TO_VIEW[window.location.pathname];
      if (mapped) setView(mapped);
    };
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  useEffect(() => {
    try {
      const el = document.getElementById('bootstrap');
      const embedded = el ? JSON.parse(el.textContent || 'null') : null;
      if (embedded) {
        setOffline(true);
        const simRes = embedded as SimulationResults;
        setResults(simRes);
        syncSettingsFromResults(simRes);
        setEngine({
          status: 'complete',
          progress: 100,
          message: 'Completed results loaded from archive.',
          run_id: null,
          result_id: 'offline',
          log: [],
        });
        return;
      }
    } catch {
      // Fall back to live server polling
    }

    poll();
    Promise.all([
      api<{
        config: PipelineResults['config'];
        clients: PipelineResults['clients'];
        pareto_policies: string[];
        pareto_morl: string[];
      }>('/api/pipeline/layers'),
      api<{ policies: PipelineResults['policies']; pareto_policies: string[] }>('/api/pipeline/policies'),
      api<{ morl_sweep: PipelineResults['morl_sweep']; pareto_morl: string[] }>('/api/pipeline/morl'),
    ])
      .then(([layers, pols, morl]) =>
        setPipeline({
          config: layers.config,
          clients: layers.clients,
          policies: pols.policies,
          pareto_policies: pols.pareto_policies,
          morl_sweep: morl.morl_sweep,
          pareto_morl: morl.pareto_morl,
        }),
      )
      .catch(() => {});
    const id = window.setInterval(poll, 700);
    return () => window.clearInterval(id);
  }, [poll, syncSettingsFromResults]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [engine.log]);

  const running = engine.status === 'running';

  function applyPreset(kind: 'quick' | 'full') {
    if (running || offline) return;
    setPreset(kind);
    setSettings(kind === 'quick' ? QUICK : FULL);
  }

  async function handleRun(e: React.FormEvent) {
    e.preventDefault();
    if (offline || running) return;
    setError('');
    runInFlightRef.current = true;
    setEngine((prev) => ({
      ...prev,
      status: 'running',
      progress: 4,
      message: 'Preparing client data & training RL scheduler…',
    }));
    const ticker = window.setInterval(() => {
      setEngine((prev) => {
        if (prev.status !== 'running' || prev.progress >= 92) return prev;
        return { ...prev, progress: Math.min(92, prev.progress + 4) };
      });
    }, 250);
    try {
      await api('/api/run', settings);
      runInFlightRef.current = false;
      await poll(true);
    } catch (err) {
      runInFlightRef.current = false;
      setError(err instanceof Error ? err.message : 'Run failed');
      setEngine((prev) => ({ ...prev, status: 'failed' }));
    } finally {
      runInFlightRef.current = false;
      window.clearInterval(ticker);
    }
  }

  async function handleStop() {
    try {
      await api('/api/cancel', {});
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Stop failed');
    }
  }

  function togglePolicy(p: string) {
    setHiddenPolicies((prev) => {
      const next = new Set(prev);
      if (next.has(p)) next.delete(p);
      else next.add(p);
      return next;
    });
  }

  const kpis = useMemo(() => {
    if (!results?.summary.length) return null;
    const bestAcc = [...results.summary].sort(
      (a, b) => Number(b.final_accuracy_mean) - Number(a.final_accuracy_mean),
    )[0];
    const lowCo2 = [...results.summary].sort(
      (a, b) => Number(a.mean_total_CO2_g_MODELED) - Number(b.mean_total_CO2_g_MODELED),
    )[0];
    const lowLat = [...results.summary].sort(
      (a, b) => Number(a.mean_total_seconds_MODELED) - Number(b.mean_total_seconds_MODELED),
    )[0];
    const all4 = results.summary.find((s) => s.policy === 'FedAvg_all4');
    const rl = results.summary.find((s) => s.policy === 'Q_learning_RL');
    const rlCo2Save =
      all4 && rl
        ? ((1 - Number(rl.mean_total_CO2_g_MODELED) / Number(all4.mean_total_CO2_g_MODELED)) * 100).toFixed(1)
        : null;
    return { bestAcc, lowCo2, lowLat, all4, rl, rlCo2Save };
  }, [results]);

  const badgeClass =
    engine.status === 'running'
      ? 'running'
      : engine.status === 'failed'
        ? 'failed'
        : 'ready';
  const badgeText = offline
    ? 'Offline Archive'
    : engine.status === 'running'
      ? `Running · ${engine.progress}%`
      : engine.status === 'complete'
        ? 'Run Complete'
        : engine.status === 'failed'
          ? 'Run Failed'
          : 'Engine Ready';

  const navItems: {
    id: ViewId;
    num: string;
    label: string;
    desc: string;
    heading: string;
    plateCode: string;
    sub: string;
    theoryTitle: string;
    theoryBody: string;
  }[] = [
    {
      id: 'overview',
      num: '01',
      label: 'Policy & Pareto Matrix',
      desc: '4-policy trade-off & trajectories',
      heading: 'Empirical Policy Benchmarks & Pareto Trade-Off',
      plateCode: 'PLATE 01 · BENCHMARK',
      sub: 'Compare held-out accuracy, synchronous round latency, energy (Wh), and modeled carbon (g CO₂) across the four policies.',
      theoryTitle: 'Analytical Mechanism: Synchronous Straggler Penalty vs. Subset Diversity',
      theoryBody:
        'Selecting all 4 clients (FedAvg · All 4) maximizes per-round sample diversity but suffers straggler latency and high CO₂ because fast clients idle while PC1 finishes. Fastest 2 minimizes round time, while Lowest Energy 2 reduces active watts. Q-Learning RL learns a dynamic subset policy (2–4 clients) conditioned on (training phase, accuracy bucket).',
    },
    {
      id: 'pipeline',
      num: '02',
      label: '8-Layer MORL Pipeline',
      desc: 'fl_pipeline/ & 10k telemetry',
      heading: 'Software-Driven 8-Layer FL + MORL Architecture',
      plateCode: 'PLATE 02 · PIPELINE',
      sub: 'Execute fl_pipeline/ live with configurable Dirichlet(α) partitions, 10k telemetry calibration, 7 schedulers, and MORL preference sweeps.',
      theoryTitle: 'Analytical Mechanism: Modular 8-Layer Federated + Multi-Objective RL Stack',
      theoryBody:
        'Partitions synthetic Gaussian data via IID or Dirichlet(α) distributions (Layer 1), calibrates 4 clients from the 10,000-row telemetry CSV (Layer 2), models parallel network latency and per-client carbon intensity (Layers 3–4), evaluates 7 schedulers with Gini fairness (Layers 5–6), and sweeps MORL preference weights to extract the 3D Pareto frontier (Layers 7–8).',
    },
    {
      id: 'replay',
      num: '03',
      label: 'Topology Replay',
      desc: 'Round-by-round hub & spoke',
      heading: 'Synchronous Hub-and-Spoke Topology & Round Replay',
      plateCode: 'PLATE 03 · TOPOLOGY',
      sub: 'Step through individual training rounds to inspect scheduled uplinks (PC1–PC4), standby idle nodes, and per-round energy accounting.',
      theoryTitle: 'Analytical Mechanism: Parallel Uplink Synchronization & Idle Power Tail',
      theoryBody:
        'In synchronous FedAvg, selected clients train in parallel and transmit weight updates to the coordinator. Round duration equals the slowest selected client (training + network) plus 0.45 s aggregation. Unselected or early-finishing clients still draw idle power (7–13 W) until the round ends, while the coordinator draws 30 W continuously.',
    },
    {
      id: 'specs',
      num: '04',
      label: 'Calibration & Logs',
      desc: 'PC1 meter, sensitivity & stdout',
      heading: 'Physical Meter Calibration, Sensitivity & Execution Logs',
      plateCode: 'PLATE 04 · PROVENANCE',
      sub: 'Audit exact hyperparameters, PC1 physical meter anchor, per-client grid carbon intensities, Monte Carlo sensitivity, and live Python stdout.',
      theoryTitle: 'Analytical Mechanism: Physical Measurement Anchor & Scenario Sensitivity',
      theoryBody:
        'Only PC1’s wall-meter reading (0.065 kWh over 190 min = 20.53 W mean power) is a physical measurement. PC2–PC4 speed/power ratios, network transfer delays, and ±35% power / ±30% speed Monte Carlo sensitivity quantiles are modeled research scenarios logged with deterministic random seeds.',
    },
  ];
  const activeIdx = Math.max(0, navItems.findIndex((n) => n.id === view));
  const currentStep = navItems[activeIdx] ?? navItems[0];

  const r = results?.parameters.run;
  const t = results?.parameters.training;
  const q = results?.parameters.rl;
  const infra = results?.parameters.infrastructure;

  if (view === 'landing') {
    return (
      <MotionConfig reducedMotion="user">
        <LandingPage onEnter={(target) => navigate(target)} />
      </MotionConfig>
    );
  }

  return (
    <MotionConfig reducedMotion="user">
      <div className="app-frame">
        <aside className="nav-rail" aria-label="Monograph plate navigation">
          <div className="brand">
            <div className="emblem" aria-hidden="true">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <circle cx="12" cy="12" r="3" />
                <circle cx="5" cy="5" r="2" />
                <circle cx="19" cy="5" r="2" />
                <circle cx="5" cy="19" r="2" />
                <circle cx="19" cy="19" r="2" />
                <path d="M6.6 6.6L9.8 9.8M17.4 6.6L14.2 9.8M6.6 17.4L9.8 14.2M17.4 17.4L14.2 14.2" />
              </svg>
            </div>
            <div>
              <h1>Federated Computing Lab</h1>
              <p>MORL Workbench · 4 Plates</p>
            </div>
          </div>

          <nav className="nav-steps" role="tablist" aria-label="Ordered workbench plates">
            {navItems.map((item) => {
              const active = view === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  className={`nav-step ${active ? 'active' : ''}`}
                  onClick={() => navigate(item.id)}
                >
                  {active && (
                    <motion.div
                      layoutId="sidebar-active"
                      className="nav-step-bg"
                      transition={{ type: 'spring', stiffness: 420, damping: 34 }}
                    />
                  )}
                  <span className="step-num num">{item.num}</span>
                  <span className="step-copy">
                    <span className="step-title">{item.label}</span>
                    <span className="step-desc">{item.desc}</span>
                  </span>
                </button>
              );
            })}
          </nav>

          <div className="rail-provenance">
            <div className="rp-row">
              <span>PC1 Meter Anchor</span>
              <strong className="num">20.53 W (0.065 kWh)</strong>
            </div>
            <div className="rp-row">
              <span>PC2–PC4 Cluster</span>
              <strong>Calibrated Scenarios</strong>
            </div>
            <div className="rp-row">
              <span>Default Grid Factor</span>
              <strong className="num">0.675 kg CO₂/kWh</strong>
            </div>
          </div>

          <div className="rail-foot">
            <button type="button" className="btn-tune" onClick={() => navigate('landing')}>
              ← Monograph Cover
            </button>
            <span className={`status-pill ${badgeClass}`}>
              <motion.span
                className="status-dot"
                animate={engine.status === 'running' ? { scale: [1, 1.35, 1], opacity: [1, 0.5, 1] } : {}}
                transition={{ repeat: Infinity, duration: 1.2 }}
              />
              <span>{badgeText}</span>
            </span>
            <a
              className="btn-pill"
              href={
                offline
                  ? 'comparison_SIMULATION.csv'
                  : `/api/export?run=${encodeURIComponent(results?.result_id || 'baseline')}`
              }
            >
              <Download size={13} />
              {offline ? 'Download CSV' : 'Export Run ZIP'}
            </a>
          </div>
        </aside>

        <main className="main-col">
          <div className="stage-header">
            <div className="stage-title-group">
              <div className="head-inline">
                <h2>{currentStep.heading}</h2>
                <span className="plate-tag num">{currentStep.plateCode}</span>
              </div>
              <p>{currentStep.sub}</p>
            </div>
            <div className="pager-btns">
              <button
                type="button"
                className="btn-tune"
                disabled={activeIdx <= 0}
                onClick={() => navigate(navItems[Math.max(0, activeIdx - 1)].id)}
              >
                ← Prev Plate
              </button>
              <button
                type="button"
                className="btn-tune"
                disabled={activeIdx >= navItems.length - 1}
                onClick={() => navigate(navItems[Math.min(navItems.length - 1, activeIdx + 1)].id)}
              >
                Next Plate →
              </button>
            </div>
          </div>

          {offline && (
            <div className="offline-banner">
              Offline results archive loaded. Launch <b>START_DEMO.sh</b> (macOS/Linux) or <b>START_DEMO.bat</b> (Windows) to execute new Python simulations.
            </div>
          )}

          {(view === 'overview' || view === 'replay') && (
            <form className="deck" onSubmit={handleRun}>
              <div className="deck-main">
                <div className="deck-left">
                  <div className="seg" role="group" aria-label="Simulation presets">
                    <button
                      type="button"
                      className={preset === 'quick' ? 'active' : ''}
                      disabled={running || offline}
                      onClick={() => applyPreset('quick')}
                    >
                      Quick demo
                    </button>
                    <button
                      type="button"
                      className={preset === 'full' ? 'active' : ''}
                      disabled={running || offline}
                      onClick={() => applyPreset('full')}
                    >
                      Full experiment
                    </button>
                  </div>

                  <div className="inline-params">
                    {(
                      [
                        ['rounds', 'Rounds', 1, 100, 1],
                        ['episodes', 'Episodes', 1, 500, 1],
                        ['repeats', 'Trials', 1, 20, 1],
                        ['target', 'Target %', 0.01, 100, 'any'],
                        ['sensitivity', 'Sens n', 1, 100, 1],
                      ] as const
                    ).map(([key, label, min, max, step]) => (
                      <div className="chip-field" key={key}>
                        <label htmlFor={key}>{label}</label>
                        <input
                          id={key}
                          type="number"
                          min={min}
                          max={max}
                          step={step}
                          disabled={running || offline}
                          value={settings[key]}
                          onChange={(e) => setSettings((s) => ({ ...s, [key]: Number(e.target.value) }))}
                          required
                        />
                      </div>
                    ))}
                  </div>
                </div>

                <div className="deck-right">
                  <button type="button" className="btn-tune" onClick={() => setDrawerOpen((o) => !o)}>
                    <SlidersHorizontal size={13} />
                    {drawerOpen ? 'Hide Weights' : 'Weights'}
                  </button>
                  <motion.button
                    type="submit"
                    className="btn-run"
                    disabled={running || offline}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Play size={13} />
                    {running ? 'Running…' : offline ? 'Offline' : 'Run Simulation'}
                  </motion.button>
                  <button type="button" className="btn-stop" disabled={!running || offline} onClick={handleStop}>
                    <Square size={12} /> Stop
                  </button>
                </div>
              </div>

              <AnimatePresence>
                {drawerOpen && (
                  <motion.div
                    className="deck-drawer"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
                  >
                    {(
                      [
                        ['local_epochs', 'Local epochs', 1, 20, 1],
                        ['learning_rate', 'Learning rate (η)', 0.001, 2, 0.001],
                        ['seed', 'Random seed', 0, 2147483647, 1],
                        ['latency_weight', 'Latency penalty (w_lat)', 0, 100, 0.001],
                        ['carbon_weight', 'Carbon penalty (w_co2)', 0, 100, 0.01],
                      ] as const
                    ).map(([key, label, min, max, step]) => (
                      <div className="drawer-field" key={key}>
                        <label htmlFor={key}>{label}</label>
                        <input
                          id={key}
                          type="number"
                          min={min}
                          max={max}
                          step={step}
                          disabled={running || offline}
                          value={settings[key]}
                          onChange={(e) => setSettings((s) => ({ ...s, [key]: Number(e.target.value) }))}
                          required
                        />
                      </div>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="deck-progress">
                <div className="p-track">
                  <motion.div
                    className="p-fill"
                    animate={{ width: `${engine.progress}%` }}
                    transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                  />
                </div>
                <span className="p-pct num">{engine.progress}%</span>
                <span className="p-text" role="status">
                  {engine.message}
                </span>
              </div>
              {error && (
                <p className="error" role="alert">
                  {error}
                </p>
              )}
            </form>
          )}

          <AnimatePresence mode="wait">
            {view === 'overview' && (
              <motion.section
                key="overview"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
              >
                <div className="editorial-strip" aria-label="Empirical summary ledger">
                  <div className="rib-cell lead">
                    <span className="rib-label">Q-Learning RL Outcome</span>
                    <b className="rib-val num">
                      {kpis?.rl ? `${(Number(kpis.rl.final_accuracy_mean) * 100).toFixed(1)}%` : '—'}
                    </b>
                    <span className="rib-sub num">
                      {kpis?.rlCo2Save ? `−${kpis.rlCo2Save}% CO₂ vs. FedAvg All 4` : 'Dynamic 2–4 client subset'}
                    </span>
                  </div>
                  <div className="rib-cell">
                    <span className="rib-label">Highest Mean Accuracy</span>
                    <b className="rib-val num">
                      {kpis ? `${(Number(kpis.bestAcc.final_accuracy_mean) * 100).toFixed(1)}%` : '—'}
                    </b>
                    <span className="rib-sub">
                      {kpis ? POLICY_META[kpis.bestAcc.policy]?.label ?? kpis.bestAcc.policy : 'Loading…'}
                    </span>
                  </div>
                  <div className="rib-cell">
                    <span className="rib-label">Lowest Modeled Carbon</span>
                    <b className="rib-val num">
                      {kpis ? `${Number(kpis.lowCo2.mean_total_CO2_g_MODELED).toFixed(2)} g` : '—'}
                    </b>
                    <span className="rib-sub">
                      {kpis ? POLICY_META[kpis.lowCo2.policy]?.label ?? kpis.lowCo2.policy : 'Modeled CO₂'}
                    </span>
                  </div>
                  <div className="rib-cell">
                    <span className="rib-label">Lowest Cumulative Latency</span>
                    <b className="rib-val num">
                      {kpis ? `${Number(kpis.lowLat.mean_total_seconds_MODELED).toFixed(1)} s` : '—'}
                    </b>
                    <span className="rib-sub num">
                      {results ? `Wall exec: ${Number(results.parameters.execution_wall_seconds).toFixed(2)} s` : 'Loading…'}
                    </span>
                  </div>
                </div>

                {/* Split Empirical Plate: Side-by-side Pareto Chart + 4-Policy Comparison Matrix */}
                <div className="overview-split-grid">
                  <div className="panel">
                    <div className="panel-head">
                      <div>
                        <div className="head-inline">
                          <h2>Accuracy–Carbon Pareto &amp; Trajectories</h2>
                          <span className="plate-tag num">FIG. 1.1–1.5</span>
                        </div>
                        <p className="panel-sub">
                          {r
                            ? `${r.rounds} rounds · ${r.repeats} trials · ${(Number(r.target) * 100).toFixed(1)}% target`
                            : 'Loading…'}
                        </p>
                      </div>
                      <div className="chart-tabs" role="tablist" aria-label="Chart selector">
                        {(
                          [
                            ['tradeoff', 'Pareto'],
                            ['accuracy', 'Accuracy'],
                            ['latency', 'Latency'],
                            ['carbon', 'Carbon'],
                            ['png', 'PNGs'],
                          ] as const
                        ).map(([id, label]) => (
                          <button
                            key={id}
                            type="button"
                            role="tab"
                            aria-selected={chartTab === id}
                            className={`ctab ${chartTab === id ? 'active' : ''}`}
                            onClick={() => setChartTab(id)}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                    </div>

                    {chartTab !== 'png' && (
                      <div className="policy-pills" aria-label="Toggle policy visibility">
                        {Object.entries(POLICY_META).map(([k, m]) => (
                          <button
                            key={k}
                            type="button"
                            className={`ppill ${hiddenPolicies.has(k) ? 'muted' : ''}`}
                            onClick={() => togglePolicy(k)}
                          >
                            <i style={{ background: m.color }} />
                            <span>{m.label}</span>
                          </button>
                        ))}
                      </div>
                    )}

                    {chartTab === 'png' && results ? (
                      <div className="png-stage">
                        <div className="png-toolbar">
                          <label htmlFor="png-select">Figure:</label>
                          <select
                            id="png-select"
                            aria-label="Matplotlib figure"
                            className="pipe-select"
                            value={pngFile}
                            onChange={(e) => setPngFile(e.target.value)}
                          >
                            {PNG_FIGURES.map((f) => (
                              <option key={f.value} value={f.value}>
                                {f.label}
                              </option>
                            ))}
                          </select>
                        </div>
                        <img src={`${results.artifact_prefix}${pngFile}`} alt="Selected experiment chart" />
                      </div>
                    ) : (
                      <div className="svg-stage">
                        {results && chartTab === 'tradeoff' && (
                          <ParetoChart summary={results.summary} hiddenPolicies={hiddenPolicies} />
                        )}
                        {results && chartTab === 'accuracy' && (
                          <TrajectoryChart
                            rows={results.rows}
                            metricKey="accuracy"
                            targetY={Number(results.parameters.run.target)}
                            hiddenPolicies={hiddenPolicies}
                          />
                        )}
                        {results && chartTab === 'latency' && (
                          <TrajectoryChart
                            rows={results.rows}
                            metricKey="cum_latency"
                            hiddenPolicies={hiddenPolicies}
                          />
                        )}
                        {results && chartTab === 'carbon' && (
                          <TrajectoryChart
                            rows={results.rows}
                            metricKey="cum_co2"
                            hiddenPolicies={hiddenPolicies}
                          />
                        )}
                      </div>
                    )}
                  </div>

                  <div className="panel">
                    <div className="panel-head">
                      <div>
                        <div className="head-inline">
                          <h2>4-Policy Comparison Matrix</h2>
                          <span className="plate-tag num">TABLE 1.1</span>
                        </div>
                        <p className="panel-sub">
                          Mean over {r?.repeats ?? 5} trials vs. full-participation baseline (FedAvg · All 4).
                        </p>
                      </div>
                    </div>
                    <div className="scroll">
                      <table className="compact-matrix-table" aria-label="Policy comparison">
                        <thead>
                          <tr>
                            <th>Policy</th>
                            <th>Acc</th>
                            <th>Latency</th>
                            <th>CO₂ (Δ All-4)</th>
                            <th>Hit</th>
                          </tr>
                        </thead>
                        <tbody>
                          {results?.summary.map((s) => {
                            const m = POLICY_META[s.policy] ?? {
                              label: s.policy,
                              short: s.policy,
                              strategy: '',
                              color: '#121316',
                            };
                            const isRl = s.policy === 'Q_learning_RL';
                            const baseCo2 = Number(kpis?.all4?.mean_total_CO2_g_MODELED || s.mean_total_CO2_g_MODELED);
                            const thisCo2 = Number(s.mean_total_CO2_g_MODELED);
                            const deltaPct = ((thisCo2 - baseCo2) / Math.max(1e-6, baseCo2)) * 100;
                            const accPct = Number(s.final_accuracy_mean) * 100;
                            return (
                              <tr key={s.policy} className={isRl ? 'rl-row' : ''}>
                                <td>
                                  <div className="policy-cell-main">
                                    <span className="policy-dot" style={{ background: m.color }} />
                                    <div>
                                      <strong>{m.label}</strong>
                                      <span className="policy-strategy">{m.short}</span>
                                    </div>
                                  </div>
                                </td>
                                <td className="num">
                                  <strong>{accPct.toFixed(1)}%</strong>
                                </td>
                                <td className="num">{Number(s.mean_total_seconds_MODELED).toFixed(1)}s</td>
                                <td className="num">
                                  <strong>{thisCo2.toFixed(2)}g</strong>{' '}
                                  <span className={deltaPct < -0.5 ? 'stamp-measured' : 'stamp-modeled'}>
                                    {s.policy === 'FedAvg_all4'
                                      ? 'BASE'
                                      : `${deltaPct > 0 ? '+' : ''}${deltaPct.toFixed(0)}%`}
                                  </span>
                                </td>
                                <td className="num">{(Number(s.target_hit_rate) * 100).toFixed(0)}%</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </motion.section>
            )}

            {view === 'pipeline' && (
              <motion.section
                key="pipeline"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
              >
                <PipelineStudio
                  data={pipeline}
                  offline={offline}
                  onRun={async (cfg: PipelineConfig) => {
                    const res = await api<PipelineResults>('/api/pipeline/run', cfg);
                    setPipeline(res);
                  }}
                />
              </motion.section>
            )}

            {view === 'replay' && (
              <motion.section
                key="replay"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
              >
                <TopologyReplay results={results} />
              </motion.section>
            )}

            {view === 'specs' && (
              <motion.section
                key="specs"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
              >
                {r && t && q && infra && (
                  <>
                    <div className="spec-grid">
                      <article className="spec-card">
                        <div className="head-inline">
                          <h3>Run Settings</h3>
                          <span className="plate-tag num">EXECUTION</span>
                        </div>
                        <dl>
                          <div><dt>Federated rounds</dt><dd className="num">{r.rounds}</dd></div>
                          <div><dt>RL training episodes</dt><dd className="num">{r.episodes}</dd></div>
                          <div><dt>Performance repeats</dt><dd className="num">{r.repeats}</dd></div>
                          <div><dt>Sensitivity samples</dt><dd className="num">{r.sensitivity}</dd></div>
                          <div><dt>Comparison target</dt><dd className="num">{(Number(r.target) * 100).toFixed(1)}%</dd></div>
                          <div><dt>Master random seed</dt><dd className="num">{r.seed}</dd></div>
                        </dl>
                      </article>

                      <article className="spec-card">
                        <div className="head-inline">
                          <h3>Model &amp; Data</h3>
                          <span className="plate-tag num">FEDAVG</span>
                        </div>
                        <dl>
                          <div><dt>Classifier architecture</dt><dd>{t.model}</dd></div>
                          <div><dt>Clients / classes</dt><dd className="num">{t.clients} / {t.classes}</dd></div>
                          <div><dt>Train / test samples</dt><dd className="num">{t.training_samples} / {t.test_samples}</dd></div>
                          <div><dt>Local epochs / LR (η)</dt><dd className="num">{t.local_epochs} / {t.learning_rate}</dd></div>
                          <div><dt>Features + bias</dt><dd className="num">{t.features} + {t.bias ? 'yes' : 'no'}</dd></div>
                          <div><dt>Partition distribution</dt><dd>Synthetic non-IID</dd></div>
                        </dl>
                      </article>

                      <article className="spec-card">
                        <div className="head-inline">
                          <h3>Tabular Q-Learning</h3>
                          <span className="plate-tag num">RL AGENT</span>
                        </div>
                        <dl>
                          <div><dt>Alpha (α) / Gamma (γ)</dt><dd className="num">{q.alpha} / {q.gamma}</dd></div>
                          <div><dt>Epsilon start / floor</dt><dd className="num">{q.epsilon_start} / {q.epsilon_min}</dd></div>
                          <div><dt>Action space / subset size</dt><dd className="num">{q.actions} actions / 2–4</dd></div>
                          <div><dt>Training phases</dt><dd className="num">{q.training_phases}</dd></div>
                          <div><dt>Accuracy bucket edges</dt><dd className="num">{q.accuracy_bucket_edges.join(', ')}</dd></div>
                          <div><dt>Reward accuracy floor</dt><dd className="num">{(q.reward_accuracy_floor * 100).toFixed(1)}%</dd></div>
                        </dl>
                      </article>

                      <article className="spec-card">
                        <div className="head-inline">
                          <h3>Reward &amp; Cluster</h3>
                          <span className="plate-tag num">INFRA</span>
                        </div>
                        <dl>
                          <div><dt>Accuracy gain weight</dt><dd className="num">{q.accuracy_gain_weight}</dd></div>
                          <div><dt>Latency / carbon weight</dt><dd className="num">{q.latency_weight} / {q.carbon_weight}</dd></div>
                          <div><dt>Shortfall penalty</dt><dd className="num">{q.accuracy_shortfall_weight}</dd></div>
                          <div><dt>Base work / agg time</dt><dd className="num">{infra.settings.synthetic_pc1_train_seconds_per_round} s / {infra.settings.aggregation_seconds_per_round} s</dd></div>
                          <div><dt>Coordinator power draw</dt><dd className="num">{infra.settings.coordinator_power_w} W</dd></div>
                          <div><dt>Default grid factor</dt><dd className="num">{infra.grid_kg_co2_per_kwh} kg CO₂/kWh</dd></div>
                        </dl>
                      </article>
                    </div>

                    <div className="panel">
                      <div className="panel-head">
                        <div>
                          <div className="head-inline">
                            <h2>Client Hardware &amp; Grid Carbon Intensity Profiles</h2>
                            <span className="plate-tag num">PC1 PHYSICAL ANCHOR</span>
                          </div>
                          <p className="panel-sub">
                            Reference meter reading on PC1: {infra.reference_energy_kwh} kWh over {infra.reference_duration_min} min (
                            {infra.reference_mean_power_w.toFixed(2)} W mean power).
                          </p>
                        </div>
                        <a className="btn-pill" href={`${results.artifact_prefix}run_parameters.json`} download>
                          <Download size={13} />
                          Download run_parameters.json
                        </a>
                      </div>
                      <div className="scroll">
                        <table>
                          <thead>
                            <tr>
                              <th>Client</th>
                              <th>RAM / VRAM</th>
                              <th>Speed Ratio</th>
                              <th>Active Power</th>
                              <th>Idle Power</th>
                              <th>Network Delay</th>
                              <th>Grid Carbon Intensity (CI)</th>
                              <th>Provenance Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            {infra.clients.map((n) => {
                              const nodeCi = CALIBRATED_NODE_CI[n.name] ?? n.grid_kgco2_per_kwh ?? infra.grid_kg_co2_per_kwh;
                              return (
                                <tr key={n.name} className={n.name === 'PC1' ? 'rl-row' : ''}>
                                  <td className="num"><strong>{n.name}</strong></td>
                                  <td className="num">{n.ram_gb} GB / {n.gpu_vram_gb} GB</td>
                                  <td className="num">{n.relative_speed.toFixed(2)}×</td>
                                  <td className="num">{(infra.reference_mean_power_w * n.active_power_ratio_to_PC1).toFixed(2)} W</td>
                                  <td className="num">{n.idle_power_w_assumed} W</td>
                                  <td className="num">{n.network_seconds_assumed} s</td>
                                  <td className="num">{Number(nodeCi).toFixed(3)} kg CO₂/kWh</td>
                                  <td>
                                    <span className={n.name === 'PC1' ? 'stamp-measured' : 'stamp-modeled'}>
                                      {n.name === 'PC1' ? 'MEASURED WALL METER' : 'MODELED SCENARIO'}
                                    </span>
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    <div className="spec-two-col">
                      <div className="panel">
                        <div className="panel-head">
                          <div>
                            <div className="head-inline">
                              <h2>Scenario Sensitivity (CO₂ g)</h2>
                              <span className="plate-tag num">±35% POWER · ±30% SPEED</span>
                            </div>
                            <p className="panel-sub">
                              Monte Carlo quantiles across {r.sensitivity} perturbed hardware scenarios for PC2–PC4.
                            </p>
                          </div>
                        </div>
                        <div className="scroll">
                          <table>
                            <thead>
                              <tr>
                                <th>Policy</th>
                                <th>5th Pct</th>
                                <th>Median (50th)</th>
                                <th>95th Pct</th>
                                <th>Interval Spread</th>
                              </tr>
                            </thead>
                            <tbody>
                              {results.sensitivity.map((s) => {
                                const p05 = Number(s.co2_g_p05_ASSUMED);
                                const med = Number(s.co2_g_median_ASSUMED);
                                const p95 = Number(s.co2_g_p95_ASSUMED);
                                const leftPct = Math.max(0, Math.min(85, (p05 / 11.0) * 100));
                                const widthPct = Math.max(4, Math.min(100 - leftPct, ((p95 - p05) / 11.0) * 100));
                                return (
                                  <tr key={s.policy} className={s.policy === 'Q_learning_RL' ? 'rl-row' : ''}>
                                    <td><strong>{POLICY_META[s.policy]?.label ?? s.policy}</strong></td>
                                    <td className="num">{p05.toFixed(3)} g</td>
                                    <td className="num"><strong>{med.toFixed(3)} g</strong></td>
                                    <td className="num">{p95.toFixed(3)} g</td>
                                    <td>
                                      <div className="ci-bar-track" title={`90% interval: ${p05.toFixed(2)}g – ${p95.toFixed(2)}g`}>
                                        <div
                                          className="ci-bar-fill"
                                          style={{
                                            left: `${leftPct}%`,
                                            width: `${widthPct}%`,
                                            background: POLICY_META[s.policy]?.color ?? '#121316',
                                          }}
                                        />
                                      </div>
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </div>

                      <div className="panel">
                        <div className="panel-head">
                          <div>
                            <div className="head-inline">
                              <h2>Governing Formulas &amp; Reward</h2>
                              <span className="plate-tag num">EXACT EQUATIONS</span>
                            </div>
                            <p className="panel-sub">
                              Analytical formulations evaluated per synchronous round.
                            </p>
                          </div>
                        </div>
                        <div className="specs-equation-list">
                          <div className="eq-block">
                            <div className="eq-head">
                              <h3>Synchronous Round Latency</h3>
                              <span className="num">EQ. (1)</span>
                            </div>
                            <p className="eq-formula">{infra.latency_formula}</p>
                          </div>
                          <div className="eq-block">
                            <div className="eq-head">
                              <h3>Cluster Active + Idle + Coordinator Energy</h3>
                              <span className="num">EQ. (2)</span>
                            </div>
                            <p className="eq-formula">{infra.energy_formula}</p>
                          </div>
                          <div className="eq-block">
                            <div className="eq-head">
                              <h3>Carbon Emissions &amp; RL Scalar Reward</h3>
                              <span className="num">EQ. (3)</span>
                            </div>
                            <p className="eq-formula">{`${infra.carbon_formula}   ·   R = ${q.reward}`}</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </>
                )}

                <div className="panel">
                  <div className="panel-head">
                    <div>
                      <div className="head-inline">
                        <h2>Python Execution Log (stdout)</h2>
                        <span className="plate-tag num">LIVE STREAM</span>
                      </div>
                      <p className="panel-sub">Real-time output from run_project.py subprocess.</p>
                    </div>
                  </div>
                  <pre ref={logRef} className="log-box">
                    {engine.log.length ? engine.log.join('\n') : 'No simulation has been triggered in this browser session yet. Click "Run Simulation" above to stream live Python execution logs.'}
                  </pre>
                </div>
              </motion.section>
            )}
          </AnimatePresence>

          <div className="theory-box">
            <strong>{currentStep.theoryTitle}</strong>
            <p>{currentStep.theoryBody}</p>
          </div>

          <footer className="footer-note">
            <span>Federated Computing Lab · Synthetic non-IID training data · PC1 physical meter anchor (0.065 kWh / 190 min) · Modeled PC2–PC4 scenarios</span>
            <span className="num">SEED {r?.seed ?? 42}</span>
          </footer>
        </main>
      </div>
    </MotionConfig>
  );
}
