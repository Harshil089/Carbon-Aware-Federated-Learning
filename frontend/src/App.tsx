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
  let data;
  try {
    data = await res.json();
  } catch {
    throw new Error(
      'The simulation server is not running, so live runs are unavailable. Start it with START_DEMO.sh (macOS/Linux) or START_DEMO.bat (Windows), then reload this page.',
    );
  }
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
  const [hoverTip, setHoverTip] = useState<{ text: string; x: number; y: number } | null>(null);

  const resultIdRef = useRef<string>('');
  const runInFlightRef = useRef<boolean>(false);
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    const clampPos = (clientX: number, clientY: number) => {
      const pad = 12;
      const maxW = 320;
      const estH = 88;
      const x = Math.min(Math.max(pad, clientX + 14), window.innerWidth - maxW - pad);
      const y = Math.min(Math.max(pad, clientY + 16), window.innerHeight - estH - pad);
      return { x, y };
    };

    const onMouseOver = (e: MouseEvent) => {
      const target = (e.target as HTMLElement | null)?.closest('[data-tip]') as HTMLElement | null;
      const text = target?.getAttribute('data-tip');
      if (text) {
        const { x, y } = clampPos(e.clientX, e.clientY);
        setHoverTip({ text, x, y });
      }
    };

    const onMouseMove = (e: MouseEvent) => {
      const target = (e.target as HTMLElement | null)?.closest('[data-tip]') as HTMLElement | null;
      const text = target?.getAttribute('data-tip');
      if (text) {
        const { x, y } = clampPos(e.clientX, e.clientY);
        setHoverTip({ text, x, y });
      }
    };

    const onMouseOut = (e: MouseEvent) => {
      const target = (e.target as HTMLElement | null)?.closest('[data-tip]') as HTMLElement | null;
      if (!target) return;
      const related = (e.relatedTarget as HTMLElement | null)?.closest('[data-tip]') as HTMLElement | null;
      if (related !== target) {
        setHoverTip(related?.getAttribute('data-tip') ? {
          text: related.getAttribute('data-tip')!,
          ...clampPos(e.clientX, e.clientY),
        } : null);
      }
    };

    document.addEventListener('mouseover', onMouseOver);
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseout', onMouseOut);
    return () => {
      document.removeEventListener('mouseover', onMouseOver);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseout', onMouseOut);
    };
  }, []);

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
      setEngine((prev) => ({ ...prev, status: 'failed', message: 'Server not running' }));
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
  const hostedLimits = engine.hosted_limits;

  // The hosted server rejects runs above its limits, so keep the form inside them.
  useEffect(() => {
    if (!hostedLimits) return;
    const keys = Object.keys(hostedLimits) as (keyof typeof hostedLimits)[];
    if (!keys.some((k) => settings[k] > (hostedLimits[k] ?? Infinity))) return;
    setSettings((s) => {
      const next = { ...s };
      for (const k of keys) next[k] = Math.min(s[k], hostedLimits[k] ?? s[k]);
      return next;
    });
    setPreset('quick');
  }, [hostedLimits, settings]);

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
    ? 'Saved results'
    : engine.status === 'running'
      ? `Running · ${engine.progress}%`
      : engine.status === 'complete'
        ? 'Run complete'
        : engine.status === 'failed'
          ? engine.message === 'Server not running' ? 'Server offline' : 'Run failed'
          : 'Ready';

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
      label: 'Results',
      desc: 'Compare 4 ways of picking PCs',
      heading: 'Which way of picking PCs works best?',
      plateCode: 'MAIN SIMULATION',
      sub: 'Four rules for choosing which PCs train each round, compared on accuracy, total time, electricity (Wh) and CO₂ (g). All time, energy and CO₂ figures are simulated.',
      theoryTitle: 'Why the rules differ',
      theoryBody:
        'Using all 4 PCs sees the most varied data, but every round waits for the slowest PC (PC1) while the others sit idle and draw power. "Fastest 2" finishes rounds quickly, and "Lowest-power 2" uses the least electricity, but both see less varied data. The RL picker learns, from how training is going, when to use 2, 3 or 4 PCs.',
    },
    {
      id: 'pipeline',
      num: '02',
      label: 'Extended simulator',
      desc: 'Separate model · 7 rules',
      heading: 'Extended simulator: more rules, network and fairness',
      plateCode: 'SEPARATE EXPERIMENT',
      sub: 'A second, more detailed simulator (fl_pipeline/). It adds network transfer times, 7 picking rules and a fairness score. Its numbers are not comparable with the Results page.',
      theoryTitle: 'How the extended simulator works',
      theoryBody:
        'It splits synthetic data unevenly across 4 PCs, sets each PC’s hardware from a generated 10,000-row profile (based on the meter readings), adds download/upload time and per-PC grid carbon, then compares 7 picking rules. It also tries RL agents that weigh accuracy, time, CO₂ and fairness differently, and shows which settings give the best trade-offs.',
    },
    {
      id: 'replay',
      num: '03',
      label: 'Round replay',
      desc: 'Which PCs trained each round',
      heading: 'Replay a training run, round by round',
      plateCode: 'MAIN SIMULATION',
      sub: 'Step through the rounds of one run to see which PCs were picked, which waited, and how much time and energy each round used.',
      theoryTitle: 'What happens in one round',
      theoryBody:
        'Picked PCs train at the same time, then send their model updates to the coordinator, which combines them (0.45 s). A round lasts as long as the slowest picked PC. PCs that were not picked, or finished early, still draw idle power (7–13 W) until the round ends, and the coordinator draws 30 W the whole time.',
    },
    {
      id: 'specs',
      num: '04',
      label: 'Settings & sources',
      desc: 'What is measured vs. estimated',
      heading: 'Settings, data sources and uncertainty',
      plateCode: 'MAIN SIMULATION',
      sub: 'Exact settings behind the results, which hardware numbers are measured and which are estimated, how much the CO₂ result moves if estimates are off, and the raw run log.',
      theoryTitle: 'Measured vs. estimated',
      theoryBody:
        'Active power for PC1 (0.065 kWh over 190 min = 20.53 W), PC2 and PC3 comes from wall-plug meter readings; PC4 is estimated. Speeds, idle power and network delays are estimates for all PCs. The uncertainty table shows how CO₂ changes if PC2–PC4 power is off by up to ±35% and speed by up to ±30%.',
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
        <aside className="nav-rail" aria-label="Console sections">
          <div
            className="brand"
            data-tip="Federated Computing Lab: a simulation of choosing which computers join each round of federated training, to save time and CO₂."
          >
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
              <p>Experiment console</p>
            </div>
          </div>

          <nav className="nav-steps" role="tablist" aria-label="Console sections">
            {navItems.map((item) => {
              const active = view === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  className={`nav-step ${active ? 'active' : ''}`}
                  data-tip={`${item.label}: ${item.desc}.`}
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
            <div
              className="rp-row"
              data-tip="PC1 used 0.065 kWh over 190 minutes on a wall-plug meter (20.53 W average). Other PCs’ power is expressed relative to this."
            >
              <span>PC1 meter reading</span>
              <strong className="num">20.53 W (0.065 kWh)</strong>
            </div>
            <div
              className="rp-row"
              data-tip="PC2 and PC3 power comes from wall-plug meter readings on a CIFAR-10 run. PC4 is an estimated profile. All are expressed relative to PC1."
            >
              <span>PC2–PC4</span>
              <strong>PC2–3 metered · PC4 estimated</strong>
            </div>
            <div
              className="rp-row"
              data-tip="How much CO₂ the electricity grid emits per kWh (Indian grid average). Electricity used × this factor = CO₂."
            >
              <span>Grid CO₂ per kWh</span>
              <strong className="num">0.675 kg CO₂/kWh</strong>
            </div>
          </div>

          <div className="rail-foot">
            <button
              type="button"
              className="btn-tune"
              data-tip="Back to the project overview page. Loaded results are kept."
              onClick={() => navigate('landing')}
            >
              ← Project overview
            </button>
            <span
              className={`status-pill ${badgeClass}`}
              data-tip="Whether a simulation is running, finished, failed, or whether you are viewing saved results without the simulation server."
            >
              <motion.span
                className="status-dot"
                animate={engine.status === 'running' ? { scale: [1, 1.35, 1], opacity: [1, 0.5, 1] } : {}}
                transition={{ repeat: Infinity, duration: 1.2 }}
              />
              <span>{badgeText}</span>
            </span>
            <a
              className="btn-pill"
              data-tip="Download the results shown here: tables (CSV), settings (JSON) and charts."
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
                data-tip="Go to the previous section."
                onClick={() => navigate(navItems[Math.max(0, activeIdx - 1)].id)}
              >
                ← Previous
              </button>
              <button
                type="button"
                className="btn-tune"
                disabled={activeIdx >= navItems.length - 1}
                data-tip="Go to the next section."
                onClick={() => navigate(navItems[Math.min(navItems.length - 1, activeIdx + 1)].id)}
              >
                Next →
              </button>
            </div>
          </div>

          {offline && (
            <div className="offline-banner">
              Offline results archive loaded. Launch <b>START_DEMO.sh</b> (macOS/Linux) or <b>START_DEMO.bat</b> (Windows) to execute new Python simulations.
            </div>
          )}

          {view === 'overview' && (
            <form className="deck" onSubmit={handleRun}>
              <div className="step-kicker num">OPTIONAL · RUN YOUR OWN SIMULATION (RESULTS BELOW ARE ALREADY LOADED)</div>
              <div className="deck-main">
                <div className="deck-left">
                  <div className="seg" role="group" aria-label="Simulation presets">
                    <button
                      type="button"
                      className={preset === 'quick' ? 'active' : ''}
                      disabled={running || offline}
                      data-tip="Smaller settings that finish in seconds. Good for a first try."
                      onClick={() => applyPreset('quick')}
                    >
                      Quick demo
                    </button>
                    <button
                      type="button"
                      className={preset === 'full' ? 'active' : ''}
                      disabled={running || offline || !!hostedLimits}
                      data-tip={
                        hostedLimits
                          ? 'Full-size runs take too long for the hosted demo. Run the project locally to use them; the published results below already use these settings.'
                          : 'The settings used for the published results (20 rounds, 130 RL training episodes).'
                      }
                      onClick={() => applyPreset('full')}
                    >
                      {hostedLimits ? 'Full (local only)' : 'Full experiment'}
                    </button>
                  </div>

                  <div className="inline-params">
                    {(
                      [
                        ['rounds', 'Rounds', 1, 100, 1, 'How many training rounds each run has. In each round, the picked PCs train and the coordinator combines their updates.'],
                        ['episodes', 'RL practice runs', 1, 500, 1, 'How many practice runs the RL picker gets to learn which PCs to choose. More runs, better choices, slower.'],
                        ['target', 'Target accuracy %', 0.01, 100, 'any', 'Accuracy goal. Used to report how often, and how fast, each rule reaches it.'],
                      ] as const
                    ).map(([key, label, min, max, step, tip]) => (
                      <div className="chip-field" key={key} data-tip={tip}>
                        <label htmlFor={key}>{label}</label>
                        <input
                          id={key}
                          type="number"
                          min={min}
                          max={hostedLimits?.[key as keyof typeof hostedLimits] ?? max}
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
                  <button
                    type="button"
                    className="btn-tune"
                    data-tip="Show advanced settings: repeats, uncertainty samples, training settings, random seed and how much the RL picker cares about time vs. CO₂."
                    onClick={() => setDrawerOpen((o) => !o)}
                  >
                    <SlidersHorizontal size={13} />
                    {drawerOpen ? 'Hide Options' : 'More Options'}
                  </button>
                  <motion.button
                    type="submit"
                    className="btn-run"
                    disabled={running || offline}
                    whileTap={{ scale: 0.98 }}
                    data-tip="Run the simulation with these settings. Needs the simulation server (START_DEMO) running."
                  >
                    <Play size={13} />
                    {running ? 'Running…' : offline ? 'Offline' : 'Run Simulation'}
                  </motion.button>
                  <button
                    type="button"
                    className="btn-stop"
                    disabled={!running || offline}
                    data-tip="Stop the running simulation."
                    onClick={handleStop}
                  >
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
                        ['repeats', 'Repeat runs', 1, 20, 1, 'How many times each rule is run with random timing variation. Results are averaged.'],
                        ['sensitivity', 'Uncertainty samples', 1, 100, 1, 'How many what-if variations of PC power and speed to test. Shown in the uncertainty table under Settings & sources.'],
                        ['local_epochs', 'Local epochs', 1, 20, 1, 'How many passes over its own data each picked PC makes per round. More passes take more time and energy.'],
                        ['learning_rate', 'Learning rate (η)', 0.001, 2, 0.001, 'How big each training step is. Too high is unstable, too low is slow.'],
                        ['seed', 'Random seed', 0, 2147483647, 1, 'Same seed and settings give exactly the same results.'],
                        ['latency_weight', 'Time penalty', 0, 100, 0.001, 'How much the RL picker is punished for slow rounds. Higher means it favours faster PCs.'],
                        ['carbon_weight', 'CO₂ penalty', 0, 100, 0.01, 'How much the RL picker is punished for CO₂. Higher means it favours lower-power PCs.'],
                      ] as const
                    ).map(([key, label, min, max, step, tip]) => (
                      <div className="drawer-field" key={key} data-tip={tip}>
                        <label htmlFor={key}>{label}</label>
                        <input
                          id={key}
                          type="number"
                          min={min}
                          max={hostedLimits?.[key as keyof typeof hostedLimits] ?? max}
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

              <div
                className="deck-progress"
                data-tip="Progress of the running simulation."
              >
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
                <div className="step-kicker num">KEY RESULTS (SIMULATED)</div>
                <div className="editorial-strip" aria-label="Key results">
                  <div
                    className="rib-cell lead"
                    data-tip="Final accuracy of the RL picker, and how much CO₂ it saved compared with using all 4 PCs every round."
                  >
                    <span className="rib-label">RL picker</span>
                    <b className="rib-val num">
                      {kpis?.rl ? `${(Number(kpis.rl.final_accuracy_mean) * 100).toFixed(1)}%` : '—'}
                    </b>
                    <span className="rib-sub num">
                      {kpis?.rlCo2Save ? `−${kpis.rlCo2Save}% CO₂ vs. all 4 PCs` : 'Picks 2–4 PCs each round'}
                    </span>
                  </div>
                  <div
                    className="rib-cell"
                    data-tip="The best final accuracy of the 4 rules, averaged over repeat runs. Differences under ~1 point are small."
                  >
                    <span className="rib-label">Best accuracy</span>
                    <b className="rib-val num">
                      {kpis ? `${(Number(kpis.bestAcc.final_accuracy_mean) * 100).toFixed(1)}%` : '—'}
                    </b>
                    <span className="rib-sub">
                      {kpis ? POLICY_META[kpis.bestAcc.policy]?.label ?? kpis.bestAcc.policy : 'Loading…'}
                    </span>
                  </div>
                  <div
                    className="rib-cell"
                    data-tip="The lowest total CO₂ of the 4 rules over a whole run (simulated)."
                  >
                    <span className="rib-label">Lowest CO₂</span>
                    <b className="rib-val num">
                      {kpis ? `${Number(kpis.lowCo2.mean_total_CO2_g_MODELED).toFixed(2)} g` : '—'}
                    </b>
                    <span className="rib-sub">
                      {kpis ? POLICY_META[kpis.lowCo2.policy]?.label ?? kpis.lowCo2.policy : 'Modeled CO₂'}
                    </span>
                  </div>
                  <div
                    className="rib-cell"
                    data-tip="The shortest total training time of the 4 rules (simulated). Below it: how long your computer took to run the simulation itself."
                  >
                    <span className="rib-label">Shortest training time</span>
                    <b className="rib-val num">
                      {kpis ? `${Number(kpis.lowLat.mean_total_seconds_MODELED).toFixed(1)} s` : '—'}
                    </b>
                    <span className="rib-sub num">
                      {results ? `Simulation took ${Number(results.parameters.execution_wall_seconds).toFixed(2)} s` : 'Loading…'}
                    </span>
                  </div>
                </div>

                {/* Split Empirical Plate: Side-by-side Pareto Chart + 4-Policy Comparison Matrix */}
                <div className="overview-split-grid">
                  <div className="panel">
                    <div className="panel-head">
                      <div>
                        <div className="head-inline">
                          <h2>Accuracy vs. CO₂, and progress per round</h2>
                          <span className="plate-tag num">CHARTS</span>
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
                            ['tradeoff', 'Trade-off', 'Each rule as one dot: accuracy vs. total CO₂. The dashed line joins rules that no other rule beats on both.'],
                            ['accuracy', 'Accuracy', 'Accuracy after each round, for each rule.'],
                            ['latency', 'Latency', 'Total training time so far, after each round.'],
                            ['carbon', 'Carbon', 'Total CO₂ so far, after each round.'],
                            ['png', 'Images', 'The same charts as saved image files.'],
                          ] as const
                        ).map(([id, label, tip]) => (
                          <button
                            key={id}
                            type="button"
                            role="tab"
                            aria-selected={chartTab === id}
                            className={`ctab ${chartTab === id ? 'active' : ''}`}
                            data-tip={tip}
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
                            data-tip={`Show or hide ${m.label} (${m.strategy.toLowerCase()}).`}
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
                          <h2>The 4 rules side by side</h2>
                          <span className="plate-tag num">AVERAGE OF REPEAT RUNS</span>
                        </div>
                        <p className="panel-sub">
                          Average of {r?.repeats ?? 5} repeat runs. CO₂ change is compared with using all 4 PCs.
                        </p>
                      </div>
                    </div>
                    <div className="scroll">
                      <table className="compact-matrix-table" aria-label="Policy comparison">
                        <thead>
                          <tr>
                            <th data-tip="The rule for choosing which PCs train each round.">Rule</th>
                            <th data-tip="Final accuracy on test data the model never trained on.">Accuracy</th>
                            <th data-tip="Total training time for the whole run, in seconds (simulated).">Time</th>
                            <th data-tip="Total CO₂ for the whole run in grams (simulated), and the change compared with using all 4 PCs.">CO₂ (vs. all 4)</th>
                            <th data-tip="Share of repeat runs that reached the target accuracy.">Reached target</th>
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
                              <tr
                                key={s.policy}
                                className={isRl ? 'rl-row' : ''}
                                data-tip={`${m.label} (${m.strategy || m.short}): ${accPct.toFixed(1)}% accuracy, ${thisCo2.toFixed(2)} g CO₂, averaged over repeat runs.`}
                              >
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
                      <article
                        className="spec-card"
                        data-tip="Settings used to produce the results on this page."
                      >
                        <div className="head-inline">
                          <h3>Run settings</h3>
                          <span className="plate-tag num">RUN</span>
                        </div>
                        <dl>
                          <div><dt>Training rounds</dt><dd className="num">{r.rounds}</dd></div>
                          <div><dt>RL practice runs</dt><dd className="num">{r.episodes}</dd></div>
                          <div><dt>Repeat runs</dt><dd className="num">{r.repeats}</dd></div>
                          <div><dt>Uncertainty samples</dt><dd className="num">{r.sensitivity}</dd></div>
                          <div><dt>Target accuracy</dt><dd className="num">{(Number(r.target) * 100).toFixed(1)}%</dd></div>
                          <div><dt>Random seed</dt><dd className="num">{r.seed}</dd></div>
                        </dl>
                      </article>

                      <article
                        className="spec-card"
                        data-tip="Summarizes the FedAvg classifier architecture and synthetic non-IID dataset split. Read from results.parameters.training."
                      >
                        <div className="head-inline">
                          <h3>Model and data</h3>
                          <span className="plate-tag num">SYNTHETIC DATA</span>
                        </div>
                        <dl>
                          <div><dt>Model</dt><dd>{t.model}</dd></div>
                          <div><dt>PCs / classes</dt><dd className="num">{t.clients} / {t.classes}</dd></div>
                          <div><dt>Train / test samples</dt><dd className="num">{t.training_samples} / {t.test_samples}</dd></div>
                          <div><dt>Local epochs / LR (η)</dt><dd className="num">{t.local_epochs} / {t.learning_rate}</dd></div>
                          <div><dt>Features + bias</dt><dd className="num">{t.features} + {t.bias ? 'yes' : 'no'}</dd></div>
                          <div><dt>Data split</dt><dd>Synthetic, different mix per PC</dd></div>
                        </dl>
                      </article>

                      <article
                        className="spec-card"
                        data-tip="Summarizes the FedAvg classifier architecture and synthetic non-IID dataset split. Read from results.parameters.training."
                      >
                        <div className="head-inline">
                          <h3>RL picker (Q-learning)</h3>
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

                      <article
                        className="spec-card"
                        data-tip="Summarizes the FedAvg classifier architecture and synthetic non-IID dataset split. Read from results.parameters.training."
                      >
                        <div className="head-inline">
                          <h3>RL goals and hardware model</h3>
                          <span className="plate-tag num">GOALS</span>
                        </div>
                        <dl>
                          <div><dt>Reward for accuracy gain</dt><dd className="num">{q.accuracy_gain_weight}</dd></div>
                          <div><dt>Time / CO₂ penalty</dt><dd className="num">{q.latency_weight} / {q.carbon_weight}</dd></div>
                          <div><dt>Penalty below accuracy floor</dt><dd className="num">{q.accuracy_shortfall_weight}</dd></div>
                          <div><dt>PC1 training / combine time per round</dt><dd className="num">{infra.settings.synthetic_pc1_train_seconds_per_round} s / {infra.settings.aggregation_seconds_per_round} s</dd></div>
                          <div><dt>Coordinator power draw</dt><dd className="num">{infra.settings.coordinator_power_w} W</dd></div>
                          <div><dt>Default grid CO₂ per kWh</dt><dd className="num">{infra.grid_kg_co2_per_kwh} kg CO₂/kWh</dd></div>
                        </dl>
                      </article>
                    </div>

                    <div className="panel">
                      <div className="panel-head">
                        <div>
                          <div className="head-inline">
                            <h2>The 4 PCs: hardware and power</h2>
                            <span className="plate-tag num">PC1–3 METERED · PC4 ESTIMATED</span>
                          </div>
                          <p className="panel-sub">
                            PC1 meter reading: {infra.reference_energy_kwh} kWh over {infra.reference_duration_min} min (
                            {infra.reference_mean_power_w.toFixed(2)} W average). PC2 and PC3 power comes from their own meter
                            readings on a CIFAR-10 test. Idle power and network delay are estimates for all PCs.
                          </p>
                        </div>
                        <a className="btn-pill" href={`${results.artifact_prefix}run_parameters.json`} download>
                          <Download size={13} />
                          Download settings (JSON)
                        </a>
                      </div>
                      <div className="scroll">
                        <table>
                          <thead>
                            <tr>
                              <th data-tip="One of the 4 computers in the simulation.">PC</th>
                              <th data-tip="Memory and graphics-card memory. Shown for reference; training runs on the CPU.">RAM / GPU memory</th>
                              <th data-tip="How much faster than PC1 this PC trains (PC1 = 1.00×).">Speed vs. PC1</th>
                              <th data-tip="Power drawn while training.">Training power</th>
                              <th data-tip="Estimated power drawn while waiting for the round to end.">Idle power (est.)</th>
                              <th data-tip="Estimated time to send the model update to the coordinator.">Network delay (est.)</th>
                              <th data-tip="CO₂ emitted per kWh of electricity where this PC is.">Grid CO₂ per kWh</th>
                              <th data-tip="Whether this PC's training power was measured with a wall-plug meter or estimated.">Training power source</th>
                            </tr>
                          </thead>
                          <tbody>
                            {infra.clients.map((n) => {
                              const nodeCi = CALIBRATED_NODE_CI[n.name] ?? n.grid_kgco2_per_kwh ?? infra.grid_kg_co2_per_kwh;
                              return (
                                <tr
                                  key={n.name}
                                  className={n.name === 'PC1' ? 'rl-row' : ''}
                                  data-tip={`${n.name}: ${n.relative_speed.toFixed(2)}× speed, ${(infra.reference_mean_power_w * n.active_power_ratio_to_PC1).toFixed(2)} W active, ${Number(nodeCi).toFixed(3)} kg CO₂/kWh. Used to simulate this PC's time and energy.`}
                                >
                                  <td className="num"><strong>{n.name}</strong></td>
                                  <td className="num">{n.ram_gb} GB / {n.gpu_vram_gb} GB</td>
                                  <td className="num">{n.relative_speed.toFixed(2)}×</td>
                                  <td className="num">{(infra.reference_mean_power_w * n.active_power_ratio_to_PC1).toFixed(2)} W</td>
                                  <td className="num">{n.idle_power_w_assumed} W</td>
                                  <td className="num">{n.network_seconds_assumed} s</td>
                                  <td className="num">{Number(nodeCi).toFixed(3)} kg CO₂/kWh</td>
                                  <td>
                                    <span className={n.status?.startsWith('MEASURED') ? 'stamp-measured' : 'stamp-modeled'}>
                                      {n.status?.startsWith('MEASURED') ? 'METERED' : 'ESTIMATED'}
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
                              <h2>How sure is the CO₂ result?</h2>
                              <span className="plate-tag num">±35% POWER · ±30% SPEED</span>
                            </div>
                            <p className="panel-sub">
                              Total CO₂ (g) when PC2–PC4 power and speed are varied at random, over {r.sensitivity} what-if runs. 90% of runs fall between the low and high values.
                            </p>
                          </div>
                        </div>
                        <div className="scroll">
                          <table data-tip="If our power and speed estimates are off, CO₂ could land anywhere in this range.">
                            <thead>
                              <tr>
                                <th>Rule</th>
                                <th>Low (5%)</th>
                                <th>Typical</th>
                                <th>High (95%)</th>
                                <th>Range</th>
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
                              <h2>How time, energy and CO₂ are calculated</h2>
                              <span className="plate-tag num">FORMULAS</span>
                            </div>
                            <p className="panel-sub">
                              Applied to every round of the simulation.
                            </p>
                          </div>
                        </div>
                        <div className="specs-equation-list">
                          <div className="eq-block">
                            <div className="eq-head">
                              <h3>Round time</h3>
                              <span className="num">EQ. (1)</span>
                            </div>
                            <p className="eq-formula">{infra.latency_formula}</p>
                          </div>
                          <div className="eq-block">
                            <div className="eq-head">
                              <h3>Round energy: training + idle + coordinator</h3>
                              <span className="num">EQ. (2)</span>
                            </div>
                            <p className="eq-formula">{infra.energy_formula}</p>
                          </div>
                          <div className="eq-block">
                            <div className="eq-head">
                              <h3>CO₂, and the RL picker’s score</h3>
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
                        <h2>Run log</h2>
                        <span className="plate-tag num">RAW OUTPUT</span>
                      </div>
                      <p className="panel-sub">Text output from the simulation (run_project.py), for checking details.</p>
                    </div>
                  </div>
                  <pre
                    ref={logRef}
                    className="log-box"
                    data-tip="Raw output from the simulation."
                  >
                    {engine.log.length ? engine.log.join('\n') : 'No simulation run yet in this session. The results shown are from a saved run; click "Run Simulation" on the Results page to start a new one.'}
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
            <span>Federated Computing Lab · Synthetic training data · Time, energy and CO₂ are simulated · Training power metered on PC1–PC3, estimated on PC4</span>
            <span className="num">SEED {r?.seed ?? 42}</span>
          </footer>
        </main>
      </div>
      {hoverTip && (
        <div
          className="hover-tip-card"
          role="tooltip"
          style={{ left: hoverTip.x, top: hoverTip.y }}
        >
          {hoverTip.text}
        </div>
      )}
    </MotionConfig>
  );
}
