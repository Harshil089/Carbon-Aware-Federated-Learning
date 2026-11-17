import { useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, ArrowUpRight, Play } from 'lucide-react';
import { ViewId } from '../types';

const BASE_EASE: [number, number, number, number] = [0.22, 1, 0.36, 1];

type PolicyDemoId = 'rl' | 'all4' | 'fast2' | 'low2';

const DEMO_POLICIES: Record<
  PolicyDemoId,
  {
    name: string;
    subset: string;
    roundSeconds: number;
    roundWh: number;
    roundCo2: number;
    accPct: number;
    co2Delta: string;
    note: string;
    activeNodes: Record<string, { workS: number; netS: number }>;
  }
> = {
  rl: {
    name: 'Q-Learning RL (Dynamic Subset)',
    subset: 'PC2 + PC4 (Phase 2 Equilibrium)',
    roundSeconds: 15.45,
    roundWh: 0.422,
    roundCo2: 0.285,
    accPct: 82.4,
    co2Delta: '−26.9% CO₂ vs. All-4',
    note: 'Avoids PC1 straggler delay and PC3 50.3W peak draw while maintaining class coverage.',
    activeNodes: {
      PC2: { workS: 14.55, netS: 0.45 },
      PC4: { workS: 11.85, netS: 0.45 },
    },
  },
  all4: {
    name: 'FedAvg · All 4 (Naive Baseline)',
    subset: 'PC1 + PC2 + PC3 + PC4',
    roundSeconds: 16.75,
    roundWh: 0.578,
    roundCo2: 0.390,
    accPct: 83.2,
    co2Delta: 'Baseline Reference (100%)',
    note: 'PC1 (16.3s) bottlenecks the entire cluster; PC2–PC4 and the 30W coordinator burn idle power waiting.',
    activeNodes: {
      PC1: { workS: 16.0, netS: 0.3 },
      PC2: { workS: 14.55, netS: 0.45 },
      PC3: { workS: 8.89, netS: 0.6 },
      PC4: { workS: 11.85, netS: 0.45 },
    },
  },
  fast2: {
    name: 'Fastest 2 (Greedy Latency)',
    subset: 'PC3 + PC4',
    roundSeconds: 12.75,
    roundWh: 0.391,
    roundCo2: 0.264,
    accPct: 78.9,
    co2Delta: '−32.3% CO₂ · Misses 80% Target',
    note: 'Finishes in 12.75s using GPU-accelerated nodes, but starves non-IID classes held on PC1 and PC2.',
    activeNodes: {
      PC3: { workS: 8.89, netS: 0.6 },
      PC4: { workS: 11.85, netS: 0.45 },
    },
  },
  low2: {
    name: 'Lowest Energy 2 (Greedy Active W)',
    subset: 'PC1 + PC2',
    roundSeconds: 16.75,
    roundWh: 0.358,
    roundCo2: 0.242,
    accPct: 77.5,
    co2Delta: '−37.9% CO₂ · Severe Straggler',
    note: 'Minimizes active watts on paper, but PC1 stretches round duration to 16.75s and degrades convergence.',
    activeNodes: {
      PC1: { workS: 16.0, netS: 0.3 },
      PC2: { workS: 14.55, netS: 0.45 },
    },
  },
};

const CLUSTER_NODES = [
  {
    name: 'PC1',
    hw: '8 GB RAM · Intel Core i7-12700',
    speed: '1.00×',
    activeW: 20.53,
    idleW: 7.0,
    netS: 0.3,
    ci: '0.675',
    provenance: 'MEASURED ANCHOR (RE-VERIFY)',
    isMeasured: true,
  },
  {
    name: 'PC2',
    hw: '16 GB RAM · Intel Core Ultra 5 235',
    speed: '1.10×',
    activeW: 86.22,
    idleW: 9.0,
    netS: 0.45,
    ci: '0.675',
    provenance: 'MEASURED (CIFAR-10 · 186.3g CO₂)',
    isMeasured: true,
  },
  {
    name: 'PC3',
    hw: '32 GB · Ultra 7 265K (8GB GPU, CUDA off)',
    speed: '1.54×',
    activeW: 209.4,
    idleW: 13.0,
    netS: 0.6,
    ci: '0.675',
    provenance: 'MEASURED (CIFAR-10 · 324.0g CO₂)',
    isMeasured: true,
  },
  {
    name: 'PC4',
    hw: '16 GB RAM · 4 GB GPU Profile',
    speed: '1.35×',
    activeW: 35.92,
    idleW: 11.0,
    netS: 0.45,
    ci: '0.420',
    provenance: 'MODELED SCENARIO',
    isMeasured: false,
  },
];

export function LandingPage({ onEnter }: { onEnter: (target: ViewId) => void }) {
  const [policyId, setPolicyId] = useState<PolicyDemoId>('rl');
  const activeDemo = DEMO_POLICIES[policyId];
  const maxTimelineS = 17.0;

  const rqs: { id: string; question: string; mechanism: string; finding: string; view: ViewId; plate: string }[] = [
    {
      id: 'RQ1',
      question: 'How do heterogeneous CPU/GPU capabilities govern synchronous round latency and idle energy?',
      mechanism: 'Parallel straggler max duration + idle power tail accounting across PC1–PC4 + 30W coordinator.',
      finding: 'Slow nodes inflate idle energy on fast nodes; active watts alone mispredict total round Wh.',
      view: 'replay',
      plate: 'Plate 03 · Topology',
    },
    {
      id: 'RQ2',
      question: 'How does per-client grid carbon intensity alter optimal client scheduling?',
      mechanism: 'Node-specific grid factors (0.420 to 0.820 kg CO₂/kWh) weighted into per-round carbon cost.',
      finding: 'High-speed high-carbon nodes (PC3) trade off against moderate-speed low-carbon nodes (PC4).',
      view: 'specs',
      plate: 'Plate 04 · Calibration',
    },
    {
      id: 'RQ3',
      question: 'How does subset selection impact convergence under IID vs. Dirichlet(α) non-IID partitions?',
      mechanism: 'Configurable Dirichlet(α) label skew across 4 partitions with sample-weighted FedAvg.',
      finding: 'Static 2-node heuristics stall under skewed α; dynamic subset rotation recovers class diversity.',
      view: 'pipeline',
      plate: 'Plate 02 · Pipeline',
    },
    {
      id: 'RQ4',
      question: 'Can Tabular Q-Learning & MORL outperform static speed, energy, and fairness heuristics?',
      mechanism: 'State-conditioned policy over (training phase, accuracy bucket) across 11 client-subset actions.',
      finding: 'RL adapts client count by training phase, reaching the 80% accuracy target with lower CO₂.',
      view: 'overview',
      plate: 'Plate 01 · Benchmarks',
    },
    {
      id: 'RQ5',
      question: 'What is the empirical 3D Pareto frontier across Accuracy, Latency, and Carbon?',
      mechanism: 'Multi-objective preference scalarization sweep and non-dominated frontier extraction.',
      finding: 'Traces continuous trade-off from latency-priority to carbon-priority with Gini fairness tracking.',
      view: 'pipeline',
      plate: 'Plate 02 · MORL Sweep',
    },
    {
      id: 'RQ6',
      question: 'How can one physically metered PC anchor a reproducible multi-node FL benchmark?',
      mechanism: 'PC1 metered at 0.065 kWh / 190 min (20.53W) with ±35% power and ±30% speed Monte Carlo sensitivity.',
      finding: 'Separates physical measurement from scenario assumptions with full seed & parameter provenance.',
      view: 'specs',
      plate: 'Plate 04 · Provenance',
    },
  ];

  const plates: { id: ViewId; code: string; title: string; summary: string; metrics: string }[] = [
    {
      id: 'overview',
      code: 'PLATE 01',
      title: 'Policy Benchmarks & Pareto Matrix',
      summary: 'Execute custom NumPy FedAvg + Q-Learning simulations, inspect Pareto curves, and compare 4 policies.',
      metrics: '4 Policies · 5 PNG Figures · Interactive SVG',
    },
    {
      id: 'pipeline',
      code: 'PLATE 02',
      title: '8-Layer Modular FL + MORL Studio',
      summary: 'Run fl_pipeline/ live with Dirichlet(α) splits, 10,000-row telemetry calibration, 7 schedulers, and MORL sweep.',
      metrics: 'Layers L1–L8 · 7 Schedulers · Gini Index',
    },
    {
      id: 'replay',
      code: 'PLATE 03',
      title: 'Synchronous Hub-and-Spoke Replay',
      summary: 'Step round-by-round through recorded client selections, active uplinks, and per-node idle power tails.',
      metrics: 'PC1–PC4 + 30W Coordinator · Round Timeline',
    },
    {
      id: 'specs',
      code: 'PLATE 04',
      title: 'Calibration, Sensitivity & Run Logs',
      summary: 'Audit PC1 physical meter provenance, 5th–95th percentile sensitivity bounds, equations, and Python stdout.',
      metrics: '0.065 kWh Anchor · Monte Carlo Quantiles',
    },
  ];

  return (
    <div className="mono-landing">
      <a className="skip-link" href="#mono-main">
        Skip to primary research content
      </a>

      <header className="mono-masthead">
        <div className="mono-masthead-inner">
          <div className="mono-brand-block">
            <span className="mono-mark" aria-hidden="true">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="3" />
                <circle cx="5" cy="5" r="2" />
                <circle cx="19" cy="5" r="2" />
                <circle cx="5" cy="19" r="2" />
                <circle cx="19" cy="19" r="2" />
                <path d="M6.6 6.6L9.8 9.8M17.4 6.6L14.2 9.8M6.6 17.4L9.8 14.2M17.4 17.4L14.2 14.2" />
              </svg>
            </span>
            <span className="mono-brand-name">Federated Computing Lab</span>
            <span className="mono-brand-divider" aria-hidden="true">/</span>
            <span className="mono-brand-edition">Research Monograph &amp; Simulation Workbench</span>
          </div>

          <nav className="mono-top-nav" aria-label="Primary sections">
            <button type="button" className="mono-nav-link" onClick={() => onEnter('overview')}>
              01 Benchmarks
            </button>
            <button type="button" className="mono-nav-link" onClick={() => onEnter('pipeline')}>
              02 8-Layer Pipeline
            </button>
            <button type="button" className="mono-nav-link" onClick={() => onEnter('replay')}>
              03 Topology Replay
            </button>
            <button type="button" className="mono-nav-link" onClick={() => onEnter('specs')}>
              04 Provenance &amp; Specs
            </button>
            <button type="button" className="mono-nav-cta" onClick={() => onEnter('overview')}>
              Enter Console
              <ArrowRight size={14} />
            </button>
          </nav>
        </div>
      </header>

      <main id="mono-main" className="mono-container">
        {/* First Viewport: Split Monograph Thesis + Live Synchronous Round Energy Dissector */}
        <section className="mono-hero-grid" aria-label="Research thesis and interactive round dissector">
          <div className="mono-hero-thesis">
            <motion.h1
              className="mono-h1"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: BASE_EASE }}
            >
              Multi-Objective Reinforcement Learning for Latency–Carbon Optimization in Federated Computing.
            </motion.h1>

            <motion.p
              className="mono-lead"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.05, ease: BASE_EASE }}
            >
              In synchronous Federated Averaging, selecting all nodes forces faster machines and the
              30 W coordinator to sit idle while waiting for the slowest straggler. Our state-conditioned
              RL scheduler dynamically selects 2-to-4 client subsets by training phase and accuracy bucket—cutting
              modeled carbon emissions and wall-clock latency while preserving target convergence.
            </motion.p>

            <motion.div
              className="mono-hero-actions"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.1, ease: BASE_EASE }}
            >
              <button type="button" className="btn-vermilion" onClick={() => onEnter('overview')}>
                <Play size={14} />
                Open Interactive Console
              </button>
              <button type="button" className="btn-architectural" onClick={() => onEnter('pipeline')}>
                Inspect 8-Layer Architecture
                <ArrowUpRight size={14} />
              </button>
            </motion.div>

            <dl className="mono-provenance-ledger" aria-label="Experimental provenance summary">
              <div className="prov-row">
                <dt>Physical Meter Anchor</dt>
                <dd className="num">
                  <strong>PC1 (i7-12700 · 8 GB)</strong> — 0.065 kWh over 190 min (20.53 W mean) ·{' '}
                  <span className="stamp-measured">MEASURED</span>
                </dd>
              </div>
              <div className="prov-row">
                <dt>Calibrated Cluster</dt>
                <dd className="num">
                  <strong>10,000 Telemetry Rows</strong> — PC2–PC4 speed (1.10×–1.80×) &amp; grid CI (0.42–0.82 kg/kWh) ·{' '}
                  <span className="stamp-modeled">MODELED</span>
                </dd>
              </div>
              <div className="prov-row">
                <dt>Decision &amp; Test Suite</dt>
                <dd className="num">
                  <strong>11 Subset Actions · 7 Policies</strong> — Tabular Q-Learning + MORL Pareto Sweep ·{' '}
                  <span className="stamp-verified">26/26 TESTS</span>
                </dd>
              </div>
            </dl>
          </div>

          {/* Right Plate: Interactive Synchronous Round Straggler & Idle Energy Dissector */}
          <motion.div
            className="mono-dissector-plate"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.12, ease: BASE_EASE }}
          >
            <div className="plate-title-bar">
              <h2>Synchronous Round Timing &amp; Idle Power Dissector</h2>
              <span className="plate-code num">FIG. 01 · INTERACTIVE MECHANISM</span>
            </div>

            <div className="dissector-tabs" role="tablist" aria-label="Select scheduling policy">
              {(
                [
                  ['rl', 'Q-Learning RL'],
                  ['all4', 'FedAvg · All 4'],
                  ['fast2', 'Fastest 2'],
                  ['low2', 'Lowest Energy 2'],
                ] as const
              ).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  role="tab"
                  aria-selected={policyId === id}
                  className={`dtab ${policyId === id ? 'active' : ''}`}
                  onClick={() => setPolicyId(id)}
                >
                  {label}
                </button>
              ))}
            </div>

            <div className="dissector-meta-strip">
              <span>
                Scheduled Subset: <strong className="num">{activeDemo.subset}</strong>
              </span>
              <span className="stamp-delta num">{activeDemo.co2Delta}</span>
            </div>

            <div className="gantt-ledger" aria-label="Per-node active compute vs idle waiting duration">
              {CLUSTER_NODES.map((node) => {
                const activeTiming = activeDemo.activeNodes[node.name];
                const isSelected = Boolean(activeTiming);
                const activeDur = activeTiming ? activeTiming.workS + activeTiming.netS : 0;
                const idleDur = Math.max(0, activeDemo.roundSeconds - activeDur);
                const activePct = (activeDur / maxTimelineS) * 100;
                const idlePct = (idleDur / maxTimelineS) * 100;
                const nodeWh = ((node.activeW * activeDur + node.idleW * idleDur) / 3600).toFixed(3);

                return (
                  <div className={`gantt-row ${isSelected ? 'selected' : 'unselected'}`} key={node.name}>
                    <div className="gantt-node-id">
                      <div className="gantt-node-top">
                        <strong className="num">{node.name}</strong>
                        <span className={node.isMeasured ? 'mini-tag measured' : 'mini-tag modeled'}>
                          {node.isMeasured ? 'METER' : 'SCEN'}
                        </span>
                      </div>
                      <span className="gantt-node-sub num">
                        {node.speed} · {node.activeW.toFixed(1)}W / {node.idleW.toFixed(0)}W idle
                      </span>
                    </div>

                    <div className="gantt-track-wrap">
                      <div className="gantt-track">
                        <motion.div
                          className="gantt-bar-active"
                          animate={{ width: `${activePct}%` }}
                          transition={{ duration: 0.35, ease: BASE_EASE }}
                          title={`Active compute + uplink: ${activeDur.toFixed(2)}s at ${node.activeW.toFixed(1)}W`}
                        />
                        <motion.div
                          className="gantt-bar-idle"
                          animate={{ width: `${idlePct}%` }}
                          transition={{ duration: 0.35, ease: BASE_EASE }}
                          title={`Idle waiting until straggler finishes: ${idleDur.toFixed(2)}s at ${node.idleW}W`}
                        />
                      </div>
                      <div className="gantt-timing-caption num">
                        {isSelected
                          ? `${activeDur.toFixed(1)}s active + ${idleDur.toFixed(1)}s idle wait`
                          : `0.0s active + ${idleDur.toFixed(1)}s unselected idle (${node.idleW}W)`}
                      </div>
                    </div>

                    <div className="gantt-wh num">{nodeWh} Wh</div>
                  </div>
                );
              })}

              {/* Coordinator Row */}
              <div className="gantt-row coordinator">
                <div className="gantt-node-id">
                  <div className="gantt-node-top">
                    <strong className="num">COORD</strong>
                    <span className="mini-tag modeled">30W</span>
                  </div>
                  <span className="gantt-node-sub num">+0.45s FedAvg agg</span>
                </div>
                <div className="gantt-track-wrap">
                  <div className="gantt-track">
                    <motion.div
                      className="gantt-bar-coord"
                      animate={{ width: `${(activeDemo.roundSeconds / maxTimelineS) * 100}%` }}
                      transition={{ duration: 0.35, ease: BASE_EASE }}
                    />
                  </div>
                  <div className="gantt-timing-caption num">
                    Powered continuously for entire {activeDemo.roundSeconds.toFixed(2)}s synchronous round
                  </div>
                </div>
                <div className="gantt-wh num">{((30 * activeDemo.roundSeconds) / 3600).toFixed(3)} Wh</div>
              </div>
            </div>

            <div className="gantt-legend">
              <span><i className="leg-swatch active" /> Active Local Training + Uplink</span>
              <span><i className="leg-swatch idle" /> Idle Power Tail (Straggler Wait)</span>
              <span><i className="leg-swatch coord" /> 30 W Coordinator Overhead</span>
            </div>

            <p className="dissector-note">{activeDemo.note}</p>

            <div className="dissector-readout-table">
              <div>
                <span>Round Latency</span>
                <b className="num">{activeDemo.roundSeconds.toFixed(2)} s</b>
              </div>
              <div>
                <span>Total Energy</span>
                <b className="num">{activeDemo.roundWh.toFixed(3)} Wh</b>
              </div>
              <div>
                <span>Round Carbon</span>
                <b className="num">{activeDemo.roundCo2.toFixed(3)} g CO₂</b>
              </div>
              <div>
                <span>Final Accuracy</span>
                <b className="num">{activeDemo.accPct.toFixed(1)}%</b>
              </div>
            </div>
          </motion.div>
        </section>

        {/* Section 2: Mathematical Formulation & Hardware Calibration Ledger */}
        <section className="mono-section" aria-label="Governing equations and hardware calibration">
          <div className="section-head-row">
            <h2>Governing Physical Formulations &amp; 4-Node Calibration Roster</h2>
            <span className="plate-code num">PLATE II · ANALYTICAL MODEL</span>
          </div>

          <div className="math-and-roster-grid">
            <div className="equation-sheet">
              <div className="eq-block">
                <div className="eq-head">
                  <h3>1. Synchronous Parallel Straggler Latency</h3>
                  <span className="num">EQ. (1)</span>
                </div>
                <p className="eq-formula">
                  T<sub>round</sub> = max<sub>i ∈ S</sub> ( T<sub>down, i</sub> + T<sub>base</sub> · ξ<sub>i</sub> / speed<sub>i</sub> + T<sub>up, i</sub> ) + T<sub>agg</sub>
                </p>
                <p className="eq-desc">
                  Selected clients train in parallel. Total round duration is governed by the slowest selected node plus 0.45 s sample-weighted aggregation.
                </p>
              </div>

              <div className="eq-block">
                <div className="eq-head">
                  <h3>2. Active, Idle Tail &amp; Coordinator Energy Accounting</h3>
                  <span className="num">EQ. (2)</span>
                </div>
                <p className="eq-formula">
                  E<sub>total</sub> = ( P<sub>coord</sub> · T<sub>round</sub> + ∑<sub>i=1..4</sub> [ P<sub>active, i</sub> · T<sub>work, i</sub> + P<sub>idle, i</sub> · (T<sub>round</sub> − T<sub>work, i</sub>) ] ) / 3600
                </p>
                <p className="eq-desc">
                  Every finished or unselected client draws idle power (7–13 W) until T<sub>round</sub> completes; Wh multiplied by each node’s grid factor yields grams CO₂.
                </p>
              </div>

              <div className="eq-block">
                <div className="eq-head">
                  <h3>3. Multi-Objective Scalarized RL Reward</h3>
                  <span className="num">EQ. (3)</span>
                </div>
                <p className="eq-formula">
                  R<sub>t</sub> = 15 · ΔAcc<sub>t</sub> − w<sub>lat</sub> · T<sub>round, t</sub> − w<sub>co2</sub> · CO<sub>2, t</sub> − 0.6 · max(0, 0.78 − Acc<sub>t</sub>)
                </p>
                <p className="eq-desc">
                  Tabular Q-Learning (α=0.24, γ=0.86) and MORL preference sweeps balance accuracy gains against latency and carbon penalties above a 78% reward floor.
                </p>
              </div>
            </div>

            <div className="roster-sheet">
              <div className="sheet-subhead">
                <h3>Calibrated Client Hardware &amp; Grid Carbon Profiles</h3>
                <button type="button" className="link-inline" onClick={() => onEnter('specs')}>
                  Full Specs →
                </button>
              </div>
              <div className="scroll">
                <table className="mono-table" aria-label="Client hardware profiles">
                  <thead>
                    <tr>
                      <th>Node</th>
                      <th>Hardware Profile</th>
                      <th>Speed</th>
                      <th>Active / Idle</th>
                      <th>Grid CI</th>
                      <th>Provenance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {CLUSTER_NODES.map((n) => (
                      <tr key={n.name} className={n.isMeasured ? 'row-measured' : ''}>
                        <td className="num"><strong>{n.name}</strong></td>
                        <td>{n.hw}</td>
                        <td className="num">{n.speed}</td>
                        <td className="num">{n.activeW.toFixed(1)}W / {n.idleW.toFixed(0)}W</td>
                        <td className="num">{n.ci} kg/kWh</td>
                        <td>
                          <span className={n.isMeasured ? 'stamp-measured' : 'stamp-modeled'}>
                            {n.provenance}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="roster-footnote">
                <strong>Data Provenance Note:</strong> Only <code>PC1</code> (0.065 kWh over 190 minutes = 20.526 W mean power) is a physical wall-meter measurement. <code>PC2–PC4</code> ratios and the 10,000-row telemetry CSV are calibrated simulation scenarios.
              </p>
            </div>
          </div>
        </section>

        {/* Section 3: Research Questions Ledger Table */}
        <section className="mono-section" aria-label="Research questions">
          <div className="section-head-row">
            <h2>Six Core Research Questions &amp; Empirical Verification</h2>
            <span className="plate-code num">PLATE III · RESEARCH INDEX (RQ1–RQ6)</span>
          </div>

          <div className="rq-ledger-list">
            {rqs.map((r) => (
              <div key={r.id} className="rq-ledger-row">
                <div className="rq-col-id num">{r.id}</div>
                <div className="rq-col-main">
                  <h3>{r.question}</h3>
                  <p className="rq-mech"><strong>Mechanism:</strong> {r.mechanism}</p>
                </div>
                <div className="rq-col-finding">
                  <p>{r.finding}</p>
                </div>
                <div className="rq-col-action">
                  <button type="button" className="btn-plate-jump" onClick={() => onEnter(r.view)}>
                    {r.plate}
                    <ArrowUpRight size={13} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Section 4: Four Interactive Console Plates */}
        <section className="mono-section" aria-label="Interactive console plates">
          <div className="section-head-row">
            <h2>Ordered Four-Plate Experimental Walkthrough</h2>
            <span className="plate-code num">PLATE IV · WORKBENCH MODULES</span>
          </div>

          <div className="walkthrough-grid">
            {plates.map((p) => (
              <button
                key={p.id}
                type="button"
                className="walkthrough-plate"
                onClick={() => onEnter(p.id)}
              >
                <div className="wp-top">
                  <h3>{p.title}</h3>
                  <span className="wp-code num">{p.code}</span>
                </div>
                <p className="wp-desc">{p.summary}</p>
                <div className="wp-foot">
                  <span>{p.metrics}</span>
                  <span className="wp-enter">Open Plate →</span>
                </div>
              </button>
            ))}
          </div>
        </section>

        {/* Colophon / Footer CTA */}
        <footer className="mono-colophon">
          <div className="colophon-copy">
            <h2>Execute Live Simulations or Inspect the Reproducible Run Archive</h2>
            <p>
              Adjust federated rounds, Dirichlet non-IID α, Q-learning episodes, or latency/carbon reward penalties directly in the console.
            </p>
          </div>
          <div className="colophon-actions">
            <button type="button" className="btn-vermilion" onClick={() => onEnter('overview')}>
              <Play size={14} />
              Enter Experiment Console
            </button>
            <button type="button" className="btn-architectural" onClick={() => onEnter('specs')}>
              View PC1 Meter &amp; Provenance
            </button>
          </div>
        </footer>
      </main>
    </div>
  );
}
