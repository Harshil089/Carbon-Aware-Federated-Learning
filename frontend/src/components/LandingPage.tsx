import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import { ViewId } from '../types';

const EASE: [number, number, number, number] = [0.22, 1, 0.36, 1];

const STRAGGLER_BARS = [
  { node: 'PC1 · i7-12700 (20.53W)', workPct: 86, idlePct: 0, label: 'Straggler (16.75s)' },
  { node: 'PC2 · Ultra 5 235 (86.22W)', workPct: 68, idlePct: 18, label: 'Waits 3.1s idle' },
  { node: 'PC3 · Ultra 7 265K (209.4W)', workPct: 48, idlePct: 38, label: 'Burns 209.4W + waits' },
  { node: 'PC4 · 4GB GPU Profile (35.92W)', workPct: 56, idlePct: 30, label: 'Waits 5.2s idle' },
];

const POLICY_POINTS = [
  { name: 'LowestEnergy2', co2: 6.346, acc: 81.6, color: '#B45309', highlight: false },
  { name: 'Fastest2', co2: 12.063, acc: 82.2, color: '#1D4ED8', highlight: false },
  { name: 'Q_learning_RL', co2: 13.214, acc: 83.4, color: '#0F766E', highlight: true },
  { name: 'FedAvg_all4', co2: 18.196, acc: 83.3, color: '#C82D14', highlight: false },
];

const HARDWARE_BARS = [
  { name: 'PC1', sub: 'i7-12700', watts: 20.53, co2: null, acc: 'Anchor' },
  { name: 'PC2', sub: 'Ultra 5 235', watts: 86.22, co2: 186.3, acc: '72.11%' },
  { name: 'PC3', sub: 'Ultra 7 265K', watts: 209.4, co2: 324.0, acc: '72.46%' },
];

export function LandingPage({ onEnter }: { onEnter: (target: ViewId) => void }) {
  return (
    <div className="lp-page">
      <header className="lp-topbar">
        <div className="lp-topbar-inner">
          <div className="lp-brand">
            <span className="lp-mark" aria-hidden="true">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="3" />
                <circle cx="5" cy="5" r="2" />
                <circle cx="19" cy="5" r="2" />
                <circle cx="5" cy="19" r="2" />
                <circle cx="19" cy="19" r="2" />
                <path d="M6.6 6.6L9.8 9.8M17.4 6.6L14.2 9.8M6.6 17.4L9.8 14.2M17.4 17.4L14.2 14.2" />
              </svg>
            </span>
            <span className="lp-brand-name">Federated Computing Lab</span>
          </div>

          <nav className="lp-nav" aria-label="Console navigation">
            <button type="button" className="lp-nav-btn" onClick={() => onEnter('overview')}>
              Console
              <ArrowRight size={14} />
            </button>
          </nav>
        </div>
      </header>

      <main className="lp-main">
        {/* 1. What the project is */}
        <section className="lp-hero-split" aria-labelledby="lp-project-title">
          <motion.div
            className="lp-hero-copy"
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, ease: EASE }}
          >
            <p className="lp-kicker">01 · What The Project Is</p>
            <h1 id="lp-project-title" className="lp-title">
              Carbon- and Latency-Aware Federated Learning via Multi-Objective RL.
            </h1>
            <p className="lp-lead">
              Instead of training every round across all heterogeneous devices, this project builds a
              Multi-Objective Reinforcement Learning (MORL) coordinator that dynamically selects the best
              client subset each round—jointly optimizing model accuracy, wall-clock latency, and grid CO₂ emissions.
            </p>
            <div className="lp-hero-actions">
              <button type="button" className="lp-primary-btn" onClick={() => onEnter('overview')}>
                Open Experiment Console
                <ArrowRight size={15} />
              </button>
            </div>
          </motion.div>

          {/* Animated Topology Selection Graphic */}
          <motion.div
            className="lp-visual-card"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1, ease: EASE }}
            aria-label="Animated MORL client selection topology"
          >
            <div className="lp-visual-head">
              <span className="lp-visual-label">LIVE MORL ROUND SELECTION</span>
              <span className="lp-visual-pill">PC2 + PC4 Active · −27.4% CO₂</span>
            </div>

            <svg viewBox="0 0 400 250" className="lp-topo-svg" role="img" aria-label="Coordinator selecting PC2 and PC4">
              {/* Inactive links */}
              <line x1="200" y1="125" x2="72" y2="52" stroke="#D5CFC2" strokeWidth="1.5" strokeDasharray="4 4" />
              <line x1="200" y1="125" x2="72" y2="198" stroke="#D5CFC2" strokeWidth="1.5" strokeDasharray="4 4" />

              {/* Active links */}
              <line x1="200" y1="125" x2="328" y2="52" stroke="#0F766E" strokeWidth="2.5" />
              <line x1="200" y1="125" x2="328" y2="198" stroke="#0F766E" strokeWidth="2.5" />

              {/* Animated packets on active links */}
              <motion.circle
                r="4.5"
                fill="#0F766E"
                animate={{ cx: [200, 328, 200], cy: [125, 52, 125] }}
                transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
              />
              <motion.circle
                r="4.5"
                fill="#C82D14"
                animate={{ cx: [328, 200, 328], cy: [198, 125, 198] }}
                transition={{ duration: 2.2, delay: 0.5, repeat: Infinity, ease: 'easeInOut' }}
              />

              {/* Coordinator */}
              <motion.circle
                cx="200"
                cy="125"
                r="36"
                fill="none"
                stroke="#C82D14"
                strokeOpacity="0.28"
                strokeWidth="1.5"
                animate={{ r: [35, 45, 35], strokeOpacity: [0.35, 0, 0.35] }}
                transition={{ duration: 2.6, repeat: Infinity, ease: 'easeOut' }}
              />
              <circle cx="200" cy="125" r="32" fill="#121316" />
              <text x="200" y="121" textAnchor="middle" fill="#FAF8F3" fontSize="11" fontWeight="700">
                MORL
              </text>
              <text x="200" y="135" textAnchor="middle" fill="#D5CFC2" fontSize="9" fontFamily="monospace">
                30W Coord
              </text>

              {/* PC1 Standby */}
              <g>
                <rect x="8" y="24" width="128" height="52" rx="3" fill="#FAF8F3" stroke="#D5CFC2" />
                <text x="18" y="44" fill="#57534A" fontSize="11" fontWeight="700">PC1 · Standby</text>
                <text x="18" y="61" fill="#57534A" fontSize="9.5" fontFamily="monospace">20.53W · Straggler</text>
              </g>

              {/* PC3 Standby */}
              <g>
                <rect x="8" y="172" width="128" height="52" rx="3" fill="#FAF8F3" stroke="#D5CFC2" />
                <text x="18" y="192" fill="#57534A" fontSize="11" fontWeight="700">PC3 · Skipped</text>
                <text x="18" y="209" fill="#C82D14" fontSize="9.5" fontFamily="monospace">209.4W · Peak CO₂</text>
              </g>

              {/* PC2 Selected */}
              <g>
                <rect x="264" y="24" width="128" height="52" rx="3" fill="#CCFBF1" stroke="#0F766E" strokeWidth="1.5" />
                <text x="274" y="44" fill="#121316" fontSize="11" fontWeight="700">PC2 · Selected</text>
                <text x="274" y="61" fill="#0F766E" fontSize="9.5" fontFamily="monospace">86.22W · 1.10× speed</text>
              </g>

              {/* PC4 Selected */}
              <g>
                <rect x="264" y="172" width="128" height="52" rx="3" fill="#CCFBF1" stroke="#0F766E" strokeWidth="1.5" />
                <text x="274" y="192" fill="#121316" fontSize="11" fontWeight="700">PC4 · Selected</text>
                <text x="274" y="209" fill="#0F766E" fontSize="9.5" fontFamily="monospace">35.92W · 1.35× speed</text>
              </g>
            </svg>
          </motion.div>
        </section>

        {/* 2. What is the need for it */}
        <section className="lp-section" aria-labelledby="lp-need-title">
          <div className="lp-section-head">
            <p className="lp-kicker">02 · Why It Is Needed</p>
            <h2 id="lp-need-title" className="lp-h2">
              Standard Federated Averaging wastes time, idle power, and grid carbon.
            </h2>
          </div>

          {/* Animated Straggler & Idle Energy Chart */}
          <div className="lp-visual-card">
            <div className="lp-visual-head">
              <span className="lp-visual-label">SYNCHRONOUS FEDAVG ROUND BOTTLENECK (ALL-4 BASELINE)</span>
              <div className="lp-legend">
                <span className="lp-legend-item"><i className="lp-dot lp-dot-work" /> Active Compute</span>
                <span className="lp-legend-item"><i className="lp-dot lp-dot-idle" /> Wasted Idle Wait</span>
              </div>
            </div>

            <div className="lp-bar-list">
              {STRAGGLER_BARS.map((row, idx) => (
                <div key={row.node} className="lp-bar-row">
                  <div className="lp-bar-meta">
                    <span className="lp-bar-name">{row.node}</span>
                    <span className="lp-bar-note">{row.label}</span>
                  </div>
                  <div className="lp-bar-track">
                    <motion.div
                      className="lp-bar-work"
                      initial={{ width: 0 }}
                      whileInView={{ width: `${row.workPct}%` }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.65, delay: idx * 0.08, ease: EASE }}
                    />
                    {row.idlePct > 0 && (
                      <motion.div
                        className="lp-bar-idle"
                        initial={{ width: 0 }}
                        whileInView={{ width: `${row.idlePct}%` }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.55, delay: 0.35 + idx * 0.08, ease: EASE }}
                      />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="lp-grid-3">
            <article className="lp-card">
              <span className="lp-card-num">01</span>
              <h3 className="lp-card-title">Synchronous Straggler Bottleneck</h3>
              <p className="lp-card-body">
                In synchronous FedAvg, every round stalls until the slowest client finishes local training,
                inflating total wall-clock convergence time across heterogeneous hardware.
              </p>
            </article>

            <article className="lp-card">
              <span className="lp-card-num">02</span>
              <h3 className="lp-card-title">Idle Power &amp; Carbon Multiplication</h3>
              <p className="lp-card-body">
                Faster clients and the 30 W coordinator burn idle energy while waiting on stragglers,
                while high-wattage or carbon-intensive grid nodes drive up CO₂ per round.
              </p>
            </article>

            <article className="lp-card">
              <span className="lp-card-num">03</span>
              <h3 className="lp-card-title">Naive Heuristics Break Under Non-IID Data</h3>
              <p className="lp-card-body">
                Always picking the fastest 2 or lowest-energy 2 nodes skews class distribution and drops
                accuracy (82.2% and 81.6%) compared to adaptive RL selection (83.4%).
              </p>
            </article>
          </div>
        </section>

        {/* 3. What we have achieved so far */}
        <section className="lp-section" aria-labelledby="lp-achieved-title">
          <div className="lp-section-head">
            <p className="lp-kicker">03 · What We Have Achieved So Far</p>
            <h2 id="lp-achieved-title" className="lp-h2">
              Physical power benchmarks, an 8-layer MORL pipeline, and verified CO₂ savings.
            </h2>
          </div>

          <div className="lp-stats-row">
            <motion.div className="lp-stat" initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.35, ease: EASE }}>
              <span className="lp-stat-val">83.4%</span>
              <span className="lp-stat-label">Q-Learning RL accuracy (vs. 83.3% All-4 FedAvg)</span>
            </motion.div>
            <motion.div className="lp-stat" initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.35, delay: 0.06, ease: EASE }}>
              <span className="lp-stat-val lp-stat-good">−27.4%</span>
              <span className="lp-stat-label">CO₂ emissions (13.214 g vs. 18.196 g All-4)</span>
            </motion.div>
            <motion.div className="lp-stat" initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.35, delay: 0.12, ease: EASE }}>
              <span className="lp-stat-val">−13.1%</span>
              <span className="lp-stat-label">Training latency (294.3 s vs. 338.8 s All-4)</span>
            </motion.div>
            <motion.div className="lp-stat" initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.35, delay: 0.18, ease: EASE }}>
              <span className="lp-stat-val">10,000</span>
              <span className="lp-stat-label">Calibrated telemetry rows across 4 hardware nodes</span>
            </motion.div>
          </div>

          {/* Two animated SVG data charts */}
          <div className="lp-charts-grid">
            {/* Chart 1: Accuracy vs CO2 Policy Trade-Off */}
            <div className="lp-visual-card">
              <div className="lp-visual-head">
                <span className="lp-visual-label">SIMULATION POLICY TRADE-OFF (CO₂ VS. ACCURACY)</span>
                <span className="lp-visual-pill">Pareto Winner: Q_learning_RL</span>
              </div>
              <svg viewBox="0 0 420 215" className="lp-chart-svg" role="img" aria-label="Policy CO2 vs Accuracy chart">
                {/* Gridlines */}
                {[40, 80, 120, 160].map((y) => (
                  <line key={y} x1="48" y1={y} x2="396" y2={y} stroke="#EFECE4" strokeWidth="1" />
                ))}
                {/* Axes */}
                <line x1="48" y1="24" x2="48" y2="170" stroke="#D5CFC2" strokeWidth="1.5" />
                <line x1="48" y1="170" x2="396" y2="170" stroke="#D5CFC2" strokeWidth="1.5" />

                {/* Y labels (Accuracy 81% - 84%) */}
                <text x="40" y="34" textAnchor="end" fill="#57534A" fontSize="10" fontFamily="monospace">84%</text>
                <text x="40" y="82" textAnchor="end" fill="#57534A" fontSize="10" fontFamily="monospace">83%</text>
                <text x="40" y="130" textAnchor="end" fill="#57534A" fontSize="10" fontFamily="monospace">82%</text>
                <text x="40" y="172" textAnchor="end" fill="#57534A" fontSize="10" fontFamily="monospace">81%</text>

                {/* X labels (CO2 5g - 20g) */}
                <text x="60" y="188" textAnchor="middle" fill="#57534A" fontSize="10" fontFamily="monospace">5g</text>
                <text x="168" y="188" textAnchor="middle" fill="#57534A" fontSize="10" fontFamily="monospace">10g</text>
                <text x="276" y="188" textAnchor="middle" fill="#57534A" fontSize="10" fontFamily="monospace">15g</text>
                <text x="384" y="188" textAnchor="middle" fill="#57534A" fontSize="10" fontFamily="monospace">20g CO₂</text>

                {/* Animated Pareto Frontier Curve */}
                <motion.path
                  d="M 89 144 Q 160 112 238 58 L 345 63"
                  fill="none"
                  stroke="#D5CFC2"
                  strokeWidth="1.5"
                  strokeDasharray="4 4"
                  initial={{ pathLength: 0 }}
                  whileInView={{ pathLength: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.9, ease: EASE }}
                />

                {POLICY_POINTS.map((p, i) => {
                  const cx = 60 + ((p.co2 - 5) / 15) * 324;
                  const cy = 170 - ((p.acc - 81) / 3) * 140;
                  const labelPos =
                    p.name === 'LowestEnergy2'
                      ? { x: cx + 10, y: cy + 4, anchor: 'start' as const }
                      : p.name === 'Fastest2'
                        ? { x: cx, y: cy + 20, anchor: 'middle' as const }
                        : p.name === 'Q_learning_RL'
                          ? { x: cx - 10, y: cy - 12, anchor: 'end' as const }
                          : { x: cx + 8, y: cy + 20, anchor: 'end' as const };
                  return (
                    <g key={p.name}>
                      {p.highlight && (
                        <motion.circle
                          cx={cx}
                          cy={cy}
                          r="13"
                          fill="none"
                          stroke="#0F766E"
                          strokeWidth="1.5"
                          animate={{ r: [10, 16, 10], opacity: [0.6, 0.15, 0.6] }}
                          transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
                        />
                      )}
                      <motion.circle
                        cx={cx}
                        cy={cy}
                        r={p.highlight ? 7 : 5.5}
                        fill={p.color}
                        initial={{ scale: 0, opacity: 0 }}
                        whileInView={{ scale: 1, opacity: 1 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.35, delay: 0.2 + i * 0.1, ease: EASE }}
                      />
                      <text
                        x={labelPos.x}
                        y={labelPos.y}
                        textAnchor={labelPos.anchor}
                        fill="#121316"
                        fontSize="10"
                        fontWeight={p.highlight ? '700' : '600'}
                        fontFamily="monospace"
                      >
                        {p.name} ({p.acc}%)
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>

            {/* Chart 2: Physical Hardware Power & CIFAR-10 CO2 */}
            <div className="lp-visual-card">
              <div className="lp-visual-head">
                <span className="lp-visual-label">MEASURED WALL-PLUG POWER &amp; CIFAR-10 CO₂</span>
                <span className="lp-visual-pill">+73.91% CO₂ on PC3 for +0.35 pp</span>
              </div>
              <div className="lp-hw-bars">
                {HARDWARE_BARS.map((node, i) => (
                  <div key={node.name} className="lp-hw-row">
                    <div className="lp-hw-label">
                      <strong>{node.name}</strong>
                      <span>{node.sub}</span>
                    </div>
                    <div className="lp-hw-metrics">
                      <div className="lp-hw-bar-wrap">
                        <motion.div
                          className="lp-hw-fill-watts"
                          initial={{ width: 0 }}
                          whileInView={{ width: `${(node.watts / 220) * 100}%` }}
                          viewport={{ once: true }}
                          transition={{ duration: 0.65, delay: i * 0.1, ease: EASE }}
                        />
                        <span className="lp-hw-val">{node.watts} W</span>
                      </div>
                      <div className="lp-hw-sub">
                        {node.co2 !== null
                          ? `${node.co2} g CO₂ · ${node.acc} CIFAR-10`
                          : '0.065 kWh / 190 min anchor'}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="lp-grid-3">
            <article className="lp-card">
              <span className="lp-card-tag">PHYSICAL BENCHMARKS</span>
              <h3 className="lp-card-title">Measured PC1–PC3 Wall-Plug Anchors</h3>
              <p className="lp-card-body">
                Benchmarked real CIFAR-10 runs on PC1 (20.53 W), PC2 Ultra 5 235 (86.22 W, 186.3 g CO₂,
                72.11% acc), and PC3 Ultra 7 265K (209.4 W, 324.0 g CO₂, 72.46% acc)—proving a 28.40%
                speedup costs +73.91% more CO₂ for just +0.35 pp accuracy.
              </p>
            </article>

            <article className="lp-card">
              <span className="lp-card-tag">CORE ENGINE</span>
              <h3 className="lp-card-title">8-Layer PyTorch &amp; MORL Pipeline</h3>
              <p className="lp-card-body">
                Built the modular 8-layer <code>fl_pipeline/</code> supporting Non-IID MNIST partitions,
                MLP/CNN models, Q-learning &amp; DQN selectors, preference-vector Pareto sweeps, and
                multi-seed hypervolume evaluation—verified by 26 passing tests.
              </p>
            </article>

            <article className="lp-card">
              <span className="lp-card-tag">INTERACTIVE CONSOLE</span>
              <h3 className="lp-card-title">Live Policy, Pipeline &amp; Topology Workbench</h3>
              <p className="lp-card-body">
                Shipped a full experiment console to run custom RL sweeps, inspect the 4-node synchronous
                round timeline, and audit physical vs. simulated hardware provenance end-to-end.
              </p>
            </article>
          </div>
        </section>

        <footer className="lp-footer">
          <span>Federated Computing Lab · Multi-Objective RL Research Workbench</span>
          <button type="button" className="lp-footer-btn" onClick={() => onEnter('overview')}>
            Go to Console
            <ArrowRight size={14} />
          </button>
        </footer>
      </main>
    </div>
  );
}
