import { useRef, useState, type MouseEvent, type ReactNode } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import { ViewId } from '../types';

const SPRING_SNAPPY = { type: 'spring' as const, stiffness: 380, damping: 30, mass: 0.5 };
const SPRING_SMOOTH = { type: 'spring' as const, stiffness: 200, damping: 28, mass: 0.85 };
const SPRING_MORPH = { type: 'spring' as const, stiffness: 220, damping: 32, mass: 1.0 };

type RoundMode = 'rl' | 'all4';

const STRAGGLER_BARS = [
  { node: 'PC1 · i7-12700', watts: '20.53 W', workPct: 86, idlePct: 0, note: '16.75s straggler' },
  { node: 'PC2 · Ultra 5 235', watts: '86.22 W', workPct: 68, idlePct: 18, note: '3.1s idle wait' },
  { node: 'PC3 · Ultra 7 265K', watts: '209.4 W', workPct: 48, idlePct: 38, note: '209.4W + 6.4s idle' },
  { node: 'PC4 · 4GB GPU Profile', watts: '35.92 W', workPct: 56, idlePct: 30, note: '5.2s idle wait' },
];

const POLICY_POINTS = [
  { name: 'LowestEnergy2', co2: 6.346, acc: 81.6, color: '#F59E0B', highlight: false },
  { name: 'Fastest2', co2: 12.063, acc: 82.2, color: '#60A5FA', highlight: false },
  { name: 'Q_learning_RL', co2: 13.214, acc: 83.4, color: '#10B981', highlight: true },
  { name: 'FedAvg_all4', co2: 18.196, acc: 83.3, color: '#EF4444', highlight: false },
];

const HARDWARE_BARS = [
  { name: 'PC1', sub: 'i7-12700', watts: 20.53, co2: null, acc: '0.065 kWh / 190 min' },
  { name: 'PC2', sub: 'Ultra 5 235', watts: 86.22, co2: 186.3, acc: '72.11% CIFAR-10' },
  { name: 'PC3', sub: 'Ultra 7 265K', watts: 209.4, co2: 324.0, acc: '72.46% CIFAR-10' },
];

function SpotlightCard({
  children,
  className = '',
}: {
  children: ReactNode;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [spot, setSpot] = useState({ x: 0, y: 0, active: false });

  const onMove = (e: MouseEvent<HTMLDivElement>) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    setSpot({ x: e.clientX - rect.left, y: e.clientY - rect.top, active: true });
  };

  return (
    <motion.div
      ref={ref}
      onMouseMove={onMove}
      onMouseLeave={() => setSpot((s) => ({ ...s, active: false }))}
      className={`lp-spotlight ${className}`}
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.2 }}
      transition={SPRING_SMOOTH}
    >
      <div
        className="lp-spotlight-glow"
        style={{
          opacity: spot.active ? 1 : 0,
          background: `radial-gradient(360px circle at ${spot.x}px ${spot.y}px, rgba(255,255,255,0.065), transparent 70%)`,
        }}
        aria-hidden="true"
      />
      <div className="lp-spotlight-inner">{children}</div>
    </motion.div>
  );
}

export function LandingPage({ onEnter }: { onEnter: (target: ViewId) => void }) {
  const [roundMode, setRoundMode] = useState<RoundMode>('rl');
  const isRl = roundMode === 'rl';

  return (
    <div className="lp-page">
      <header className="lp-topbar">
        <div className="lp-topbar-inner">
          <div className="lp-brand">
            <span className="lp-mark" aria-hidden="true">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
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
            <motion.button
              type="button"
              className="lp-nav-btn"
              onClick={() => onEnter('overview')}
              whileHover={{ y: -1 }}
              whileTap={{ scale: 0.97 }}
              transition={SPRING_SNAPPY}
            >
              Console
              <ArrowRight size={14} />
            </motion.button>
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
            transition={SPRING_SMOOTH}
          >
            <span className="lp-eyebrow">What the project is</span>
            <h1 id="lp-project-title" className="lp-title">
              Carbon- and Latency-Aware Federated Learning via Multi-Objective RL.
            </h1>
            <p className="lp-lead">
              A Multi-Objective Reinforcement Learning (MORL) coordinator selects the optimal client
              subset each training round—jointly balancing convergence accuracy, wall-clock latency,
              and grid CO₂ emissions across heterogeneous devices.
            </p>
            <div className="lp-hero-actions">
              <motion.button
                type="button"
                className="lp-primary-btn"
                onClick={() => onEnter('overview')}
                whileHover={{ y: -1 }}
                whileTap={{ scale: 0.97 }}
                transition={SPRING_SNAPPY}
              >
                Open Experiment Console
                <ArrowRight size={15} />
              </motion.button>
            </div>
          </motion.div>

          <SpotlightCard className="lp-visual-card">
            <div className="lp-visual-head">
              <div className="lp-segmented" role="tablist" aria-label="Round selection mode">
                {(['rl', 'all4'] as const).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    role="tab"
                    aria-selected={roundMode === mode}
                    className={`lp-seg-btn ${roundMode === mode ? 'is-active' : ''}`}
                    onClick={() => setRoundMode(mode)}
                  >
                    {roundMode === mode && (
                      <motion.span
                        layoutId="lp-round-pill"
                        className="lp-seg-pill"
                        transition={SPRING_MORPH}
                      />
                    )}
                    <span className="lp-seg-text">
                      {mode === 'rl' ? 'MORL Subset' : 'FedAvg All-4'}
                    </span>
                  </button>
                ))}
              </div>
              <span className={`lp-status-pill ${isRl ? 'is-good' : 'is-warn'}`}>
                {isRl ? '−27.4% CO₂ · PC2 + PC4' : '18.196g CO₂ · All Nodes'}
              </span>
            </div>

            <svg viewBox="0 0 400 246" className="lp-topo-svg" role="img" aria-label="Coordinator client selection topology">
              {/* Links */}
              <line
                x1="200" y1="123" x2="74" y2="50"
                stroke={isRl ? '#2E2E2E' : '#EF4444'}
                strokeWidth={isRl ? 1.5 : 2}
                strokeDasharray={isRl ? '4 4' : undefined}
              />
              <line
                x1="200" y1="123" x2="74" y2="196"
                stroke={isRl ? '#2E2E2E' : '#EF4444'}
                strokeWidth={isRl ? 1.5 : 2}
                strokeDasharray={isRl ? '4 4' : undefined}
              />
              <line x1="200" y1="123" x2="326" y2="50" stroke="#10B981" strokeWidth="2.2" />
              <line x1="200" y1="123" x2="326" y2="196" stroke="#10B981" strokeWidth="2.2" />

              {/* Animated packets on active links */}
              <motion.circle
                r="4"
                fill="#10B981"
                animate={{ cx: [200, 326, 200], cy: [123, 50, 123] }}
                transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
              />
              <motion.circle
                r="4"
                fill="#E5E5E5"
                animate={{ cx: [326, 200, 326], cy: [196, 123, 196] }}
                transition={{ duration: 2.2, delay: 0.45, repeat: Infinity, ease: 'easeInOut' }}
              />
              {!isRl && (
                <>
                  <motion.circle
                    r="4"
                    fill="#EF4444"
                    animate={{ cx: [200, 74, 200], cy: [123, 50, 123] }}
                    transition={{ duration: 2.8, repeat: Infinity, ease: 'easeInOut' }}
                  />
                  <motion.circle
                    r="4"
                    fill="#EF4444"
                    animate={{ cx: [74, 200, 74], cy: [196, 123, 196] }}
                    transition={{ duration: 2.4, delay: 0.3, repeat: Infinity, ease: 'easeInOut' }}
                  />
                </>
              )}

              {/* Coordinator */}
              <motion.circle
                cx="200"
                cy="123"
                r="36"
                fill="none"
                stroke={isRl ? '#10B981' : '#EF4444'}
                strokeOpacity="0.28"
                strokeWidth="1.5"
                animate={{ r: [34, 44, 34], strokeOpacity: [0.35, 0, 0.35] }}
                transition={{ duration: 2.6, repeat: Infinity, ease: 'easeOut' }}
              />
              <circle cx="200" cy="123" r="32" fill="#242424" stroke="#3A3A3A" strokeWidth="1.5" />
              <text x="200" y="119" textAnchor="middle" fill="#F5F5F5" fontSize="11" fontWeight="700">
                MORL
              </text>
              <text x="200" y="133" textAnchor="middle" fill="#A3A3A3" fontSize="9" fontFamily="monospace">
                30W Coord
              </text>

              {/* PC1 */}
              <g>
                <rect
                  x="8" y="22" width="128" height="52" rx="8"
                  fill="#242424"
                  stroke={isRl ? '#2E2E2E' : '#EF4444'}
                  strokeWidth={isRl ? 1 : 1.5}
                />
                <text x="18" y="42" fill={isRl ? '#A3A3A3' : '#F5F5F5'} fontSize="11" fontWeight="600">
                  {isRl ? 'PC1 · Standby' : 'PC1 · Bottleneck'}
                </text>
                <text x="18" y="59" fill={isRl ? '#737373' : '#FCA5A5'} fontSize="9.5" fontFamily="monospace">
                  20.53W · Straggler
                </text>
              </g>

              {/* PC3 */}
              <g>
                <rect
                  x="8" y="170" width="128" height="52" rx="8"
                  fill="#242424"
                  stroke={isRl ? '#2E2E2E' : '#EF4444'}
                  strokeWidth={isRl ? 1 : 1.5}
                />
                <text x="18" y="190" fill={isRl ? '#A3A3A3' : '#F5F5F5'} fontSize="11" fontWeight="600">
                  {isRl ? 'PC3 · Skipped' : 'PC3 · Active'}
                </text>
                <text x="18" y="207" fill="#FCA5A5" fontSize="9.5" fontFamily="monospace">
                  209.4W · Peak CO₂
                </text>
              </g>

              {/* PC2 */}
              <g>
                <rect x="264" y="22" width="128" height="52" rx="8" fill="#182722" stroke="#10B981" strokeWidth="1.5" />
                <text x="274" y="42" fill="#F5F5F5" fontSize="11" fontWeight="600">PC2 · Selected</text>
                <text x="274" y="59" fill="#6EE7B7" fontSize="9.5" fontFamily="monospace">86.22W · 1.10× speed</text>
              </g>

              {/* PC4 */}
              <g>
                <rect x="264" y="170" width="128" height="52" rx="8" fill="#182722" stroke="#10B981" strokeWidth="1.5" />
                <text x="274" y="190" fill="#F5F5F5" fontSize="11" fontWeight="600">PC4 · Selected</text>
                <text x="274" y="207" fill="#6EE7B7" fontSize="9.5" fontFamily="monospace">35.92W · 1.35× speed</text>
              </g>
            </svg>
          </SpotlightCard>
        </section>

        {/* 2. Why it is needed */}
        <section className="lp-section" aria-labelledby="lp-need-title">
          <div className="lp-section-head">
            <span className="lp-eyebrow">Why it is needed</span>
            <h2 id="lp-need-title" className="lp-h2">
              Standard FedAvg wastes wall-clock time, idle power, and grid carbon.
            </h2>
          </div>

          <SpotlightCard className="lp-visual-card">
            <div className="lp-visual-head">
              <span className="lp-visual-label">Synchronous All-4 Round Timeline</span>
              <div className="lp-legend">
                <span className="lp-legend-item"><i className="lp-dot lp-dot-work" /> Compute</span>
                <span className="lp-legend-item"><i className="lp-dot lp-dot-idle" /> Wasted Idle Wait</span>
              </div>
            </div>

            <div className="lp-bar-list">
              {STRAGGLER_BARS.map((row, idx) => (
                <div key={row.node} className="lp-bar-row">
                  <div className="lp-bar-meta">
                    <span className="lp-bar-name">
                      {row.node} <span className="lp-bar-watts">{row.watts}</span>
                    </span>
                    <span className="lp-bar-note">{row.note}</span>
                  </div>
                  <div className="lp-bar-track">
                    <motion.div
                      className="lp-bar-work"
                      initial={{ width: 0 }}
                      whileInView={{ width: `${row.workPct}%` }}
                      viewport={{ once: true }}
                      transition={{ ...SPRING_SMOOTH, delay: idx * 0.06 }}
                    />
                    {row.idlePct > 0 && (
                      <motion.div
                        className="lp-bar-idle"
                        initial={{ width: 0 }}
                        whileInView={{ width: `${row.idlePct}%` }}
                        viewport={{ once: true }}
                        transition={{ ...SPRING_SMOOTH, delay: 0.25 + idx * 0.06 }}
                      />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </SpotlightCard>

          <div className="lp-grid-3">
            <SpotlightCard className="lp-card">
              <h3 className="lp-card-title">Synchronous Straggler Stall</h3>
              <p className="lp-card-body">
                Every FedAvg round waits on the slowest device to finish local epochs before aggregation can begin.
              </p>
            </SpotlightCard>

            <SpotlightCard className="lp-card">
              <h3 className="lp-card-title">Idle Power &amp; Carbon Overhead</h3>
              <p className="lp-card-body">
                Waiting clients and the 30 W coordinator draw continuous idle power while high-wattage nodes inflate CO₂.
              </p>
            </SpotlightCard>

            <SpotlightCard className="lp-card">
              <h3 className="lp-card-title">Non-IID Heuristic Collapse</h3>
              <p className="lp-card-body">
                Static Fastest-2 (82.2%) and Lowest-Energy-2 (81.6%) rules skew class coverage below RL selection (83.4%).
              </p>
            </SpotlightCard>
          </div>
        </section>

        {/* 3. What we have achieved so far */}
        <section className="lp-section" aria-labelledby="lp-achieved-title">
          <div className="lp-section-head">
            <span className="lp-eyebrow">What we have achieved so far</span>
            <h2 id="lp-achieved-title" className="lp-h2">
              Measured hardware power anchors, an 8-layer MORL engine, and verified CO₂ savings.
            </h2>
          </div>

          <div className="lp-stats-row">
            <SpotlightCard className="lp-stat">
              <span className="lp-stat-val">83.4%</span>
              <span className="lp-stat-label">Q-Learning RL accuracy (vs. 83.3% All-4)</span>
            </SpotlightCard>
            <SpotlightCard className="lp-stat">
              <span className="lp-stat-val lp-stat-good">−27.4%</span>
              <span className="lp-stat-label">CO₂ emissions (13.214 g vs. 18.196 g)</span>
            </SpotlightCard>
            <SpotlightCard className="lp-stat">
              <span className="lp-stat-val">−13.1%</span>
              <span className="lp-stat-label">Round latency (294.3 s vs. 338.8 s)</span>
            </SpotlightCard>
            <SpotlightCard className="lp-stat">
              <span className="lp-stat-val">10,000</span>
              <span className="lp-stat-label">Calibrated telemetry rows across 4 nodes</span>
            </SpotlightCard>
          </div>

          <div className="lp-charts-grid">
            <SpotlightCard className="lp-visual-card">
              <div className="lp-visual-head">
                <span className="lp-visual-label">Simulation Policy Trade-Off (CO₂ vs. Accuracy)</span>
                <span className="lp-status-pill is-good">Pareto Winner: Q_learning_RL</span>
              </div>
              <svg viewBox="0 0 420 215" className="lp-chart-svg" role="img" aria-label="Policy CO2 vs Accuracy chart">
                {[40, 80, 120, 160].map((y) => (
                  <line key={y} x1="48" y1={y} x2="396" y2={y} stroke="#2E2E2E" strokeWidth="1" />
                ))}
                <line x1="48" y1="24" x2="48" y2="170" stroke="#3A3A3A" strokeWidth="1.5" />
                <line x1="48" y1="170" x2="396" y2="170" stroke="#3A3A3A" strokeWidth="1.5" />

                <text x="40" y="34" textAnchor="end" fill="#A3A3A3" fontSize="10" fontFamily="monospace">84%</text>
                <text x="40" y="82" textAnchor="end" fill="#A3A3A3" fontSize="10" fontFamily="monospace">83%</text>
                <text x="40" y="130" textAnchor="end" fill="#A3A3A3" fontSize="10" fontFamily="monospace">82%</text>
                <text x="40" y="172" textAnchor="end" fill="#A3A3A3" fontSize="10" fontFamily="monospace">81%</text>

                <text x="60" y="188" textAnchor="middle" fill="#A3A3A3" fontSize="10" fontFamily="monospace">5g</text>
                <text x="168" y="188" textAnchor="middle" fill="#A3A3A3" fontSize="10" fontFamily="monospace">10g</text>
                <text x="276" y="188" textAnchor="middle" fill="#A3A3A3" fontSize="10" fontFamily="monospace">15g</text>
                <text x="384" y="188" textAnchor="middle" fill="#A3A3A3" fontSize="10" fontFamily="monospace">20g CO₂</text>

                <motion.path
                  d="M 89 144 Q 160 112 238 58 L 345 63"
                  fill="none"
                  stroke="#3A3A3A"
                  strokeWidth="1.5"
                  strokeDasharray="4 4"
                  initial={{ pathLength: 0 }}
                  whileInView={{ pathLength: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
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
                          stroke="#10B981"
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
                        transition={{ ...SPRING_SNAPPY, delay: 0.12 + i * 0.08 }}
                      />
                      <text
                        x={labelPos.x}
                        y={labelPos.y}
                        textAnchor={labelPos.anchor}
                        fill="#F5F5F5"
                        fontSize="10"
                        fontWeight={p.highlight ? '700' : '500'}
                        fontFamily="monospace"
                      >
                        {p.name} ({p.acc}%)
                      </text>
                    </g>
                  );
                })}
              </svg>
            </SpotlightCard>

            <SpotlightCard className="lp-visual-card">
              <div className="lp-visual-head">
                <span className="lp-visual-label">Measured Wall-Plug Power &amp; CIFAR-10 CO₂</span>
                <span className="lp-status-pill is-warn">+73.91% CO₂ on PC3 for +0.35 pp</span>
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
                          transition={{ ...SPRING_SMOOTH, delay: i * 0.08 }}
                        />
                        <span className="lp-hw-val">{node.watts} W</span>
                      </div>
                      <div className="lp-hw-sub">
                        {node.co2 !== null ? `${node.co2} g CO₂ · ${node.acc}` : node.acc}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </SpotlightCard>
          </div>

          <div className="lp-grid-3">
            <SpotlightCard className="lp-card">
              <h3 className="lp-card-title">Physical PC1–PC3 Wall-Plug Anchors</h3>
              <p className="lp-card-body">
                Metered CIFAR-10 training on PC2 (86.22 W, 186.3 g CO₂) vs. PC3 (209.4 W, 324.0 g CO₂)
                proves 28.40% faster runs cost +73.91% more CO₂ for +0.35 pp accuracy.
              </p>
            </SpotlightCard>

            <SpotlightCard className="lp-card">
              <h3 className="lp-card-title">8-Layer PyTorch &amp; MORL Engine</h3>
              <p className="lp-card-body">
                Built <code>fl_pipeline/</code> with Non-IID MNIST partitions, MLP/CNN models, Q-learning
                &amp; DQN selectors, preference sweeps, and 26 passing tests.
              </p>
            </SpotlightCard>

            <SpotlightCard className="lp-card">
              <h3 className="lp-card-title">Interactive Experiment Console</h3>
              <p className="lp-card-body">
                Run custom MORL sweeps, inspect 4-node synchronous round timelines, and audit physical vs.
                modeled hardware provenance live in the console.
              </p>
            </SpotlightCard>
          </div>
        </section>

        <footer className="lp-footer">
          <span>Federated Computing Lab</span>
          <motion.button
            type="button"
            className="lp-footer-btn"
            onClick={() => onEnter('overview')}
            whileHover={{ y: -1 }}
            whileTap={{ scale: 0.97 }}
            transition={SPRING_SNAPPY}
          >
            Go to Console
            <ArrowRight size={14} />
          </motion.button>
        </footer>
      </main>
    </div>
  );
}
