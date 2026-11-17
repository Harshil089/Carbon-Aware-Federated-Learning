import { useRef, useState, type MouseEvent, type ReactNode } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import { ViewId } from '../types';

const SPRING_SNAPPY = { type: 'spring' as const, stiffness: 380, damping: 30, mass: 0.5 };
const SPRING_SMOOTH = { type: 'spring' as const, stiffness: 200, damping: 28, mass: 0.85 };
const SPRING_MORPH = { type: 'spring' as const, stiffness: 220, damping: 32, mass: 1.0 };

type RoundMode = 'rl' | 'all4';

const STRAGGLER_BARS = [
  { node: 'PC1 · i7-12700', watts: '20.53 W', workPct: 86, idlePct: 0, note: '16.75 s · slowest, sets round time' },
  { node: 'PC2 · Ultra 5 235', watts: '86.22 W', workPct: 68, idlePct: 18, note: '3.1 s idle' },
  { node: 'PC3 · Ultra 7 265K', watts: '209.4 W', workPct: 48, idlePct: 38, note: '6.4 s idle at high power' },
  { node: 'PC4 · 4GB GPU Profile', watts: '35.92 W', workPct: 56, idlePct: 30, note: '5.2 s idle' },
];

const POLICY_POINTS = [
  { name: 'LowestEnergy2', label: 'Lowest-power 2', co2: 6.346, acc: 81.6, color: '#B45309', highlight: false },
  { name: 'Fastest2', label: 'Fastest 2', co2: 12.063, acc: 82.2, color: '#1D4ED8', highlight: false },
  { name: 'Q_learning_RL', label: 'RL picker', co2: 13.214, acc: 83.4, color: '#0F766E', highlight: true },
  { name: 'FedAvg_all4', label: 'All 4 (FedAvg)', co2: 18.196, acc: 83.3, color: '#C82D14', highlight: false },
];

const HARDWARE_BARS = [
  { name: 'PC1', sub: 'i7-12700', watts: 20.53, co2: null, acc: '0.065 kWh over 190 min' },
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
          background: `radial-gradient(360px circle at ${spot.x}px ${spot.y}px, rgba(18, 19, 22, 0.045), transparent 70%)`,
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
              Open console
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
            <span className="lp-eyebrow">Research prototype · federated learning</span>
            <h1 id="lp-project-title" className="lp-title">
              Train one AI model across many computers, with less waiting and less CO₂.
            </h1>
            <p className="lp-lead">
              In federated learning, several computers each train a shared model on their own data and
              send only the updates to a coordinator. Normally the coordinator waits for every computer,
              every round. Here, a reinforcement-learning agent learns which computers to pick each round,
              keeping accuracy while cutting time and electricity.
            </p>
            <p className="lp-lead">
              Results below come from a simulation of 4 PCs, with power figures taken from electricity-meter
              readings on 3 of them.
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
                Run the experiments
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
                      {mode === 'rl' ? 'RL picks PCs' : 'All 4 PCs'}
                    </span>
                  </button>
                ))}
              </div>
              <span className={`lp-status-pill ${isRl ? 'is-good' : 'is-warn'}`}>
                {isRl ? '−27% CO₂ · picks PC2 + PC4' : '18.2 g CO₂ · all 4 PCs'}
              </span>
            </div>

            <svg viewBox="0 0 400 246" className="lp-topo-svg" role="img" aria-label="Coordinator client selection topology">
              {/* Links */}
              <line
                x1="200" y1="123" x2="74" y2="50"
                stroke={isRl ? '#D5CFC2' : '#C82D14'}
                strokeWidth={isRl ? 1.5 : 2}
                strokeDasharray={isRl ? '4 4' : undefined}
              />
              <line
                x1="200" y1="123" x2="74" y2="196"
                stroke={isRl ? '#D5CFC2' : '#C82D14'}
                strokeWidth={isRl ? 1.5 : 2}
                strokeDasharray={isRl ? '4 4' : undefined}
              />
              <line x1="200" y1="123" x2="326" y2="50" stroke="#0F766E" strokeWidth="2.2" />
              <line x1="200" y1="123" x2="326" y2="196" stroke="#0F766E" strokeWidth="2.2" />

              {/* Animated packets on active links */}
              <motion.circle
                cx={200}
                cy={123}
                r="4"
                fill="#0F766E"
                animate={{ cx: [200, 326, 200], cy: [123, 50, 123] }}
                transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
              />
              <motion.circle
                cx={326}
                cy={196}
                r="4"
                fill="#121316"
                animate={{ cx: [326, 200, 326], cy: [196, 123, 196] }}
                transition={{ duration: 2.2, delay: 0.45, repeat: Infinity, ease: 'easeInOut' }}
              />
              <motion.circle
                cx={200}
                cy={123}
                r="4"
                fill="#C82D14"
                animate={{
                  cx: [200, 74, 200],
                  cy: [123, 50, 123],
                  opacity: isRl ? 0 : 1,
                }}
                transition={{ duration: 2.8, repeat: Infinity, ease: 'easeInOut' }}
              />
              <motion.circle
                cx={74}
                cy={196}
                r="4"
                fill="#C82D14"
                animate={{
                  cx: [74, 200, 74],
                  cy: [196, 123, 196],
                  opacity: isRl ? 0 : 1,
                }}
                transition={{ duration: 2.4, delay: 0.3, repeat: Infinity, ease: 'easeInOut' }}
              />

              {/* Coordinator */}
              <motion.circle
                cx="200"
                cy="123"
                r="36"
                fill="none"
                stroke={isRl ? '#0F766E' : '#C82D14'}
                strokeOpacity="0.28"
                strokeWidth="1.5"
                animate={{ r: [34, 44, 34], strokeOpacity: [0.35, 0, 0.35] }}
                transition={{ duration: 2.6, repeat: Infinity, ease: 'easeOut' }}
              />
              <circle cx="200" cy="123" r="32" fill="#121316" stroke="#3A3832" strokeWidth="1.5" />
              <text x="200" y="119" textAnchor="middle" fill="#FAF8F3" fontSize="9.5" fontWeight="700">
                Coordinator
              </text>
              <text x="200" y="133" textAnchor="middle" fill="#D5CFC2" fontSize="9" fontFamily="monospace">
                30 W
              </text>

              {/* PC1 */}
              <g>
                <rect
                  x="8" y="22" width="128" height="52" rx="4"
                  fill={isRl ? '#F5F2EB' : '#FDECEC'}
                  stroke={isRl ? '#D5CFC2' : '#C82D14'}
                  strokeWidth={isRl ? 1 : 1.5}
                />
                <text x="18" y="42" fill={isRl ? '#3A3832' : '#121316'} fontSize="11" fontWeight="600">
                  {isRl ? 'PC1 · Left out' : 'PC1 · Slowest'}
                </text>
                <text x="18" y="59" fill={isRl ? '#57534A' : '#C82D14'} fontSize="9.5" fontFamily="monospace">
                  20.53 W · all wait
                </text>
              </g>

              {/* PC3 */}
              <g>
                <rect
                  x="8" y="170" width="128" height="52" rx="4"
                  fill={isRl ? '#F5F2EB' : '#FDECEC'}
                  stroke={isRl ? '#D5CFC2' : '#C82D14'}
                  strokeWidth={isRl ? 1 : 1.5}
                />
                <text x="18" y="190" fill={isRl ? '#3A3832' : '#121316'} fontSize="11" fontWeight="600">
                  {isRl ? 'PC3 · Left out' : 'PC3 · Training'}
                </text>
                <text x="18" y="207" fill="#C82D14" fontSize="9.5" fontFamily="monospace">
                  209.4 W · most CO₂
                </text>
              </g>

              {/* PC2 */}
              <g>
                <rect x="264" y="22" width="128" height="52" rx="4" fill="#CCFBF1" stroke="#0F766E" strokeWidth="1.5" />
                <text x="274" y="42" fill="#121316" fontSize="11" fontWeight="600">PC2 · Picked</text>
                <text x="274" y="59" fill="#0F766E" fontSize="9.5" fontFamily="monospace">86 W · 1.1× faster</text>
              </g>

              {/* PC4 */}
              <g>
                <rect x="264" y="170" width="128" height="52" rx="4" fill="#CCFBF1" stroke="#0F766E" strokeWidth="1.5" />
                <text x="274" y="190" fill="#121316" fontSize="11" fontWeight="600">PC4 · Picked</text>
                <text x="274" y="207" fill="#0F766E" fontSize="9.5" fontFamily="monospace">36 W · 1.35× faster</text>
              </g>
            </svg>
          </SpotlightCard>
        </section>

        {/* 2. Why it is needed */}
        <section className="lp-section" aria-labelledby="lp-need-title">
          <div className="lp-section-head">
            <span className="lp-eyebrow">The problem</span>
            <h2 id="lp-need-title" className="lp-h2">
              When every computer joins every round, the fast ones sit idle waiting for the slowest.
            </h2>
          </div>

          <SpotlightCard className="lp-visual-card">
            <div className="lp-visual-head">
              <span className="lp-visual-label">One round with all 4 PCs (simulated)</span>
              <div className="lp-legend">
                <span className="lp-legend-item"><i className="lp-dot lp-dot-work" /> Training</span>
                <span className="lp-legend-item"><i className="lp-dot lp-dot-idle" /> Idle, waiting</span>
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
                      viewport={{ once: true, amount: 0.05 }}
                      transition={{ ...SPRING_SMOOTH, delay: idx * 0.05 }}
                    />
                    {row.idlePct > 0 && (
                      <motion.div
                        className="lp-bar-idle"
                        initial={{ width: 0 }}
                        whileInView={{ width: `${row.idlePct}%` }}
                        viewport={{ once: true, amount: 0.05 }}
                        transition={{ ...SPRING_SMOOTH, delay: 0.12 + idx * 0.05 }}
                      />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </SpotlightCard>

          <div className="lp-grid-3">
            <SpotlightCard className="lp-card">
              <h3 className="lp-card-title">Everyone waits for the slowest</h3>
              <p className="lp-card-body">
                A round ends only when the slowest computer has finished training and sent its update.
              </p>
            </SpotlightCard>

            <SpotlightCard className="lp-card">
              <h3 className="lp-card-title">Waiting still uses power</h3>
              <p className="lp-card-body">
                Idle computers and the 30 W coordinator keep drawing electricity, and power-hungry PCs add the most CO₂.
              </p>
            </SpotlightCard>

            <SpotlightCard className="lp-card">
              <h3 className="lp-card-title">Simple shortcuts lose accuracy</h3>
              <p className="lp-card-body">
                Each PC holds different data. Always using the 2 fastest (82.2%) or the 2 lowest-power PCs (81.6%)
                sees less variety, so accuracy drops. The RL picker kept 83.4%.
              </p>
            </SpotlightCard>
          </div>
        </section>

        {/* 3. What we have achieved so far */}
        <section className="lp-section" aria-labelledby="lp-achieved-title">
          <div className="lp-section-head">
            <span className="lp-eyebrow">Results so far</span>
            <h2 id="lp-achieved-title" className="lp-h2">
              In simulation: the same accuracy as using every PC, with 27% less CO₂ and 13% less time.
            </h2>
          </div>

          <div className="lp-stats-row">
            <SpotlightCard className="lp-stat">
              <span className="lp-stat-val">83.4%</span>
              <span className="lp-stat-label">Accuracy with RL picking PCs (83.3% using all 4, effectively equal)</span>
            </SpotlightCard>
            <SpotlightCard className="lp-stat">
              <span className="lp-stat-val lp-stat-good">−27.4%</span>
              <span className="lp-stat-label">CO₂ per training run vs. all 4 PCs (13.2 g vs. 18.2 g)</span>
            </SpotlightCard>
            <SpotlightCard className="lp-stat">
              <span className="lp-stat-val">−13.1%</span>
              <span className="lp-stat-label">Total training time vs. all 4 PCs (294 s vs. 339 s)</span>
            </SpotlightCard>
            <SpotlightCard className="lp-stat">
              <span className="lp-stat-val">3 of 4</span>
              <span className="lp-stat-label">PCs with real electricity-meter readings (PC4 is estimated)</span>
            </SpotlightCard>
          </div>

          <div className="lp-charts-grid">
            <SpotlightCard className="lp-visual-card">
              <div className="lp-visual-head">
                <span className="lp-visual-label">Simulated: CO₂ vs. accuracy for 4 ways of picking PCs</span>
                <span className="lp-status-pill is-good">RL ≈ all-4 accuracy, 27% less CO₂</span>
              </div>
              <svg viewBox="0 0 420 215" className="lp-chart-svg" role="img" aria-label="Policy CO2 vs Accuracy chart">
                {[40, 80, 120, 160].map((y) => (
                  <line key={y} x1="48" y1={y} x2="396" y2={y} stroke="#EFECE4" strokeWidth="1" />
                ))}
                <line x1="48" y1="24" x2="48" y2="170" stroke="#D5CFC2" strokeWidth="1.5" />
                <line x1="48" y1="170" x2="396" y2="170" stroke="#D5CFC2" strokeWidth="1.5" />

                <text x="40" y="34" textAnchor="end" fill="#57534A" fontSize="10" fontFamily="monospace">84%</text>
                <text x="40" y="82" textAnchor="end" fill="#57534A" fontSize="10" fontFamily="monospace">83%</text>
                <text x="40" y="130" textAnchor="end" fill="#57534A" fontSize="10" fontFamily="monospace">82%</text>
                <text x="40" y="172" textAnchor="end" fill="#57534A" fontSize="10" fontFamily="monospace">81%</text>

                <text x="60" y="188" textAnchor="middle" fill="#57534A" fontSize="10" fontFamily="monospace">5g</text>
                <text x="168" y="188" textAnchor="middle" fill="#57534A" fontSize="10" fontFamily="monospace">10g</text>
                <text x="276" y="188" textAnchor="middle" fill="#57534A" fontSize="10" fontFamily="monospace">15g</text>
                <text x="384" y="188" textAnchor="middle" fill="#57534A" fontSize="10" fontFamily="monospace">20g CO₂</text>

                <motion.path
                  d="M 89 144 Q 160 112 238 58 L 345 63"
                  fill="none"
                  stroke="#D5CFC2"
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
                        transition={{ ...SPRING_SNAPPY, delay: 0.12 + i * 0.08 }}
                      />
                      <text
                        x={labelPos.x}
                        y={labelPos.y}
                        textAnchor={labelPos.anchor}
                        fill="#121316"
                        fontSize="10"
                        fontWeight={p.highlight ? '700' : '500'}
                        fontFamily="monospace"
                      >
                        {p.label} ({p.acc}%)
                      </text>
                    </g>
                  );
                })}
              </svg>
            </SpotlightCard>

            <SpotlightCard className="lp-visual-card">
              <div className="lp-visual-head">
                <span className="lp-visual-label">Real meter readings (separate CIFAR-10 image test)</span>
                <span className="lp-status-pill is-warn">PC3: 28% faster, +74% CO₂</span>
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
              <h3 className="lp-card-title">Real power measurements</h3>
              <p className="lp-card-body">
                PC2 and PC3 ran the same image-recognition training job (CIFAR-10) on a wall-plug meter. PC3
                finished 28% faster but used 74% more electricity, for only 0.35 points more accuracy. These
                readings set the power figures in the simulation.
              </p>
            </SpotlightCard>

            <SpotlightCard className="lp-card">
              <h3 className="lp-card-title">A working simulation</h3>
              <p className="lp-card-body">
                Python code that gives 4 virtual PCs different slices of data, trains a small classifier on
                synthetic data, and compares the RL picker against simple rules over repeated runs.
              </p>
            </SpotlightCard>

            <SpotlightCard className="lp-card">
              <h3 className="lp-card-title">Try it yourself</h3>
              <p className="lp-card-body">
                Open the console to run your own experiments, replay which PCs were picked each round, and
                see which numbers are measured and which are estimated.
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
