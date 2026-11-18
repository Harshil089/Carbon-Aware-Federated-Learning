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
  { name: 'LowestEnergy2', label: 'Lowest-power 2', co2: 6.346, acc: 82.2, color: '#B45309', highlight: false },
  { name: 'Fastest2', label: 'Fastest 2', co2: 12.063, acc: 82.3, color: '#1D4ED8', highlight: false },
  { name: 'Q_learning_RL', label: 'RL picker', co2: 10.242, acc: 82.6, color: '#0F766E', highlight: true },
  { name: 'FedAvg_all4', label: 'All 4 (FedAvg)', co2: 18.196, acc: 83.9, color: '#C82D14', highlight: false },
];

const chartX = (co2: number) => 60 + ((co2 - 5) / 15) * 324;
const chartY = (acc: number) => 170 - ((acc - 81) / 3) * 140;
const POLICY_PATH = [...POLICY_POINTS]
  .sort((a, b) => a.co2 - b.co2)
  .map((p, i) => `${i === 0 ? 'M' : 'L'} ${chartX(p.co2).toFixed(1)} ${chartY(p.acc).toFixed(1)}`)
  .join(' ');

// Hero diagram: link end points (lx, ly) are node box centres; timings match the original packets
const NODES_VIEW = [
  { id: 'PC1', x: 8, y: 22, lx: 74, ly: 50, dur: 2.8, delay: 0, watts: '20.53 W', speed: '1.0×', allTag: 'Slowest', allSub: '20.53 W · all wait' },
  { id: 'PC2', x: 264, y: 22, lx: 326, ly: 50, dur: 2.2, delay: 0, watts: '86 W', speed: '1.1×', allTag: 'Training', allSub: '86 W · 1.1× faster' },
  { id: 'PC3', x: 8, y: 170, lx: 74, ly: 196, dur: 2.4, delay: 0.3, watts: '209.4 W', speed: '1.54×', allTag: 'Training', allSub: '209.4 W · most CO₂' },
  { id: 'PC4', x: 264, y: 170, lx: 326, ly: 196, dur: 2.2, delay: 0.45, watts: '36 W', speed: '1.35×', allTag: 'Training', allSub: '36 W · 1.35× faster' },
];
// Most frequent RL pick: PC1 + PC4 in 29 of 100 simulated rounds
const RL_PICKED = ['PC1', 'PC4'];

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
              cutting electricity and time at a small cost in accuracy.
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
                {isRl ? '−44% CO₂ · −1.3 pts accuracy' : '18.2 g CO₂ · all 4 PCs'}
              </span>
            </div>

            <svg viewBox="0 0 400 246" className="lp-topo-svg" role="img" aria-label="Coordinator client selection topology">
              {/* Links and animated packets, one per PC */}
              {NODES_VIEW.map((n) => {
                const picked = RL_PICKED.includes(n.id);
                const off = isRl && !picked;
                return (
                  <g key={n.id}>
                    <line
                      x1="200" y1="123" x2={n.lx} y2={n.ly}
                      stroke={!isRl ? '#C82D14' : picked ? '#0F766E' : '#D5CFC2'}
                      strokeWidth={!isRl ? 2 : picked ? 2.2 : 1.5}
                      strokeDasharray={off ? '4 4' : undefined}
                    />
                    <motion.circle
                      cx={200}
                      cy={123}
                      r="4"
                      fill={!isRl ? '#C82D14' : n.id === 'PC4' ? '#121316' : '#0F766E'}
                      animate={{ cx: [200, n.lx, 200], cy: [123, n.ly, 123], opacity: off ? 0 : 1 }}
                      transition={{ duration: n.dur, delay: n.delay, repeat: Infinity, ease: 'easeInOut' }}
                    />
                  </g>
                );
              })}

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

              {/* PC boxes */}
              {NODES_VIEW.map((n) => {
                const picked = RL_PICKED.includes(n.id);
                const dim = isRl && !picked;
                const title = !isRl ? `${n.id} · ${n.allTag}` : `${n.id} · ${picked ? 'Picked' : 'Left out'}`;
                const sub = !isRl ? n.allSub : picked ? `${n.watts} · ${n.speed}` : `${n.watts} · idle`;
                return (
                  <g key={n.id}>
                    <rect
                      x={n.x} y={n.y} width="128" height="52" rx="4"
                      fill={!isRl ? '#FDECEC' : picked ? '#CCFBF1' : '#F5F2EB'}
                      stroke={!isRl ? '#C82D14' : picked ? '#0F766E' : '#D5CFC2'}
                      strokeWidth={dim ? 1 : 1.5}
                    />
                    <text x={n.x + 10} y={n.y + 20} fill={dim ? '#3A3832' : '#121316'} fontSize="11" fontWeight="600">
                      {title}
                    </text>
                    <text
                      x={n.x + 10} y={n.y + 37}
                      fill={!isRl ? '#C82D14' : picked ? '#0F766E' : '#57534A'}
                      fontSize="9.5" fontFamily="monospace"
                    >
                      {sub}
                    </text>
                  </g>
                );
              })}
            </svg>
            <p className="lp-card-body">
              {isRl
                ? 'Most common RL choice (29% of simulated rounds). The picker changes its pair as training progresses.'
                : 'Every PC trains every round; the slowest PC sets the round time.'}
            </p>
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
                Each PC holds different data. Always using the 2 fastest (82.3%) or the 2 lowest-power PCs (82.2%)
                sees less variety, so accuracy drops. The RL picker reached 82.6%; using all 4 PCs gave 83.9%.
              </p>
            </SpotlightCard>
          </div>
        </section>

        {/* 3. What we have achieved so far */}
        <section className="lp-section" aria-labelledby="lp-achieved-title">
          <div className="lp-section-head">
            <span className="lp-eyebrow">Results so far</span>
            <h2 id="lp-achieved-title" className="lp-h2">
              In simulation: 44% less CO₂ and 8% less time than using every PC, at a small accuracy cost (about 1.3 points).
            </h2>
          </div>

          <div className="lp-stats-row">
            <SpotlightCard className="lp-stat">
              <span className="lp-stat-val">82.6%</span>
              <span className="lp-stat-label">Accuracy with RL picking PCs (83.9% using all 4; about 1.3 points lower)</span>
            </SpotlightCard>
            <SpotlightCard className="lp-stat">
              <span className="lp-stat-val lp-stat-good">−43.7%</span>
              <span className="lp-stat-label">CO₂ per training run vs. all 4 PCs (10.2 g vs. 18.2 g)</span>
            </SpotlightCard>
            <SpotlightCard className="lp-stat">
              <span className="lp-stat-val">−8.4%</span>
              <span className="lp-stat-label">Total training time vs. all 4 PCs (310 s vs. 339 s)</span>
            </SpotlightCard>
            <SpotlightCard className="lp-stat">
              <span className="lp-stat-val">3 of 4</span>
              <span className="lp-stat-label">PCs with user-reported electricity-meter readings (PC4 is estimated)</span>
            </SpotlightCard>
          </div>

          <div className="lp-charts-grid">
            <SpotlightCard className="lp-visual-card">
              <div className="lp-visual-head">
                <span className="lp-visual-label">Simulated: CO₂ vs. accuracy for 4 ways of picking PCs</span>
                <span className="lp-status-pill is-good">RL: −44% CO₂, about 1.3 points less accurate than all 4</span>
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
                  d={POLICY_PATH}
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
                  const cx = chartX(p.co2);
                  const cy = chartY(p.acc);
                  const labelPos =
                    p.name === 'LowestEnergy2'
                      ? { x: cx, y: cy + 20, anchor: 'start' as const }
                      : p.name === 'Fastest2'
                        ? { x: cx + 10, y: cy + 4, anchor: 'start' as const }
                        : p.name === 'Q_learning_RL'
                          ? { x: cx - 14, y: cy - 16, anchor: 'end' as const }
                          : { x: cx - 10, y: cy - 8, anchor: 'end' as const };
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
