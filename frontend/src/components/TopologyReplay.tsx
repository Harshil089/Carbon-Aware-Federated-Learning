import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { ChevronLeft, ChevronRight, Pause, Play } from 'lucide-react';
import { POLICY_META, SimulationResults } from '../types';

export function TopologyReplay({ results }: { results: SimulationResults | null }) {
  const [policy, setPolicy] = useState('Q_learning_RL');
  const [trialSeed, setTrialSeed] = useState<number>(0);
  const [round, setRound] = useState(1);
  const [playing, setPlaying] = useState(false);

  const seeds = useMemo(() => {
    if (!results) return [];
    return [...new Set(results.rows.map((r) => Number(r.scenario_seed)))].sort((a, b) => a - b);
  }, [results]);

  useEffect(() => {
    if (seeds.length && !seeds.includes(trialSeed)) setTrialSeed(seeds[0]);
  }, [seeds, trialSeed]);

  const maxRounds = results?.parameters.run.rounds ?? 20;

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setRound((prev) => {
        if (prev >= maxRounds) {
          setPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 620);
    return () => window.clearInterval(id);
  }, [playing, maxRounds]);

  if (!results) {
    return <div className="panel">Loading synchronous federated topology replay…</div>;
  }

  const trialRows = results.rows
    .filter((r) => r.policy === policy && Number(r.scenario_seed) === trialSeed)
    .sort((a, b) => Number(a.round) - Number(b.round));

  const row = trialRows.find((r) => Number(r.round) === round) ?? trialRows[0];
  const selected = row ? row.clients_selected.split('+') : [];
  const nodes = results.parameters.infrastructure.clients;
  const baseTrainS = results.parameters.infrastructure.settings.synthetic_pc1_train_seconds_per_round;
  const aggS = results.parameters.infrastructure.settings.aggregation_seconds_per_round;
  const coordW = results.parameters.infrastructure.settings.coordinator_power_w;
  const refMeanW = results.parameters.infrastructure.reference_mean_power_w;

  const pos = [
    { x: 26, y: 24 },
    { x: 424, y: 24 },
    { x: 26, y: 238 },
    { x: 424, y: 238 },
  ];
  const cx = 320;
  const cy = 180;

  return (
    <div>
      <div className="topo-grid">
        <div className="panel">
          <div className="panel-head">
            <div>
              <div className="head-inline">
                <h2>Synchronous Hub-and-Spoke FedAvg Topology</h2>
                <span className="plate-tag num">ROUND {round} / {maxRounds}</span>
              </div>
              <p className="panel-sub">
                Scheduled nodes transmit weight updates in vermilion; unselected nodes remain on standby and draw idle power until round completion.
              </p>
            </div>
          </div>

          <div className="topo-canvas">
            <svg className="topo-svg" viewBox="0 0 640 356" aria-label="Federated client topology schematic">
              <defs>
                <pattern id="topoGrid" width="24" height="24" patternUnits="userSpaceOnUse">
                  <path d="M 24 0 L 0 0 0 24" fill="none" stroke="#EAE4D7" strokeWidth="0.75" />
                </pattern>
              </defs>
              <rect x="0" y="0" width="640" height="356" fill="url(#topoGrid)" />

              {nodes.map((n, i) => {
                const on = selected.includes(n.name);
                const p = pos[i];
                const nx = p.x + 95;
                const ny = p.y + 46;
                const d = `M ${nx} ${ny} Q ${(nx + cx) / 2} ${cy} ${cx} ${cy}`;
                return (
                  <motion.path
                    key={`link-${n.name}`}
                    d={d}
                    fill="none"
                    stroke={on ? '#D9381E' : '#C8C2B4'}
                    strokeWidth={on ? 2.6 : 1.4}
                    strokeDasharray={on ? '7 4' : '3 3'}
                    animate={
                      on
                        ? { strokeDashoffset: [0, -22], opacity: 1 }
                        : { strokeDashoffset: 0, opacity: 0.65 }
                    }
                    transition={
                      on
                        ? { strokeDashoffset: { repeat: Infinity, duration: 0.85, ease: 'linear' }, opacity: { duration: 0.2 } }
                        : { duration: 0.2 }
                    }
                  />
                );
              })}

              {nodes.map((n, i) => {
                const on = selected.includes(n.name);
                const p = pos[i];
                const calCiMap: Record<string, number> = { PC1: 0.675, PC2: 0.52, PC3: 0.82, PC4: 0.42 };
                const ci = (n.grid_kgco2_per_kwh ?? calCiMap[n.name] ?? results.parameters.infrastructure.grid_kg_co2_per_kwh).toFixed(3);
                const activeW = (refMeanW * n.active_power_ratio_to_PC1).toFixed(1);
                return (
                  <motion.g
                    key={n.name}
                    animate={{ scale: on ? 1.015 : 0.99, opacity: on ? 1 : 0.72 }}
                    transition={{ type: 'spring', stiffness: 360, damping: 28 }}
                    style={{ transformOrigin: `${p.x + 95}px ${p.y + 46}px` }}
                  >
                    <rect
                      x={p.x}
                      y={p.y}
                      width={190}
                      height={94}
                      rx={4}
                      fill={on ? '#FFFFFF' : '#F3EFE6'}
                      stroke={on ? '#D9381E' : '#121316'}
                      strokeWidth={on ? 2 : 1}
                    />
                    <text x={p.x + 12} y={p.y + 23} fill="#121316" fontSize="14" fontWeight="700">
                      {n.name} {n.name === 'PC1' ? '· METER' : ''}
                    </text>
                    <rect
                      x={p.x + 122}
                      y={p.y + 9}
                      width={56}
                      height={18}
                      rx={2}
                      fill={on ? '#D9381E' : '#E2DDD1'}
                    />
                    <text
                      x={p.x + 150}
                      y={p.y + 21}
                      textAnchor="middle"
                      fill={on ? '#FFFFFF' : '#3F3D37'}
                      fontSize="10"
                      fontFamily="var(--mono)"
                      fontWeight="700"
                    >
                      {on ? 'ACTIVE' : 'STANDBY'}
                    </text>
                    <text x={p.x + 12} y={p.y + 46} fill="#3F3D37" fontSize="11.5">
                      {n.ram_gb}GB RAM · {n.relative_speed}× speed
                    </text>
                    <text x={p.x + 12} y={p.y + 64} fill="#57534A" fontSize="10.5" fontFamily="var(--mono)">
                      {activeW}W act / {n.idle_power_w_assumed}W idle
                    </text>
                    <text x={p.x + 12} y={p.y + 81} fill="#57534A" fontSize="10.5" fontFamily="var(--mono)">
                      Net {n.network_seconds_assumed}s · CI {ci} kg/kWh
                    </text>
                  </motion.g>
                );
              })}

              <g>
                <rect
                  x={170}
                  y={148}
                  width={300}
                  height={64}
                  rx={4}
                  fill="#121316"
                  stroke="#121316"
                  strokeWidth={1.5}
                />
                <text x={320} y={171} textAnchor="middle" fill="#FAF8F3" fontSize="12.5" fontWeight="700">
                  FedAvg Coordinator ({coordW}W · +{aggS}s)
                </text>
                <text x={320} y={191} textAnchor="middle" fill="#F59E0B" fontSize="11.5" fontFamily="var(--mono)" fontWeight="600">
                  AGGREGATING: {selected.join(' + ') || '—'}
                </text>
              </g>
            </svg>
          </div>

          <div className="replay-bar">
            <button
              type="button"
              className="btn-run"
              onClick={() => {
                if (!playing && round >= maxRounds) setRound(1);
                setPlaying((p) => !p);
              }}
            >
              {playing ? <Pause size={13} /> : <Play size={13} />}
              {playing ? 'Pause' : 'Play Rounds'}
            </button>
            <button
              type="button"
              className="btn-tune"
              aria-label="Previous round"
              onClick={() => {
                setPlaying(false);
                setRound((r) => Math.max(1, r - 1));
              }}
            >
              <ChevronLeft size={14} />
            </button>
            <button
              type="button"
              className="btn-tune"
              aria-label="Next round"
              onClick={() => {
                setPlaying(false);
                setRound((r) => Math.min(maxRounds, r + 1));
              }}
            >
              <ChevronRight size={14} />
            </button>
            <input
              aria-label="Replay round"
              type="range"
              min={1}
              max={maxRounds}
              value={round}
              onChange={(e) => {
                setPlaying(false);
                setRound(Number(e.target.value));
              }}
            />
            <span className="round-readout num">
              R{round} / R{maxRounds}
            </span>
          </div>

          {/* Round Participation Barcode / Matrix Strip */}
          <div className="participation-matrix" aria-label="Round-by-round client participation matrix">
            <div className="pm-head">
              <h3>Client Participation Schedule Matrix (Click any round column)</h3>
              <span className="num">{POLICY_META[policy]?.label ?? policy}</span>
            </div>
            <div className="pm-table-wrap">
              <table className="pm-table">
                <thead>
                  <tr>
                    <th>Node</th>
                    {trialRows.map((tr) => {
                      const rNum = Number(tr.round);
                      return (
                        <th
                          key={rNum}
                          className={rNum === round ? 'pm-active-col num' : 'num'}
                          onClick={() => {
                            setPlaying(false);
                            setRound(rNum);
                          }}
                        >
                          {rNum}
                        </th>
                      );
                    })}
                  </tr>
                </thead>
                <tbody>
                  {nodes.map((n) => (
                    <tr key={n.name}>
                      <td className="num"><strong>{n.name}</strong></td>
                      {trialRows.map((tr) => {
                        const rNum = Number(tr.round);
                        const isPicked = tr.clients_selected.split('+').includes(n.name);
                        return (
                          <td
                            key={`${n.name}-${rNum}`}
                            className={`pm-cell ${isPicked ? 'on' : 'off'} ${rNum === round ? 'curr' : ''}`}
                            onClick={() => {
                              setPlaying(false);
                              setRound(rNum);
                            }}
                            title={`Round ${rNum}: ${n.name} ${isPicked ? 'SELECTED' : 'IDLE'}`}
                          >
                            <span />
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <div>
              <div className="head-inline">
                <h2>Round Telemetry &amp; Node Timing Ledger</h2>
                <span className="plate-tag num">SEED {trialSeed}</span>
              </div>
              <p className="panel-sub">Select policy and performance trial seed to inspect round-level physics.</p>
            </div>
          </div>

          <div className="replay-selectors">
            <div className="drawer-field">
              <label htmlFor="replay-policy">Scheduling Policy</label>
              <select
                id="replay-policy"
                value={policy}
                onChange={(e) => {
                  setPlaying(false);
                  setPolicy(e.target.value);
                }}
              >
                {Object.entries(POLICY_META).map(([k, m]) => (
                  <option key={k} value={k}>
                    {m.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="drawer-field">
              <label htmlFor="replay-trial">Performance Trial</label>
              <select
                id="replay-trial"
                value={trialSeed}
                onChange={(e) => {
                  setPlaying(false);
                  setTrialSeed(Number(e.target.value));
                }}
              >
                {seeds.map((s, i) => (
                  <option key={s} value={s}>
                    Trial {i + 1} · seed {s}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {row && (
            <>
              <div className="rkpi-grid">
                <div className="rkpi-box">
                  <span>Held-Out Test Accuracy</span>
                  <b className="num">{(Number(row.test_accuracy_REAL_MODEL_SYNTHETIC_DATA) * 100).toFixed(2)}%</b>
                </div>
                <div className="rkpi-box">
                  <span>Synchronous Round Latency</span>
                  <b className="num">{Number(row.round_seconds_MODELED).toFixed(2)} s</b>
                </div>
                <div className="rkpi-box">
                  <span>Round Cluster Energy</span>
                  <b className="num">{Number(row.round_Wh_MODELED).toFixed(4)} Wh</b>
                </div>
                <div className="rkpi-box">
                  <span>Round Carbon Emissions</span>
                  <b className="num">{Number(row.round_CO2_g_MODELED).toFixed(4)} g CO₂</b>
                </div>
              </div>

              <div className="node-timing-breakdown">
                <h3>Round {round} Synchronous Node Breakdown (Nominal Timing)</h3>
                <div className="scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Node</th>
                        <th>State</th>
                        <th>Est. Active s</th>
                        <th>Power Profile</th>
                      </tr>
                    </thead>
                    <tbody>
                      {nodes.map((n) => {
                        const isSel = selected.includes(n.name);
                        const nomWork = baseTrainS / n.relative_speed + n.network_seconds_assumed;
                        const actW = (refMeanW * n.active_power_ratio_to_PC1).toFixed(1);
                        return (
                          <tr key={n.name} className={isSel ? 'rl-row' : ''}>
                            <td className="num"><strong>{n.name}</strong></td>
                            <td>
                              <span className={isSel ? 'pareto-badge' : 'dom-badge'}>
                                {isSel ? 'ACTIVE + IDLE TAIL' : '100% STANDBY IDLE'}
                              </span>
                            </td>
                            <td className="num">{isSel ? `${nomWork.toFixed(2)} s` : '0.00 s'}</td>
                            <td className="num">{isSel ? `${actW} W → ${n.idle_power_w_assumed} W` : `${n.idle_power_w_assumed} W idle`}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
