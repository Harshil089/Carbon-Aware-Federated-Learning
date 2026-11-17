import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Play } from 'lucide-react';
import { PipelineConfig, PipelineResults } from '../types';
import { MorlSweepChart } from './Charts';

export function PipelineStudio({
  data,
  offline,
  onRun,
}: {
  data: PipelineResults | null;
  offline: boolean;
  onRun: (cfg: PipelineConfig) => Promise<void>;
}) {
  const [mode, setMode] = useState<'dirichlet' | 'iid'>('dirichlet');
  const [alpha, setAlpha] = useState(0.5);
  const [rounds, setRounds] = useState(15);
  const [qEp, setQEp] = useState(35);
  const [mEp, setMEp] = useState(20);
  const [useCal, setUseCal] = useState(true);
  const [running, setRunning] = useState(false);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [error, setError] = useState('');

  async function handleRun() {
    if (offline || running) return;
    setError('');
    setRunning(true);
    try {
      await onRun({
        mode,
        alpha,
        rounds,
        q_episodes: qEp,
        morl_episodes: mEp,
        use_calibrated_telemetry: useCal,
        seed: 42,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Pipeline execution failed');
    } finally {
      setRunning(false);
    }
  }

  const c = data?.config;
  const cl = data?.clients ?? [];
  const pol = data?.policies ?? [];
  const morl = data?.morl_sweep ?? [];

  const layers =
    data && c
      ? [
          {
            n: 'L1',
            module: 'fl_pipeline/data.py',
            t: 'Data & Partitions',
            s: `${c.mode.toUpperCase()} · α=${c.alpha}`,
            d: `${cl.reduce((a, x) => a + x.num_samples, 0)} synthetic Gaussian training samples across ${cl.length} non-empty partitions (seed ${c.seed}).`,
            detail: cl.map((x) => `${x.name}: ${x.num_samples} samples`).join(' · '),
          },
          {
            n: 'L2',
            module: 'fl_pipeline/clients.py',
            t: 'Calibrated Clients',
            s: c.use_calibrated_telemetry ? '10k Telemetry CSV' : 'Config Baseline',
            d: cl
              .map(
                (x) =>
                  `${x.name} (${x.ram_gb}GB, ${x.relative_speed.toFixed(2)}×, ${x.active_power_w.toFixed(1)}W, ${x.grid_kgco2_per_kwh.toFixed(3)} kg/kWh)`,
              )
              .join(' · '),
            detail: 'Numerically stable full-batch NumPy softmax gradient descent on each client partition.',
          },
          {
            n: 'L3',
            module: 'fl_pipeline/network_energy.py',
            t: 'Network & Latency',
            s: `Parallel Max · ${c.rounds}r`,
            d: `Synchronous round latency = max(T_download + T_train + T_upload) + T_agg across selected clients.`,
            detail: `Mean cluster uplink/downlink bandwidth: ${(
              cl.reduce((a, x) => a + x.bandwidth_mbps, 0) / Math.max(1, cl.length)
            ).toFixed(1)} Mbps.`,
          },
          {
            n: 'L4',
            module: 'fl_pipeline/network_energy.py',
            t: 'Energy & Carbon',
            s: 'Per-Client Grid CI',
            d: "Active work + idle waiting tail + 30W coordinator Wh multiplied by each client's specific grid carbon intensity.",
            detail: `PC1 anchored to physical meter (0.065 kWh / 190m); PC2–PC4 calibrated via scenario ratios.`,
          },
          {
            n: 'L5',
            module: 'fl_pipeline/fedavg.py',
            t: 'FedAvg Aggregation',
            s: 'Sample-Weighted',
            d: 'Global classifier weights averaged proportionally to participating client sample counts (∑ w_k = 1.0).',
            detail: 'Exact sample-weighted convex combination verified by unit tests.',
          },
          {
            n: 'L6',
            module: 'fl_pipeline/schedulers.py',
            t: '7 Schedulers',
            s: `${pol.length} Policies Evaluated`,
            d: `Evaluates Random, Round-Robin, Fastest, Lowest Carbon, Weighted, Fairness Gini, and Q-Learning.`,
            detail: `Pareto-optimal policies in current run: ${data.pareto_policies.join(', ') || 'none'}.`,
          },
          {
            n: 'L7',
            module: 'fl_pipeline/rl_env.py',
            t: 'RL / MORL Env',
            s: `${c.q_episodes} Q-Ep · ${c.morl_episodes} MORL`,
            d: `Gym-style FederatedRLEnv with 15-D state observation vector and 4-D vector reward.`,
            detail: `Sweeps ${morl.length} linear scalarization preference weights from latency-priority to carbon-priority.`,
          },
          {
            n: 'L8',
            module: 'fl_pipeline/schedulers.py',
            t: 'Pareto & Fairness',
            s: `${data.pareto_morl.length} Nondominated`,
            d: 'Computes participation Gini fairness index and extracts non-dominated 3D (Accuracy, Latency, Carbon) Pareto frontier.',
            detail: `Nondominated MORL preferences: ${data.pareto_morl.join(', ') || 'none'}.`,
          },
        ]
      : [];

  return (
    <div>
      <div className="pipe-bar">
        <div className="pipe-controls">
          <div className="chip-field">
            <label htmlFor="pipe-mode">Partition</label>
            <select
              id="pipe-mode"
              className="pipe-select"
              aria-label="Partition mode"
              data-tip="Selects IID or non-IID Dirichlet data partitioning across clients. Passed as cfg.mode to fl_pipeline/data.py to split synthetic Gaussian samples."
              value={mode}
              onChange={(e) => setMode(e.target.value as 'dirichlet' | 'iid')}
            >
              <option value="dirichlet">Dirichlet (Non-IID)</option>
              <option value="iid">IID Uniform</option>
            </select>
          </div>
          <div className="chip-field">
            <label htmlFor="pipe-alpha">Alpha α</label>
            <input
              id="pipe-alpha"
              type="number"
              min={0.05}
              max={50}
              step={0.1}
              data-tip="Sets the Dirichlet concentration α (lower means more heterogeneous client data). Sent as cfg.alpha to fl_pipeline/data.py when non-IID mode is active."
              value={alpha}
              disabled={mode === 'iid'}
              onChange={(e) => setAlpha(Number(e.target.value))}
            />
          </div>
          <div className="chip-field">
            <label htmlFor="pipe-rounds">Rounds</label>
            <input
              id="pipe-rounds"
              type="number"
              min={2}
              max={50}
              step={1}
              data-tip="Controls how many synchronous FedAvg communication rounds each policy runs. Passed as cfg.rounds to the backend simulator and RL environment."
              value={rounds}
              onChange={(e) => setRounds(Number(e.target.value))}
            />
          </div>
          <div className="chip-field">
            <label htmlFor="pipe-qep">Q-Ep</label>
            <input
              id="pipe-qep"
              type="number"
              min={5}
              max={200}
              step={5}
              data-tip="Sets the number of training episodes for the tabular Q-Learning scheduler. Sent as cfg.q_episodes to fl_pipeline/schedulers.py."
              value={qEp}
              onChange={(e) => setQEp(Number(e.target.value))}
            />
          </div>
          <div className="chip-field">
            <label htmlFor="pipe-mep">MORL-Ep</label>
            <input
              id="pipe-mep"
              type="number"
              min={5}
              max={100}
              step={5}
              data-tip="Sets episodes per preference vector in the MORL scalarization sweep. Sent as cfg.morl_episodes to fl_pipeline/rl_env.py."
              value={mEp}
              onChange={(e) => setMEp(Number(e.target.value))}
            />
          </div>
          <label
            className="telemetry-toggle"
            data-tip="Toggles between 10,000-row calibrated hardware telemetry and static config defaults. Sets cfg.use_calibrated_telemetry for fl_pipeline/clients.py."
          >
            <input
              type="checkbox"
              checked={useCal}
              onChange={(e) => setUseCal(e.target.checked)}
            />
            <span>10k Calibrated Telemetry CSV</span>
          </label>
        </div>

        <motion.button
          type="button"
          className="btn-run"
          disabled={offline || running}
          whileTap={{ scale: 0.98 }}
          data-tip="Executes a live 8-layer FL and MORL sweep with the current parameters. Posts PipelineConfig to the backend /api/run_pipeline endpoint."
          onClick={handleRun}
        >
          <Play size={13} />
          {running ? 'Executing Pipeline…' : 'Run 8-Layer Pipeline'}
        </motion.button>
      </div>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      <div className="panel">
        <div className="panel-head">
          <div>
            <div className="head-inline">
              <h2>8-Layer Modular FL + MORL Schematic</h2>
              <span className="plate-tag num">PACKAGE: fl_pipeline/</span>
            </div>
            <p className="panel-sub">
              {offline
                ? 'Offline archive mode: launch START_DEMO.sh or demo_server.py to execute live 8-layer sweeps.'
                : 'Select any architectural layer (L1–L8) below to inspect its module binding and live telemetry output.'}
            </p>
          </div>
        </div>

        <div className="pipe-flow" role="tablist" aria-label="8-layer pipeline modules">
          {layers.map((l, i) => {
            const isSelected = i === selectedLayer;
            return (
              <button
                key={l.n}
                type="button"
                role="tab"
                aria-selected={isSelected}
                className={`layer-node ${isSelected ? 'active' : ''}`}
                data-tip={`Selects architectural layer ${l.n} (${l.t}) to inspect its live telemetry. Maps to backend module ${l.module}.`}
                onClick={() => setSelectedLayer(i)}
              >
                {isSelected && (
                  <motion.div
                    layoutId="layer-active-ring"
                    className="layer-node-ring"
                    transition={{ type: 'spring', stiffness: 420, damping: 32 }}
                  />
                )}
                <div className="layer-top">
                  <span className="layer-title">{l.t}</span>
                  <span className="layer-num num">{l.n}</span>
                </div>
                <span className="layer-stat num">{l.s}</span>
                <span className="layer-mod num">{l.module}</span>
              </button>
            );
          })}
        </div>

        <AnimatePresence mode="wait">
          {layers[selectedLayer] && (
            <motion.div
              key={layers[selectedLayer].n}
              className="pipe-inspector"
              data-tip={`Displays detailed diagnostics for ${layers[selectedLayer].n} (${layers[selectedLayer].t}). Summarizes live output from ${layers[selectedLayer].module}.`}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.18 }}
            >
              <div className="inspector-head">
                <h3>
                  {layers[selectedLayer].n} — {layers[selectedLayer].t}
                </h3>
                <code className="num">{layers[selectedLayer].module}</code>
              </div>
              <p className="inspector-main">{layers[selectedLayer].d}</p>
              <p className="inspector-sub num">{layers[selectedLayer].detail}</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="stage-grid">
        <div className="panel">
          <div className="panel-head">
            <div>
              <div className="head-inline">
                <h2>7-Scheduler Evaluation &amp; Participation Fairness</h2>
                <span className="plate-tag num">LAYER 6 &amp; 8</span>
              </div>
              <p className="panel-sub">
                Compares final accuracy, total latency, total CO₂, client participation counts [PC1..PC4], and Gini index.
              </p>
            </div>
          </div>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th data-tip="Displays the client-selection scheduler name. Corresponds to policy implementations in fl_pipeline/schedulers.py.">Policy</th>
                  <th data-tip="Shows final global model test accuracy after all rounds. Computed by evaluating aggregated FedAvg weights on held-out test data.">Accuracy</th>
                  <th data-tip="Shows cumulative synchronous wall-clock latency across all rounds. Summed from per-round max client compute+network plus aggregation time.">Latency</th>
                  <th data-tip="Shows total cluster carbon emissions in grams of CO2. Computed from active, idle-tail, and coordinator energy weighted by client grid intensity.">CO₂</th>
                  <th data-tip="Lists how many rounds each client [PC1..PC4] was selected. Tracked by the scheduler loop in fl_pipeline/schedulers.py.">Client Counts</th>
                  <th data-tip="Shows the Gini coefficient of client participation (0 = equal, 1 = skewed). Calculated in Layer 8 to quantify scheduling fairness.">Gini</th>
                  <th data-tip="Indicates whether the policy is non-dominated in (Accuracy, Latency, CO2). Determined by 3D Pareto frontier extraction in Layer 8.">Status</th>
                </tr>
              </thead>
              <tbody>
                {pol.map((p) => (
                  <tr
                    key={p.policy_name}
                    className={p.pareto_optimal ? 'rl-row' : ''}
                    data-tip={`${p.policy_name}: ${(p.final_accuracy * 100).toFixed(1)}% accuracy, ${p.total_latency_s.toFixed(1)}s latency, ${p.total_co2_g.toFixed(2)}g CO2. ${p.pareto_optimal ? 'Pareto-optimal' : 'Dominated'} in Layer 6/8 evaluation.`}
                  >
                    <td><strong>{p.policy_name}</strong></td>
                    <td className="num">{(p.final_accuracy * 100).toFixed(1)}%</td>
                    <td className="num">{p.total_latency_s.toFixed(1)} s</td>
                    <td className="num">{p.total_co2_g.toFixed(2)} g</td>
                    <td className="num">[{p.participation_counts.join(', ')}]</td>
                    <td className="num">{p.participation_gini.toFixed(3)}</td>
                    <td>
                      {p.pareto_optimal ? (
                        <span className="pareto-badge">PARETO</span>
                      ) : (
                        <span className="dom-badge">DOMINATED</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <div>
              <div className="head-inline">
                <h2>MORL Preference Scalarization Sweep</h2>
                <span className="plate-tag num">LAYER 7 &amp; 8</span>
              </div>
              <p className="panel-sub">
                Traces non-dominated solutions across multi-objective weight vectors (w_acc, w_lat, w_co2, w_fair).
              </p>
            </div>
          </div>
          <div className="svg-stage">
            <MorlSweepChart morl={morl} />
          </div>
          {morl.length > 0 && (
            <div className="scroll" style={{ marginTop: 12 }}>
              <table>
                <thead>
                  <tr>
                    <th data-tip="Names the MORL scalarization preference weight vector. Passed to FederatedRLEnv in fl_pipeline/rl_env.py.">Preference Profile</th>
                    <th data-tip="Shows final global accuracy achieved under this preference vector. Evaluated on the held-out test split after training.">Accuracy</th>
                    <th data-tip="Shows total synchronous round latency for this MORL policy. Aggregated across all rounds in the evaluation episode.">Latency</th>
                    <th data-tip="Shows total cluster CO2 emissions for this MORL policy. Modeled from per-client energy and grid carbon intensity.">CO₂</th>
                    <th data-tip="Shows participation Gini fairness index for this MORL policy. Computed by Layer 8 from client selection counts.">Gini</th>
                    <th data-tip="Marks whether this preference profile lies on the non-dominated 3D Pareto frontier. Extracted by Layer 8.">Frontier</th>
                  </tr>
                </thead>
                <tbody>
                  {morl.map((m) => (
                    <tr
                      key={m.preference_name}
                      className={m.pareto_optimal ? 'rl-row' : ''}
                      data-tip={`MORL ${m.preference_name}: ${(m.final_accuracy * 100).toFixed(1)}% accuracy, ${m.total_latency_s.toFixed(1)}s, ${m.total_co2_g.toFixed(2)}g CO2. ${m.pareto_optimal ? 'Nondominated Pareto point' : 'Dominated point'} from fl_pipeline/rl_env.py.`}
                    >
                      <td><strong>{m.preference_name}</strong></td>
                      <td className="num">{(m.final_accuracy * 100).toFixed(1)}%</td>
                      <td className="num">{m.total_latency_s.toFixed(1)} s</td>
                      <td className="num">{m.total_co2_g.toFixed(2)} g</td>
                      <td className="num">{m.participation_gini.toFixed(3)}</td>
                      <td>{m.pareto_optimal ? <span className="pareto-badge">PARETO</span> : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Calibrated 4-Client Telemetry Strip from Pipeline */}
      {cl.length > 0 && (
        <div className="panel">
          <div className="panel-head">
            <div>
              <div className="head-inline">
                <h2>Layer 1 &amp; 2 Client Partition &amp; Telemetry Calibration</h2>
                <span className="plate-tag num">{c?.use_calibrated_telemetry ? '10,000-ROW CSV CALIBRATED' : 'CONFIG BASELINE'}</span>
              </div>
              <p className="panel-sub">
                Sample distribution and calibrated hardware/carbon parameters fed into Layers 3–8.
              </p>
            </div>
          </div>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th data-tip="Identifies the federated client node (PC1–PC4). Instantiated as a calibrated client object in fl_pipeline/clients.py.">Client</th>
                  <th data-tip="Shows local training sample count n_k assigned to this client. Partitioned by fl_pipeline/data.py and used for FedAvg weighting.">Samples (n_k)</th>
                  <th data-tip="Displays client system RAM and GPU VRAM capacity. Loaded from hardware telemetry or config in fl_pipeline/clients.py.">RAM / GPU</th>
                  <th data-tip="Shows relative compute speed multiplier compared to PC1. Scales local training duration in fl_pipeline/network_energy.py.">Speed Ratio</th>
                  <th data-tip="Displays active training power and idle standby power in watts. Used by Layer 4 to compute active and idle-tail energy.">Active / Idle W</th>
                  <th data-tip="Shows client network link bandwidth in Mbps. Determines model download and upload transfer times in Layer 3.">Bandwidth</th>
                  <th data-tip="Shows regional grid carbon intensity in kg CO2 per kWh. Multiplied by client Wh in Layer 4 to compute emissions.">Grid Carbon Intensity</th>
                  <th data-tip="Indicates whether parameters come from the PC1 physical meter or calibrated scenario ratios. Set in fl_pipeline/clients.py.">Provenance</th>
                </tr>
              </thead>
              <tbody>
                {cl.map((node) => (
                  <tr
                    key={node.name}
                    className={node.name === 'PC1' ? 'rl-row' : ''}
                    data-tip={`${node.name}: ${node.num_samples} samples, ${node.relative_speed.toFixed(2)}x speed, ${node.grid_kgco2_per_kwh.toFixed(3)} kg CO2/kWh. ${node.name === 'PC1' ? 'Physical meter anchor' : 'Calibrated scenario'} in Layer 1 & 2.`}
                  >
                    <td className="num"><strong>{node.name}</strong></td>
                    <td className="num">{node.num_samples}</td>
                    <td className="num">{node.ram_gb} GB / {node.gpu_vram_gb} GB</td>
                    <td className="num">{node.relative_speed.toFixed(2)}×</td>
                    <td className="num">{node.active_power_w.toFixed(1)} W / {node.idle_power_w.toFixed(1)} W</td>
                    <td className="num">{node.bandwidth_mbps.toFixed(1)} Mbps</td>
                    <td className="num">{node.grid_kgco2_per_kwh.toFixed(3)} kg CO₂/kWh</td>
                    <td>
                      <span className={node.name === 'PC1' ? 'stamp-measured' : 'stamp-modeled'}>
                        {node.name === 'PC1' ? 'PC1 METER ANCHOR' : 'CALIBRATED SCENARIO'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
