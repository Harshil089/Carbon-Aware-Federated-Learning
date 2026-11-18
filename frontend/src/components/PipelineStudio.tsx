import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Play } from 'lucide-react';
import { PipelineConfig, PipelineResults } from '../types';
import { MorlSweepChart } from './Charts';

const PROVENANCE_LABEL: Record<string, string> = {
  SYNTHETIC_GAUSSIAN_CLUSTERS: 'SYNTHETIC CLUSTERS · NUMPY MODEL',
  TORCHVISION_MNIST: 'REAL MNIST · PYTORCH MODEL',
  SYNTHETIC_28x28_FALLBACK: 'SYNTHETIC 28×28 IMAGES (MNIST NOT FOUND) · PYTORCH MODEL',
};

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
            t: 'Split data',
            s: `${c.mode.toUpperCase()} · α=${c.alpha}`,
            d: `${cl.reduce((a, x) => a + x.num_samples, 0)} synthetic training samples split across ${cl.length} PCs (seed ${c.seed}).`,
            detail: cl.map((x) => `${x.name}: ${x.num_samples} samples`).join(' · '),
          },
          {
            n: 'L2',
            module: 'fl_pipeline/clients.py',
            t: 'PC hardware',
            s: c.use_calibrated_telemetry ? 'Generated 10k-row profile' : 'Fixed config values',
            d: cl
              .map(
                (x) =>
                  `${x.name} (${x.ram_gb}GB, ${x.relative_speed.toFixed(2)}×, ${x.active_power_w.toFixed(1)}W, ${x.grid_kgco2_per_kwh.toFixed(3)} kg/kWh)`,
              )
              .join(' · '),
            detail: 'Each PC trains a small classifier on its own data.',
          },
          {
            n: 'L3',
            module: 'fl_pipeline/network_energy.py',
            t: 'Round time',
            s: `${c.rounds} rounds`,
            d: `Round time = slowest picked PC (download + train + upload) + time to combine updates.`,
            detail: `Average network speed: ${(
              cl.reduce((a, x) => a + x.bandwidth_mbps, 0) / Math.max(1, cl.length)
            ).toFixed(1)} Mbps.`,
          },
          {
            n: 'L4',
            module: 'fl_pipeline/network_energy.py',
            t: 'Energy and CO₂',
            s: 'Per-PC grid CO₂',
            d: "Training + idle waiting + 30 W coordinator electricity, times each PC's grid CO₂ per kWh.",
            detail: `Training power metered on PC1–PC3; PC4 estimated. Idle power and network are estimates.`,
          },
          {
            n: 'L5',
            module: 'fl_pipeline/fedavg.py',
            t: 'Combine updates',
            s: 'FedAvg',
            d: 'The coordinator averages the picked PCs’ models, giving more weight to PCs with more data.',
            detail: 'Checked by unit tests.',
          },
          {
            n: 'L6',
            module: 'fl_pipeline/schedulers.py',
            t: '7 picking rules',
            s: `${pol.length} rules compared`,
            d: `Random, take turns, fastest, lowest CO₂, weighted mix, fairest, and an RL picker.`,
            detail: `Best trade-off rules in this run: ${data.pareto_policies.join(', ') || 'none'}.`,
          },
          {
            n: 'L7',
            module: 'fl_pipeline/rl_env.py',
            t: 'RL with different goals',
            s: `${c.q_episodes} + ${c.morl_episodes} practice runs`,
            d: `RL agents score each round on accuracy, time, CO₂ and fairness.`,
            detail: `Tries ${morl.length} goal mixes, from "time matters most" to "CO₂ matters most".`,
          },
          {
            n: 'L8',
            module: 'fl_pipeline/schedulers.py',
            t: 'Best trade-offs',
            s: `${data.pareto_morl.length} best mixes`,
            d: 'Finds results no other result beats on accuracy, time and CO₂ at once, and scores how evenly PCs were used.',
            detail: `Best-trade-off goal mixes: ${data.pareto_morl.join(', ') || 'none'}.`,
          },
        ]
      : [];

  return (
    <div>
      <div className="pipe-bar">
        <div className="pipe-controls">
          <div className="chip-field">
            <label htmlFor="pipe-mode">Data split</label>
            <select
              id="pipe-mode"
              className="pipe-select"
              aria-label="Partition mode"
              data-tip="Give each PC a different mix of data (uneven, more realistic) or the same mix (even)."
              value={mode}
              onChange={(e) => setMode(e.target.value as 'dirichlet' | 'iid')}
            >
              <option value="dirichlet">Uneven per PC</option>
              <option value="iid">Even</option>
            </select>
          </div>
          <div className="chip-field">
            <label htmlFor="pipe-alpha">Unevenness α</label>
            <input
              id="pipe-alpha"
              type="number"
              min={0.05}
              max={50}
              step={0.1}
              data-tip="Lower means each PC's data is more different from the others. Used only with an uneven split."
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
              data-tip="How many training rounds each rule runs."
              value={rounds}
              onChange={(e) => setRounds(Number(e.target.value))}
            />
          </div>
          <div className="chip-field">
            <label htmlFor="pipe-qep">RL practice runs</label>
            <input
              id="pipe-qep"
              type="number"
              min={5}
              max={200}
              step={5}
              data-tip="How many practice runs the RL picker gets before it is compared."
              value={qEp}
              onChange={(e) => setQEp(Number(e.target.value))}
            />
          </div>
          <div className="chip-field">
            <label htmlFor="pipe-mep">Runs per goal mix</label>
            <input
              id="pipe-mep"
              type="number"
              min={5}
              max={100}
              step={5}
              data-tip="Practice runs for each goal mix (how much accuracy, time, CO₂ and fairness matter)."
              value={mEp}
              onChange={(e) => setMEp(Number(e.target.value))}
            />
          </div>
          <label
            className="telemetry-toggle"
            data-tip="Use a generated 10,000-row hardware profile (based on the meter readings, with random variation) instead of fixed config values. These rows are generated, not recorded."
          >
            <input
              type="checkbox"
              checked={useCal}
              onChange={(e) => setUseCal(e.target.checked)}
            />
            <span>Use generated 10k-row hardware profile</span>
          </label>
        </div>

        <motion.button
          type="button"
          className="btn-run"
          disabled={offline || running}
          whileTap={{ scale: 0.98 }}
          data-tip="Run the extended simulator with these settings. Needs the simulation server running."
          onClick={handleRun}
        >
          <Play size={13} />
          {running ? 'Running…' : 'Run extended simulator'}
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
              <h2>How it works, step by step</h2>
              <span className="plate-tag num">8 STEPS</span>
            </div>
            <p className="panel-sub">
              {offline
                ? 'Saved results only. Start the simulation server (START_DEMO.sh or START_DEMO.bat) to run this simulator.'
                : data
                  ? 'Click a step to see what it does and its output from the last run.'
                  : 'No results yet. Click "Run extended simulator" above.'}
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
                data-tip={`Step ${l.n}: ${l.t}.`}
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
              data-tip={`Details for step ${layers[selectedLayer].n}: ${layers[selectedLayer].t}.`}
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
                <h2>7 picking rules compared</h2>
                <span className="plate-tag num">STEPS 6 &amp; 8</span>
              </div>
              <p className="panel-sub">
                Final accuracy, total time, total CO₂, how often each PC was picked, and how unevenly PCs were used.
              </p>
            </div>
          </div>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th data-tip="The rule for picking PCs each round.">Rule</th>
                  <th data-tip="Final accuracy on test data the model never trained on.">Accuracy</th>
                  <th data-tip="Total training time over all rounds (simulated).">Time</th>
                  <th data-tip="Total CO₂ over all rounds, in grams (simulated).">CO₂</th>
                  <th data-tip="How many rounds PC1, PC2, PC3 and PC4 were each picked.">Times picked</th>
                  <th data-tip="How unevenly PCs were used: 0 = all equally, 1 = always the same PC (Gini index).">Unevenness</th>
                  <th data-tip="Best trade-off: no other rule is better on accuracy, time and CO₂ at once.">Trade-off</th>
                </tr>
              </thead>
              <tbody>
                {pol.map((p) => (
                  <tr
                    key={p.policy_name}
                    className={p.pareto_optimal ? 'rl-row' : ''}
                    data-tip={`${p.policy_name}: ${(p.final_accuracy * 100).toFixed(1)}% accuracy, ${p.total_latency_s.toFixed(1)}s latency, ${p.total_co2_g.toFixed(2)}g CO2. ${p.pareto_optimal ? 'Best trade-off' : 'Outperformed'} in Layer 6/8 evaluation.`}
                  >
                    <td><strong>{p.policy_name}</strong></td>
                    <td className="num">{(p.final_accuracy * 100).toFixed(1)}%</td>
                    <td className="num">{p.total_latency_s.toFixed(1)} s</td>
                    <td className="num">{p.total_co2_g.toFixed(2)} g</td>
                    <td className="num">[{p.participation_counts.join(', ')}]</td>
                    <td className="num">{p.participation_gini.toFixed(3)}</td>
                    <td>
                      {p.pareto_optimal ? (
                        <span className="pareto-badge">BEST TRADE-OFF</span>
                      ) : (
                        <span className="dom-badge">OUTPERFORMED</span>
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
                <h2>RL with different goals</h2>
                <span className="plate-tag num">STEPS 7 &amp; 8</span>
              </div>
              <p className="panel-sub">
                Each row is an RL agent told to care about accuracy, time, CO₂ and fairness in a different mix.
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
                    <th data-tip="What this agent was told matters most.">Goal mix</th>
                    <th data-tip="Final accuracy on test data.">Accuracy</th>
                    <th data-tip="Total training time (simulated).">Time</th>
                    <th data-tip="Total CO₂ in grams (simulated).">CO₂</th>
                    <th data-tip="How unevenly PCs were used (0 = all equally).">Unevenness</th>
                    <th data-tip="Best trade-off: no other goal mix is better on accuracy, time and CO₂ at once.">Trade-off</th>
                  </tr>
                </thead>
                <tbody>
                  {morl.map((m) => (
                    <tr
                      key={m.preference_name}
                      className={m.pareto_optimal ? 'rl-row' : ''}
                      data-tip={`Goal mix ${m.preference_name}: ${(m.final_accuracy * 100).toFixed(1)}% accuracy, ${m.total_latency_s.toFixed(1)}s, ${m.total_co2_g.toFixed(2)}g CO2. ${m.pareto_optimal ? 'Best trade-off' : 'Dominated point'} from fl_pipeline/rl_env.py.`}
                    >
                      <td><strong>{m.preference_name}</strong></td>
                      <td className="num">{(m.final_accuracy * 100).toFixed(1)}%</td>
                      <td className="num">{m.total_latency_s.toFixed(1)} s</td>
                      <td className="num">{m.total_co2_g.toFixed(2)} g</td>
                      <td className="num">{m.participation_gini.toFixed(3)}</td>
                      <td>{m.pareto_optimal ? <span className="pareto-badge">BEST TRADE-OFF</span> : '—'}</td>
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
                <h2>The 4 PCs in this simulator</h2>
                <span className="plate-tag num">{c?.use_calibrated_telemetry ? 'GENERATED 10K-ROW PROFILE' : 'FIXED CONFIG VALUES'}</span>
                {c?.data_provenance && (
                  <span
                    className="plate-tag num"
                    title="Where the training data came from. The fallback is generated locally because MNIST is not downloaded."
                  >
                    DATA: {PROVENANCE_LABEL[c.data_provenance] ?? c.data_provenance}
                  </span>
                )}
              </div>
              <p className="panel-sub">
                {`How much data each PC got, and the hardware values used in steps 3–8.${
                  c?.data_split
                    ? ` Scheduler decisions use a separate ${c.data_split.n_val}-sample validation set; accuracy shown is on ${c.data_split.n_test} held-out test samples.`
                    : ''
                }`}
              </p>
            </div>
          </div>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th data-tip="One of the 4 PCs.">PC</th>
                  <th data-tip="How many training examples this PC holds.">Samples</th>
                  <th data-tip="Memory and graphics-card memory, for reference.">RAM / GPU</th>
                  <th data-tip="How much faster than PC1 this PC trains.">Speed vs. PC1</th>
                  <th data-tip="Power while training / while waiting.">Training / idle W</th>
                  <th data-tip="Network speed, which sets download and upload time.">Network</th>
                  <th data-tip="CO₂ emitted per kWh of electricity where this PC is.">Grid CO₂ per kWh</th>
                  <th data-tip="Whether this PC's training power was measured with a wall-plug meter or estimated.">Training power source</th>
                </tr>
              </thead>
              <tbody>
                {cl.map((node) => (
                  <tr
                    key={node.name}
                    className={node.name === 'PC1' ? 'rl-row' : ''}
                    data-tip={`${node.name}: ${node.num_samples} samples, ${node.relative_speed.toFixed(2)}× PC1 speed, ${node.grid_kgco2_per_kwh.toFixed(3)} kg CO₂/kWh, ${node.name !== 'PC4' ? 'training power metered' : 'training power estimated'}.`}
                  >
                    <td className="num"><strong>{node.name}</strong></td>
                    <td className="num">{node.num_samples}</td>
                    <td className="num">{node.ram_gb} GB / {node.gpu_vram_gb} GB</td>
                    <td className="num">{node.relative_speed.toFixed(2)}×</td>
                    <td className="num">{node.active_power_w.toFixed(1)} W / {node.idle_power_w.toFixed(1)} W</td>
                    <td className="num">{node.bandwidth_mbps.toFixed(1)} Mbps</td>
                    <td className="num">{node.grid_kgco2_per_kwh.toFixed(3)} kg CO₂/kWh</td>
                    <td>
                      <span className={node.name !== 'PC4' ? 'stamp-measured' : 'stamp-modeled'}>
                        {node.name !== 'PC4' ? 'METERED' : 'ESTIMATED'}
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
