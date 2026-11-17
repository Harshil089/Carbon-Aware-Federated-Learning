import { ArrowRight } from 'lucide-react';
import { ViewId } from '../types';

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
        <section className="lp-hero" aria-labelledby="lp-project-title">
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
        </section>

        {/* 2. What is the need for it */}
        <section className="lp-section" aria-labelledby="lp-need-title">
          <div className="lp-section-head">
            <p className="lp-kicker">02 · Why It Is Needed</p>
            <h2 id="lp-need-title" className="lp-h2">
              Standard Federated Averaging wastes time, idle power, and grid carbon.
            </h2>
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
            <div className="lp-stat">
              <span className="lp-stat-val">83.4%</span>
              <span className="lp-stat-label">Q-Learning RL accuracy (vs. 83.3% All-4 FedAvg)</span>
            </div>
            <div className="lp-stat">
              <span className="lp-stat-val lp-stat-good">−27.4%</span>
              <span className="lp-stat-label">CO₂ emissions (13.214 g vs. 18.196 g All-4)</span>
            </div>
            <div className="lp-stat">
              <span className="lp-stat-val">−13.1%</span>
              <span className="lp-stat-label">Training latency (294.3 s vs. 338.8 s All-4)</span>
            </div>
            <div className="lp-stat">
              <span className="lp-stat-val">10,000</span>
              <span className="lp-stat-label">Calibrated telemetry rows across 4 hardware nodes</span>
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
