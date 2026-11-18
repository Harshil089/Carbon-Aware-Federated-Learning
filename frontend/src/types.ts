export type ViewId = 'landing' | 'overview' | 'pipeline' | 'replay' | 'specs';

export interface SimSettings {
  rounds: number;
  episodes: number;
  repeats: number;
  target: number;
  sensitivity: number;
  local_epochs: number;
  learning_rate: number;
  seed: number;
  latency_weight: number;
  carbon_weight: number;
}

export interface EngineState {
  status: 'idle' | 'running' | 'complete' | 'failed' | 'cancelled';
  progress: number;
  message: string;
  run_id: string | null;
  result_id: string;
  log: string[];
  /** Set only by the hosted (Vercel) server: upper bounds a run may use. */
  hosted_limits?: Partial<Record<'rounds' | 'episodes' | 'repeats' | 'sensitivity', number>>;
}

export interface PolicySummaryRow {
  policy: string;
  final_accuracy_mean: string;
  mean_total_seconds_MODELED: string;
  mean_total_Wh_MODELED: string;
  mean_total_CO2_g_MODELED: string;
  target_hit_rate: string;
}

export interface RoundRow {
  policy: string;
  round: string;
  scenario_seed: string;
  clients_selected: string;
  test_accuracy_REAL_MODEL_SYNTHETIC_DATA: string;
  round_seconds_MODELED: string;
  round_Wh_MODELED: string;
  round_CO2_g_MODELED: string;
}

export interface ClientProfile {
  name: string;
  ram_gb: number;
  gpu_vram_gb: number;
  relative_speed: number;
  active_power_ratio_to_PC1: number;
  idle_power_w_assumed: number;
  status?: string;
  network_seconds_assumed: number;
  grid_kgco2_per_kwh?: number;
}

export interface SensitivityRow {
  policy: string;
  co2_g_p05_ASSUMED: string;
  co2_g_median_ASSUMED: string;
  co2_g_p95_ASSUMED: string;
  seconds_p05_ASSUMED?: string;
  seconds_p95_ASSUMED?: string;
}

export interface SimulationResults {
  summary: PolicySummaryRow[];
  rows: RoundRow[];
  sensitivity: SensitivityRow[];
  artifact_prefix: string;
  result_id?: string;
  offline?: boolean;
  parameters: {
    execution_wall_seconds: number;
    run: Record<string, number>;
    training: {
      model: string;
      clients: number;
      classes: number;
      training_samples: number;
      test_samples: number;
      local_epochs: number;
      learning_rate: number;
      features: number;
      bias: boolean;
    };
    rl: {
      alpha: number;
      gamma: number;
      epsilon_start: number;
      epsilon_min: number;
      actions: number;
      training_phases: number;
      accuracy_bucket_edges: number[];
      reward_accuracy_floor: number;
      accuracy_gain_weight: number;
      latency_weight: number;
      carbon_weight: number;
      accuracy_shortfall_weight: number;
      reward: string;
    };
    infrastructure: {
      reference_energy_kwh: number;
      reference_duration_min: number;
      reference_mean_power_w: number;
      grid_kg_co2_per_kwh: number;
      latency_formula: string;
      energy_formula: string;
      carbon_formula: string;
      settings: {
        synthetic_pc1_train_seconds_per_round: number;
        aggregation_seconds_per_round: number;
        coordinator_power_w: number;
      };
      clients: ClientProfile[];
    };
  };
}

export interface PipelineConfig {
  mode: 'iid' | 'dirichlet';
  alpha: number;
  rounds: number;
  q_episodes: number;
  morl_episodes: number;
  use_calibrated_telemetry: boolean;
  seed: number;
  // Response-only; optional so older cached/offline results and the request payload still type-check.
  model?: string;
  data_provenance?: string;
  train_data_seeds?: number;
  data_split?: {
    n_train: number;
    n_val: number;
    n_test: number;
    rl_selection_metric: string;
    reported_metric: string;
  };
}

export interface PipelineClient {
  client_id: number;
  name: string;
  ram_gb: number;
  gpu_vram_gb: number;
  relative_speed: number;
  active_power_w: number;
  idle_power_w: number;
  bandwidth_mbps: number;
  grid_kgco2_per_kwh: number;
  num_samples: number;
  provenance: string;
}

export interface PipelinePolicy {
  policy_name: string;
  final_accuracy: number;
  final_loss: number;
  total_latency_s: number;
  total_wh: number;
  total_co2_g: number;
  participation_counts: number[];
  participation_gini: number;
  constraint_violations: number;
  pareto_optimal: boolean;
}

export interface MorlSweepPoint {
  preference_name: string;
  final_accuracy: number;
  total_latency_s: number;
  total_co2_g: number;
  participation_gini: number;
  pareto_optimal: boolean;
}

export interface PipelineResults {
  config: PipelineConfig;
  clients: PipelineClient[];
  policies: PipelinePolicy[];
  pareto_policies: string[];
  morl_sweep: MorlSweepPoint[];
  pareto_morl: string[];
}

export const POLICY_META: Record<string, { label: string; short: string; strategy: string; color: string }> = {
  Q_learning_RL: {
    label: 'RL picker',
    short: 'Learns which 2–4 PCs to pick',
    strategy: 'Reinforcement learning picks PCs each round',
    color: '#0F766E',
  },
  FedAvg_all4: {
    label: 'All 4 (FedAvg)',
    short: 'Every PC, every round',
    strategy: 'Standard approach: waits for the slowest PC',
    color: '#C82D14',
  },
  Fastest2: {
    label: 'Fastest 2',
    short: 'Always PC3 + PC4',
    strategy: 'Fixed rule: the 2 fastest PCs',
    color: '#1D4ED8',
  },
  LowestModelEnergy2: {
    label: 'Lowest-power 2',
    short: 'Always PC1 + PC2',
    strategy: 'Fixed rule: the 2 lowest-power PCs',
    color: '#B45309',
  },
};
