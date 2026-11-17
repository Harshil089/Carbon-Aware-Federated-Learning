import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import {
  MorlSweepPoint,
  POLICY_META,
  PolicySummaryRow,
  RoundRow,
} from '../types';

export interface TooltipState {
  x: number;
  y: number;
  title: string;
  subtitle?: string;
  rows: { label: string; value: string; color?: string }[];
}

export function ChartTooltip({ tip }: { tip: TooltipState | null }) {
  if (!tip) return null;
  const left = typeof window !== 'undefined' ? Math.min(window.innerWidth - 230, tip.x + 16) : tip.x + 16;
  const top = typeof window !== 'undefined' ? Math.max(12, tip.y - 20) : tip.y;
  return (
    <div className="tooltip" style={{ left, top }}>
      <div className="tt-head">
        <span className="tt-h">{tip.title}</span>
        {tip.subtitle && <span className="tt-sub num">{tip.subtitle}</span>}
      </div>
      {tip.rows.map((r) => (
        <div className="tt-r" key={r.label}>
          <span className="tt-lbl">
            {r.color && (
              <i
                className="tt-swatch"
                style={{ background: r.color }}
              />
            )}
            {r.label}
          </span>
          <b className="num">{r.value}</b>
        </div>
      ))}
    </div>
  );
}

export function ParetoChart({
  summary,
  hiddenPolicies,
}: {
  summary: PolicySummaryRow[];
  hiddenPolicies: Set<string>;
}) {
  const [tip, setTip] = useState<TooltipState | null>(null);
  const visible = useMemo(
    () => summary.filter((s) => !hiddenPolicies.has(s.policy)),
    [summary, hiddenPolicies],
  );

  if (!visible.length) {
    return (
      <div className="empty-chart">
        Turn on at least one rule above to show the chart.
      </div>
    );
  }

  const W = 740;
  const H = 330;
  const P = { t: 28, r: 44, b: 46, l: 64 };
  const iw = W - P.l - P.r;
  const ih = H - P.t - P.b;

  const co2s = visible.map((s) => Number(s.mean_total_CO2_g_MODELED));
  const accs = visible.map((s) => Number(s.final_accuracy_mean));
  const minX = Math.min(...co2s) * 0.88;
  const maxX = Math.max(...co2s) * 1.1;
  const minY = Math.max(0, Math.min(...accs) - 0.035);
  const maxY = Math.min(1, Math.max(...accs) + 0.028);
  const sx = (v: number) => P.l + ((v - minX) / Math.max(1e-6, maxX - minX)) * iw;
  const sy = (v: number) => P.t + ih - ((v - minY) / Math.max(1e-6, maxY - minY)) * ih;

  const sorted = [...visible].sort(
    (a, b) => Number(a.mean_total_CO2_g_MODELED) - Number(b.mean_total_CO2_g_MODELED),
  );
  const pareto: PolicySummaryRow[] = [];
  let bestA = -1;
  for (const s of sorted) {
    const a = Number(s.final_accuracy_mean);
    if (a > bestA) {
      pareto.push(s);
      bestA = a;
    }
  }
  const paretoD =
    pareto.length > 1
      ? pareto
          .map(
            (s, i) =>
              `${i ? 'L' : 'M'}${sx(Number(s.mean_total_CO2_g_MODELED)).toFixed(1)} ${sy(
                Number(s.final_accuracy_mean),
              ).toFixed(1)}`,
          )
          .join(' ')
      : '';

  const paretoSet = new Set(pareto.map((p) => p.policy));

  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Accuracy vs simulated CO₂ for each rule">
        <defs>
          <pattern id="plateGrid" width="28" height="28" patternUnits="userSpaceOnUse">
            <path d="M 28 0 L 0 0 0 28" fill="none" stroke="#EBE6DC" strokeWidth="0.75" />
          </pattern>
        </defs>

        <rect x={P.l} y={P.t} width={iw} height={ih} fill="url(#plateGrid)" stroke="#CFC9BC" strokeWidth="1" />

        {[0, 1, 2, 3, 4].map((i) => {
          const vy = minY + (maxY - minY) * (i / 4);
          const y = sy(vy);
          const vx = minX + (maxX - minX) * (i / 4);
          const x = sx(vx);
          return (
            <g key={i}>
              <line x1={P.l} y1={y} x2={W - P.r} y2={y} stroke="#DED8CA" strokeDasharray="2 3" />
              <line x1={x} y1={P.t} x2={x} y2={P.t + ih} stroke="#DED8CA" strokeDasharray="2 3" />
              <text x={P.l - 10} y={y + 4} textAnchor="end" fill="#57534A" fontSize="11" fontFamily="var(--mono)">
                {(vy * 100).toFixed(1)}%
              </text>
              <text x={x} y={H - 16} textAnchor="middle" fill="#57534A" fontSize="11" fontFamily="var(--mono)">
                {vx.toFixed(2)} g
              </text>
            </g>
          );
        })}

        <text x={P.l + 10} y={P.t + 16} fill="#57534A" fontSize="10.5" fontFamily="var(--mono)">
          BEST CORNER: TOP-LEFT (HIGH ACCURACY, LOW CO₂)
        </text>

        <text x={W / 2} y={H - 2} textAnchor="middle" fill="#3F3D37" fontSize="11" fontWeight="600">
          Total CO₂ over the run, g (simulated)
        </text>

        {paretoD && (
          <motion.path
            d={paretoD}
            fill="none"
            stroke="#D9381E"
            strokeWidth={2}
            strokeDasharray="6 4"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.65, ease: [0.22, 1, 0.36, 1] }}
          />
        )}

        {(() => {
          const placed: { x: number; y: number }[] = [];
          const offsets = [
            { dx: 0, dy: -13, anchor: 'middle' as const },
            { dx: 0, dy: 21, anchor: 'middle' as const },
            { dx: 12, dy: 4, anchor: 'start' as const },
            { dx: -12, dy: 4, anchor: 'end' as const },
            { dx: 12, dy: -11, anchor: 'start' as const },
            { dx: -12, dy: 18, anchor: 'end' as const },
          ];
          return visible.map((s) => {
            const m = POLICY_META[s.policy] ?? { label: s.policy, short: s.policy, strategy: '', color: '#121316' };
            const x = sx(Number(s.mean_total_CO2_g_MODELED));
            const y = sy(Number(s.final_accuracy_mean));
            const isRl = s.policy === 'Q_learning_RL';
            const isPareto = paretoSet.has(s.policy);
            const r = isRl ? 8.5 : 6.5;

            let chosen = offsets[0];
            for (const cand of offsets) {
              const lx = x + cand.dx;
              const ly = y + cand.dy;
              const collides = placed.some((p) => Math.abs(p.x - lx) < 78 && Math.abs(p.y - ly) < 16);
              if (!collides) {
                chosen = cand;
                break;
              }
            }
            placed.push({ x: x + chosen.dx, y: y + chosen.dy });

            return (
              <motion.g
                key={s.policy}
                initial={{ scale: 0.6, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
                style={{ cursor: 'pointer' }}
                onMouseMove={(e) =>
                  setTip({
                    x: e.clientX,
                    y: e.clientY,
                    title: m.label,
                    subtitle: isPareto ? 'BEST TRADE-OFF: NO RULE BEATS IT ON BOTH' : 'ANOTHER RULE BEATS IT ON BOTH',
                    rows: [
                      { label: 'Final Accuracy', value: `${(Number(s.final_accuracy_mean) * 100).toFixed(2)}%`, color: m.color },
                      { label: 'CO₂ (simulated)', value: `${Number(s.mean_total_CO2_g_MODELED).toFixed(3)} g` },
                      { label: 'Training time (simulated)', value: `${Number(s.mean_total_seconds_MODELED).toFixed(1)} s` },
                      { label: 'Energy (simulated)', value: `${Number(s.mean_total_Wh_MODELED).toFixed(3)} Wh` },
                      { label: 'Reached target', value: `${(Number(s.target_hit_rate) * 100).toFixed(0)}%` },
                    ],
                  })
                }
                onMouseLeave={() => setTip(null)}
              >
                <circle cx={x} cy={y} r={r + 9} fill="transparent" />
                {isPareto && (
                  <circle
                    cx={x}
                    cy={y}
                    r={r + 4.5}
                    fill="none"
                    stroke={m.color}
                    strokeWidth={1.2}
                    strokeDasharray="2 2"
                  />
                )}
                <circle cx={x} cy={y} r={r} fill={m.color} stroke="#FAF8F3" strokeWidth={2.2} />
                <text
                  x={x + chosen.dx}
                  y={y + chosen.dy}
                  textAnchor={chosen.anchor}
                  fill="#121316"
                  fontSize="11.5"
                  fontWeight="700"
                >
                  {m.label}
                </text>
              </motion.g>
            );
          });
        })()}
      </svg>
      <ChartTooltip tip={tip} />
    </>
  );
}

export function TrajectoryChart({
  rows,
  metricKey,
  targetY,
  hiddenPolicies,
}: {
  rows: RoundRow[];
  metricKey: 'accuracy' | 'cum_latency' | 'cum_co2';
  targetY?: number;
  hiddenPolicies: Set<string>;
}) {
  const [hoverRound, setHoverRound] = useState<number | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const series = useMemo(() => {
    const byPolicy: Record<string, Record<number, { acc: number[]; lat: number[]; co2: number[] }>> = {};
    for (const p of Object.keys(POLICY_META)) byPolicy[p] = {};
    for (const r of rows) {
      if (!byPolicy[r.policy]) continue;
      const rnd = Number(r.round);
      if (!byPolicy[r.policy][rnd]) byPolicy[r.policy][rnd] = { acc: [], lat: [], co2: [] };
      byPolicy[r.policy][rnd].acc.push(Number(r.test_accuracy_REAL_MODEL_SYNTHETIC_DATA));
      byPolicy[r.policy][rnd].lat.push(Number(r.round_seconds_MODELED));
      byPolicy[r.policy][rnd].co2.push(Number(r.round_CO2_g_MODELED));
    }
    const out: Record<string, { round: number; accuracy: number; cum_latency: number; cum_co2: number }[]> = {};
    for (const [p, roundsObj] of Object.entries(byPolicy)) {
      const keys = Object.keys(roundsObj).map(Number).sort((a, b) => a - b);
      let cumLat = 0;
      let cumCo2 = 0;
      out[p] = keys.map((rnd) => {
        const d = roundsObj[rnd];
        const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
        cumLat += mean(d.lat);
        cumCo2 += mean(d.co2);
        return { round: rnd, accuracy: mean(d.acc), cum_latency: cumLat, cum_co2: cumCo2 };
      });
    }
    return out;
  }, [rows]);

  const visibleEntries = Object.entries(series).filter(([p]) => !hiddenPolicies.has(p));
  const allPts = visibleEntries.flatMap(([, pts]) => pts);
  if (!allPts.length) {
    return (
      <div className="empty-chart">
        Turn on at least one rule above to show the chart.
      </div>
    );
  }

  const W = 740;
  const H = 330;
  const P = { t: 24, r: 34, b: 44, l: 64 };
  const iw = W - P.l - P.r;
  const ih = H - P.t - P.b;
  const maxR = Math.max(...allPts.map((d) => d.round));
  const minY = metricKey === 'accuracy' ? 0.4 : 0;
  let maxY = metricKey === 'accuracy' ? 0.95 : Math.max(...allPts.map((d) => d[metricKey])) * 1.06;
  if (maxY === minY) maxY = minY + 1;

  const sx = (r: number) => P.l + ((r - 1) / Math.max(1, maxR - 1)) * iw;
  const sy = (v: number) => P.t + ih - ((v - minY) / (maxY - minY)) * ih;
  const fmtY = (v: number) =>
    metricKey === 'accuracy'
      ? `${(v * 100).toFixed(1)}%`
      : metricKey === 'cum_latency'
        ? `${v.toFixed(0)} s`
        : `${v.toFixed(2)} g`;

  const stepR = Math.max(1, Math.ceil(maxR / 8));
  const xTicks: number[] = [];
  for (let r = 1; r <= maxR; r += stepR) xTicks.push(r);
  if (xTicks[xTicks.length - 1] !== maxR) xTicks.push(maxR);

  const tip: TooltipState | null =
    hoverRound !== null
      ? {
          x: mousePos.x,
          y: mousePos.y,
          title: `Federated Round ${hoverRound}`,
          subtitle: `R${hoverRound}/${maxR}`,
          rows: visibleEntries
            .map(([p, pts]) => {
              const pt = pts.find((d) => d.round === hoverRound);
              if (!pt) return null;
              return {
                label: POLICY_META[p].label,
                value: fmtY(pt[metricKey]),
                color: POLICY_META[p].color,
              };
            })
            .filter(Boolean) as { label: string; value: string; color?: string }[],
        }
      : null;

  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Round trajectory chart for ${metricKey}`}>
        <defs>
          <linearGradient id="rlAreaGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#0F766E" stopOpacity="0.16" />
            <stop offset="100%" stopColor="#0F766E" stopOpacity="0.01" />
          </linearGradient>
        </defs>

        <rect x={P.l} y={P.t} width={iw} height={ih} fill="#FAF8F3" stroke="#CFC9BC" strokeWidth="1" />

        {[0, 1, 2, 3, 4].map((i) => {
          const v = minY + (maxY - minY) * (i / 4);
          const y = sy(v);
          return (
            <g key={i}>
              <line x1={P.l} y1={y} x2={W - P.r} y2={y} stroke="#E2DDD1" strokeDasharray="2 3" />
              <text x={P.l - 10} y={y + 4} textAnchor="end" fill="#57534A" fontSize="11" fontFamily="var(--mono)">
                {fmtY(v)}
              </text>
            </g>
          );
        })}

        {xTicks.map((r) => (
          <g key={r}>
            <line x1={sx(r)} y1={P.t} x2={sx(r)} y2={P.t + ih} stroke="#EBE6DC" />
            <text x={sx(r)} y={H - 16} textAnchor="middle" fill="#57534A" fontSize="11" fontFamily="var(--mono)">
              R{r}
            </text>
          </g>
        ))}

        {metricKey === 'accuracy' && targetY !== undefined && (
          <g>
            <line
              x1={P.l}
              y1={sy(targetY)}
              x2={W - P.r}
              y2={sy(targetY)}
              stroke="#121316"
              strokeWidth="1.2"
              strokeDasharray="5 4"
            />
            <text
              x={W - P.r - 6}
              y={sy(targetY) - 6}
              textAnchor="end"
              fill="#121316"
              fontSize="10.5"
              fontFamily="var(--mono)"
              fontWeight="600"
            >
              TARGET {(targetY * 100).toFixed(0)}%
            </text>
          </g>
        )}

        {visibleEntries.map(([p, pts]) => {
          const m = POLICY_META[p];
          const d = pts
            .map((pt, i) => `${i ? 'L' : 'M'}${sx(pt.round).toFixed(1)} ${sy(pt[metricKey]).toFixed(1)}`)
            .join(' ');
          return (
            <g key={p}>
              {p === 'Q_learning_RL' && pts.length > 1 && (
                <path
                  d={`${d} L${sx(pts[pts.length - 1].round).toFixed(1)} ${(P.t + ih).toFixed(1)} L${sx(pts[0].round).toFixed(1)} ${(P.t + ih).toFixed(1)} Z`}
                  fill="url(#rlAreaGrad)"
                />
              )}
              <motion.path
                d={d}
                fill="none"
                stroke={m.color}
                strokeWidth={p === 'Q_learning_RL' ? 2.8 : 2}
                strokeLinejoin="round"
                strokeLinecap="round"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
              />
              {pts.map((pt) => (
                <circle
                  key={`${p}-${pt.round}`}
                  cx={sx(pt.round)}
                  cy={sy(pt[metricKey])}
                  r={p === 'Q_learning_RL' ? 3.2 : 2.4}
                  fill={m.color}
                />
              ))}
            </g>
          );
        })}

        {hoverRound !== null && (
          <g>
            <line
              x1={sx(hoverRound)}
              y1={P.t}
              x2={sx(hoverRound)}
              y2={P.t + ih}
              stroke="#121316"
              strokeWidth="1"
              strokeDasharray="3 3"
            />
            {visibleEntries.map(([p, pts]) => {
              const pt = pts.find((d) => d.round === hoverRound);
              if (!pt) return null;
              return (
                <circle
                  key={p}
                  cx={sx(hoverRound)}
                  cy={sy(pt[metricKey])}
                  r={5}
                  fill={POLICY_META[p].color}
                  stroke="#FAF8F3"
                  strokeWidth={2}
                />
              );
            })}
          </g>
        )}

        {Array.from({ length: maxR }, (_, idx) => idx + 1).map((r) => (
          <rect
            key={r}
            x={sx(r) - iw / (maxR * 2)}
            y={P.t}
            width={iw / Math.max(1, maxR - 1)}
            height={ih}
            fill="transparent"
            onMouseMove={(e) => {
              setHoverRound(r);
              setMousePos({ x: e.clientX, y: e.clientY });
            }}
            onMouseLeave={() => setHoverRound(null)}
          />
        ))}
      </svg>
      <ChartTooltip tip={tip} />
    </>
  );
}

export function MorlSweepChart({ morl }: { morl: MorlSweepPoint[] }) {
  const [tip, setTip] = useState<TooltipState | null>(null);
  if (!morl.length) return null;

  const W = 580;
  const H = 290;
  const P = { t: 26, r: 34, b: 42, l: 58 };
  const iw = W - P.l - P.r;
  const ih = H - P.t - P.b;
  const co2s = morl.map((m) => Number(m.total_co2_g));
  const accs = morl.map((m) => Number(m.final_accuracy));
  const minX = Math.min(...co2s) * 0.9;
  const maxX = Math.max(...co2s) * 1.1;
  const minY = Math.max(0, Math.min(...accs) - 0.035);
  const maxY = Math.min(1, Math.max(...accs) + 0.03);
  const sx = (v: number) => P.l + ((v - minX) / Math.max(1e-6, maxX - minX)) * iw;
  const sy = (v: number) => P.t + ih - ((v - minY) / Math.max(1e-6, maxY - minY)) * ih;

  const paretoPts = [...morl]
    .filter((m) => m.pareto_optimal)
    .sort((a, b) => Number(a.total_co2_g) - Number(b.total_co2_g));
  const pLine =
    paretoPts.length > 1
      ? paretoPts
          .map(
            (m, i) =>
              `${i ? 'L' : 'M'}${sx(Number(m.total_co2_g)).toFixed(1)} ${sy(
                Number(m.final_accuracy),
              ).toFixed(1)}`,
          )
          .join(' ')
      : '';

  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="RL goal mixes: results">
        <rect x={P.l} y={P.t} width={iw} height={ih} fill="#FAF8F3" stroke="#CFC9BC" strokeWidth="1" />
        {[0, 1, 2, 3, 4].map((i) => {
          const vy = minY + (maxY - minY) * (i / 4);
          const y = sy(vy);
          const vx = minX + (maxX - minX) * (i / 4);
          const x = sx(vx);
          return (
            <g key={i}>
              <line x1={P.l} y1={y} x2={W - P.r} y2={y} stroke="#E2DDD1" strokeDasharray="2 3" />
              <text x={P.l - 8} y={y + 3.5} textAnchor="end" fill="#57534A" fontSize="10.5" fontFamily="var(--mono)">
                {(vy * 100).toFixed(1)}%
              </text>
              <text x={x} y={H - 12} textAnchor="middle" fill="#57534A" fontSize="10.5" fontFamily="var(--mono)">
                {vx.toFixed(1)} g
              </text>
            </g>
          );
        })}
        {pLine && (
          <motion.path
            d={pLine}
            fill="none"
            stroke="#D9381E"
            strokeWidth={2}
            strokeDasharray="5 4"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          />
        )}
        {(() => {
          const placed: { x: number; y: number }[] = [];
          const offsets = [
            { dx: 0, dy: -12, anchor: 'middle' as const },
            { dx: 0, dy: 20, anchor: 'middle' as const },
            { dx: 11, dy: 4, anchor: 'start' as const },
            { dx: -11, dy: 4, anchor: 'end' as const },
            { dx: 11, dy: -10, anchor: 'start' as const },
            { dx: -11, dy: 16, anchor: 'end' as const },
          ];
          return morl.map((m) => {
            const x = sx(Number(m.total_co2_g));
            const y = sy(Number(m.final_accuracy));
            const col = m.pareto_optimal ? '#D9381E' : '#1D4ED8';

            let chosen = offsets[0];
            for (const cand of offsets) {
              const lx = x + cand.dx;
              const ly = y + cand.dy;
              const collides = placed.some((p) => Math.abs(p.x - lx) < 82 && Math.abs(p.y - ly) < 15);
              if (!collides) {
                chosen = cand;
                break;
              }
            }
            placed.push({ x: x + chosen.dx, y: y + chosen.dy });

            return (
              <motion.g
                key={m.preference_name}
                initial={{ scale: 0.6, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
                style={{ cursor: 'pointer' }}
                onMouseMove={(e) =>
                  setTip({
                    x: e.clientX,
                    y: e.clientY,
                    title: m.preference_name,
                    subtitle: m.pareto_optimal ? 'BEST TRADE-OFF' : 'OUTPERFORMED',
                    rows: [
                      { label: 'Final Accuracy', value: `${(m.final_accuracy * 100).toFixed(1)}%`, color: col },
                      { label: 'Total CO₂', value: `${m.total_co2_g.toFixed(3)} g` },
                      { label: 'Total Latency', value: `${m.total_latency_s.toFixed(1)} s` },
                      { label: 'Unevenness (0 = fair)', value: m.participation_gini.toFixed(3) },
                    ],
                  })
                }
                onMouseLeave={() => setTip(null)}
              >
                <circle cx={x} cy={y} r={11} fill="transparent" />
                {m.pareto_optimal && (
                  <circle cx={x} cy={y} r={10} fill="none" stroke="#D9381E" strokeWidth={1.2} strokeDasharray="2 2" />
                )}
                <circle
                  cx={x}
                  cy={y}
                  r={m.pareto_optimal ? 7 : 5.5}
                  fill={col}
                  stroke="#FAF8F3"
                  strokeWidth={2}
                />
                <text
                  x={x + chosen.dx}
                  y={y + chosen.dy}
                  textAnchor={chosen.anchor}
                  fill="#121316"
                  fontSize="10.5"
                  fontWeight="700"
                >
                  {m.preference_name}
                </text>
              </motion.g>
            );
          });
        })()}
      </svg>
      <ChartTooltip tip={tip} />
    </>
  );
}
