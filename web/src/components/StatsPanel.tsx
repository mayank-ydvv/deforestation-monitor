import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { IndexRow } from "../lib/types";
import { formatMonth } from "./DateRangeSlider";

const CLEARED = "#ef4444"; // same red as the map's cleared layer (validated vs the dark panel)
const OUT_OF_RANGE = "#475569";
const INK_MUTED = "#94a3b8";
const GRID = "#1e293b";

interface StatsPanelProps {
  index: IndexRow[];
  before: string;
  after: string;
  hectares: number;
  patches: number;
}

interface TipProps {
  active?: boolean;
  payload?: { payload: IndexRow }[];
}

function BarTip({ active, payload }: TipProps) {
  if (!active || !payload?.length) return null;
  const r = payload[0].payload;
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs text-slate-200 shadow-lg">
      <div className="font-medium">{formatMonth(r.month)}</div>
      <div>{r.hectares.toFixed(1)} ha in {r.patches} patches</div>
      <div className="text-slate-400">vs {formatMonth(r.t1)}</div>
    </div>
  );
}

/** Hectares cleared per month for the region, plus the total for the selected range. */
export default function StatsPanel({ index, before, after, hectares, patches }: StatsPanelProps) {
  const inRange = (m: string) => m > before && m <= after;
  const data = index.map((r) => ({ ...r, label: formatMonth(r.month) }));
  return (
    <section aria-labelledby="stats-title" className="flex flex-col gap-3">
      <h2 id="stats-title" className="text-xs font-medium uppercase tracking-wide text-slate-400">
        Forest cleared
      </h2>
      <div>
        <div className="text-3xl font-semibold tabular-nums text-slate-100">
          {hectares.toFixed(1)} <span className="text-base font-normal text-slate-400">ha</span>
        </div>
        <p className="text-xs text-slate-400">
          {patches} patch{patches === 1 ? "" : "es"} between {formatMonth(before)} and {formatMonth(after)}.
          Smallest patch mapped: 0.5 ha.
        </p>
      </div>
      {data.length > 0 && (
        <div className="h-36" aria-hidden>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -18 }}>
              <CartesianGrid vertical={false} stroke={GRID} />
              <XAxis dataKey="label" tick={{ fill: INK_MUTED, fontSize: 11 }} tickLine={false} axisLine={{ stroke: GRID }} />
              <YAxis tick={{ fill: INK_MUTED, fontSize: 11 }} tickLine={false} axisLine={false} unit=" ha" width={64} />
              <Tooltip content={<BarTip />} cursor={{ fill: "rgba(148,163,184,0.08)" }} />
              <Bar dataKey="hectares" radius={[4, 4, 0, 0]} maxBarSize={28} isAnimationActive={false}>
                {data.map((r) => (
                  <Cell key={r.month} fill={inRange(r.month) ? CLEARED : OUT_OF_RANGE} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      <details className="text-xs text-slate-300">
        <summary className="cursor-pointer text-slate-400 hover:text-slate-200">Show as table</summary>
        <table className="mt-2 w-full text-left">
          <thead className="text-slate-400">
            <tr>
              <th className="font-normal">Month</th>
              <th className="font-normal">Compared with</th>
              <th className="text-right font-normal">Hectares</th>
              <th className="text-right font-normal">Patches</th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {index.map((r) => (
              <tr key={r.month} className={inRange(r.month) ? "" : "text-slate-500"}>
                <td>{formatMonth(r.month)}</td>
                <td>{formatMonth(r.t1)}</td>
                <td className="text-right">{r.hectares.toFixed(1)}</td>
                <td className="text-right">{r.patches}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </section>
  );
}
