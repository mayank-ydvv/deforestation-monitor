import { useMemo } from "react";

const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "2025-07" -> "Jul 2025" */
export function formatMonth(month: string): string {
  const [y, m] = month.split("-").map(Number);
  return `${MONTH_NAMES[m - 1]} ${y}`;
}

/** Every calendar month from `first` to `last`, inclusive. */
export function calendarMonths(first: string, last: string): string[] {
  const out: string[] = [];
  let [y, m] = first.split("-").map(Number);
  const [ly, lm] = last.split("-").map(Number);
  while (y < ly || (y === ly && m <= lm)) {
    out.push(`${y}-${String(m).padStart(2, "0")}`);
    m += 1;
    if (m > 12) [y, m] = [y + 1, 1];
  }
  return out;
}

interface DateRangeSliderProps {
  months: string[];
  before: string;
  after: string;
  onChange: (before: string, after: string) => void;
}

/**
 * Pick the "before" and "after" months. Both thumbs snap to months that have data; the strip
 * below shows every calendar month, with gaps (cloudy or skipped months) marked "no data".
 */
export default function DateRangeSlider({ months, before, after, onChange }: DateRangeSliderProps) {
  const timeline = useMemo(
    () => (months.length ? calendarMonths(months[0], months[months.length - 1]) : []),
    [months],
  );
  if (months.length < 2) {
    return <p className="text-sm text-slate-400">Not enough months with data yet for this region.</p>;
  }
  const last = months.length - 1;
  const bi = Math.max(0, months.indexOf(before));
  const ai = Math.max(0, months.indexOf(after));
  const available = new Set(months);

  return (
    <fieldset className="flex flex-col gap-3">
      <legend className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-400">Dates</legend>
      <div className="flex items-center justify-between text-sm">
        <span>
          <span className="text-slate-400">Before </span>
          <span className="font-medium">{formatMonth(months[bi])}</span>
        </span>
        <span aria-hidden className="text-slate-500">→</span>
        <span>
          <span className="text-slate-400">After </span>
          <span className="font-medium">{formatMonth(months[ai])}</span>
        </span>
      </div>
      <label className="flex flex-col gap-1 text-xs text-slate-400">
        Before
        <input
          type="range"
          min={0}
          max={last}
          value={bi}
          aria-valuetext={formatMonth(months[bi])}
          onChange={(e) => {
            const i = Math.min(Number(e.target.value), last - 1); // before stays left of after
            onChange(months[i], months[Math.max(ai, i + 1)]);
          }}
          className="accent-emerald-400"
        />
      </label>
      <label className="flex flex-col gap-1 text-xs text-slate-400">
        After
        <input
          type="range"
          min={0}
          max={last}
          value={ai}
          aria-valuetext={formatMonth(months[ai])}
          onChange={(e) => {
            const i = Math.max(Number(e.target.value), 1); // after stays right of before
            onChange(months[Math.min(bi, i - 1)], months[i]);
          }}
          className="accent-red-400"
        />
      </label>
      <div aria-hidden className="flex h-2 gap-px overflow-hidden rounded">
        {timeline.map((m) => {
          const inRange = m >= months[bi] && m <= months[ai];
          const color = !available.has(m)
            ? "bg-slate-700/60"
            : inRange
              ? "bg-emerald-400"
              : "bg-slate-400";
          return <div key={m} title={available.has(m) ? formatMonth(m) : `${formatMonth(m)}: no data`} className={`flex-1 ${color}`} />;
        })}
      </div>
      <p className="text-xs text-slate-500">
        Grey gaps: no usable composite (cloud or season). Clearing shown is everything detected after
        the before month, up to the after month.
      </p>
    </fieldset>
  );
}
