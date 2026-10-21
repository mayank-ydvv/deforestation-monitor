import type { Aoi } from "../lib/types";

interface RegionPickerProps {
  aois: Aoi[];
  value: string;
  onChange: (id: string) => void;
}

/** Dropdown of preset AOIs from aois.json; the map flies to the chosen region. */
export default function RegionPicker({ aois, value, onChange }: RegionPickerProps) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor="region" className="text-xs font-medium uppercase tracking-wide text-slate-400">
        Region
      </label>
      <select
        id="region"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
      >
        {aois.map((a) => (
          <option key={a.id} value={a.id}>
            {a.name}
          </option>
        ))}
      </select>
    </div>
  );
}
