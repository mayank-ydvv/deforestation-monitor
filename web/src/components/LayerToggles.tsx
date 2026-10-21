export interface Layers {
  cleared: boolean;
  forest: boolean;
  swipe: boolean;
}

const OPTIONS: { key: keyof Layers; label: string; hint: string; swatch: string }[] = [
  { key: "cleared", label: "Cleared areas", hint: "Patches of at least 0.5 ha", swatch: "bg-red-500" },
  { key: "forest", label: "Forest mask", hint: "Model forest, after month", swatch: "bg-emerald-500" },
  { key: "swipe", label: "Before / after swipe", hint: "Drag the divider to compare", swatch: "bg-slate-300" },
];

interface LayerTogglesProps {
  value: Layers;
  onChange: (value: Layers) => void;
}

/** Checkboxes for the map overlays. */
export default function LayerToggles({ value, onChange }: LayerTogglesProps) {
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-400">Layers</legend>
      {OPTIONS.map((o) => (
        <label key={o.key} className="flex cursor-pointer items-start gap-3 rounded-lg px-1 py-1 hover:bg-slate-800/60">
          <input
            type="checkbox"
            checked={value[o.key]}
            onChange={(e) => onChange({ ...value, [o.key]: e.target.checked })}
            className="mt-1 h-4 w-4 accent-emerald-500"
          />
          <span className="flex flex-col">
            <span className="flex items-center gap-2 text-sm">
              <span aria-hidden className={`inline-block h-2.5 w-2.5 rounded-sm ${o.swatch}`} />
              {o.label}
            </span>
            <span className="text-xs text-slate-400">{o.hint}</span>
          </span>
        </label>
      ))}
    </fieldset>
  );
}
