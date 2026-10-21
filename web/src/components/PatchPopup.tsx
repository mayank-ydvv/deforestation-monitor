import { Popup } from "react-leaflet";

import type { ClearedFeature } from "../lib/types";
import { formatMonth } from "./DateRangeSlider";

interface PatchPopupProps {
  feature: ClearedFeature;
  position: [number, number];
  onClose: () => void;
}

/** Details for one cleared patch: area, detection dates, NDVI drop, confidence, confirmation. */
export default function PatchPopup({ feature, position, onClose }: PatchPopupProps) {
  const p = feature.properties;
  return (
    <Popup position={position} eventHandlers={{ remove: onClose }} maxWidth={260}>
      <div className="min-w-[200px] text-slate-100">
        <div className="mb-2 flex items-baseline justify-between gap-3">
          <span className="text-lg font-semibold">{p.area_ha.toFixed(2)} ha</span>
          <span
            className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${
              p.confirmed ? "bg-red-500/20 text-red-300" : "bg-slate-700 text-slate-300"
            }`}
          >
            {p.confirmed ? "Confirmed" : "Unconfirmed"}
          </span>
        </div>
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
          <dt className="text-slate-400">Detected</dt>
          <dd>
            {formatMonth(p.t1)} → {formatMonth(p.t2)}
          </dd>
          <dt className="text-slate-400">NDVI drop</dt>
          <dd>{p.ndvi_drop.toFixed(2)}</dd>
          <dt className="text-slate-400">Confidence</dt>
          <dd>{Math.round(p.confidence * 100)}%</dd>
          <dt className="text-slate-400">Patch</dt>
          <dd className="truncate font-mono text-[11px] text-slate-400">{p.id}</dd>
        </dl>
        {!p.confirmed && (
          <p className="mt-2 text-[11px] leading-snug text-slate-400">
            Unconfirmed: not yet checked against the following month's image.
          </p>
        )}
      </div>
    </Popup>
  );
}
