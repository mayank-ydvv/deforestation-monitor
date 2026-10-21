import { useCallback, useEffect, useState } from "react";

import ClearedLayer from "./components/ClearedLayer";
import DateRangeSlider from "./components/DateRangeSlider";
import MapView from "./components/MapView";
import Panel from "./components/Panel";
import PatchPopup from "./components/PatchPopup";
import RegionPicker from "./components/RegionPicker";
import { useAoiData } from "./hooks/useAoiData";
import { getJson } from "./lib/data";
import type { Aoi, ClearedFeature } from "./lib/types";

export default function App() {
  const [aois, setAois] = useState<Aoi[]>([]);
  const [aoiId, setAoiId] = useState<string>("");
  const [range, setRange] = useState<{ before: string; after: string }>({ before: "", after: "" });
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<{ feature: ClearedFeature; at: [number, number] } | null>(null);

  useEffect(() => {
    getJson<Aoi[]>("aois.json")
      .then((list) => {
        setAois(list);
        setAoiId((id) => id || list[0]?.id || "");
      })
      .catch((e: unknown) => setError(String(e)));
  }, []);

  const aoi = aois.find((a) => a.id === aoiId) ?? null;

  // New region: default to its full date range.
  useEffect(() => {
    if (aoi) setRange({ before: aoi.months[0] ?? "", after: aoi.months[aoi.months.length - 1] ?? "" });
  }, [aoi]);

  const data = useAoiData(aoi, range.before, range.after);
  const onSelect = useCallback((feature: ClearedFeature, at: [number, number]) => setSelected({ feature, at }), []);

  // A new region or date range invalidates the open popup.
  useEffect(() => setSelected(null), [aoiId, range]);

  return (
    <div className="relative h-full w-full">
      <MapView aoi={aoi}>
        <ClearedLayer features={data.features} onSelect={onSelect} />
        {selected && (
          <PatchPopup feature={selected.feature} position={selected.at} onClose={() => setSelected(null)} />
        )}
      </MapView>
      <Panel>
        {error && <p role="alert" className="text-sm text-red-300">Could not load regions: {error}</p>}
        {aois.length > 0 && <RegionPicker aois={aois} value={aoiId} onChange={setAoiId} />}
        {aoi && (
          <DateRangeSlider
            months={aoi.months}
            before={range.before}
            after={range.after}
            onChange={(before, after) => setRange({ before, after })}
          />
        )}
      </Panel>
    </div>
  );
}
