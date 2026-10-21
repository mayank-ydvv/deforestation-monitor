import { useEffect, useState } from "react";

import MapView from "./components/MapView";
import Panel from "./components/Panel";
import RegionPicker from "./components/RegionPicker";
import { getJson } from "./lib/data";
import type { Aoi } from "./lib/types";

export default function App() {
  const [aois, setAois] = useState<Aoi[]>([]);
  const [aoiId, setAoiId] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getJson<Aoi[]>("aois.json")
      .then((list) => {
        setAois(list);
        setAoiId((id) => id || list[0]?.id || "");
      })
      .catch((e: unknown) => setError(String(e)));
  }, []);

  const aoi = aois.find((a) => a.id === aoiId) ?? null;

  return (
    <div className="relative h-full w-full">
      <MapView aoi={aoi} />
      <Panel>
        {error && <p role="alert" className="text-sm text-red-300">Could not load regions: {error}</p>}
        {aois.length > 0 && <RegionPicker aois={aois} value={aoiId} onChange={setAoiId} />}
      </Panel>
    </div>
  );
}
