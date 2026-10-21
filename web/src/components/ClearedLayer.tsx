import type { PathOptions } from "leaflet";
import { memo, useMemo } from "react";
import { GeoJSON } from "react-leaflet";

import type { ClearedFeature } from "../lib/types";

const STYLE: PathOptions = {
  color: "#ef4444",
  weight: 1.5,
  fillColor: "#ef4444",
  fillOpacity: 0.35,
};

interface ClearedLayerProps {
  features: ClearedFeature[];
  onSelect?: (feature: ClearedFeature, latlng: [number, number]) => void;
}

/** Red polygons for cleared patches. Re-created only when the feature set changes. */
function ClearedLayer({ features, onSelect }: ClearedLayerProps) {
  const data = useMemo(() => ({ type: "FeatureCollection" as const, features }), [features]);
  // GeoJSON layers are immutable in react-leaflet: a new key swaps the layer when data changes.
  const key = useMemo(() => features.map((f) => f.properties.id).join("|"), [features]);
  if (!features.length) return null;
  return (
    <GeoJSON
      key={key}
      data={data}
      style={() => STYLE}
      onEachFeature={(feature, layer) => {
        layer.on("click", (e) => {
          onSelect?.(feature as ClearedFeature, [e.latlng.lat, e.latlng.lng]);
        });
      }}
    />
  );
}

export default memo(ClearedLayer);
