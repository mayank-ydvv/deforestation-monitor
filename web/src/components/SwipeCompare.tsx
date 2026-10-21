import L from "leaflet";
import { useEffect } from "react";
import { useMap } from "react-leaflet";

import { formatMonth } from "./DateRangeSlider";

/** 1x1 transparent PNG for tiles outside the AOI (they 404 on the raster store). */
const EMPTY_TILE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

interface SwipeCompareProps {
  beforeUrl: string;
  afterUrl: string;
  before: string;
  after: string;
}

/**
 * Before/after true-colour composites side by side with a draggable divider
 * (leaflet-side-by-side, imported on first use). Tiles exist for z 8-14; deeper zooms upscale.
 */
export default function SwipeCompare({ beforeUrl, afterUrl, before, after }: SwipeCompareProps) {
  const map = useMap();

  useEffect(() => {
    const opts: L.TileLayerOptions = {
      minNativeZoom: 8,
      maxNativeZoom: 14,
      errorTileUrl: EMPTY_TILE,
      zIndex: 5,
    };
    const left = L.tileLayer(beforeUrl, { ...opts, attribution: `Left: ${formatMonth(before)}` }).addTo(map);
    const right = L.tileLayer(afterUrl, { ...opts, attribution: `Right: ${formatMonth(after)}` }).addTo(map);
    let control: L.Control | null = null;
    let cancelled = false;
    import("leaflet-side-by-side").then(() => {
      if (cancelled) return;
      control = (L.control as unknown as SideBySideFactory).sideBySide(left, right).addTo(map);
    });
    return () => {
      cancelled = true;
      control?.remove();
      map.removeLayer(left);
      map.removeLayer(right);
    };
  }, [map, beforeUrl, afterUrl, before, after]);

  return null;
}

interface SideBySideFactory {
  sideBySide: (left: L.Layer, right: L.Layer) => L.Control;
}
