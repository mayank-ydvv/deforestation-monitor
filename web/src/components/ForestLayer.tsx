import type { Layer } from "leaflet";
import { useEffect } from "react";
import { useMap } from "react-leaflet";

const FOREST_RGBA = "rgba(34, 197, 94, 0.45)";

/**
 * Green overlay of the model's forest mask (uint8 COG: 1 forest, 0 not, 255 no data).
 * georaster + georaster-layer-for-leaflet are imported on first use, so they stay out of the
 * initial bundle; the COG is read with HTTP range requests (only the tiles in view).
 */
export default function ForestLayer({ url, onError }: { url: string; onError?: (msg: string) => void }) {
  const map = useMap();

  useEffect(() => {
    let layer: Layer | null = null;
    let cancelled = false;
    (async () => {
      const [{ default: parseGeoraster }, { default: GeoRasterLayer }] = await Promise.all([
        import("georaster"),
        import("georaster-layer-for-leaflet"),
      ]);
      const georaster = await parseGeoraster(url);
      if (cancelled) return;
      layer = new GeoRasterLayer({
        georaster,
        opacity: 1,
        resolution: 256,
        pixelValuesToColorFn: (values: number[]) => (values[0] === 1 ? FOREST_RGBA : null),
      });
      layer.addTo(map);
    })().catch((err: unknown) => {
      if (!cancelled) onError?.(`Forest mask unavailable: ${err instanceof Error ? err.message : String(err)}`);
    });
    return () => {
      cancelled = true;
      if (layer) map.removeLayer(layer);
    };
  }, [url, map, onError]);

  return null;
}
