import { useEffect, useMemo, useState } from "react";

import { getJson } from "../lib/data";
import type { Aoi, ClearedCollection, ClearedFeature, IndexRow } from "../lib/types";

export interface AoiData {
  index: IndexRow[];
  /** Patches detected in months after `before`, up to and including `after`. */
  features: ClearedFeature[];
  hectares: number;
  loading: boolean;
  error: string | null;
}

/** Months whose cleared layer belongs to the (before, after] range. */
export function monthsInRange(index: IndexRow[], before: string, after: string): string[] {
  return index.filter((r) => r.month > before && r.month <= after).map((r) => r.month);
}

/**
 * Lazily load one AOI's index and the cleared.geojson files for the selected date range.
 * Files are cached per aoi/month; changing the selection aborts requests still in flight.
 */
export function useAoiData(aoi: Aoi | null, before: string, after: string): AoiData {
  const [index, setIndex] = useState<IndexRow[]>([]);
  const [layers, setLayers] = useState<ClearedCollection[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const aoiId = aoi?.id;

  useEffect(() => {
    if (!aoiId) return;
    const ctrl = new AbortController();
    setLoading(true);
    setError(null);
    (async () => {
      const rows = await getJson<IndexRow[]>(`${aoiId}/index.json`, ctrl.signal);
      const months = monthsInRange(rows, before, after);
      const collections = await Promise.all(
        months.map((m) => getJson<ClearedCollection>(`${aoiId}/${m}/cleared.geojson`, ctrl.signal)),
      );
      if (ctrl.signal.aborted) return;
      setIndex(rows);
      setLayers(collections);
      setLoading(false);
    })().catch((err: unknown) => {
      if (ctrl.signal.aborted) return;
      setError(err instanceof Error ? err.message : String(err));
      setLoading(false);
    });
    return () => ctrl.abort();
  }, [aoiId, before, after]);

  const features = useMemo(() => layers.flatMap((c) => c.features), [layers]);
  const hectares = useMemo(
    () => features.reduce((sum, f) => sum + f.properties.area_ha, 0),
    [features],
  );
  return { index, features, hectares, loading, error };
}
