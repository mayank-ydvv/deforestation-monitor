/** Fetching the precomputed data files, with a per-file cache. */
import type { Aoi } from "./types";

/** Where the JSON data lives (Vite serves web/public/data at /data/). */
export const DATA_ROOT: string = import.meta.env.VITE_DATA_ROOT ?? "/data/";

/** Resolved files, keyed by their path under DATA_ROOT ("<aoi>/<month>/cleared.geojson"). */
const cache = new Map<string, unknown>();

export function clearCache(): void {
  cache.clear();
}

/**
 * Fetch and parse a JSON file under DATA_ROOT, once. Only successful responses are cached,
 * so an aborted or failed request is simply retried next time.
 */
export async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  if (cache.has(path)) return cache.get(path) as T;
  const res = await fetch(DATA_ROOT + path, { signal });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  const data = (await res.json()) as T;
  cache.set(path, data);
  return data;
}

/** URL of a raster for one AOI month: forest COG or RGB XYZ tile template. */
export function rasterUrl(aoi: Aoi, month: string, kind: "forest" | "rgb"): string {
  const base = aoi.raster_base || DATA_ROOT;
  const tail = kind === "forest" ? "forest.tif" : "rgb/{z}/{x}/{y}.png";
  return `${base}${aoi.id}/${month}/${tail}`;
}
