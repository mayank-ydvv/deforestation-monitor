import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { clearCache } from "../lib/data";
import type { Aoi } from "../lib/types";
import { monthsInRange, useAoiData } from "./useAoiData";

const aoi: Aoi = {
  id: "hasdeo",
  name: "Hasdeo",
  bbox: [82.66, 22.71, 82.96, 22.97],
  center: [22.84, 82.81],
  months: ["2025-01", "2025-02", "2026-01"],
  compare: "yoy",
  raster_base: "",
};

const index = [
  { month: "2025-02", t1: "2025-01", hectares: 1, patches: 1 },
  { month: "2026-01", t1: "2025-01", hectares: 2, patches: 1 },
];

function feature(id: string, area: number) {
  return {
    type: "Feature",
    geometry: { type: "Polygon", coordinates: [[[82.8, 22.8], [82.81, 22.8], [82.81, 22.81], [82.8, 22.8]]] },
    properties: { id, area_ha: area, t1: "2025-01", t2: "2026-01", ndvi_drop: 0.4, confidence: 0.8, confirmed: false },
  };
}

const files: Record<string, unknown> = {
  "/data/hasdeo/index.json": index,
  "/data/hasdeo/2025-02/cleared.geojson": { type: "FeatureCollection", features: [feature("a", 1)] },
  "/data/hasdeo/2026-01/cleared.geojson": { type: "FeatureCollection", features: [feature("b", 2.5)] },
};

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  clearCache();
  fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (init?.signal?.aborted) throw new DOMException("aborted", "AbortError");
    return new Response(JSON.stringify(files[url]), { status: files[url] ? 200 : 404 });
  });
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => vi.unstubAllGlobals());

describe("monthsInRange", () => {
  it("includes months after `before` up to and including `after`", () => {
    expect(monthsInRange(index, "2025-01", "2026-01")).toEqual(["2025-02", "2026-01"]);
    expect(monthsInRange(index, "2025-02", "2026-01")).toEqual(["2026-01"]);
    expect(monthsInRange(index, "2025-01", "2025-01")).toEqual([]);
  });
});

describe("useAoiData", () => {
  it("merges the range's layers and caches files per aoi/month", async () => {
    const { result, rerender } = renderHook(({ before, after }) => useAoiData(aoi, before, after), {
      initialProps: { before: "2025-01", after: "2026-01" },
    });
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.features.map((f) => f.properties.id)).toEqual(["a", "b"]);
    expect(result.current.hectares).toBeCloseTo(3.5);
    expect(fetchMock).toHaveBeenCalledTimes(3);

    // Narrow the range, then widen it again: everything comes from the cache.
    rerender({ before: "2025-02", after: "2026-01" });
    await waitFor(() => expect(result.current.features.map((f) => f.properties.id)).toEqual(["b"]));
    rerender({ before: "2025-01", after: "2026-01" });
    await waitFor(() => expect(result.current.features).toHaveLength(2));
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("aborts the previous request when the selection changes", async () => {
    const signals: AbortSignal[] = [];
    fetchMock.mockImplementation(async (url: string, init?: RequestInit) => {
      signals.push(init!.signal!);
      await new Promise((r) => setTimeout(r, 20));
      if (init?.signal?.aborted) throw new DOMException("aborted", "AbortError");
      return new Response(JSON.stringify(files[url]), { status: 200 });
    });
    const { result, rerender } = renderHook(({ after }) => useAoiData(aoi, "2025-01", after), {
      initialProps: { after: "2025-02" },
    });
    rerender({ after: "2026-01" });
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(signals[0].aborted).toBe(true);
    expect(result.current.error).toBeNull();
    expect(result.current.features).toHaveLength(2);
  });
});
