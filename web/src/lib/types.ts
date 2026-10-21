/** Types for the data contract in docs/data-contract.json (pipeline writes, site reads). */
import type { Feature, FeatureCollection, MultiPolygon, Polygon } from "geojson";

/** One entry of aois.json. `center` is [lat, lon]; `bbox` is [west, south, east, north]. */
export interface Aoi {
  id: string;
  name: string;
  bbox: [number, number, number, number];
  center: [number, number];
  months: string[];
  compare: "consecutive" | "yoy";
  /** "" = rasters live next to the JSON; otherwise an absolute URL ending in "/". */
  raster_base: string;
}

/** One row of <aoi>/index.json: clearing detected in `month` relative to `t1`. */
export interface IndexRow {
  month: string;
  t1: string;
  hectares: number;
  patches: number;
}

export interface ClearedProps {
  id: string;
  area_ha: number;
  t1: string;
  t2: string;
  ndvi_drop: number;
  confidence: number;
  confirmed: boolean;
}

export type ClearedFeature = Feature<Polygon | MultiPolygon, ClearedProps>;
export type ClearedCollection = FeatureCollection<Polygon | MultiPolygon, ClearedProps>;

export interface Scores {
  iou: number;
  f1: number;
  precision: number;
  recall: number;
  baseline_iou?: number;
}

export interface Metrics extends Scores {
  evaluated_on: string;
  per_region: Record<string, Scores & { split: string; n_tiles: number; months?: Record<string, Scores> }>;
  change_validation?: Record<
    string,
    { t1: string; t2: string; hansen_years: number[]; pixel_precision: number; pixel_recall: number; patch_precision: number }
  >;
}
