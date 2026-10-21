// Minimal typings for untyped map plugins (loaded lazily).
declare module "georaster" {
  const parseGeoraster: (input: string | ArrayBuffer) => Promise<unknown>;
  export default parseGeoraster;
}

declare module "georaster-layer-for-leaflet" {
  import type { GridLayer, GridLayerOptions } from "leaflet";
  interface GeoRasterLayerOptions extends GridLayerOptions {
    georaster: unknown;
    resolution?: number;
    pixelValuesToColorFn?: (values: number[]) => string | null;
  }
  export default class GeoRasterLayer extends GridLayer {
    constructor(options: GeoRasterLayerOptions);
  }
}
