import type { LatLngBoundsExpression } from "leaflet";
import { useEffect, type ReactNode } from "react";
import { MapContainer, TileLayer, useMap } from "react-leaflet";

import type { Aoi } from "../lib/types";

const ESRI_IMAGERY =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const ESRI_ATTRIBUTION =
  "Basemap &copy; Esri, Maxar, Earthstar Geographics &amp; the GIS User Community";
export const DATA_ATTRIBUTION =
  'Contains modified <a href="https://sentinel.esa.int/">Copernicus Sentinel</a> data (ESA) · ' +
  '<a href="https://esa-worldcover.org/">ESA WorldCover</a> 2021 (CC BY 4.0) · ' +
  'validation: <a href="https://glad.earthengine.app/view/global-forest-change">Hansen/UMD GFC</a>';

/** [[south, west], [north, east]] for Leaflet from a [west, south, east, north] bbox. */
export function bboxToBounds(bbox: Aoi["bbox"]): LatLngBoundsExpression {
  const [w, s, e, n] = bbox;
  return [
    [s, w],
    [n, e],
  ];
}

function FlyToAoi({ aoi }: { aoi: Aoi | null }) {
  const map = useMap();
  useEffect(() => {
    if (aoi) map.flyToBounds(bboxToBounds(aoi.bbox), { padding: [24, 24], duration: 1.2 });
  }, [aoi, map]);
  return null;
}

interface MapViewProps {
  aoi: Aoi | null;
  children?: ReactNode;
}

/** Full-screen map: Esri World Imagery basemap plus whatever data layers are passed in. */
export default function MapView({ aoi, children }: MapViewProps) {
  return (
    <MapContainer
      className="h-full w-full"
      center={aoi?.center ?? [0, 20]}
      zoom={aoi ? 11 : 3}
      minZoom={3}
      maxZoom={17}
      zoomControl={false}
      worldCopyJump
      attributionControl
    >
      <TileLayer url={ESRI_IMAGERY} attribution={`${ESRI_ATTRIBUTION} · ${DATA_ATTRIBUTION}`} maxNativeZoom={17} />
      <FlyToAoi aoi={aoi} />
      {children}
    </MapContainer>
  );
}
