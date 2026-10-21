/// <reference types="vitest" />
import { existsSync } from "node:fs";
import { join } from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

/**
 * In dev, answer 404 for missing files under /data/ instead of Vite's SPA fallback
 * (index.html with 200). Static hosts 404 there too; georaster probes `forest.tif.ovr`
 * and would otherwise try to parse HTML as a TIFF.
 */
function dataNotFound(): Plugin {
  return {
    name: "data-not-found",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const path = decodeURIComponent((req.url ?? "").split("?")[0]);
        if (path.startsWith("/data/") && !existsSync(join(server.config.publicDir, path))) {
          res.statusCode = 404;
          res.end("Not found");
          return;
        }
        next();
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), dataNotFound()],
  build: {
    target: "es2020",
    rollupOptions: {
      output: {
        // Keep the map core in its own long-cached chunk; charts/swipe/COG load on demand.
        manualChunks: { leaflet: ["leaflet", "react-leaflet"] },
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
  },
});
