/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
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
