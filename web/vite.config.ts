import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The built interface ships inside the Python package and is served by the station itself.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../src/avavision/station/ui", emptyOutDir: true, chunkSizeWarningLimit: 800 },
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
});
