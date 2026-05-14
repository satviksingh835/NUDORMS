import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, "") },
    },
  },
  resolve: {
    // Force a single Three.js instance — PSV bundles three internally and
    // having two copies causes NaN geometry and a blank panorama.
    dedupe: ["three"],
  },
});
