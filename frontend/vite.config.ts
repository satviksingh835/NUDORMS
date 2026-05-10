import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, "") },
    },
  },
  optimizeDeps: {
    // Both viewers use ESM bundles that Vite can't pre-bundle; exclude both.
    exclude: ["@mkkellogg/gaussian-splats-3d", "@sparkjsdev/spark"],
  },
});
