import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Dev-only proxy to avoid CORS when UI runs on :5173 and API on :8008.
      "/health": "http://127.0.0.1:8000",
      "/pipeline": "http://127.0.0.1:8000",
      "/orchestrate": "http://127.0.0.1:8000",
      "/debate": "http://127.0.0.1:8000",
    },
  },
});
