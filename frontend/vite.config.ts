import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The dev server proxies to the FastAPI backend so the browser talks to one origin.
// Proxying the socket as well as the REST calls means no CORS preflight and no
// second URL to configure.
const backend = process.env.REPOPILOT_BACKEND ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: backend, changeOrigin: true },
      '/ws': { target: backend, ws: true, changeOrigin: true },
    },
  },
})
