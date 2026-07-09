import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    // Injected at build time. Set VITE_BACKEND_URL in Vercel dashboard.
    // Example: "your-org--aegis-backend.modal.run"  (no https://)
    // In local dev, leave unset — frontend falls back to same-origin.
    __BACKEND_URL__: JSON.stringify(process.env.VITE_BACKEND_URL || ''),
  },
})
