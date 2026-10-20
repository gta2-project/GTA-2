import { defineConfig } from 'vite';
import type { Plugin } from 'vite';
import react from '@vitejs/plugin-react';

const libraryPages: Plugin = {
  name: 'library-pages',
  configureServer(server) {
    server.middlewares.use((request, _response, next) => {
      const path = request.url?.split('?')[0];
      if (path && ['/', '/admin', '/library', '/run'].includes(path)) request.url = '/library.html';
      next();
    });
  }
};

export default defineConfig(({ mode }) => ({
  plugins: mode === 'library' ? [react(), libraryPages] : [react()],
  build: {
    rollupOptions: { input: { main: 'index.html', library: 'library.html' } }
  },
  server: {
    port: mode === 'library' ? 5174 : 5173,
    proxy: {
      '/api': {
        target: mode === 'library' ? 'http://127.0.0.1:8001' : 'http://127.0.0.1:8000',
        ws: true,
        changeOrigin: true
      }
    }
  }
}));
