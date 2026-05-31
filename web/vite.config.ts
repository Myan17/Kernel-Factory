/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// On `vite build` (GitHub Pages) assets must resolve under the project path
// https://myan17.github.io/QuackHacks/ ; local dev stays at root.
export default defineConfig(({ command }) => ({
  base: command === "build" ? "/QuackHacks/" : "/",
  plugins: [react()],
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
}));
