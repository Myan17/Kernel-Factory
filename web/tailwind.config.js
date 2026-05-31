/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "rgb(var(--canvas) / <alpha-value>)",
        surface: "rgb(var(--surface) / <alpha-value>)",
        ink: "rgb(var(--ink) / <alpha-value>)",
        muted: "rgb(var(--muted) / <alpha-value>)",
        line: "rgb(var(--line) / <alpha-value>)",
        indigo: "rgb(var(--indigo) / <alpha-value>)",
        violet: "rgb(var(--violet) / <alpha-value>)",
        duck: "rgb(var(--duck) / <alpha-value>)",
        mint: "rgb(var(--mint) / <alpha-value>)",
        coral: "rgb(var(--coral) / <alpha-value>)",
      },
      fontFamily: {
        display: ['"Bricolage Grotesque"', "ui-sans-serif", "system-ui", "sans-serif"],
        sans: ['"Plus Jakarta Sans"', "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        soft: "0 1px 2px rgb(16 16 29 / 0.04), 0 8px 24px -8px rgb(16 16 29 / 0.10)",
        lift: "0 2px 4px rgb(16 16 29 / 0.06), 0 24px 48px -16px rgb(91 91 214 / 0.28)",
        glow: "0 0 0 1px rgb(91 91 214 / 0.20), 0 0 32px -4px rgb(91 91 214 / 0.45)",
      },
      borderRadius: {
        xl2: "1.25rem",
      },
      keyframes: {
        floaty: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-8px)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "200% 0" },
          "100%": { backgroundPosition: "-200% 0" },
        },
        gridpan: {
          "0%": { backgroundPosition: "0 0" },
          "100%": { backgroundPosition: "48px 48px" },
        },
      },
      animation: {
        floaty: "floaty 4s ease-in-out infinite",
        shimmer: "shimmer 8s linear infinite",
        gridpan: "gridpan 6s linear infinite",
      },
    },
  },
  plugins: [],
};
