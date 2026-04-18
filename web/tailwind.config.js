/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        parchment: "var(--color-parchment, #f5f0e8)",
        "parchment-dark": "var(--color-parchment-dark, #ede6d6)",
        ink: "var(--color-ink, #1a1108)",
        "ink-light": "var(--color-ink-light, #4a3f2f)",
        burgundy: "var(--color-burgundy, #8b1a1a)",
        "burgundy-dark": "#6b1414",
        "burgundy-light": "#cc4444",
        navy: "#2d4a6b",
        cream: "var(--color-cream, #fffdf9)",
      },
      fontFamily: {
        serif: ["var(--font-playfair)", "Georgia", "serif"],
        body: ["var(--font-crimson)", "Georgia", "serif"],
        sans: ["var(--font-lato)", "system-ui", "sans-serif"],
      },
      boxShadow: {
        book: "0 2px 8px rgba(30,15,0,0.10), 0 0 0 1px rgba(30,15,0,0.06)",
        "book-hover":
          "0 8px 24px rgba(30,15,0,0.14), 0 0 0 1px rgba(30,15,0,0.08)",
        "book-cover":
          "3px 3px 10px rgba(0,0,0,0.25), inset -2px 0 4px rgba(0,0,0,0.1)",
      },
    },
  },
  plugins: [],
};
