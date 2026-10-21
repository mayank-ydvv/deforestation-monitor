/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        forest: "#22c55e",
        cleared: "#ef4444",
      },
    },
  },
  plugins: [],
};
