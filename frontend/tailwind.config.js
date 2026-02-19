/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#458EE2',
        secondary: '#41C185',
        accent: '#FFBD59',
        success: '#41C185',
        warning: '#FFCF87',
        danger: '#E85D5D',
        surface: '#FFFFFF',
        canvas: '#F5F5F5',
        body: '#333333',
        muted: '#666666',
        subtle: '#999999',
        'accent-light': '#FFF2DF',
        'accent-soft': '#FFE7C2',
        brand: {
          yellow: '#FFBD59',
          yellowLight: '#FFCF87',
          yellowSoft: '#FFE7C2',
          yellowPale: '#FFF2DF',
          green: '#41C185',
          blue: '#458EE2',
          text: '#333333',
          muted: '#666666',
          subtle: '#999999',
          white: '#FFFFFF',
          canvas: '#F5F5F5',
          danger: '#E85D5D',
          dangerLight: '#FDE8E8',
        },
      }
    },
  },
  plugins: [],
}
