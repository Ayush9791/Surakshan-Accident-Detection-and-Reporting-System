import "./globals.css";

export const metadata = {
  title: "Surakshan | Traffic Incident Operations",
  description: "Real-time traffic incident monitoring and simulated response coordination.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
