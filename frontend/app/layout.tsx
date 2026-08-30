import "./globals.css";

export const metadata = {
  title: "Demand Intelligence",
  description: "AI-powered demand forecasting and inventory optimization dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
