import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Spark Learning - AI Learning Platform',
  description: 'Learn Your Way',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}