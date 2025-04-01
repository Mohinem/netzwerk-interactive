import Link from 'next/link';

export default function Navbar() {
  return (
    <nav className="p-4 shadow bg-white">
      <div className="container mx-auto flex justify-between items-center">
        <Link href="/" className="font-bold text-xl">
          Netzwerk Interactive
        </Link>
        <div className="flex gap-4">
          <Link href="/exercises">Exercises</Link>
          <Link href="/ocr">OCR</Link>
          <Link href="/grammar">Grammar AI</Link>
          <Link href="/dashboard">Dashboard</Link>
        </div>
      </div>
    </nav>
  );
}
