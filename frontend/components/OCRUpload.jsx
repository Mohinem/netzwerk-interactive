import { useState } from 'react';
import axios from 'axios';

export default function OCRUpload() {
  const [file, setFile] = useState(null);
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    const res = await axios.post('http://localhost:8000/ocr/digitize', formData);
    setText(res.data.extracted_text);
    setLoading(false);
  };

  return (
    <div className="bg-white shadow-xl rounded-xl p-6">
      <input type="file" onChange={(e) => setFile(e.target.files[0])} accept=".pdf" />
      <button
        onClick={handleUpload}
        className="mt-4 bg-blue-500 text-white px-4 py-2 rounded-lg"
        disabled={loading}
      >
        {loading ? 'Processing...' : 'Digitize'}
      </button>
      {text && <pre className="mt-4 overflow-auto h-64">{text}</pre>}
    </div>
  );
}
