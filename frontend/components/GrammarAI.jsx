import { useState } from 'react';
import axios from 'axios';

export default function GrammarAI() {
  const [topic, setTopic] = useState('');
  const [explanation, setExplanation] = useState('');
  const [loading, setLoading] = useState(false);

  const getExplanation = async () => {
    setLoading(true);
    const res = await axios.post('http://localhost:8000/ai/grammar', { topic });
    setExplanation(res.data.explanation);
    setLoading(false);
  };

  return (
    <div className="bg-white shadow-xl rounded-xl p-6">
      <input
        value={topic}
        onChange={(e) => setTopic(e.target.value)}
        placeholder="Enter grammar topic"
        className="border p-2 rounded w-full"
      />
      <button
        onClick={getExplanation}
        className="mt-4 bg-green-500 text-white px-4 py-2 rounded-lg"
      >
        Explain
      </button>
      {loading && <p>Loading...</p>}
      {explanation && <pre className="mt-4">{explanation}</pre>}
    </div>
  );
}
