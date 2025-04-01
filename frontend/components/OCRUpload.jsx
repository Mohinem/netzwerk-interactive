import React, { useState, useEffect } from 'react';
import axios from 'axios';

function PageViewer({ page }) {
  const [scale, setScale] = useState(1);
  const [scaledWidth, setScaledWidth] = useState(page.width);
  const [scaledHeight, setScaledHeight] = useState(page.height);

  useEffect(() => {
    // Use the full viewport width for scaling
    const containerWidth = window.innerWidth;
    const newScale = containerWidth / page.width;
    setScale(newScale);
    setScaledWidth(page.width * newScale);
    setScaledHeight(page.height * newScale);
  }, [page]);

  return (
    // Outer container takes full viewport width
    <div
      style={{
        width: '100vw',
        marginBottom: '20px', // Adjust margin as needed
      }}
    >
      <div
        style={{
          position: 'relative',
          width: scaledWidth,    // Exactly the scaled width
          height: scaledHeight,  // Exactly the scaled height
          overflow: 'hidden',
        }}
      >
        {/* Inner container at original size, then scaled */}
        <div
          style={{
            position: 'absolute',
            width: page.width,
            height: page.height,
            transform: `scale(${scale})`,
            transformOrigin: 'top left',
          }}
        >
          <img
            src={page.image}
            alt={`Page ${page.page_number}`}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: page.width,
              height: page.height,
            }}
          />
          {/* Overlay each recognized word with your original styling */}
          {page.words && page.words.map((word, idx) => (
            <div
              key={idx}
              style={{
                position: 'absolute',
                left: word.left,
                top: word.top,
                width: word.width,
                height: word.height,
                pointerEvents: 'none',
                overflow: 'hidden',
                fontSize: `${word.height}px`,  // dynamic font size based on word height
                lineHeight: `${word.height}px`,
                color: 'black',
                whiteSpace: 'nowrap',
                backgroundColor: 'rgba(255, 255, 255, 1)' // opaque white background
              }}
            >
              {word.text}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function OCRUpload() {
  const [file, setFile] = useState(null);
  const [pages, setPages] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await axios.post('http://localhost:8000/ocr/digitize', formData);
      if (res.data && res.data.pages) {
        setPages(res.data.pages);
      } else {
        setPages([]);
      }
    } catch (error) {
      console.error('Upload error:', error);
      setPages([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <input
        type="file"
        accept=".pdf"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button
        onClick={handleUpload}
        style={{ marginLeft: '10px' }}
      >
        {loading ? 'Processing...' : 'Digitize'}
      </button>

      <div style={{ marginTop: '20px' }}>
        {pages.map((page) => (
          <PageViewer key={page.page_number} page={page} />
        ))}
      </div>
    </div>
  );
}
