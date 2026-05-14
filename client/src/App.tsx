import { useEffect, useState } from 'react';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { ChatPage } from './pages/ChatPage';
import { HomePage } from './pages/HomePage';

function ThemeToggle() {
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'dark');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggle = () => setTheme(theme === 'dark' ? 'light' : 'dark');

  return (
    <button 
      onClick={toggle} 
      style={{ position: 'absolute', top: '1rem', right: '1rem', zIndex: 1000, background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '8px', borderRadius: '50%', cursor: 'pointer' }}
      title="Toggle Theme"
    >
      {theme === 'dark' ? '☀️' : '🌙'}
    </button>
  );
}

export default function App() {
  return (
    <>
      <ThemeToggle />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/chat" element={<ChatPage />} />
        </Routes>
      </BrowserRouter>
    </>
  );
}
