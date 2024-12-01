import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

const root = ReactDOM.createRoot(document.getElementById('root'));

if (import.meta.VITE_ENVIRONMENT === 'development') {
  root.render(<React.StrictMode><App /></React.StrictMode>);
} else {
  root.render(<App />);
}