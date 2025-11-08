import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';

console.log('index.jsx loaded');
console.log('React:', React);
console.log('createRoot:', createRoot);

const container = document.getElementById('root');
console.log('Container element:', container);

if (container) {
  const root = createRoot(container);
  console.log('Root created, rendering App...');
  root.render(<App />);
  console.log('App rendered');
} else {
  console.error('Root element not found!');
}
