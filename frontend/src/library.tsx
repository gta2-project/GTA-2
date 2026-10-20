import React from 'react';
import ReactDOM from 'react-dom/client';
import LibraryShell from './LibraryShell';
import './styles.css';
import './library.css';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode><LibraryShell /></React.StrictMode>
);
