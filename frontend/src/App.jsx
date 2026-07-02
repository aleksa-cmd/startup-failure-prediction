import { useState } from 'react'
import './App.css'
import Header from './components/Header'
import Predictor from './components/Predictor'
import Dashboard from './components/Dashboard'

export default function App() {
  const [view, setView] = useState('predictor')

  return (
    <div className="page">
      <Header view={view} onNavigate={setView} />
      <div className="view-content">
        {view === 'predictor' && <Predictor />}
        {view === 'dashboard' && <Dashboard />}
      </div>
    </div>
  )
}
