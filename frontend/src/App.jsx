import { useState } from 'react'
import Predictor from './components/Predictor'

export default function App() {
  const [view, setView] = useState('predictor')

  return (
    <div>
      <nav>
        <button onClick={() => setView('predictor')} disabled={view === 'predictor'}>
          Predictor
        </button>
        <button onClick={() => setView('dashboard')} disabled={view === 'dashboard'}>
          Dashboard
        </button>
      </nav>
      {view === 'predictor' && <Predictor />}
      {view === 'dashboard' && <p>Dashboard coming soon.</p>}
    </div>
  )
}
