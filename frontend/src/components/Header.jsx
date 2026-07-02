export default function Header({ view, onNavigate }) {
  return (
    <div className="header">
      <div className="wordmark">
        startup<span className="fail">/fail</span>
      </div>
      <div className="nav">
        <button
          type="button"
          className={`nav-pill ${view === 'predictor' ? 'active' : ''}`}
          onClick={() => onNavigate('predictor')}
        >
          Predict
        </button>
        <button
          type="button"
          className={`nav-pill ${view === 'dashboard' ? 'active' : ''}`}
          onClick={() => onNavigate('dashboard')}
        >
          Results
        </button>
      </div>
    </div>
  )
}
