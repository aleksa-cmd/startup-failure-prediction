import { useState } from 'react'
import regressionLeaderboard from '../data/regression_leaderboard.json'
import classificationLeaderboard from '../data/classification_leaderboard.json'
import regressionImportance from '../data/regression_feature_importance.json'
import classificationImportance from '../data/classification_feature_importance.json'
import './Dashboard.css'

const MODEL_DISPLAY_NAMES = {
  DecisionTree: 'Tree',
  XGBoost: 'XGBoost',
  LassoCV: 'Lasso',
  LogisticRegressionCV: 'LogReg',
  Dummy: 'Guessing',
}

const MEDALS = { 1: '🥇', 2: '🥈', 3: '🥉' }

// Curated, human-readable feature lists per view -- NOT simply "top 5 by
// magnitude" (that would surface n_flags/n_flags_sq on the classification
// side, which are engineered meta-features with no intuitive story for a
// non-technical reader). Order matches descending real importance for both
// views (verified against the source JSON).
const REGRESSION_FEATURES = [
  { raw: 'decade_started', label: 'When founded' },
  { raw: 'raised_musd', label: 'Money raised' },
  { raw: 'Sector', label: 'Sector' },
  { raw: 'Giants', label: 'Giants pressure' },
  { raw: 'No Budget', label: 'No Budget pressure' },
]

const CLASSIFICATION_FEATURES = [
  { raw: 'raised_musd', label: 'Money raised' },
  { raw: 'decade_started', label: 'When founded' },
  { raw: 'Sector', label: 'Sector' },
  { raw: 'big_tech_pressure', label: 'Big Tech pressure' },
  { raw: 'No Budget', label: 'No Budget pressure' },
]

// ".38" style: drop the leading zero, keep the sign, collapse near-zero to a
// clean ".00" (the baseline "guessing the average scores 0" story shouldn't
// render as a confusing "-.00").
function formatScore(v) {
  const rounded = v.toFixed(2)
  if (rounded === '0.00' || rounded === '-0.00') return '.00'
  return rounded.startsWith('0.') ? rounded.slice(1) : rounded
}

// "0.38" style for the big stat-card numbers (keeps the leading zero).
function formatStat(v) {
  return v.toFixed(2)
}

function buildLeaderboardRows(data, valueKey) {
  const sorted = [...data].sort((a, b) => b[valueKey] - a[valueKey])
  const maxValue = sorted[0][valueKey]
  return sorted.map((row, i) => {
    const rank = i + 1
    const isTree = row.model === 'DecisionTree'
    const isDummy = row.model === 'Dummy'
    return {
      model: row.model,
      displayName: MODEL_DISPLAY_NAMES[row.model] ?? row.model,
      value: row[valueKey],
      rank,
      isTree,
      isDummy,
      // Proportional to the card's max value; floored so a near-zero/negative
      // baseline still renders as a visible sliver rather than disappearing.
      widthPct: Math.max((row[valueKey] / maxValue) * 100, 1.5),
      // Medals are earned by the Tree model's OWN rank in this round, not by
      // whichever model actually finished 1st/2nd -- confirmed against the
      // reference: Round 1 Tree wins and gets the only medal (🥇); Round 2
      // XGBoost wins with no medal shown, Tree places 2nd and gets 🥈. This
      // is the "Tree" story throughout, not a generic leaderboard.
      medal: isTree ? MEDALS[rank] : null,
    }
  })
}

function buildFeatureRows(features, importanceData) {
  const lookup = Object.fromEntries(importanceData.map((d) => [d.feature, d.importance]))
  const withValues = features.map((f) => ({ ...f, value: lookup[f.raw] ?? 0 }))
  const maxAbs = Math.max(...withValues.map((f) => Math.abs(f.value)))
  return withValues.map((f, i) => ({
    ...f,
    widthPct: (Math.abs(f.value) / maxAbs) * 100,
    rank: i,
  }))
}

function featureBarColor(rank) {
  if (rank === 0) return 'var(--accent)'
  if (rank === 1) return 'oklch(0.55 0.2 275 / .55)'
  return 'var(--bar-muted)'
}

function LeaderboardCard({ title, caption, rows, formatValue }) {
  return (
    <div className="card leaderboard-card">
      <div className="leaderboard-title">{title}</div>
      <div className="leaderboard-caption">{caption}</div>
      <div className="leaderboard-rows">
        {rows.map((row) => (
          <div key={row.model} className="leaderboard-row">
            <span
              className="leaderboard-label"
              style={{
                fontWeight: row.isTree ? 700 : row.isDummy ? 400 : 600,
                color: row.isTree
                  ? 'var(--accent-deep)'
                  : row.isDummy
                    ? 'var(--faint)'
                    : row.rank === 1
                      ? 'var(--ink)'
                      : 'var(--ink-soft)',
              }}
            >
              {row.displayName} {row.medal ?? ''}
            </span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{
                  width: `${row.widthPct}%`,
                  background: row.isTree
                    ? 'var(--accent)'
                    : row.isDummy
                      ? 'var(--line)'
                      : row.rank === 1
                        ? 'var(--ink)'
                        : 'var(--bar-muted)',
                }}
              />
            </div>
            <span
              className="leaderboard-value"
              style={{
                fontWeight: row.isTree || row.rank === 1 ? 700 : 400,
                color: row.isDummy ? 'var(--faint)' : row.isTree || row.rank === 1 ? 'var(--ink)' : 'var(--muted)',
              }}
            >
              {formatValue(row.value)}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

function FeatureImportanceCard({ featureView, onToggle }) {
  const rows =
    featureView === 'regression'
      ? buildFeatureRows(REGRESSION_FEATURES, regressionImportance.xgb_permutation)
      : buildFeatureRows(CLASSIFICATION_FEATURES, classificationImportance.xgb_permutation)

  return (
    <div className="card feature-card">
      <div className="feature-card-header">
        <div className="leaderboard-title">What actually matters</div>
        <div className="feature-toggle">
          <button
            type="button"
            className={`toggle-pill ${featureView === 'regression' ? 'active' : ''}`}
            onClick={() => onToggle('regression')}
          >
            Years survived
          </button>
          <button
            type="button"
            className={`toggle-pill ${featureView === 'classification' ? 'active' : ''}`}
            onClick={() => onToggle('classification')}
          >
            Failure timing
          </button>
        </div>
      </div>
      <div className="feature-rows">
        {rows.map((f) => (
          <div key={f.raw} className="feature-row">
            <span className={`feature-label ${f.rank <= 1 ? 'feature-label-strong' : ''}`}>{f.label}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${f.widthPct}%`, background: featureBarColor(f.rank) }} />
            </div>
            <span className="feature-annotation">{f.rank === 0 ? 'strongest' : ''}</span>
          </div>
        ))}
      </div>
      <div className="feature-footnote">
        <strong>Surprise:</strong> the era a startup was born in matters more than any reason it failed.
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [featureView, setFeatureView] = useState('regression')

  const regressionRows = buildLeaderboardRows(regressionLeaderboard, 'test_R2')
  const classificationRows = buildLeaderboardRows(classificationLeaderboard, 'f1_macro')

  const bestRegression = regressionRows[0]
  const bestClassification = classificationRows[0]
  const treeClassification = classificationRows.find((r) => r.isTree)

  return (
    <div>
      <div className="stat-row">
        <div className="card headline-card">
          <div className="eyebrow panel-eyebrow-dark">Headline</div>
          <div className="headline-text">The simple Decision Tree beat XGBoost on regression 🏆</div>
        </div>
        <div className="card stat-card">
          <div className="eyebrow section-label">Best R²</div>
          <div className="stat-number">{formatStat(bestRegression.value)}</div>
          <div className="stat-caption">{bestRegression.displayName} · years survived</div>
        </div>
        <div className="card stat-card">
          <div className="eyebrow section-label">Best F1</div>
          <div className="stat-number">{formatStat(bestClassification.value)}</div>
          <div className="stat-caption">
            {bestClassification.displayName} · timing
            {!bestClassification.isTree && ` (Tree: ${formatStat(treeClassification.value)})`}
          </div>
        </div>
      </div>

      <div className="leaderboard-grid">
        <LeaderboardCard
          title="Round 1: predicting years survived"
          caption="R² — guessing the average scores 0"
          rows={regressionRows}
          formatValue={formatScore}
        />
        <LeaderboardCard
          title="Round 2: timing the failure"
          caption="Early / typical / long-run · F1 score"
          rows={classificationRows}
          formatValue={formatScore}
        />
      </div>

      <FeatureImportanceCard featureView={featureView} onToggle={setFeatureView} />
    </div>
  )
}
