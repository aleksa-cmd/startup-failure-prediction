import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'
import regressionLeaderboard from '../data/regression_leaderboard.json'
import classificationLeaderboard from '../data/classification_leaderboard.json'
import regressionImportance from '../data/regression_feature_importance.json'
import classificationImportance from '../data/classification_feature_importance.json'

const DUMMY_REGRESSION_R2 = regressionLeaderboard.find((row) => row.model === 'Dummy').test_R2
const DUMMY_CLASSIFICATION_F1 = classificationLeaderboard.find((row) => row.model === 'Dummy').f1_macro

export default function Dashboard() {
  return (
    <div>
      <h2>Model Findings</h2>

      <section>
        <h3>Regression: test R2 by model (dashed line = Dummy baseline)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={regressionLeaderboard}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="model" />
            <YAxis />
            <Tooltip />
            <Legend />
            <ReferenceLine y={DUMMY_REGRESSION_R2} stroke="red" strokeDasharray="4 4" label="Dummy baseline" />
            <Bar dataKey="test_R2" fill="#4c72b0" />
          </BarChart>
        </ResponsiveContainer>
      </section>

      <section>
        <h3>Classification: test F1-macro by model (dashed line = Dummy baseline)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={classificationLeaderboard}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="model" />
            <YAxis />
            <Tooltip />
            <Legend />
            <ReferenceLine y={DUMMY_CLASSIFICATION_F1} stroke="red" strokeDasharray="4 4" label="Dummy baseline" />
            <Bar dataKey="f1_macro" fill="#dd8452" />
          </BarChart>
        </ResponsiveContainer>
      </section>

      <section>
        <h3>Regression: top features (Lasso coefficient magnitude)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={regressionImportance.lasso} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="feature" width={200} />
            <Tooltip />
            <Bar dataKey="importance" fill="#55a868" />
          </BarChart>
        </ResponsiveContainer>
      </section>

      <section>
        <h3>Regression: top features (XGBoost permutation importance)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={regressionImportance.xgb_permutation} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="feature" width={200} />
            <Tooltip />
            <Bar dataKey="importance" fill="#8172b2" />
          </BarChart>
        </ResponsiveContainer>
      </section>

      <section>
        <h3>Classification: top features (XGBoost permutation importance)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={classificationImportance.xgb_permutation} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="feature" width={200} />
            <Tooltip />
            <Bar dataKey="importance" fill="#c44e52" />
          </BarChart>
        </ResponsiveContainer>
      </section>
    </div>
  )
}
