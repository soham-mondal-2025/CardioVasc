import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts';

export default function Dashboard({ predictionData, loading }) {
  if (loading) return <section className="panel dashboard-panel"><h3>Prediction results</h3><p>Updating demo output…</p></section>;
  if (!predictionData) return <section className="panel dashboard-panel empty-dashboard">
    <div><p className="eyebrow">MODEL OUTPUT</p><h3>Prediction dashboard</h3><p>Enter example features and run a prediction to populate vessel values and feature signals.</p></div>
    <div className="empty-mark">↗</div>
  </section>;

  const vessels = predictionData.vesselStenosis || {};
  const features = Object.entries(predictionData.featureContributions || {})
    .map(([name, value]) => ({ name, value: Math.round(Number(value || 0) * 100) / 1 }))
    .filter(item => Number.isFinite(item.value))
    .sort((a, b) => b.value - a.value);

  return <section className="panel dashboard-panel">
    <div className="panel-heading dashboard-heading">
      <div><p className="eyebrow">MODEL OUTPUT</p><h3>Prediction dashboard</h3></div>
      <span className="demo-chip">{predictionData.mode || 'DEMO'}</span>
    </div>
    <div className="results-grid">
      <div className="overall-result">
        <span className="result-label">Overall CAD demo classification</span>
        <strong className={predictionData.overallCAD ? 'risk-high' : 'risk-low'}>{predictionData.overallCAD ? 'Higher demo score' : 'Lower demo score'}</strong>
        <div className="probability-track"><span style={{ width: `${Math.round((predictionData.overallProbability || 0) * 100)}%` }} /></div>
        <small>{Math.round((predictionData.overallProbability || 0) * 100)}% illustrative score · not calibrated</small>
      </div>
      {['LAD', 'LCX', 'RCA'].map(name => {
        const value = Number(vessels[name]);
        const pct = Number.isFinite(value) ? Math.round(value * 100) : null;
        return <div className="vessel-card" key={name}>
          <div className={`vessel-symbol ${name.toLowerCase()}`}>{name}</div>
          <div><span>{name} stenosis demo value</span><strong>{pct === null ? '—' : `${pct}%`}</strong></div>
          <div className="mini-track"><span style={{ width: `${pct ?? 0}%` }} /></div>
        </div>;
      })}
    </div>
    <div className="explanation-section">
      <div className="explanation-heading"><div><p className="eyebrow">INTERPRETABILITY</p><h4>Input signal overview</h4></div><span>Heuristic signal · not SHAP</span></div>
      {features.length ? <div className="chart-container">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={features} margin={{ top: 8, right: 8, left: -20, bottom: 30 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5eaf2" />
            <XAxis dataKey="name" interval={0} angle={-18} textAnchor="end" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip formatter={value => [`${Number(value).toFixed(1)} relative units`, 'Demo signal']} />
            <Bar dataKey="value" fill="#6276e8" radius={[5, 5, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div> : <p>No feature signals returned.</p>}
      <p className="chart-note">These bars reflect the demo heuristic's input signals. They are not validated feature attributions or causal effects.</p>
    </div>
  </section>;
}
