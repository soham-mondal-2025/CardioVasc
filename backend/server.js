const express = require('express');
const cors = require('cors');

const app = express();
const PORT = Number(process.env.PORT || 5002);

app.use(cors());
app.use(express.json({ limit: '1mb' }));

app.get('/api/hello', (_req, res) => {
  res.json({ message: 'Hello from Cardiovascular Backend!', mode: 'DEMO' });
});

// Explicitly a deterministic demo heuristic, NOT a trained or clinically validated model.
app.post('/api/predict', (req, res) => {
  const body = req.body || {};
  const age = clampNumber(body.age, 18, 100, 45);
  const systolicBP = clampNumber(body.systolicBP, 70, 250, 120);
  const cholesterol = clampNumber(body.cholesterol, 80, 500, 180);
  const pulse = clampNumber(body.pulse, 30, 220, 72);
  const stElevation = Boolean(body.stElevation);
  const tInversion = Boolean(body.tInversion);
  const rwma = Boolean(body.rwma);

  const score = sigmoid(
    -3.1 +
    Math.max(0, age - 40) * 0.025 +
    Math.max(0, systolicBP - 120) * 0.012 +
    Math.max(0, cholesterol - 180) * 0.006 +
    Math.max(0, pulse - 75) * 0.008 +
    (stElevation ? 0.65 : 0) +
    (tInversion ? 0.45 : 0) +
    (rwma ? 0.85 : 0)
  );

  const vessels = {
    LAD: clamp01(score * 0.95 + (stElevation ? 0.08 : 0)),
    LCX: clamp01(score * 0.78 + (tInversion ? 0.06 : 0)),
    RCA: clamp01(score * 0.88 + (rwma ? 0.07 : 0))
  };

  const features = {
    Age: Math.max(0, (age - 40) / 60),
    'Systolic BP': Math.max(0, (systolicBP - 120) / 130),
    Cholesterol: Math.max(0, (cholesterol - 180) / 320),
    Pulse: Math.max(0, (pulse - 75) / 145),
    'ST Elevation': stElevation ? 0.65 : 0,
    'T Inversion': tInversion ? 0.45 : 0,
    RWMA: rwma ? 0.85 : 0
  };

  res.json({
    mode: 'DEMO',
    disclaimer: 'Illustrative output only. Not a medical diagnosis or trained-model prediction.',
    overallCAD: score >= 0.5,
    overallProbability: Number(score.toFixed(4)),
    vesselStenosis: Object.fromEntries(
      Object.entries(vessels).map(([key, value]) => [key, Number(value.toFixed(4))])
    ),
    featureContributions: features,
    modelName: 'Demo heuristic (not clinically validated)'
  });
});

function clampNumber(value, min, max, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(min, Math.min(max, n)) : fallback;
}
function clamp01(value) { return Math.max(0, Math.min(1, value)); }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

app.listen(PORT, '0.0.0.0', () => {
  console.log(`CardioVas API listening on http://localhost:${PORT}`);
});
