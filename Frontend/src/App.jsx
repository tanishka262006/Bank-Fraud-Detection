import { useState } from "react";
import "./App.css";
const TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"];
const API_URL = "http://localhost:5000/api/predict";
const INITIAL_FORM = {
  step: "",
  amount: "",
  type: "PAYMENT",
  oldbalanceOrg: "",
  newbalanceOrig: "",
  oldbalanceDest: "",
  newbalanceDest: "",
};
export default function App() {
  const [form, setForm] = useState(INITIAL_FORM);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiErrors, setApiErrors] = useState([]);

  function handleChange(e) {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
    setResult(null);
    setApiErrors([]);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setResult(null);
    setApiErrors([]);
    setLoading(true);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          step: Number(form.step),
          amount: Number(form.amount),
          type: form.type,
          oldbalanceOrg: Number(form.oldbalanceOrg),
          newbalanceOrig: Number(form.newbalanceOrig),
          oldbalanceDest: Number(form.oldbalanceDest),
          newbalanceDest: Number(form.newbalanceDest),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        setApiErrors(data.errors || [data.error || "An error occurred."]);
      } else {
        setResult(data);
      }
    } catch (err) {
      setApiErrors([
        "Could not connect to the prediction server. Make sure the API server is running on port 5000.",
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setForm(INITIAL_FORM);
    setResult(null);
    setApiErrors([]);
  }

  const pctLabel =
    result !== null
      ? `${(result.fraud_probability * 100).toFixed(2)}%`
      : null;

  let riskClass = "";
  let riskVerdict = "";
  let barClass = "";
  let iconContent = null;
  let statColorClass = "";
  let showSuspiciousNote = false;

  if (result) {
    if (result.fraud_probability >= result.threshold) {
      riskClass = "result-fraud";
      riskVerdict = "Potential Fraud Detected";
      barClass = "bar-fraud";
      statColorClass = "value-danger";
      iconContent = (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
        </svg>
      );
    } else if (result.fraud_probability >= 0.50 && result.fraud_probability < result.threshold) {
      riskClass = "result-suspicious";
      riskVerdict = "Suspicious Transaction — Manual Review Recommended";
      barClass = "bar-suspicious";
      statColorClass = "value-suspicious";
      showSuspiciousNote = true;
      iconContent = (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      );
    } else {
      riskClass = "result-safe";
      riskVerdict = "Transaction Appears Legitimate";
      barClass = "bar-safe";
      statColorClass = "value-safe";
      iconContent = (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      );
    }
  }

  return (
    <div className="page">
            <header className="header">
        <div className="header-inner">
          <div className="header-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
            </svg>
          </div>
          <div>
            <h1 className="header-title">Bank Fraud Detection</h1>
            <p className="header-subtitle">Machine Learning Transaction Risk Prediction</p>
          </div>
        </div>
      </header>

      <main className="main">
        <div className="card">
          <h2 className="section-title">Transaction Details</h2>
          <p className="section-desc">
            Enter the transaction information below to assess fraud risk using the trained XGBoost model.
          </p>

          <form id="prediction-form" onSubmit={handleSubmit} noValidate>
                        <div className="form-row">
              <div className="form-group">
                <label htmlFor="step" className="form-label">
                  Transaction Hour
                  <span className="form-hint">24-hour format (0–23)</span>
                </label>
                <input
                  id="step"
                  name="step"
                  type="number"
                  min="0"
                  max="23"
                  step="1"
                  className="form-input"
                  placeholder="e.g. 14"
                  value={form.step}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="amount" className="form-label">
                  Transaction Amount
                  <span className="form-hint">Positive value</span>
                </label>
                <input
                  id="amount"
                  name="amount"
                  type="number"
                  min="0"
                  step="0.01"
                  className="form-input"
                  placeholder="e.g. 9839.64"
                  value={form.amount}
                  onChange={handleChange}
                  required
                />
              </div>
            </div>

                        <div className="form-group">
              <label htmlFor="type" className="form-label">
                Transaction Type
              </label>
              <select
                id="type"
                name="type"
                className="form-input form-select"
                value={form.type}
                onChange={handleChange}
                required
              >
                {TRANSACTION_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </div>

                        <div className="balance-section">
              <h3 className="balance-heading">Sender Account</h3>
              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="oldbalanceOrg" className="form-label">
                    Sender Balance Before
                  </label>
                  <input
                    id="oldbalanceOrg"
                    name="oldbalanceOrg"
                    type="number"
                    min="0"
                    step="0.01"
                    className="form-input"
                    placeholder="e.g. 170136.00"
                    value={form.oldbalanceOrg}
                    onChange={handleChange}
                    required
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="newbalanceOrig" className="form-label">
                    Sender Balance After
                  </label>
                  <input
                    id="newbalanceOrig"
                    name="newbalanceOrig"
                    type="number"
                    min="0"
                    step="0.01"
                    className="form-input"
                    placeholder="e.g. 160296.36"
                    value={form.newbalanceOrig}
                    onChange={handleChange}
                    required
                  />
                </div>
              </div>
            </div>

                        <div className="balance-section">
              <h3 className="balance-heading">Receiver Account</h3>
              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="oldbalanceDest" className="form-label">
                    Receiver Balance Before
                  </label>
                  <input
                    id="oldbalanceDest"
                    name="oldbalanceDest"
                    type="number"
                    min="0"
                    step="0.01"
                    className="form-input"
                    placeholder="e.g. 0.00"
                    value={form.oldbalanceDest}
                    onChange={handleChange}
                    required
                  />
                </div>
                <div className="form-group">
                  <label htmlFor="newbalanceDest" className="form-label">
                    Receiver Balance After
                  </label>
                  <input
                    id="newbalanceDest"
                    name="newbalanceDest"
                    type="number"
                    min="0"
                    step="0.01"
                    className="form-input"
                    placeholder="e.g. 0.00"
                    value={form.newbalanceDest}
                    onChange={handleChange}
                    required
                  />
                </div>
              </div>
            </div>

                        {apiErrors.length > 0 && (
              <div className="error-box" role="alert">
                <strong>Please correct the following:</strong>
                <ul>
                  {apiErrors.map((err, i) => (
                    <li key={i}>{err}</li>
                  ))}
                </ul>
              </div>
            )}

                        <div className="btn-row">
              <button
                id="predict-btn"
                type="submit"
                className="btn btn-primary"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="spinner" aria-hidden="true" />
                    Analysing…
                  </>
                ) : (
                  "Predict Fraud"
                )}
              </button>
              <button
                id="reset-btn"
                type="button"
                className="btn btn-secondary"
                onClick={handleReset}
                disabled={loading}
              >
                Reset
              </button>
            </div>
          </form>
        </div>

                {result && (
          <div className={`result-card ${riskClass}`}>
            <div className="result-icon" aria-hidden="true">
              {iconContent}
            </div>

            <div className="result-body">
              <h2 className="result-verdict">
                {riskVerdict}
              </h2>

              <div className="result-stats">
                <div className="stat">
                  <span className="stat-label">Fraud Risk Score</span>
                  <span className={`stat-value ${statColorClass}`}>
                    {pctLabel}
                  </span>
                </div>
                <div className="stat-divider" aria-hidden="true" />
                <div className="stat">
                  <span className="stat-label">Decision Threshold</span>
                  <span className="stat-value value-neutral">
                    {(result.threshold * 100).toFixed(2)}%
                  </span>
                </div>
              </div>

                            <div className="prob-bar-track" role="progressbar"
                aria-valuenow={Math.round(result.fraud_probability * 100)}
                aria-valuemin="0" aria-valuemax="100">
                <div
                  className={`prob-bar-fill ${barClass}`}
                  style={{ width: `${Math.min(result.fraud_probability * 100, 100)}%` }}
                />
                <div
                  className="threshold-marker"
                  style={{ left: `${result.threshold * 100}%` }}
                  title={`Threshold: ${(result.threshold * 100).toFixed(1)}%`}
                />
              </div>
              <div className="prob-bar-labels">
                <span>0%</span>
                <span>Threshold {(result.threshold * 100).toFixed(1)}%</span>
                <span>100%</span>
              </div>

              {showSuspiciousNote && (
                <p className="suspicious-note">
                  This transaction has elevated fraud risk but does not cross the model's final fraud decision threshold. Additional verification or manual review is recommended.
                </p>
              )}

              <p className="disclaimer">
                This prediction is generated by a machine-learning model and should be treated as a
                risk indicator, not a final banking decision.
              </p>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Bank Fraud Detection &mdash; XGBoost ML Model &mdash; Interview Demo Project</p>
      </footer>
    </div>
  );
}
