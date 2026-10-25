
const express = require("express");
const cors = require("cors");
const { spawn } = require("child_process");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 5000;
app.use(cors({ origin: ["http://localhost:5173", "http://127.0.0.1:5173"] }));
app.use(express.json());
const PREDICT_SCRIPT = path.join(__dirname, "ml_training", "predict.py");
app.post("/api/predict", (req, res) => {
  const body = req.body;
  const VALID_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"];

  const { step, amount, type, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest } = body;

  const errors = [];
  if (step === undefined || step === null || step === "") {
    errors.push("Transaction Hour is required.");
  } else if (!Number.isInteger(Number(step)) || Number(step) < 0 || Number(step) > 23) {
    errors.push("Transaction Hour must be an integer between 0 and 23.");
  }
  if (amount === undefined || amount === null || amount === "") {
    errors.push("Transaction Amount is required.");
  } else if (isNaN(Number(amount)) || Number(amount) < 0) {
    errors.push("Transaction Amount must be a non-negative number.");
  }
  if (!type || !VALID_TYPES.includes(type)) {
    errors.push(`Transaction Type must be one of: ${VALID_TYPES.join(", ")}.`);
  }
  const balanceFields = [
    ["oldbalanceOrg", "Sender Balance Before"],
    ["newbalanceOrig", "Sender Balance After"],
    ["oldbalanceDest", "Receiver Balance Before"],
    ["newbalanceDest", "Receiver Balance After"],
  ];

  for (const [field, label] of balanceFields) {
    const val = body[field];
    if (val === undefined || val === null || val === "") {
      errors.push(`${label} is required.`);
    } else if (isNaN(Number(val)) || Number(val) < 0) {
      errors.push(`${label} must be a non-negative number.`);
    }
  }

  if (errors.length > 0) {
    return res.status(400).json({ errors });
  }
  const payload = {
    step: Number(step),
    amount: Number(amount),
    type,
    oldbalanceOrg: Number(oldbalanceOrg),
    newbalanceOrig: Number(newbalanceOrig),
    oldbalanceDest: Number(oldbalanceDest),
    newbalanceDest: Number(newbalanceDest),
  };
  const python = spawn("python", [PREDICT_SCRIPT], { cwd: path.join(__dirname, "ml_training") });

  let stdout = "";
  let stderr = "";

  python.stdout.on("data", (data) => { stdout += data.toString(); });
  python.stderr.on("data", (data) => { stderr += data.toString(); });

  python.on("close", (code) => {
    if (code !== 0) {
      console.error("Python stderr:", stderr);
      return res.status(500).json({ error: "Prediction script failed.", detail: stderr });
    }

    try {
      const output = stdout.trim();
      const startIndex = output.indexOf('{');
      const jsonStr = startIndex >= 0 ? output.slice(startIndex) : output;

      const result = JSON.parse(jsonStr);
      if (result.error) {
        return res.status(400).json({ error: result.error });
      }
      return res.json(result);
    } catch (e) {
      return res.status(500).json({ error: "Failed to parse prediction result.", raw: stdout, stderr });
    }
  });
  python.stdin.write(JSON.stringify(payload));
  python.stdin.end();
});
app.get("/health", (_req, res) => {
  res.json({ status: "ok", message: "Bank Fraud Detection API is running." });
});
app.listen(PORT, () => {
  console.log(`Bank Fraud Detection API running on http://localhost:${PORT}`);
  console.log(`POST /api/predict  — fraud prediction endpoint`);
});
