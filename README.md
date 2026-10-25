# Bank Fraud Detection System

### ML-Based Transaction Risk Analytics

An enterprise-grade financial security platform that leverages **XGBoost machine learning, dynamic feature engineering, transaction analytics, and a MERN-based dashboard** to detect potentially fraudulent transactions in real time.

---

## 📌 Overview

The **Bank Fraud Detection System** is designed to support real-time financial fraud detection and risk assessment.

The system processes transaction data through a **Node.js/Express API**, performs machine learning inference using a **Python-based XGBoost model**, and presents the resulting risk assessment through an interactive **React dashboard**.

The platform categorizes transactions into different risk levels, identifies anomalous activity, and provides alerts to support faster manual review.

### Key Capabilities

- Real-time transaction risk assessment
- XGBoost-based fraud prediction
- Dynamic transaction feature engineering
- Automated anomaly detection
- Three-level risk classification
- Real-time fraud alerts
- React-based monitoring dashboard
- Python ML inference pipeline
- Historical transaction analytics
- Model retraining and optimization

---

# 🏗️ System Architecture

The application follows a decoupled architecture where the frontend, backend, and machine learning inference engine operate as separate components.

```text
┌───────────────────────────────────────┐
│            React Frontend             │
│         Vite + React + CSS            │
└───────────────────┬───────────────────┘
                    │
                    │ REST API / JSON
                    ▼
┌───────────────────────────────────────┐
│       Node.js + Express Backend       │
│                                       │
│  • Request Validation                 │
│  • Bounds Checking                    │
│  • API Management                     │
│  • Python Process Management          │
└───────────────────┬───────────────────┘
                    │
                    │ JSON Input
                    ▼
┌───────────────────────────────────────┐
│       Python ML Inference Engine      │
│              predict.py               │
│                                       │
│  • Feature Engineering                │
│  • Data Processing                    │
│  • XGBoost Prediction                 │
└───────────────────┬───────────────────┘
                    │
                    │ Load Model
                    ▼
┌───────────────────────────────────────┐
│       XGBoost Model Artifact           │
│       best_fraud_model_v2.pkl         │
└───────────────────────────────────────┘
```

---

# 📂 Project Structure

```text
Bank-Fraud-Detection/
│
├── Backend/
│   ├── ml_training/
│   │   ├── predict.py
│   │   └── retrain_model.py
│   │
│   ├── models/
│   │   └── best_fraud_model_v2.pkl
│   │
│   ├── server.js
│   └── package.json
│
├── Frontend/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── main.jsx
│   │   └── style.css
│   │
│   ├── package.json
│   └── vite.config.js
│
├── README.md
├── requirements.txt
└── .gitignore
```

> The structure above can be adjusted to match the actual files in the repository.

---

# ⚙️ Technology Stack

## Frontend

- **React**
- **Vite**
- **JavaScript / JSX**
- **CSS**
- REST API integration

## Backend

- **Node.js**
- **Express.js**
- REST API
- Python child-process integration
- Request validation

## Machine Learning

- **Python**
- **XGBoost**
- Pandas
- NumPy
- Scikit-learn
- Feature engineering
- Model evaluation

## Data & Analytics

- Historical transaction records
- `Fraud.csv`
- Stratified train/validation/test splitting
- Transaction anomaly analysis

---

# 🔄 Transaction Prediction Workflow

The system processes a transaction through the following pipeline:

```text
User Transaction
       │
       ▼
React Dashboard
       │
       │ POST /api/predict
       ▼
Express.js API
       │
       ▼
Input Validation
       │
       ▼
Feature Engineering
       │
       ▼
Python Prediction Pipeline
       │
       ▼
XGBoost Model
       │
       ▼
Fraud Probability
       │
       ▼
Risk Classification
       │
       ▼
React Dashboard
```

---

# 🚨 Risk Classification

The prediction engine generates a fraud probability that is evaluated against configured decision thresholds.

Transactions can be classified into categories such as:

| Risk Level | Description |
|---|---|
| 🟢 **Low Risk** | Transaction appears normal |
| 🟡 **Suspicious** | Transaction requires additional review |
| 🔴 **High Risk** | Transaction has a high probability of fraudulent activity |

The dashboard dynamically visualizes the calculated risk level and probability.

---

# 🧠 Machine Learning Pipeline

The fraud detection model is based on **XGBoost**, which is trained using historical transaction data.

The pipeline includes:

1. Data ingestion
2. Data cleaning
3. Feature engineering
4. Train/validation/test splitting
5. Model training
6. Model evaluation
7. Decision threshold optimization
8. Model artifact generation
9. Real-time prediction

### Example Engineered Features

The prediction pipeline can incorporate transaction-derived features such as:

- Sender balance errors
- Receiver balance errors
- Transaction amount
- Transaction type
- Time-of-day indicators
- Account balance changes
- Historical transaction characteristics

---

# 📊 Historical Analytics & Model Tuning

Historical transaction data from `Fraud.csv` is used to analyze fraud patterns and improve model performance.

The training process uses a **stratified train/validation/test split** to maintain the distribution of fraudulent and legitimate transactions across datasets.

Model tuning focuses on balancing:

- Precision
- Recall
- Fraud detection rate
- False-positive rate
- Decision threshold

The system is designed to prioritize high recall while maintaining practical precision for financial fraud detection.

---

# 🔔 Fraud Alerting

The backend evaluates prediction probabilities against predefined decision thresholds.

For example:

```text
Fraud Probability
       │
       ├── Low ────────► Safe
       │
       ├── Medium ─────► Manual Review
       │
       └── High ───────► Fraud Alert
```

The system can also perform bounds and anomaly checks to identify transactions that fall significantly outside expected training-data ranges.

When a transaction exceeds configured risk limits, the dashboard can provide:

- Fraud alerts
- Risk indicators
- Manual review recommendations
- Transaction probability scores

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/tanishka262006/Bank-Fraud-Detection.git
cd Bank-Fraud-Detection
```

---

## 2. Backend Setup

Navigate to the backend directory:

```bash
cd Backend
```

Install Node.js dependencies:

```bash
npm install
```

Start the backend server:

```bash
npm start
```

---

## 3. Python Environment

From the project root, install the Python dependencies:

```bash
pip install -r requirements.txt
```

If using a virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

Then:

```bash
pip install -r requirements.txt
```

---

## 4. Frontend Setup

Open another terminal:

```bash
cd Frontend
```

Install dependencies:

```bash
npm install
```

Start the Vite development server:

```bash
npm run dev
```

The terminal will provide the local URL for the React application.

---

# 🔌 API

### Prediction Endpoint

```http
POST /api/predict
```

The endpoint accepts transaction information in JSON format and returns a fraud-risk prediction.

### Example Request

```json
{
  "amount": 15000,
  "transactionType": "TRANSFER",
  "oldBalance": 20000,
  "newBalance": 5000
}
```

### Example Response

```json
{
  "probability": 0.92,
  "risk": "High Risk",
  "fraud": true
}
```

> Request and response fields should be updated to match the actual backend implementation.

---

# 📈 Model Performance

The model is evaluated using metrics suitable for highly imbalanced fraud-detection datasets.

Important metrics include:

- **Precision**
- **Recall**
- **F1 Score**
- **ROC-AUC**
- **Confusion Matrix**

Because fraudulent transactions typically represent a small percentage of total transactions, **recall and precision are considered alongside overall accuracy**.

---

# 🔐 Security Considerations

For production deployment, the following should be considered:

- API authentication
- Input sanitization
- Rate limiting
- HTTPS
- Secure model storage
- Environment variables for secrets
- Logging and monitoring
- Database security
- Access control
- Protection of sensitive transaction data

---

# 🔮 Future Improvements

Potential future enhancements include:

- Real-time streaming with Kafka
- Database-backed transaction history
- Automated model retraining
- Explainable AI using SHAP
- User authentication and role-based access
- Advanced anomaly detection
- Cloud deployment
- Docker containerization
- Model monitoring and drift detection
- Email/SMS fraud notifications
- Advanced fraud investigation dashboard

---

# 👩‍💻 Project

**Bank Fraud Detection System**

Built using:

```text
React + Vite
Node.js + Express
Python
XGBoost
Machine Learning
REST APIs
```

The goal of the project is to provide an intelligent transaction monitoring system capable of identifying potentially fraudulent activity and presenting actionable risk information through a real-time dashboard.
