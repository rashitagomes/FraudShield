# FraudShield —  Streamlit demo

FraudShield is an educational credit-card transaction fraud-screening prototype. The interface uses Streamlit and a trained scikit-learn pipeline.

## Features
- Estimated fraud score, risk flag, and recent checks for the current session
- Training and evaluation kept separate from the UI

## Setup (Windows PowerShell)
Open this folder in VS Code and run these commands in the project-root terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser.


