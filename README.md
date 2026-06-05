# 🧠 NephroAI – Kidney Disease Prediction System

## 📌 Project Description
NephroAI is a machine learning-based web application designed to predict kidney disease (ESRD) using clinical patient data.  
The system uses a trained ML model (XGBoost) integrated with a Flask backend and a simple HTML frontend.

---

## ⚙️ Technologies Used
- Python
- Flask
- Scikit-learn
- Pandas & NumPy
- XGBoost
- HTML / CSS / JavaScript
- SQLite

---

## 📁 Project Structure
NephroAI/
│
├── backend/
│     ├── app.py
│     ├── train.py
│     ├── nephroai.db
│
├── data/
│     ├── esrs_prediction_dataset.csv
│
├── model/
│     ├── esrd_pipeline.pkl
│
├── frontend/
│     ├── Nephroai_final.html
│
├── requirements.txt
├── README.md
├── .gitignore

---

## 🚀 How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Train model (optional):
python backend/train.py

3. Run application:
python backend/app.py

4. Open in browser:
http://127.0.0.1:5000

---

## 🧠 Features
- Kidney disease prediction
- Machine learning model (XGBoost)
- Web interface
- SQLite database
- Real-time prediction

---

## 👩‍💻 Author
Dalila Rahim

---

## 📌 Status
Project ready for academic presentation (comité)
