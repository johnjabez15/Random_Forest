# Random Forest Project – Loan Approval Prediction  

## 📌 Overview  
This project implements a **Random Forest Classifier** to predict whether a loan application will be **Approved** or **Denied** based on applicant details.  
The model is trained using a custom dataset and deployed through a **Flask web application**, allowing users to input loan application details and get instant predictions.  

---

## 📂 Project Structure  
```
DataScience/
│
├── Random Forest/
│   ├── data/
│   │   └── loan_approval_dataset.csv
│   ├── model/
│   │   ├── random_forest_model.pkl
│   │   └── label_encoders.pkl
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   ├── model.py
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
```

---

## ⚙️ Installation & Setup  

### 1️⃣ Clone the repository  
```bash
git clone <your-repo-url>
cd "DataScience/Random Forest"
```

### 2️⃣ Create a virtual environment (recommended)  
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 3️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset  
The dataset contains details of loan applicants with the following features:  

- **Age** *(numeric)*  
- **Income** *(numeric)*  
- **Credit Score** *(numeric)*  
- **Loan Amount** *(numeric)*  
- **Employment Type** *(categorical: Salaried, Self-Employed, Unemployed)*  
- **Marital Status** *(categorical: Single, Married, Divorced)*  
- **Loan Term** *(categorical: 12, 24, 36, 48, 60 months)*  
- **Previous Default** *(categorical: Yes, No)*  
- **Loan Approved** *(Target: 0 = Denied, 1 = Approved)*  

---

## 🎯 Problem Statement  
Banks and financial institutions need a reliable way to assess whether a loan applicant should be approved based on their financial history, income, and other demographic details.  
This project aims to automate the loan approval process using **machine learning** to save time and reduce errors in decision-making.  

---

## 💡 Why Random Forest?  
- **High Accuracy**: Combines multiple decision trees to improve prediction accuracy.  
- **Robust to Noise**: Handles noisy data better than single decision trees.  
- **Feature Importance**: Automatically ranks features based on their contribution to the prediction.  
- **Overfitting Prevention**: Reduces overfitting through bagging and averaging.  

---

## 🚀 How to Run  

### 1️⃣ Train the Model  
```bash
python model.py
```
This will create:  
- `random_forest_model.pkl` (trained model)  
- `label_encoders.pkl` (categorical encoders)  

### 2️⃣ Run the Flask App  
```bash
python app.py
```
Visit **`http://127.0.0.1:5000/`** in your browser.  

---

## 🖥️ Frontend Input Example  
Example loan application form input:  
```
Age: 35
Income: 75000
Credit Score: 720
Loan Amount: 20000
Employment Type: Salaried
Marital Status: Married
Loan Term: 36
Previous Default: No
```

---

## 📌 Prediction Goal  
The application predicts:  
- **Loan Approved ✅** — if the applicant meets the approval criteria.  
- **Loan Denied ❌** — if the applicant fails to meet the criteria.  

---

## 🛠 Tech Stack  
- **Python** – Core programming language  
- **Pandas & NumPy** – Data manipulation  
- **Scikit-learn** – Machine learning model training  
- **Flask** – Web framework for deployment  
- **HTML/CSS** – Frontend UI design  

---

## 📌 Future Scope  
- 🔹 Deploy the model on **Heroku** or **Render** for public access.  
- 🔹 Improve accuracy using hyperparameter tuning and additional data.  
- 🔹 Add data visualization dashboard for loan trends and statistics.  
- 🔹 Integrate with banking APIs for real-time loan applications.  
- 🔹 Implement authentication for secure loan approval access.  


## Screen Shots:

**Home Page:**

<img width="1920" height="1080" alt="Screenshot (19)" src="https://github.com/user-attachments/assets/7db0d68f-eb7b-4d1d-a263-65cf2303e76d" />

**Result Page:**

<img width="1920" height="1080" alt="Screenshot (20)" src="https://github.com/user-attachments/assets/c8aa69be-f911-4eb2-a53b-e2c168a44700" />

