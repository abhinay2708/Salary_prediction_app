📈 Salary Prediction App
A simple web application built using Streamlit that predicts the salary of an employee based on their years of experience using Linear Regression.

🚀 Demo
Try the live app here 👉 [Streamlit Cloud App URL]
(Replace with your actual app link after deployment)

🧠 How It Works
This app uses a machine learning model trained on a dataset of employees' salaries and their years of experience. It uses Linear Regression to predict salaries for new inputs.

🗂️ Project Structure
bash
Copy
Edit
salary_prediction_app/
│
├── app.py                      # Streamlit web app
├── train_model.py              # Script to train and save the model
├── Salary_Data.csv             # Dataset used for training
├── linear_regression_model.pkl # Trained model file
└── requirements.txt            # Python dependencies
📦 Requirements
To run this app locally, install the dependencies:

bash
Copy
Edit
pip install -r requirements.txt
▶️ How to Run
Clone the repository

bash
Copy
Edit
git clone https://github.com/your-username/Salary_prediction_app.git
cd Salary_prediction_app
Run the app

bash
Copy
Edit
streamlit run app.py
The app will open in your browser at http://localhost:8501

📊 Sample Input/Output
Input: 3.5 years of experience

Output: 💰 Predicted Salary: $65,479.50

🛠️ Built With
Python 🐍

Streamlit 🌐

Scikit-learn 📚

Pandas & NumPy 🔢

✍️ Author
Abhinay Mahato
GitHub: @abhinay2708

📃 License
This project is open-source and available under the MIT License.
