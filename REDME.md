🚨 AI Disaster Relief Resource Allocation System

An AI-powered disaster management decision-support system that predicts essential relief resource requirements and helps prioritize resource allocation based on disaster severity, affected population, injuries, and displacement.

📌 Project Overview

Natural disasters such as floods, earthquakes, cyclones, landslides, droughts, and wildfires can create urgent demands for food, drinking water, medical supplies, rescue teams, and shelter.

This project combines Data Analysis, Machine Learning, Deep Learning, and resource allocation logic to estimate resource requirements and identify potential shortages through an interactive Streamlit dashboard.

🎯 Objectives

- Analyze disaster data and identify important patterns.
- Predict food, water, medicine, rescue team, and shelter requirements.
- Compare Random Forest and Artificial Neural Network predictions.
- Calculate a disaster priority score.
- Estimate resource allocations based on available inventory.
- Identify shortages in essential relief resources.
- Visualize results through a Streamlit dashboard.

🛠️ Technologies Used

Technology| Purpose
Python| Core programming
Pandas| Data manipulation
NumPy| Numerical operations
Matplotlib| Data visualization
Seaborn| Exploratory data analysis
Scikit-learn| Machine learning and preprocessing
TensorFlow / Keras| Deep learning using ANN
SQL / MySQL| Disaster and resource data storage
Streamlit| Interactive web application
Joblib| Saving trained ML models

🧠 Machine Learning

The project uses Random Forest Regression to predict:

- Food requirement
- Water requirement
- Medicine requirement
- Rescue team requirement
- Shelter capacity requirement

The input features include disaster type, location, affected population, severity, injured people, displaced people, duration, temperature, rainfall, and infrastructure damage.

🤖 Deep Learning

An Artificial Neural Network (ANN) built with TensorFlow/Keras predicts multiple resource requirements using the same input features.

The project compares Random Forest and ANN predictions to explore their differences.

📊 Data Analysis

Exploratory Data Analysis (EDA) includes:

- Dataset inspection and descriptive statistics
- Missing-value analysis
- Disaster type distribution
- Disaster severity distribution
- Population versus food requirement analysis
- Correlation analysis
- Resource requirement visualization

🚑 Resource Allocation

The application calculates a priority score using disaster severity, affected population, injured people, displaced people, and infrastructure damage.

It then compares predicted requirements against available inventory to estimate allocations and highlight shortages.

Note: The current allocation logic is a prototype for academic demonstration. It is not an operational emergency-response optimization system.

📁 Project Structure

AI_Disaster_Relief_Resource_Allocation/
│
├── dataset/
│   └── disaster_relief_dataset.csv
│
├── models/
│   ├── food_required_model.pkl
│   ├── water_required_litres_model.pkl
│   ├── medicine_required_model.pkl
│   ├── rescue_teams_required_model.pkl
│   ├── shelter_required_model.pkl
│   ├── disaster_relief_ann.keras
│   └── dl_preprocessor.pkl
│
├── src/
│   ├── generate_dataset.py
│   ├── data_analysis.py
│   ├── train_ml.py
│   ├── train_dl.py
│   └── allocation.py
│
├── database/
│   └── disaster_relief.sql
│
├── app.py
├── requirements.txt
└── README.md

⚙️ Installation and Setup

1. Clone the repository

git clone https://github.com/YOUR-USERNAME/AI-Disaster-Relief-Resource-Allocation.git
cd AI-Disaster-Relief-Resource-Allocation

Replace "YOUR-USERNAME" with your GitHub username.

2. Install dependencies

pip install -r requirements.txt

3. Generate the development dataset

python src/generate_dataset.py

4. Perform data analysis

python src/data_analysis.py

5. Train machine learning models

python src/train_ml.py

6. Train the deep learning model

python src/train_dl.py

7. Launch the Streamlit application

streamlit run app.py

Open the local URL displayed in the terminal, usually "http://localhost:8501".

📈 Model Evaluation

The training scripts report:

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- R² Score: Coefficient of determination

Actual model performance must be evaluated using reliable, representative disaster data. Results from synthetic data do not establish real-world predictive accuracy.

🗄️ Database

The planned MySQL integration can store disaster records, predictions, resource inventory, and allocation history.

Run the SQL schema in "database/disaster_relief.sql" using a configured MySQL connection. The current Streamlit application does not automatically save records to MySQL unless database integration is implemented.

📂 Dataset Information

The initial development dataset is generated using Python and NumPy. It contains synthetic disaster scenarios for testing the data-processing, model-training, and dashboard workflow.

Important: Synthetic records are not verified historical disaster records. For meaningful real-world analysis, replace or supplement them with documented datasets from reliable public sources.

🔮 Future Enhancements

- Integration with verified historical disaster datasets
- Live weather API integration
- Geographic disaster mapping
- SQL-backed prediction history
- Multi-location resource optimization
- Improved model validation and monitoring
- Role-based access for relief coordinators
- Real-time resource inventory updates

⚠️ Disclaimer

This project is an academic and educational prototype. Predictions and priority scores are estimates and must not be treated as verified emergency guidance. Actual disaster response decisions require validated data, qualified emergency personnel, and established safety protocols.

👨‍💻 Author

Bala Murugan

Aspiring Data Scientist | Machine Learning | Deep Learning | Data Analysis

GitHub: https://github.com/balamurugan-ds

---

⭐ If you find this project useful, consider starring the repository.