# 🛒 Retail Shelf Optimization using Machine Learning

> **An End-to-End AI & Machine Learning Project for Intelligent Retail Category Management**

Retail Shelf Optimization is a production-style Machine Learning project that helps retailers make data-driven merchandising decisions. The system analyzes real retail transaction data to forecast demand, optimize shelf placement, recommend discounts, identify customer buying patterns, and provide interactive business insights through a Streamlit dashboard.

The project demonstrates the complete Machine Learning lifecycle—from data collection and preprocessing to model training, deployment, visualization, and business recommendation generation.

---

# Project Objectives

Modern retailers face several challenges:

* Which products should receive premium shelf space?
* Which products are likely to experience high demand?
* When should discounts be applied?
* Which products are frequently purchased together?
* How do seasonal trends influence customer purchases?

This project answers these questions using Machine Learning and Data Analytics.

---

# Key Features

### Sales Forecasting

Predict future product sales using historical transaction data and engineered business features.

### Shelf Placement Optimization

Recommend optimal shelf positions based on:

* Predicted demand
* Product profitability
* Sales velocity
* Impulse buying score
* Shelf capacity constraints

### Discount Recommendation Engine

Generate intelligent discount suggestions by considering:

* Expected sales uplift
* Current inventory
* Profit margin
* Seasonal demand
* Promotional impact

### Customer Purchase Pattern Analysis

Use Market Basket Analysis (Apriori Algorithm) to discover products frequently purchased together for cross-selling opportunities.

### Exploratory Data Analysis (EDA)

Automatically generate business insights including:

* Revenue trends
* Monthly sales analysis
* Country-wise revenue
* Best-selling products
* Top customers
* Order value distribution
* Basket size analysis

### Interactive Dashboard

A Streamlit dashboard provides:

* KPI cards
* Sales visualizations
* Demand forecasts
* Shelf optimization recommendations
* Discount recommendations
* Association rule explorer
* Product analytics

---

# Machine Learning Pipeline

```
Raw UCI Dataset
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Model Training
(Random Forest Regressor)
        │
        ▼
Demand Prediction
        │
        ▼
Shelf Optimization
        │
        ▼
Discount Recommendation
        │
        ▼
Interactive Dashboard
```

---

# Tech Stack

### Programming

* Python

### Machine Learning

* Scikit-Learn
* Random Forest Regressor
* Feature Engineering

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Plotly

### Dashboard

* Streamlit

### Model Storage

* Joblib

---

# Project Structure

```
Retail-Shelf-Optimization
│
├── app.py
├── requirements.txt
├── README.md
│
├── data
│   ├── raw
│   └── processed
│
├── models
│
├── reports
│
├── scripts
│   ├── generate_dataset.py
│   └── train_models.py
│
└── src
    └── retail_optimizer
        ├── association_rules.py
        ├── config.py
        ├── data_generation.py
        ├── eda.py
        ├── features.py
        ├── modeling.py
        ├── recommendations.py
        ├── shelf_optimizer.py
        └── real_data.py
```

---

# Dataset

This project uses the **UCI Online Retail Dataset**, containing over **500,000 real retail transactions** from a UK-based online retailer.

### Data Cleaning

The preprocessing pipeline automatically:

* Removes cancelled orders
* Removes invalid quantities
* Removes missing values
* Creates daily sales tables
* Generates customer baskets
* Engineers retail business features

### Additional Business Features

To simulate real-world retail decision making, the project generates:

* Inventory level
* Profit margin
* Product cost
* Shelf position
* Shelf capacity
* Weather signal
* Promotion indicator
* Holiday indicator

These simulated features enable realistic optimization scenarios while keeping the original transaction data intact.

---

# Machine Learning Model

### Target

Predict future product sales.

### Input Features

* Product
* Category
* Store
* Day
* Month
* Week
* Holiday
* Promotion
* Weather
* Inventory
* Margin
* Historical sales

### Model

Random Forest Regressor

### Evaluation Metrics

* R² Score
* RMSE
* MAE
* Feature Importance

---

# Business Outputs

The trained model generates:

* Demand Forecast
* Sales Prediction
* Shelf Placement Recommendation
* Discount Recommendation
* Cross-selling Opportunities
* Revenue Insights
* Inventory Priorities

---

# Dashboard Features

The Streamlit dashboard provides:

* Executive KPI Overview
* Product Performance
* Sales Trends
* Revenue Analysis
* Country Analysis
* Product Forecasts
* Shelf Optimization Results
* Discount Recommendations
* Association Rules
* Interactive Filters

---

# Installation

```bash
git clone https://github.com/amajali-784/Retail-Shelf-Optimization.git

cd Retail-Shelf-Optimization

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt
```

---

# Run the Project

Generate the processed dataset

```bash
python scripts/generate_dataset.py
```

Train the Machine Learning model

```bash
python scripts/train_models.py
```

Launch the dashboard

```bash
streamlit run app.py
```

---

# Generated Outputs

After execution, the project automatically creates:

```
data/processed/
models/
reports/
```

Including:

* Cleaned dataset
* Daily sales dataset
* Customer baskets
* Trained ML model
* Feature importance
* EDA reports
* Model evaluation metrics

---

# Skills Demonstrated

* Machine Learning
* Data Cleaning
* Feature Engineering
* Time Series Feature Creation
* Regression Modeling
* Model Evaluation
* Business Analytics
* Data Visualization
* Recommendation Systems
* Association Rule Mining
* Streamlit Dashboard Development
* Python Software Engineering

---

# Future Enhancements

* XGBoost and LightGBM models
* Deep Learning forecasting (LSTM)
* Real-time sales prediction API
* Multi-store inventory optimization
* Computer Vision shelf monitoring
* Reinforcement Learning for shelf allocation
* Cloud deployment with Docker and AWS

---

# Author

**Ankush Majalikar**

AI Engineer | Machine Learning Engineer | Data Scientist

If you found this project useful, consider giving it a ⭐ on GitHub.
