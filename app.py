import os

# Create Streamlit config file to bypass the 200MB upload limit
os.makedirs('.streamlit', exist_ok=True)
with open('.streamlit/config.toml', 'w') as f:
    f.write('[server]\nmaxUploadSize = 1000\n')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(page_title="Fraud Detection App", layout="wide", initial_sidebar_state="expanded")

# App title and description
st.title("Fraud Detection Predictive Modeling")
st.write("A business intelligence application for processing data and training machine learning models to detect fraud.")

# Initialize session state variables to store data across interactions
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'models' not in st.session_state:
    st.session_state.models = None

# Sidebar for file upload
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV or Excel)", type=["csv", "xlsx"])

# Cache function to prevent reloading large files on every button click
@st.cache_data(show_spinner="Loading dataset...")
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

if uploaded_file is not None:
    # Clear memory if a completely new file is uploaded
    if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.processed_data = None
        st.session_state.models = None
        
    try:
        st.session_state.df = load_data(uploaded_file)
        st.sidebar.success(f"Loaded {len(st.session_state.df)} rows successfully.")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

# Application tabs
tabs = st.tabs(["Data Exploration", "Data Preprocessing", "Model Training", "Make Predictions"])

# ==========================================
# TAB: DATA EXPLORATION
# ==========================================
with tabs[0]:
    st.header("Data Exploration")
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Display basic dataset metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        st.subheader("Dataset Information")
        meta_df = pd.DataFrame({
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.notnull().sum(),
            "Missing Count": df.isnull().sum()
        })
        st.dataframe(meta_df.T, use_container_width=True)
    else:
        st.info("Please upload a dataset in the sidebar to begin.")

# ==========================================
# TAB: DATA PREPROCESSING
# ==========================================
with tabs[1]:
    st.header("Data Preprocessing and Balancing")
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        
        st.subheader("Target Variable Configuration")
        target_col = st.selectbox("Select the Target Column (Label):", df.columns, index=len(df.columns)-1)
        
        feature_cols = [col for col in df.columns if col != target_col]
        selected_features = st.multiselect("Select Features for Training:", feature_cols, default=feature_cols[:15])
        
        st.subheader("Row Limit Control")
        st.markdown("*Limit the dataset size to prevent server crashes during training.*")
        max_rows = st.slider("Maximum rows for processing:", min_value=1000, max_value=50000, value=10000, step=1000)
        
        if st.button("Process Data"):
            with st.spinner("Processing and balancing data..."):
                
                # Identify classes for balancing
                class_counts = df[target_col].value_counts()
                majority_class = class_counts.index[0]
                minority_class = class_counts.index[-1]
                
                minority_df = df[df[target_col] == minority_class]
                majority_df = df[df[target_col] == majority_class]
                
                # Calculate safe sample sizes to maintain limits
                n_minority = min(len(minority_df), int(max_rows / 3))
                n_majority = min(len(majority_df), n_minority * 2) 
                
                # Sample and combine to balance the dataset
                df_balanced = pd.concat([
                    minority_df.sample(n=n_minority, random_state=42),
                    majority_df.sample(n=n_majority, random_state=42)
                ]).sample(frac=1, random_state=42).reset_index(drop=True)
                
                # Separate features and target
                X = df_balanced[selected_features].copy()
                y = df_balanced[target_col].copy()
                
                # Handle missing values and encoding dynamically based on data type
                num_cols = X.select_dtypes(include=['int64', 'float64']).columns
                cat_cols = X.select_dtypes(include=['object', 'category']).columns
                
                # Impute numeric columns
                if len(num_cols) > 0:
                    imputer_num = SimpleImputer(strategy='median')
                    X[num_cols] = imputer_num.fit_transform(X[num_cols])
                
                # Impute and encode categorical columns
                label_encoders = {}
                if len(cat_cols) > 0:
                    imputer_cat = SimpleImputer(strategy='most_frequent')
                    X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])
                    
                    for col in cat_cols:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                        label_encoders[col] = le
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Split dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                
                # Store processed data in session state for the next tabs
                st.session_state.processed_data = {
                    "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
                    "scaler": scaler, "label_encoders": label_encoders, "features": selected_features, "target": target_col,
                    "clean_df_sample": pd.DataFrame(X_scaled, columns=selected_features).head()
                }
                
                st.success(f"Data processed and split. Training rows: {len(X_train)}, Testing rows: {len(X_test)}.")
        
        if st.session_state.processed_data is not None:
            st.write("**Preview of Processed Data:**")
            st.dataframe(st.session_state.processed_data["clean_df_sample"], use_container_width=True)
    else:
        st.info("Upload a dataset to access preprocessing options.")

# ==========================================
# TAB: MODEL TRAINING
# ==========================================
with tabs[2]:
    st.header("Model Training and Evaluation")
    if st.session_state.processed_data is not None:
        data = st.session_state.processed_data
        
        st.subheader("Model Settings")
        col1, col2, col3 = st.columns(3)
        k_val = col1.slider("KNN Neighbors (K)", 1, 11, 3)
        svm_kernel = col2.selectbox("SVM Kernel", ["rbf", "linear"])
        ann_hidden = col3.slider("ANN Hidden Layers", 10, 50, 30)
        
        if st.button("Train Models"):
            with st.spinner("Training models..."):
                # Initialize models
                knn = KNeighborsClassifier(n_neighbors=k_val)
                svm = SVC(kernel=svm_kernel, probability=True, random_state=42)
                ann = MLPClassifier(hidden_layer_sizes=(ann_hidden,), max_iter=400, random_state=42)
                
                # Train models
                knn.fit(data["X_train"], data["y_train"])
                svm.fit(data["X_train"], data["y_train"])
                ann.fit(data["X_train"], data["y_train"])
                
                # Evaluate models and store metrics
                metrics = {}
                for name, model in [("KNN", knn), ("SVM", svm), ("ANN", ann)]:
                    preds = model.predict(data["X_test"])
                    metrics[name] = {
                        "Accuracy": accuracy_score(data["y_test"], preds),
                        "Precision": precision_score(data["y_test"], preds, zero_division=0),
                        "Recall": recall_score(data["y_test"], preds, zero_division=0),
                        "F1": f1_score(data["y_test"], preds, zero_division=0),
                        "CM": confusion_matrix(data["y_test"], preds),
                        "instance": model
                    }
                st.session_state.models = metrics
                st.success("Models trained successfully.")
        
        if st.session_state.models is not None:
            st.subheader("Model Performance")
            m_data = st.session_state.models
            
            # Display metrics table
            summary_df = pd.DataFrame({
                "Algorithm": ["KNN", "SVM", "ANN"],
                "Accuracy": [m_data["KNN"]["Accuracy"], m_data["SVM"]["Accuracy"], m_data["ANN"]["Accuracy"]],
                "Precision": [m_data["KNN"]["Precision"], m_data["SVM"]["Precision"], m_data["ANN"]["Precision"]],
                "Recall": [m_data["KNN"]["Recall"], m_data["SVM"]["Recall"], m_data["ANN"]["Recall"]],
                "F1-Score": [m_data["KNN"]["F1"], m_data["SVM"]["F1"], m_data["ANN"]["F1"]]
            })
            st.table(summary_df.set_index("Algorithm"))
            
            st.subheader("Confusion Matrices")
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            for i, name in enumerate(["KNN", "SVM", "ANN"]):
                sns.heatmap(m_data[name]["CM"], annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
                axes[i].set_title(f"{name} Confusion Matrix")
                axes[i].set_xlabel("Predicted")
                axes[i].set_ylabel("Actual")
            st.pyplot(fig)
    else:
        st.info("Process the data in the Preprocessing tab before training models.")

# ==========================================
# TAB: MAKE PREDICTIONS
# ==========================================
with tabs[3]:
    st.header("Make Predictions")
    if st.session_state.models is not None and st.session_state.processed_data is not None:
        data = st.session_state.processed_data
        df_orig = st.session_state.df
        
        st.markdown("Enter custom values below to predict if a transaction is fraudulent or legitimate.")
        
        input_data = {}
        form_cols = st.columns(3)
        
        # Build input forms dynamically based on feature type
        for idx, col in enumerate(data["features"]):
            target_col_ui = form_cols[idx % 3]
            
            if "label_encoders" in data and col in data["label_encoders"]:
                # Categorical column: Create dropdown
                unique_vals = df_orig[col].dropna().astype(str).unique()
                display_vals = unique_vals[:100] if len(unique_vals) > 100 else unique_vals
                
                chosen_val = target_col_ui.selectbox(col, display_vals)
                
                try:
                    encoded_val = data["label_encoders"][col].transform([chosen_val])[0]
                except ValueError:
                    encoded_val = 0 
                input_data[col] = encoded_val
                
            else:
                # Numeric column: Create number input
                min_v = float(df_orig[col].min())
                max_v = float(df_orig[col].max())
                mean_v = float(df_orig[col].mean())
                
                if min_v == max_v:
                    chosen_val = target_col_ui.number_input(col, value=mean_v)
                else:
                    chosen_val = target_col_ui.number_input(col, min_value=min_v, max_value=max_v, value=mean_v)
                input_data[col] = chosen_val
        
        model_choice = st.radio("Select Model for Prediction:", ["KNN", "SVM", "ANN"], horizontal=True)
        
        if st.button("Run Prediction"):
            # Format and scale input data
            ordered_input = [input_data[c] for c in data["features"]]
            scaled_input = data["scaler"].transform([ordered_input])
            
            # Predict using the selected model
            selected_model_instance = st.session_state.models[model_choice]["instance"]
            prediction_idx = selected_model_instance.predict(scaled_input)[0]
            
            # Display prediction output
            if prediction_idx == 1:
                st.error(f"Fraud Detected by {model_choice}")
                st.metric(label="Prediction Result", value="Fraudulent")
            else:
                st.success(f"Transaction Cleared by {model_choice}")
                st.metric(label="Prediction Result", value="Legitimate")
    else:
        st.info("Train the models in the Model Training tab before making predictions.")