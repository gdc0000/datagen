import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ---------------------------
# Footer Function
# ---------------------------
def add_footer():
    st.markdown("---")
    st.markdown("### **Gabriele Di Cicco, PhD in Social Psychology**")
    st.markdown("""
    [GitHub](https://github.com/gdc0000) | 
    [ORCID](https://orcid.org/0000-0002-1439-5790) | 
    [LinkedIn](https://www.linkedin.com/in/gabriele-di-cicco-124067b0/)
    """)

# ---------------------------
# Data Generation Functions
# ---------------------------

def generate_linear_regression_data(sample_size, num_features, coefficients, distribution, error_mean, error_std):
    X = {}
    for i in range(num_features):
        if distribution == "Normal":
            X[f'X{i+1}'] = np.random.normal(0, 1, sample_size)
        elif distribution == "Uniform":
            X[f'X{i+1}'] = np.random.uniform(0, 1, sample_size)
        elif distribution == "Binomial":
            X[f'X{i+1}'] = np.random.binomial(1, 0.5, sample_size)
        elif distribution == "Poisson":
            X[f'X{i+1}'] = np.random.poisson(1, sample_size)
    X_df = pd.DataFrame(X)
    error = np.random.normal(error_mean, error_std, sample_size)
    y = X_df.dot(coefficients) + error
    data = X_df.copy()
    data['Y'] = y
    return data

def generate_logistic_regression_data(sample_size, num_features, coefficients, distribution, class_sep=1.0):
    X, y = make_classification(
        n_samples=sample_size,
        n_features=num_features,
        n_informative=num_features,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=class_sep,
        random_state=42
    )
    data = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(num_features)])
    data['Y'] = y
    return data

def generate_general_mixed_data(sample_size, num_features, distribution):
    data = {}
    for i in range(num_features):
        if distribution == "Normal":
            data[f'X{i+1}'] = np.random.normal(0, 1, sample_size)
        elif distribution == "Uniform":
            data[f'X{i+1}'] = np.random.uniform(0, 1, sample_size)
        elif distribution == "Binomial":
            data[f'X{i+1}'] = np.random.binomial(1, 0.5, sample_size)
        elif distribution == "Poisson":
            data[f'X{i+1}'] = np.random.poisson(1, sample_size)
    df = pd.DataFrame(data)
    return df

# Placeholder functions for advanced models
def generate_path_analysis_data(sample_size, config):
    st.warning("Path Analysis data generation is under development.")
    return pd.DataFrame()

def generate_sem_data(sample_size, config):
    st.warning("Structural Equation Modeling (SEM) data generation is under development.")
    return pd.DataFrame()

def generate_mediation_data(sample_size, config):
    st.warning("Mediation Analysis data generation is under development.")
    return pd.DataFrame()

def generate_moderation_data(sample_size, config):
    st.warning("Moderation Analysis data generation is under development.")
    return pd.DataFrame()

# ---------------------------
# Visualization Functions
# ---------------------------

def visualize_data(data, model_type):
    st.subheader("Data Visualization")
    plot_type = st.selectbox("Select Plot Type", ["Scatter Plot", "Pair Plot", "Histogram", "Box Plot"])
    
    if plot_type == "Scatter Plot":
        all_columns = data.columns.tolist()
        x_axis = st.selectbox("X-Axis", all_columns, key="scatter_x")
        y_axis = st.selectbox("Y-Axis", all_columns, key="scatter_y")
        if model_type == "Linear Regression":
            fig = px.scatter(data, x=x_axis, y=y_axis, trendline="ols")
        else:
            fig = px.scatter(data, x=x_axis, y=y_axis)
        st.plotly_chart(fig)
    
    elif plot_type == "Pair Plot":
        fig = sns.pairplot(data)
        st.pyplot(fig)
    
    elif plot_type == "Histogram":
        feature = st.selectbox("Select Feature for Histogram", data.columns, key="hist_feature")
        fig, ax = plt.subplots()
        sns.histplot(data[feature], kde=True, ax=ax)
        st.pyplot(fig)
    
    elif plot_type == "Box Plot":
        feature = st.selectbox("Select Feature for Box Plot", data.columns, key="box_feature")
        fig, ax = plt.subplots()
        sns.boxplot(y=data[feature], ax=ax)
        st.pyplot(fig)

def evaluate_model(data, model_type):
    if model_type == "Logistic Regression":
        st.subheader("Logistic Regression Evaluation")
        X = data.drop('Y', axis=1)
        y = data['Y']
        model = LogisticRegression()
        model.fit(X, y)
        y_pred_prob = model.predict_proba(X)[:,1]
        fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC = {roc_auc:.2f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, y0=0, x1=1, y1=1
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')
        st.plotly_chart(fig)
    
    elif model_type == "Linear Regression":
        st.subheader("Linear Regression Evaluation")
        X = data.drop('Y', axis=1)
        y = data['Y']
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        fig, ax = plt.subplots()
        sns.scatterplot(x=y, y=y_pred, ax=ax)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax.set_xlabel("Actual Y")
        ax.set_ylabel("Predicted Y")
        ax.set_title("Actual vs Predicted Y")
        st.pyplot(fig)

# ---------------------------
# Main Application
# ---------------------------

def main():
    st.set_page_config(page_title="Synthetic Psychological Data Generator", layout="wide")
    st.title("🧠 Synthetic Psychological Data Generator")
    
    # Sidebar Configuration
    st.sidebar.header("🔧 Configuration Panel")
    
    model_type = st.sidebar.selectbox(
        "Select Data Generation Model",
        [
            "General/Mixed Models",
            "Linear Regression",
            "Logistic Regression",
            "Path Analysis",
            "Structural Equation Modeling (SEM)",
            "Mediation Analysis",
            "Moderation Analysis"
        ]
    )
    
    sample_size = st.sidebar.number_input("📊 Sample Size", min_value=10, max_value=10000, value=100, step=10)
    distribution = st.sidebar.selectbox("🔢 Variable Distribution", ["Normal", "Uniform", "Binomial", "Poisson"])
    
    config = {
        "model_type": model_type,
        "sample_size": sample_size,
        "distribution": distribution
    }
    
    # Dynamic Configuration based on Model Type
    if model_type in ["Linear Regression", "Logistic Regression"]:
        num_features = st.sidebar.number_input("🔍 Number of Predictor Variables", min_value=1, max_value=20, value=3, step=1)
        coefficients = []
        st.sidebar.markdown("**📈 Coefficients for Predictor Variables**")
        for i in range(int(num_features)):
            coef = st.sidebar.number_input(f"Coefficient for X{i+1}", value=1.0, key=f"coef_{i}")
            coefficients.append(coef)
        config["num_features"] = num_features
        config["coefficients"] = coefficients
        
        if model_type == "Linear Regression":
            st.sidebar.markdown("**⚙️ Error Term Parameters**")
            error_mean = st.sidebar.number_input("Error Mean", value=0.0, key="error_mean")
            error_std = st.sidebar.number_input("Error Standard Deviation", value=1.0, key="error_std")
            config["error_mean"] = error_mean
            config["error_std"] = error_std
        elif model_type == "Logistic Regression":
            class_sep = st.sidebar.number_input("🔀 Class Separation", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            config["class_sep"] = class_sep
    
    elif model_type == "General/Mixed Models":
        num_features = st.sidebar.number_input("🔍 Number of Variables", min_value=1, max_value=20, value=5, step=1)
        config["num_features"] = num_features
    
    elif model_type == "Path Analysis":
        st.sidebar.markdown("**🔄 Path Analysis Configuration**")
        num_exogenous = st.sidebar.number_input("Number of Exogenous Variables", min_value=1, max_value=10, value=2, step=1)
        num_endogenous = st.sidebar.number_input("Number of Endogenous Variables", min_value=1, max_value=10, value=2, step=1)
        paths = {}
        for i in range(int(num_exogenous)):
            var = f'X{i+1}'
            paths[var] = {"type": "exogenous"}
        for i in range(int(num_endogenous)):
            var = f'M{i+1}'
            num_preds = st.sidebar.number_input(f"Number of Predictors for {var}", min_value=1, max_value=5, value=1, key=f"num_preds_{var}")
            predictors = []
            coefficients = []
            for j in range(int(num_preds)):
                pred = st.sidebar.selectbox(f"Predictor {j+1} for {var}", list(paths.keys()), key=f"pred_{var}_{j}")
                coef = st.sidebar.number_input(f"Coefficient for {pred} → {var}", value=1.0, key=f"coef_{var}_{j}")
                predictors.append(pred)
                coefficients.append(coef)
            paths[var] = {
                "type": "endogenous",
                "predictors": predictors,
                "coefficients": coefficients,
                "error_mean": 0,
                "error_std": 1
            }
        config["paths"] = paths
    
    # Placeholder configurations for advanced models
    elif model_type == "Structural Equation Modeling (SEM)":
        st.sidebar.info("🛠️ SEM configuration is under development.")
        config["sem_model"] = {}
    
    elif model_type == "Mediation Analysis":
        st.sidebar.info("🛠️ Mediation Analysis configuration is under development.")
        config["mediation_model"] = {}
    
    elif model_type == "Moderation Analysis":
        st.sidebar.info("🛠️ Moderation Analysis configuration is under development.")
        config["moderation_model"] = {}
    
    # Generate Data Based on Model
    data = pd.DataFrame()
    
    if model_type == "Linear Regression":
        data = generate_linear_regression_data(
            sample_size=sample_size,
            num_features=int(num_features),
            coefficients=coefficients,
            distribution=distribution,
            error_mean=error_mean,
            error_std=error_std
        )
    
    elif model_type == "Logistic Regression":
        data = generate_logistic_regression_data(
            sample_size=sample_size,
            num_features=int(num_features),
            coefficients=coefficients,
            distribution=distribution,
            class_sep=class_sep
        )
    
    elif model_type == "General/Mixed Models":
        data = generate_general_mixed_data(
            sample_size=sample_size,
            num_features=int(num_features),
            distribution=distribution
        )
    
    elif model_type == "Path Analysis":
        data = generate_path_analysis_data(
            sample_size=sample_size,
            config=config
        )
    
    elif model_type == "Structural Equation Modeling (SEM)":
        data = generate_sem_data(
            sample_size=sample_size,
            config=config
        )
    
    elif model_type == "Mediation Analysis":
        data = generate_mediation_data(
            sample_size=sample_size,
            config=config
        )
    
    elif model_type == "Moderation Analysis":
        data = generate_moderation_data(
            sample_size=sample_size,
            config=config
        )
    
    # Display Generated Data and Visualizations
    if not data.empty:
        st.subheader("📈 Generated Data")
        st.dataframe(data.head())
        
        with st.expander("🔍 View Summary Statistics"):
            st.write(data.describe())
        
        # Visualization Tab
        with st.expander("📊 Visualize Data"):
            visualize_data(data, model_type)
        
        # Model Evaluation Tab
        if model_type in ["Linear Regression", "Logistic Regression"]:
            with st.expander("🧪 Evaluate Model"):
                evaluate_model(data, model_type)
        
        # Download Section
        st.subheader("💾 Download Dataset")
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='synthetic_data.csv',
            mime='text/csv',
        )
    else:
        st.info("⚠️ Please configure the parameters in the sidebar to generate data.")
    
    # Educational Guidance
    with st.expander("ℹ️ How to Use This App"):
        st.markdown("""
        ### **Steps to Generate Synthetic Data:**
        
        1. **Select a Data Generation Model** from the sidebar.
        2. **Configure Parameters** relevant to the chosen model, such as sample size, number of variables, distributions, and coefficients.
        3. **Generate Data** by adjusting the parameters. The generated data will appear in the main area.
        4. **Visualize the Data** using various plot types to understand relationships and distributions.
        5. **Evaluate the Model** (if applicable) to assess performance metrics.
        6. **Download the Dataset** for further analysis or educational purposes.
        
        ### **Model Descriptions:**
        
        - **General/Mixed Models**: Generate datasets with various types of variables based on selected distributions.
        - **Linear Regression**: Create datasets suitable for modeling linear relationships between predictors and an outcome variable.
        - **Logistic Regression**: Simulate classification problems with binary outcome variables.
        - **Path Analysis**: Define and visualize direct and indirect relationships among multiple variables.
        - **Structural Equation Modeling (SEM)**: Generate complex datasets for SEM techniques.
        - **Mediation Analysis**: Include mediating variables to study indirect effects.
        - **Moderation Analysis**: Incorporate moderating variables to assess interaction effects.
        
        ### **Tips:**
        
        - **Sample Size**: A larger sample size can provide more reliable results but may require more computational resources.
        - **Variable Distribution**: Choose distributions that best represent the theoretical constructs you wish to simulate.
        - **Coefficients**: Adjusting coefficients allows you to define the strength and direction of relationships between variables.
        - **Class Separation (Logistic Regression)**: Controls the distinguishability between classes; higher values make classes more separable.
        
        **Note**: Advanced models like SEM, Mediation, and Moderation Analysis are under development and will be available in future updates.
        """)
    
    # Add Footer
    add_footer()

if __name__ == "__main__":
    main()
