import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title of the App
st.title("Simple Linear Regression Web App")

# Step 1: Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(data.head())

    # Step 2: Data Analysisz
    st.header("Data Analysis")
    if data.shape[1] >= 2:
        x_column = st.selectbox("Select X (Input) Column:", data.columns)
        y_column = st.selectbox("Select Y (Target) Column:", data.columns)

        if x_column != y_column:
            X = data[[x_column]]
            y = data[y_column]

            # Correlation
            correlation = data[x_column].corr(data[y_column])
            st.write(f"Correlation between {x_column} and {y_column}: {correlation:.2f}")

            # Visualization: Scatter Plot
            st.write("Scatter Plot:")
            plt.scatter(data[x_column], data[y_column], color="blue")
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(f"{x_column} vs {y_column}")
            st.pyplot(plt)

            # Histogram
            st.write("Histogram of Input Variable:")
            plt.hist(data[x_column], bins=10, color="skyblue", edgecolor="black")
            plt.title(f"Distribution of {x_column}")
            st.pyplot(plt)

            # Step 3: Model Training
            st.header("Model Training")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Model Evaluation
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse:.2f}")
            st.write(f"R-squared: {r2:.2f}")

            # Visualization: Regression Line
            st.write("Regression Line:")
            plt.scatter(X_test, y_test, color="blue", label="Actual")
            plt.plot(X_test, y_pred, color="red", label="Regression Line")
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.legend()
            st.pyplot(plt)

            # Step 4: Prediction
            st.header("Make a Prediction")
            user_input = st.number_input(f"Enter a value for {x_column}:")
            if user_input:
                prediction = model.predict([[user_input]])
                st.write(f"Predicted value for {y_column}: {prediction[0]:.2f}")
        else:
            st.error("X and Y columns must be different.")
    else:
        st.error("Dataset must have at least 2 columns.")
else:
    st.info("Please upload a dataset to proceed.")
