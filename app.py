import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "gradient_boosting_model.pkl"))
model_columns = joblib.load(os.path.join(BASE_DIR, "model_columns.pkl"))
df = pd.read_csv(os.path.join(BASE_DIR, "healthcare-dataset-stroke-data.csv"))

st.title("❤️ Stroke Prediction System")
st.markdown("Predict the possibility of stroke using Machine Learning.")

st.sidebar.title("📊 Project Dashboard")

option = st.sidebar.selectbox(
    "Choose Analysis",
    [
        "Prediction",
        "Dataset Preview",
        "Dataset Information",
        "Statistical Summary",
        "Correlation Heatmap",
        "Model Comparison",
        "Stroke Distribution"
    ]
)

# ----------------------------------------------------
# Dataset Preview
# ----------------------------------------------------
if option == "Dataset Preview":

    st.header("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ----------------------------------------------------
# Dataset Information
# ----------------------------------------------------
elif option == "Dataset Information":

    st.header("Dataset Information")

    c1, c2 = st.columns(2)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum().to_frame("Missing Values"))

# ----------------------------------------------------
# Statistical Summary
# ----------------------------------------------------
elif option == "Statistical Summary":

    st.header("Statistical Summary")
    st.dataframe(df.describe())

# ----------------------------------------------------
# Correlation Heatmap
# ----------------------------------------------------
elif option == "Correlation Heatmap":

    st.header("📊 Correlation Heatmap")

    cols = [
        "age",
        "hypertension",
        "heart_disease",
        "avg_glucose_level",
        "bmi",
        "stroke"
    ]

    fig, ax = plt.subplots(figsize=(8,6))

    sns.heatmap(
        df[cols].corr(),
        annot=True,
        cmap="coolwarm",
        linewidths=1,
        fmt=".2f",
        square=True,
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    st.pyplot(fig)

    st.info("""
### 📌 Insights

• Age has the strongest correlation with Stroke.

• Hypertension and Heart Disease increase stroke risk.

• BMI has very little correlation.

• Average Glucose Level shows a weak positive correlation.
""")

# ----------------------------------------------------
# Model Comparison
# ----------------------------------------------------
elif option == "Model Comparison":

    st.header("📈 Machine Learning Model Comparison")

    scores = {
        "Logistic Regression":0.939335,
        "Decision Tree":0.909980,
        "Random Forest":0.939335,
        "SVM":0.939335,
        "Naive Bayes":0.545988,
        "KNN":0.937378,
        "Gradient Boosting":0.940313
    }

    score_df = pd.DataFrame(
        scores.items(),
        columns=["Algorithm","Accuracy"]
    )

    st.dataframe(score_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10,5))

    bars = ax.bar(
        score_df["Algorithm"],
        score_df["Accuracy"]
    )

    ax.set_ylim(0,1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification Model Accuracy")

    plt.xticks(rotation=20, ha="right")

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x()+bar.get_width()/2,
            h+0.01,
            f"{h:.3f}",
            ha="center"
        )

    st.pyplot(fig)

    best = score_df.loc[score_df["Accuracy"].idxmax()]

    st.success(
        f"🏆 Best Model : {best['Algorithm']} ({best['Accuracy']:.4f})"
    )

# ----------------------------------------------------
# Stroke Distribution
# ----------------------------------------------------
elif option == "Stroke Distribution":

    st.header("Stroke Distribution")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.countplot(
        x="stroke",
        data=df,
        ax=ax
    )

    ax.set_xlabel("Stroke")
    ax.set_ylabel("Count")

    st.pyplot(fig)

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------
else:

    st.header("🩺 Stroke Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 45)
        hypertension = st.selectbox("Hypertension", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        avg_glucose_level = st.number_input(
            "Average Glucose Level",
            min_value=50.0,
            max_value=400.0,
            value=100.0
        )
        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        ever_married = st.selectbox(
            "Ever Married",
            ["Yes", "No"]
        )

        work_type = st.selectbox(
            "Work Type",
            [
                "Private",
                "Self-employed",
                "Govt_job",
                "children",
                "Never_worked"
            ]
        )

        residence_type = st.selectbox(
            "Residence Type",
            ["Urban", "Rural"]
        )

        smoking_status = st.selectbox(
            "Smoking Status",
            [
                "formerly smoked",
                "never smoked",
                "smokes",
                "Unknown"
            ]
        )

    st.write("")

    if st.button("🔍 Predict Stroke Risk", use_container_width=True):

        input_dict = {
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
        }

        # Initialize all dummy columns
        for col in model_columns:
            if col not in input_dict:
                input_dict[col] = 0

        # Gender
        if "gender_Male" in model_columns and gender == "Male":
            input_dict["gender_Male"] = 1

        # Ever Married
        if "ever_married_Yes" in model_columns and ever_married == "Yes":
            input_dict["ever_married_Yes"] = 1

        # Residence
        if "Residence_type_Urban" in model_columns and residence_type == "Urban":
            input_dict["Residence_type_Urban"] = 1

        # Work Type
        wt = f"work_type_{work_type}"
        if wt in model_columns:
            input_dict[wt] = 1

        # Smoking Status
        sm = f"smoking_status_{smoking_status}"
        if sm in model_columns:
            input_dict[sm] = 1

        # Create dataframe
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[model_columns]

        # Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        stroke_prob = probability[1] * 100
        no_stroke_prob = probability[0] * 100

        st.markdown("---")
        st.subheader("🩺 Prediction Result")

        # -----------------------------
        # Risk Classification
        # -----------------------------
        if stroke_prob >= 50:

            st.error("🚨 High Risk of Stroke")

            st.error(
                f"Estimated Stroke Probability: **{stroke_prob:.2f}%**"
            )

            st.warning("""
### Recommendation

• Consult a healthcare professional immediately.

• Monitor blood pressure and blood sugar regularly.

• Avoid smoking and excessive alcohol.

• Maintain a healthy diet and exercise routine.
""")

        elif stroke_prob >= 20:

            st.warning("⚠️ Moderate Risk of Stroke")

            st.warning(
                f"Estimated Stroke Probability: **{stroke_prob:.2f}%**"
            )

            st.info("""
### Recommendation

• Schedule a routine medical check-up.

• Exercise regularly.

• Maintain a healthy weight.

• Keep blood pressure and glucose under control.
""")

        else:

            st.success("✅ Low Risk of Stroke")

            st.success(
                f"Estimated Stroke Probability: **{stroke_prob:.2f}%**"
            )

            st.info("""
### Recommendation

• Continue following a healthy lifestyle.

• Eat a balanced diet.

• Stay physically active.

• Go for regular health check-ups.
""")

        st.markdown("---")

        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric(
                "🟢 No Stroke Probability",
                f"{no_stroke_prob:.2f}%"
            )

        with metric2:
            st.metric(
                "🔴 Stroke Probability",
                f"{stroke_prob:.2f}%"
            )

        st.progress(stroke_prob / 100)

        st.balloons()
