import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Simple Predictor", layout="wide")
st.title("Simple Predictor")
st.caption("Upload a CSV file, pick two inputs and a target, and see what a straight line predicts.")

# What the model is intended to be used for and what happens whwen its wrong.
st.info("This model was built for restaurant tips dataset, where the model estimates a server's tip "
        "is from the bill total to party size; if wrong the servers can miscalculate the earnings "
        "tipping depends on the things the model may not understand, such as service quality "
        "customers mood or any prediction.")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    try:
      df = pd.read_csv(uploaded_file)
    except Exception:
      st.error(f"Error reading CSV file. Please check the file and try again")
      st.stop()

    #A preview to understand what the viewer uploaded
    st.subheader("Preview of uploaded data")
    st.dataframe(df.head(), width = "stretch")

    #We have three numeric columns:
    #two for inputs and one to predict
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 3:
      st.error("This file needs at least three numeric columns.")
      st.stop()

    #Next chart
    #User picks a column and a chart type, and the app will draw it.
    #Scatter plot needs a column, a third box appears for scatter.
    st.subheader("Data distribution")
    col1, col2, col3 = st.columns(3)
    chart_col = col1.selectbox("Column to chart", numeric_cols)
    chart_type = col2.selectbox("Chart type", ["Line", "Scatter", "Histogram"])

    if chart_type == "Line":
      fig = px.line(df, y=chart_col, title=f"{chart_col} by row")
    elif chart_type == "Scatter":
      other_cols = [col for col in numeric_cols if col != chart_col]
      x_col = col3.selectbox("Plot it against (X axis)", other_cols)
      fig = px.scatter(df, x=x_col, y=chart_col, title=f"{chart_col} versus {x_col}")
    else:
      fig = px.histogram(df, x=chart_col, title=f"Distribution of {chart_col}")
    st.plotly_chart(fig, width="stretch")

    #Three select boxes to let the user choose two inputs and a target
    st.subheader("Predict")
    col1, col2, col3 = st.columns(3)
    x_col1 = col1.selectbox("First input: what you know", numeric_cols)
    x_col2 = col2.selectbox("Second input: what you also know", [col for col in numeric_cols if col != x_col1])
    y_col = col3.selectbox("Target: what you want to predict", [col for col in numeric_cols if col not in (x_col1, x_col2)])

    data = df[[x_col1, x_col2, y_col]].dropna()
    if len(data) < 5:
        st.error("Need at least 5 rows with values in all three columns.")
        st.stop()

    # Keep the three columns and fit the model
    x = data[[x_col1, x_col2]]
    y = data[y_col]
    model = LinearRegression()
    model.fit(x,y)

    # Three numbers go underneath: one slope for each input and R2.
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Each extra {x_col1} adds", f"{model.coef_[0]:,.2f}")
    m2.metric(f"Each extra {x_col2} adds", f"{model.coef_[1]:,.2f}")
    m3.metric("R2 on this data", f"{model.score(x, y):.2f}")

    #Now the prediction
    st.subheader("Try it")
    col1, col2 = st.columns(2)
    value1 = col1.number_input(f"Enter a value for {x_col1}", value=float(data[x_col1].mean()))
    value2 = col2.number_input(f"Enter a value for {x_col2}", value=float(data[x_col2].mean()))
    new_row = pd.DataFrame({x_col1: [value1], x_col2: [value2]})
    prediction = model.predict(new_row)[0]
    st.metric(f"Predicted {y_col}", f"{prediction:,.2f}")

    # The last lines are a warning.
    # a line does not know where its data stoped.
    for col, value in [(x_col1, value1), (x_col2, value2)]:
      low, high = data[col].min(), data[col].max()
      if value < low or value > high:
        st.warning(f"{col}: This value is outside the range the model learned from ({low:,.2f} to {high:,.2f}). Treat the number with caution.")
else:
    st.info("Upload a CSV file to get started. It needs at least three numeric columns.")


