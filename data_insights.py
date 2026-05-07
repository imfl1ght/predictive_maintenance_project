import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA

st.set_page_config(page_title="Анализ данных", layout="wide")
st.title("📈 Исследовательский анализ данных")

# Загрузка данных
uploaded = st.file_uploader("Загрузите CSV-файл с данными", type="csv", key="eda_uploader")
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("Файл успешно загружен!")
else:
    import os
    if os.path.exists("data/predictive_maintenance.csv"):
        df = pd.read_csv("data/predictive_maintenance.csv")
        st.info("Используются данные из папки data/")
    else:
        st.error("❌ Файл данных не найден. Пожалуйста, загрузите CSV-файл или поместите его в папку data/")
        st.stop()

# Визуализации
num_cols = ['Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
st.header("1. Распределение признаков")
fig1, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(col)
axes[-1].axis('off')
st.pyplot(fig1)

st.header("2. Корреляционная матрица")
corr = df[num_cols].corr()
fig2, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig2)

st.header("3. Scatter plot (интерактивный)")
x_axis = st.selectbox("Ось X", num_cols, key='x')
y_axis = st.selectbox("Ось Y", num_cols, key='y')
fig_scat = px.scatter(df, x=x_axis, y=y_axis, color='Machine failure',
                      title=f"{x_axis} vs {y_axis}")
st.plotly_chart(fig_scat)

st.header("4. Выбросы (IQR)")
for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lb = q1 - 1.5*iqr
    ub = q3 + 1.5*iqr
    outliers = df[(df[col] < lb) | (df[col] > ub)]
    st.write(f"{col}: выбросов {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

st.header("5. PCA (2 компоненты)")
pca = PCA(n_components=2)
pca_res = pca.fit_transform(df[num_cols])
df_pca = pd.DataFrame(pca_res, columns=['PC1', 'PC2'])
df_pca['Fault'] = df['Machine failure'].astype(str)
fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='Fault', title="PCA")
st.plotly_chart(fig_pca)

st.header("6. Типы отказов")
faults = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
counts = df[faults].sum()
fig_bar = px.bar(x=counts.index, y=counts.values, title="Количество отказов каждого типа")
st.plotly_chart(fig_bar)
