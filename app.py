import streamlit as st

st.set_page_config(page_title="Продвинутая аналитика отказов", layout="wide")

navigation = {
    "Моделирование": [st.Page("analysis_and_model.py", title="Обучение и прогноз")],
    "Исследование": [st.Page("data_insights.py", title="Визуализация данных")],
    "Презентация": [st.Page("presentation.py", title="Слайды проекта")],
}

current = st.navigation(navigation, position="sidebar", expanded=True)
current.run()
