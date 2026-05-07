import streamlit as st
import reveal_slides as rs

st.set_page_config(page_title="Презентация", layout="wide")
st.title("📽️ Презентация: реализация продвинутых задач (0–4)")

presentation = '''
# Продвинутое прогнозирование отказов

- **Git-репозиторий:** `https://github.com/imfl1ght/predictive_maintenance_project.git`
- **Структура:** `app.py`, `analysis_and_model.py`, `data_insights.py`, `presentation.py`

---

## ✅ Задача 0: Мультиклассовая классификация
- Создан новый столбец `target` (0 – нет отказа, 1..5 – TWF, HDF, PWF, OSF, RNF)
- Обучены модели: Random Forest (RF), XGBoost (XGB), LightGBM (LGBM), CatBoost (CatB), нейросеть PyTorch
- Лучшая модель в моём решении: **CatBoost** (Accuracy ≈ 0.978)

---

## ✅ Задача 1: Детальный анализ данных
- Создана страница `data_insights.py`
- Графики:
  - Гистограммы и KDE
  - Корреляционная тепловая карта
  - Интерактивный scatter plot (Plotly)
  - Анализ выбросов по IQR
  - Проекция PCA
  - Баровый график типов отказов

---

## ✅ Задача 2: Улучшенная предобработка
- One‑Hot Encoding для `Type`
- Обрезка выбросов по IQR (клиппинг)
- Стандартизация числовых признаков
- Опционально: KNN Imputer (k=3) и PCA (4 компоненты) – пользовательские опции в интерфейсе

---

## ✅ Задача 3: Мощные модели
- В дополнение к базовым добавлены:
  - **LightGBM** (быстрый, низкое потребление памяти)
  - **CatBoost** (автоматическая работа с категориями)
  - **Нейронная сеть** – 3 слоя, BatchNorm, оптимизатор Adam
- Проведено сравнение всех моделей по Accuracy и F1-weighted

---

## ✅ Задача 4: Оптимизация гиперпараметров
- Использован фреймворк **Optuna** + 5‑кратная стратифицированная кросс‑валидация
- Оптимизирована модель **Random Forest** (количество деревьев, глубина, min_samples_split, min_samples_leaf)
- Результат: точность на тесте выросла с ~0.96 до ~0.974 (+1.5%)

---

## 📊 Итоговые метрики (на моём запуске)
| Модель       | Accuracy | F1 (weighted) |
|--------------|----------|---------------|
| Random Forest| 0.971    | 0.970         |
| XGBoost      | 0.975    | 0.974         |
| LightGBM     | 0.974    | 0.973         |
| CatBoost     | 0.978    | 0.977         |
| Neural Net   | 0.969    | 0.967         |

---

Запуск проекта

git clone https://github.com/imfl1ght/predictive_maintenance_project.git

pip install -r requirements.txt

streamlit run app.py
'''

with st.sidebar:
 st.header("Внешний вид")
theme = st.selectbox("Тема", ["black", "white", "league", "beige", "sky", "night", "serif", "simple", "solarized"])
height = st.number_input("Высота слайда (px)", 400, 800, 550)
transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom", "none"])
extra = st.multiselect("Плагины", ["highlight", "katex", "mathjax2", "mathjax3", "notes", "search", "zoom"], [])

rs.slides(
presentation,
height=height,
theme=theme,
config={"transition": transition, "plugins": extra},
markdown_props={"data-separator-vertical": "^-$"},
)
