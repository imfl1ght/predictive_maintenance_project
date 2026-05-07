import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna

st.set_page_config(page_title="Моделирование", layout="wide")
st.title("⚙️ Продвинутое обучение моделей (задачи 0-4)")

# Загрузка данных и создание мультиклассовой цели
def load_data():
    uploaded = st.file_uploader("Загрузите CSV-файл с данными", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        st.warning("Пожалуйста, загрузите файл данных.")
        st.stop()
    # Цель: 0 - нет отказа, 1-5 - тип отказа
    fault_dict = {'TWF':1, 'HDF':2, 'PWF':3, 'OSF':4, 'RNF':5}
    def get_label(row):
        for key, val in fault_dict.items():
            if row[key] == 1:
                return val
        return 0
    df['target'] = df.apply(get_label, axis=1)
    return df
df = load_data()
st.success("Датасет загружен. Цель: target (0-5)")

# Параметры предобработки (задача 2)
st.header("Настройки предобработки")
clip_out = st.checkbox("Обрезка выбросов (IQR)", value=True)
use_pca = st.checkbox("PCA (4 компоненты)", value=False)
use_imputer = st.checkbox("KNN Imputer (k=3)", value=False)
num_feat = ['Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
cat_feat = ['Type']
X = df.drop(['target', 'Machine failure', 'UDI', 'Product ID',
             'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
y = df['target']

if clip_out:
    for col in num_feat:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5*iqr
        hi = q3 + 1.5*iqr
        X[col] = X[col].clip(lo, hi)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=5678, stratify=y)
# Pipeline
preproc = ColumnTransformer([
    ('num', StandardScaler(), num_feat),
    ('cat', OneHotEncoder(drop='first'), cat_feat)
])

X_tr_proc = preproc.fit_transform(X_tr)
X_te_proc = preproc.transform(X_te)

if use_imputer:
    imputer = KNNImputer(n_neighbors=3)
    X_tr_proc = imputer.fit_transform(X_tr_proc)
    X_te_proc = imputer.transform(X_te_proc)

if use_pca:
    pca = PCA(n_components=4)
    X_tr_proc = pca.fit_transform(X_tr_proc)
    X_te_proc = pca.transform(X_te_proc)

st.write(f"Финальная размерность: {X_tr_proc.shape[1]}")

# Модели (задача 3)
models = {
    "RF": RandomForestClassifier(n_estimators=130, random_state=5678),
    "XGB": xgb.XGBClassifier(n_estimators=130, objective='multi:softmax', num_class=6,
                             random_state=5678, eval_metric='mlogloss'),
    "LGBM": lgb.LGBMClassifier(n_estimators=130, random_state=5678, verbose=-1),
    "CatB": cb.CatBoostClassifier(iterations=130, verbose=0, random_state=5678)
}

# Нейросеть
class SimpleNN(nn.Module):
    def __init__(self, inp_dim, out_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_nn(X_tr, y_tr, X_te, y_te, inp_dim, epochs=15):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleNN(inp_dim).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.0005)
    X_tr_t = torch.tensor(X_tr.astype(np.float32))
    y_tr_t = torch.tensor(y_tr.values, dtype=torch.long)
    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
    model.eval()
    X_te_t = torch.tensor(X_te.astype(np.float32)).to(device)
    with torch.no_grad():
        pred = model(X_te_t).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_te, pred)
    return model, acc

if st.button("🚀 Запустить обучение"):
    results = {}
    for name, m in models.items():
        m.fit(X_tr_proc, y_tr)
        y_pred = m.predict(X_te_proc)
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, average='weighted')
        results[name] = {'acc': acc, 'f1': f1, 'model': m, 'pred': y_pred}
        st.write(f"**{name}** → Acc: {acc:.4f}, F1: {f1:.4f}")

    nn_mod, nn_acc = train_nn(X_tr_proc, y_tr, X_te_proc, y_te, X_tr_proc.shape[1])
    results['NN'] = {'acc': nn_acc, 'f1': 0, 'model': nn_mod}
    st.write(f"**Neural Net** → Acc: {nn_acc:.4f}")

    classical_names = [name for name in results.keys() if name != 'NN']
    best_name = max(classical_names, key=lambda name: results[name]['acc'])
    best_model = results[best_name]['model']
    st.success(f"Лучшая модель: {best_name} (Accuracy {results[best_name]['acc']:.4f})")
    # Матрица ошибок
    y_best = results[best_name]['pred'] if best_name != 'NN' else None
    if best_name == 'NN':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        X_te_t = torch.tensor(X_te_proc.astype(np.float32)).to(device)
        nn_mod.eval()
        with torch.no_grad():
            y_best = nn_mod(X_te_t).argmax(dim=1).cpu().numpy()
    cm = confusion_matrix(y_te, y_best)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    st.pyplot(fig)
    st.text(classification_report(y_te, y_best))
    st.session_state.best_model = best_model
    st.session_state.preproc = preproc
    st.session_state.use_pca = use_pca
    if use_pca:
        st.session_state.pca = pca
    st.session_state.num_feat = num_feat

# Optuna (задача 4)
st.header("🔧 Оптимизация Random Forest через Optuna")
if st.button("Запустить Optuna (20 итераций)"):
    def obj(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 80, 200),
            'max_depth': trial.suggest_int('max_depth', 6, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 12),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8)
        }
        rf = RandomForestClassifier(**params, random_state=5678, n_jobs=-1)
        cv = StratifiedKFold(5, shuffle=True, random_state=5678)
        scores = cross_val_score(rf, X_tr_proc, y_tr, cv=cv, scoring='accuracy')
        return scores.mean()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(obj, n_trials=20, show_progress_bar=True)
    st.write("**Лучшие параметры:**", study.best_params)
    st.write("**Лучшее CV accuracy:**", study.best_value)
    best_rf = RandomForestClassifier(**study.best_params, random_state=5678)
    best_rf.fit(X_tr_proc, y_tr)
    test_acc = accuracy_score(y_te, best_rf.predict(X_te_proc))
    st.write(f"**Точность на тесте после Optuna:** {test_acc:.4f}")

# Предсказание
st.header("🔮 Прогноз на новых данных")

if 'best_model' in st.session_state and st.session_state.best_model is not None:
    with st.form("pred_form"):
        tp = st.selectbox("Тип продукта", ["L", "M", "H"])
        at = st.number_input("Air temp [K]", 290.0, 310.0, 300.0)
        pt = st.number_input("Process temp [K]", 300.0, 320.0, 310.0)
        rpm = st.number_input("Rotational speed [rpm]", 1000, 3000, 1500)
        tq = st.number_input("Torque [Nm]", 0.0, 100.0, 40.0)
        wear = st.number_input("Tool wear [min]", 0, 300, 100)
        sub = st.form_submit_button("Предсказать")
        if sub:
            inp = pd.DataFrame([[tp, at, pt, rpm, tq, wear]],
                               columns=['Type'] + num_feat)
            inp_proc = st.session_state.preproc.transform(inp)
            if st.session_state.use_pca:
                inp_proc = st.session_state.pca.transform(inp_proc)
            pred = st.session_state.best_model.predict(inp_proc)[0]
            fault_map = {1:"TWF", 2:"HDF", 3:"PWF", 4:"OSF", 5:"RNF"}
            st.write(f"**Класс отказа:** {pred}")
            if pred == 0:
                st.success("✅ Нет отказа")
            else:
                st.error(f"⚠️ Отказ типа {fault_map.get(pred, '?')}")
else:
    st.info("👉 Сначала обучите модель, нажав кнопку 'запустить обучение' выше.")
