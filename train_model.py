
import pandas as pd
import joblib
import sklearn

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

print("Scikit-learn version:", sklearn.__version__)

folder = Path(__file__).parent
dataset_path = folder / "PCOS_data_without_infertility.xlsx"

df = pd.read_excel(dataset_path, sheet_name="Full_new")
df.columns = df.columns.str.strip()

cols = [
    "Age (yrs)",
    "Weight (Kg)",
    "BMI",
    "Cycle(R/I)",
    "Cycle length(days)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "PCOS (Y/N)"
]

data = df[cols].copy()

# Ensure the target contains valid binary labels
data = data.dropna(subset=["PCOS (Y/N)"])
data = data[data["PCOS (Y/N)"].isin([0, 1])]

X = data.drop(columns=["PCOS (Y/N)"])
y = data["PCOS (Y/N)"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(n_neighbors=5))
])

model.fit(X_train, y_train)

model_path = folder / "pcos_knn_model.pkl"
joblib.dump(model, model_path)

print("Model trained and saved:", model_path)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
