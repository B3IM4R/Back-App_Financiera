import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
import joblib

# Cargar y preparar los datos
df = pd.read_csv("datos_entrenamiento_balanceado.csv")

# Codificar columnas categóricas
encoders = {}
for col in ["NivelIngreso", "SaludFinanciera", "PerfilFinanciero"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Definir variables de entrada y salida
features = [
    "Ingreso", "Alimentación", "Movilidad", "Vivienda", "Salud",
    "Educación", "Entretenimiento", "Vestuario", "Ahorros", "Deudas",
    "Otros", "%Ahorro", "%Deuda", "NivelIngreso"
]

X = df[features]
y = df[["PerfilFinanciero", "SaludFinanciera"]]

# Entrenar modelo multisalida
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_base = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_multisalida = MultiOutputClassifier(modelo_base)

modelo_multisalida.fit(X_train, y_train)

# Validación cruzada individual
for i, target in enumerate(["PerfilFinanciero", "SaludFinanciera"]):
    scores = cross_val_score(modelo_multisalida.estimators_[i], X, y.iloc[:, i], cv=5)
    print(f"Precisión promedio (validación cruzada - {target}):", scores.mean())

# Guardar el modelo y los encoders
joblib.dump(modelo_multisalida, "modelo_financiero_multisalida.pkl")
joblib.dump(encoders, "encoders.pkl")

print("\nModelo multisalida entrenado, validado y guardado correctamente.")