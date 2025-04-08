import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# =============================
# 1. Cargar y preparar los datos
# =============================
df = pd.read_csv("datos_financieros.csv")

# Renombrar columnas para usar nombres consistentes si lo necesitas
df.rename(columns={
    "Ingreso": "ingresos",
    "Alimentación": "alimentacion",
    "Movilidad": "movilidad",
    "Vivienda": "vivienda",
    "Salud": "salud",
    "Educación": "educacion",
    "Entretenimiento": "entretenimiento",
    "Vestuario": "vestuario",
    "Ahorros": "ahorros",
    "Deudas": "deudas",
    "Otros": "otros"
}, inplace=True)

# =============================
# 2. Codificar columnas categóricas
# =============================
encoders = {}

# Codificar "NivelIngreso", "SaludFinanciera" y "PerfilFinanciero"
for col in ["NivelIngreso", "SaludFinanciera", "PerfilFinanciero"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# =============================
# 3. Separar variables predictoras
# =============================
features = [
    "ingresos", "alimentacion", "movilidad", "vivienda", "salud",
    "educacion", "entretenimiento", "vestuario", "ahorros", "deudas",
    "otros", "%Ahorro", "%Deuda", "NivelIngreso"
]
X = df[features]

# =============================
# 4. Entrenar modelo para Salud Financiera
# =============================
y_salud = df["SaludFinanciera"]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_salud, test_size=0.2, random_state=42)

modelo_salud = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_salud.fit(X_train_s, y_train_s)

# =============================
# 5. Entrenar modelo para Perfil Financiero
# =============================
y_perfil = df["PerfilFinanciero"]
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_perfil, test_size=0.2, random_state=42)

modelo_perfil = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_perfil.fit(X_train_p, y_train_p)

# =============================
# 6. Guardar modelos y encoders
# =============================
joblib.dump(modelo_salud, "modelo_salud_financiera.pkl")
joblib.dump(modelo_perfil, "modelo_perfil_financiero.pkl")
joblib.dump(encoders, "todos_los_encoders.pkl")

print("Modelos entrenados y guardados correctamente.")