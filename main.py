from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# =============================
# Cargar modelos y encoders
# =============================
modelo_salud = joblib.load("modelo_salud_financiera.pkl")
modelo_perfil = joblib.load("modelo_perfil_financiero.pkl")
encoders = joblib.load("todos_los_encoders.pkl")

# =============================
# Sistema experto actualizado
# =============================
def calcular_nivel_ingreso(ingresos):
    if ingresos >= 8117500:
        return "Muy alto"
    elif ingresos > 6494000 and ingresos <= 8117500:
        return "Alto"
    elif ingresos > 4870500 and ingresos <= 6494000:
        return "Medio Alto"
    elif ingresos > 3247000 and ingresos <= 4870500:
        return "Medio"
    elif ingresos > 1623500 and ingresos <= 3247000:
        return "Medio Bajo"
    else:
        return "Bajo"

def recomendaciones_expertas(data):
    ingresos = data["ingresos"]
    recomendaciones = []

    nivel = calcular_nivel_ingreso(ingresos)
    umbrales_deuda_ahorro = {
        "Bajo": 0.20,
        "Medio Bajo": 0.25,
        "Medio": 0.45,
        "Medio Alto": 0.55,
        "Alto": 0.60,
        "Muy alto": 0.65
    }

    if data["alimentacion"] > ingresos * 0.25:
        recomendaciones.append("Estás destinando una parte considerable de tus ingresos a alimentación. Intenta planificar tus compras y evitar gastos innecesarios.")

    if data["movilidad"] > ingresos * 0.15:
        recomendaciones.append("Revisa tus gastos en movilidad. Usar transporte público o compartir trayectos puede ayudarte a ahorrar.")

    if data["vivienda"] > ingresos * 0.35:
        recomendaciones.append("Tu gasto en vivienda es alto. Idealmente no debería superar el 30-35% de tus ingresos.")

    if data["salud"] < ingresos * 0.05:
        recomendaciones.append("Invertir más en salud preventiva puede evitar gastos mayores en el futuro.")
    elif data["salud"] > ingresos * 0.15:
        recomendaciones.append("Tus gastos en salud son elevados. Verifica si son recurrentes o excepcionales.")

    if data["educacion"] < ingresos * 0.05:
        recomendaciones.append("Destinar más recursos a educación puede abrirte nuevas oportunidades personales y profesionales.")

    if data["entretenimiento"] > ingresos * 0.15:
        recomendaciones.append("Intenta reducir un poco el gasto en entretenimiento. Puedes destinar ese dinero a tus metas de ahorro.")

    if data["vestuario"] > ingresos * 0.10:
        recomendaciones.append("Tus gastos en vestuario son altos. Prioriza lo esencial y aprovecha promociones.")

    if data["otros"] > ingresos * 0.10:
        recomendaciones.append("Tienes muchos gastos sin clasificar. Revisa si puedes reducir o eliminar algunos de ellos.")

    porcentaje_ahorro = data["ahorros"] / ingresos if ingresos > 0 else 0
    umbral_ahorro = umbrales_deuda_ahorro.get(nivel, 0.3)

    if porcentaje_ahorro < 0.10:
        recomendaciones.append("Intenta ahorrar al menos el 10% de tus ingresos mensuales para emergencias o metas futuras.")
    elif porcentaje_ahorro > umbral_ahorro:
        recomendaciones.append("Tu capacidad de ahorro es excelente, pero considera invertir en fondos de bajo riesgo, CDT o incluso aprender sobre inversión en acciones.")
        recomendaciones.append("También podrías usar parte de ese ahorro para educación financiera o cursos de habilidades útiles que te generen ingresos extra.")

    porcentaje_deuda = data["deudas"] / ingresos if ingresos > 0 else 0
    umbral_deuda = umbrales_deuda_ahorro.get(nivel, 0.20)

    if porcentaje_deuda > umbral_deuda:
        recomendaciones.append("Tu nivel de endeudamiento es elevado para tu nivel de ingreso. Considera consolidar o refinanciar deudas y evitar nuevas obligaciones.")

    return recomendaciones

# =============================
# API Configuración
# =============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, restringe esto a dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# Entrada esperada desde el Frontend
# =============================
class DatosUsuario(BaseModel):
    ingresos: float
    alimentacion: float
    movilidad: float
    vivienda: float
    salud: float
    educacion: float
    entretenimiento: float
    vestuario: float
    ahorros: float
    deudas: float
    otros: float

# =============================
# Lógica para predicción
# =============================
def obtener_resultados_finales(datos_usuario):
    datos_usuario["%Ahorro"] = datos_usuario["ahorros"] / datos_usuario["ingresos"] if datos_usuario["ingresos"] > 0 else 0
    datos_usuario["%Deuda"] = datos_usuario["deudas"] / datos_usuario["ingresos"] if datos_usuario["ingresos"] > 0 else 0
    datos_usuario["NivelIngreso"] = calcular_nivel_ingreso(datos_usuario["ingresos"])

    datos_usuario["NivelIngreso"] = encoders["NivelIngreso"].transform([datos_usuario["NivelIngreso"]])[0]

    df = pd.DataFrame([datos_usuario])

    pred_salud = modelo_salud.predict(df)[0]
    pred_perfil = modelo_perfil.predict(df)[0]

    salud = encoders["SaludFinanciera"].inverse_transform([pred_salud])[0]
    perfil = encoders["PerfilFinanciero"].inverse_transform([pred_perfil])[0]

    return salud, perfil

# =============================
# Endpoint POST
# =============================
@app.post("/recomendar")
def recomendar(datos: DatosUsuario):
    datos_dict = datos.dict()

    reglas = recomendaciones_expertas(datos_dict)
    salud, perfil = obtener_resultados_finales(datos_dict)

    descripciones_salud = {
        "Saludable": "Tienes un excelente manejo de tus finanzas. Tus gastos están bajo control, tienes capacidad de ahorro y bajo nivel de endeudamiento.",
        "Aceptable": "Tus finanzas son razonablemente sanas, aunque podrías optimizar algunos gastos o mejorar tus hábitos de ahorro.",
        "Riesgoso": "Tu situación financiera requiere atención. Algunos gastos o deudas podrían desestabilizar tu economía si no se controlan.",
        "Crítico": "Tus finanzas están en un punto preocupante. Necesitas tomar medidas urgentes para reducir deudas y controlar gastos."
    }

    descripciones_perfil = {
        "Ahorrador": "Tienes buenos hábitos financieros, ahorras con regularidad y mantienes tus gastos bajo control.",
        "Gastador": "Tiendes a gastar gran parte de tus ingresos y podrías tener dificultades para ahorrar o mantener estabilidad.",
        "Endeudado": "Dependes en gran medida del crédito o tienes deudas elevadas. Es importante buscar estrategias para salir de deudas.",
        "Equilibrado": "Logras un balance entre ingresos, gastos y ahorro. Mantienes una estabilidad financiera saludable."
    }

    return {
        "Clasificación Financiera": salud,
        "Descripción Clasificación": descripciones_salud.get(salud, ""),
        "Perfil Financiero": perfil,
        "Descripción Perfil": descripciones_perfil.get(perfil, ""),
        "Recomendaciones": reglas
    }