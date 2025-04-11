from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# Cargar modelo multisalida y encoders
modelo = joblib.load("modelo_financiero_multisalida.pkl")
encoders = joblib.load("encoders.pkl")

# Función para calcular el nivel de ingreso
def calcular_nivel_ingreso(ingresos: float) -> str:
    if ingresos >= 8117500:
        return "Muy Alto"
    elif ingresos > 6494000:
        return "Alto"
    elif ingresos > 4870500:
        return "Medio Alto"
    elif ingresos > 3247000:
        return "Medio"
    elif ingresos > 1623500:
        return "Medio Bajo"
    else:
        return "Bajo"

# Sistema experto de recomendaciones
def recomendaciones_expertas(data: dict) -> list:
    ingresos = data["Ingreso"]
    recomendaciones = []

    nivel = calcular_nivel_ingreso(ingresos)
    umbrales = {
        "Bajo": 0.20,
        "Medio Bajo": 0.20,
        "Medio": 0.25,
        "Medio Alto": 0.30,
        "Alto": 0.35,
        "Muy Alto": 0.40
    }

    if data["Alimentación"] > ingresos * 0.25:
        recomendaciones.append("Estás destinando una parte alta de tus ingresos a alimentación. Planea un menú semanal, evita comprar en exceso, compara precios entre supermercados y prioriza productos locales.")
    elif data["Alimentación"] < ingresos * 0.10:
        recomendaciones.append("Tu gasto en alimentación es muy bajo. Asegúrate de incluir alimentos variados y nutritivos como frutas, verduras, proteínas y carbohidratos de calidad para mantener una buena salud.")

    if data["Movilidad"] > ingresos * 0.15:
        recomendaciones.append("Tu gasto en movilidad es elevado. Considera usar transporte público, compartir vehículo o usar bicicleta en trayectos cortos.")

    if data["Vivienda"] > ingresos * 0.35:
        recomendaciones.append("Tu gasto en vivienda supera el límite recomendado. Evalúa opciones como renegociar el arriendo, buscar una vivienda más económica o compartir lugar con alguien confiable.")

    if data["Salud"] < ingresos * 0.05:
        recomendaciones.append("Estás destinando poco a salud. Procura hacerte chequeos médicos anuales, mantener un plan de EPS activo y, si es posible, contar con un seguro complementario.")
    elif data["Salud"] > ingresos * 0.15:
        recomendaciones.append("Tus gastos en salud son altos. Revisa si puedes cambiar a un proveedor más económico, aprovechar servicios incluidos en tu EPS o prevenir enfermedades con buenos hábitos.")

    if data["Educación"] < ingresos * 0.05:
        recomendaciones.append("Invertir en educación puede mejorar tus oportunidades laborales. Considera hacer cursos gratuitos o económicos en plataformas confiables.")
    elif data["Educación"] > ingresos * 0.20:
        recomendaciones.append("Estás invirtiendo bastante en educación. Asegúrate de que esta inversión esté alineada con tus metas personales o profesionales y que te sea útil.")

    if data["Entretenimiento"] > ingresos * 0.15:
        recomendaciones.append("Estás gastando bastante en entretenimiento. Puedes reducir este rubro optando por planes gratuitos o económicos como parques, eventos culturales, películas en casa o promociones.")
    elif data["Entretenimiento"] < ingresos * 0.01:
        recomendaciones.append("Recuerda que también es importante descansar y disfrutar. Incluye actividades recreativas accesibles como caminatas, lectura, música o encuentros con amigos.")

    if data["Vestuario"] > ingresos * 0.10:
        recomendaciones.append("Estás invirtiendo mucho en ropa. Antes de comprar, pregúntate si realmente lo necesitas, espera temporadas de descuento y opta por prendas versátiles de buena calidad.")

    if data["Otros"] > ingresos * 0.10:
        recomendaciones.append("Tienes muchos gastos no clasificados. Revisa extractos y recibos para identificar compras impulsivas, suscripciones innecesarias o cobros repetidos que puedas eliminar.")

    ahorro_pct = data["Ahorros"] / ingresos if ingresos > 0 else 0
    umbral_ahorro = umbrales.get(nivel, 0.3)
    if ahorro_pct < 0.10:
        recomendaciones.append("Intenta separar un 10% de tus ingresos mensuales y ahórralos para crear un fondo de emergencia.")
    elif ahorro_pct <= umbral_ahorro:
        recomendaciones.append("Tu nivel de ahorro es adecuado. Define objetivos claros como un fondo de emergencia, estudios o viajes, y automatiza el proceso para mantener el hábito.")
    else:
        recomendaciones.append("¡Excelente! Estás ahorrando por encima del promedio. Es buen momento para explorar opciones como CDT, fondos de inversión o cuentas de ahorro programado.")

    deuda_pct = data["Deudas"] / ingresos if ingresos > 0 else 0
    umbral_deuda = umbrales.get(nivel, 0.20)
    if deuda_pct > umbral_deuda:
        recomendaciones.append("Tu deuda es alta en relación a tus ingresos. Revisa tasas de interés, consolida tus créditos si es posible y prioriza pagar primero los de mayor interés.")
    elif deuda_pct > 0 and deuda_pct <= 0.10:
        recomendaciones.append("Tus compromisos financieros son bajos. Mantén este buen manejo y, si es posible, aumenta ligeramente tus abonos para saldar tus deudas más rápido.")

    return recomendaciones

# Configuración de API
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de entrada
class DatosUsuario(BaseModel):
    Ingreso: float
    Alimentación: float
    Movilidad: float
    Vivienda: float
    Salud: float
    Educación: float
    Entretenimiento: float
    Vestuario: float
    Ahorros: float
    Deudas: float
    Otros: float

@app.post("/recomendar")
def recomendar(datos: DatosUsuario):
    datos_dict = datos.dict()
    ingresos = datos_dict["Ingreso"]
    nivel_ingreso = calcular_nivel_ingreso(ingresos)

    # Codificar nivel de ingreso
    nivel_ingreso_codificado = encoders["NivelIngreso"].transform([nivel_ingreso])[0]

    # Calcular % ahorro y deuda
    porcentaje_ahorro = (datos_dict["Ahorros"] / ingresos) * 100 if ingresos > 0 else 0
    porcentaje_deuda = (datos_dict["Deudas"] / ingresos) * 100 if ingresos > 0 else 0

    # Crear vector de entrada
    X_nuevo = pd.DataFrame([[ 
        datos_dict["Ingreso"], datos_dict["Alimentación"], datos_dict["Movilidad"],
        datos_dict["Vivienda"], datos_dict["Salud"], datos_dict["Educación"],
        datos_dict["Entretenimiento"], datos_dict["Vestuario"], datos_dict["Ahorros"],
        datos_dict["Deudas"], datos_dict["Otros"],
        porcentaje_ahorro, porcentaje_deuda, nivel_ingreso_codificado
    ]], columns=[
        "Ingreso", "Alimentación", "Movilidad", "Vivienda", "Salud", "Educación",
        "Entretenimiento", "Vestuario", "Ahorros", "Deudas", "Otros",
        "%Ahorro", "%Deuda", "NivelIngreso"
    ])

    # Hacer la predicción
    pred = modelo.predict(X_nuevo)[0]

    perfil = encoders["PerfilFinanciero"].inverse_transform([pred[0]])[0]
    salud = encoders["SaludFinanciera"].inverse_transform([pred[1]])[0]

    # Validar perfil por umbrales
    def ajustar_perfil(perfil_predicho):
        condiciones = {
            "Ahorrador": porcentaje_ahorro > {
                "Bajo": 20, "Medio Bajo": 25, "Medio": 25,
                "Medio Alto": 30, "Alto": 30, "Muy Alto": 30
            }[nivel_ingreso],
            "Gastador": any((
                datos_dict["Entretenimiento"] / ingresos > {
                    "Bajo": 0.15, "Medio Bajo": 0.15, "Medio": 0.15,
                    "Medio Alto": 0.20, "Alto": 0.20, "Muy Alto": 0.30
                }[nivel_ingreso],
                datos_dict["Vestuario"] / ingresos > {
                    "Bajo": 0.15, "Medio Bajo": 0.15, "Medio": 0.15,
                    "Medio Alto": 0.20, "Alto": 0.20, "Muy Alto": 0.30
                }[nivel_ingreso],
                datos_dict["Otros"] / ingresos > {
                    "Bajo": 0.15, "Medio Bajo": 0.15, "Medio": 0.15,
                    "Medio Alto": 0.20, "Alto": 0.20, "Muy Alto": 0.30
                }[nivel_ingreso],
            )),
            "Endeudado": porcentaje_deuda > {
                "Bajo": 20, "Medio Bajo": 20, "Medio": 30,
                "Medio Alto": 35, "Alto": 40, "Muy Alto": 40
            }[nivel_ingreso],
            "Equilibrado": all((
                datos_dict["Alimentación"] < ingresos * 0.25,
                datos_dict["Movilidad"] < ingresos * 0.15,
                datos_dict["Vivienda"] < ingresos * 0.30,
                datos_dict["Salud"] < ingresos * 0.15,
                datos_dict["Educación"] < ingresos * 0.20,
                datos_dict["Entretenimiento"] < ingresos * 0.15,
                datos_dict["Otros"] < ingresos * 0.10,
            ))
        }

        for perfil_candidato, cumple in condiciones.items():
            if cumple:
                return perfil_candidato

        return perfil_predicho

    perfil_ajustado = ajustar_perfil(perfil)
    
    if perfil_ajustado != perfil:
        salud_por_perfil = {
            "Ahorrador": "Aceptable",
            "Equilibrado": "Saludable",
            "Gastador": "Riesgosa",
            "Endeudado": "Crítica"
        }
        salud = salud_por_perfil.get(perfil_ajustado, salud)

    descripciones_salud = {
        "Saludable": "Tienes un excelente manejo de tus recursos. Sigue tomando decisiones informadas y pensando a largo plazo.",
        "Aceptable": "Tus finanzas están estables, pero podrías mejorar en ahorro o reducir algunos gastos.",
        "Riesgosa": "Hay señales de advertencia en tu presupuesto. Ajusta tus hábitos antes de que los compromisos crezcan.",
        "Crítica": "Tu situación requiere atención inmediata. Reduce gastos urgentes, prioriza pagos esenciales y busca asesoría si es necesario."
    }

    descripciones_perfil = {
        "Ahorrador": "Gestionas bien tus ingresos y mantienes un enfoque claro en el ahorro. Excelente disciplina.",
        "Gastador": "Estás destinando demasiado al consumo. Controla tus impulsos, usa presupuestos y prioriza necesidades.",
        "Endeudado": "Prioriza saldar tus deudas y evita nuevos compromisos financieros.",
        "Equilibrado": "Logras mantener estabilidad entre ingresos, gastos y ahorro. Continúa con esta gestión consciente."
    }

    recomendaciones = recomendaciones_expertas(datos_dict)

    return {
        "Clasificación Financiera": salud,
        "Descripción Clasificación": descripciones_salud.get(salud, ""),
        "Perfil Financiero": perfil_ajustado,
        "Descripción Perfil": descripciones_perfil.get(perfil_ajustado, ""),
        "Recomendaciones": recomendaciones
    }