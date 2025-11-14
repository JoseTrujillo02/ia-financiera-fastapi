from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from datetime import datetime
import requests
import json
import os
import re
import unicodedata
from dotenv import load_dotenv

# ==============================
# CARGAR VARIABLES DE ENTORNO
# ==============================
load_dotenv()

# ==============================
# CONFIGURACIÓN DE OPENAI
# ==============================
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    client = None
    print("⚠ No se pudo cargar OpenAI:", e)

# ==============================
# CONFIGURACIÓN DEL BACKEND
# ==============================
BACKEND_URL = os.getenv("BACKEND_URL")

# ==============================
# FASTAPI
# ==============================
app = FastAPI(title="IA Financiera - Ajustada para Android")

class MensajeUsuario(BaseModel):
    mensaje: str

class ClasificacionRespuesta(BaseModel):
    type: str
    amount: float
    category: str
    descripcion: str
    date: str

# ==============================
# UTILIDADES
# ==============================
def eliminar_acentos(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )

def normalizar_texto(texto: str) -> str:
    texto = eliminar_acentos(texto.lower())
    reemplazos = {"@": "a", "$": "s", "3": "e", "1": "i", "4": "a", "*": ""}
    for simb, letra in reemplazos.items():
        texto = texto.replace(simb, letra)
    return re.sub(r'[^a-zñ]+', '', texto)

# ==============================
# FILTRO LENGUAJE OFENSIVO
# ==============================
PATRONES = [
    "pendej","idiot","imbecil","estupid","put","ching","verga","mamad","cabron","culer",
    "mierd","sexo","porn","follar","coger","chupar","vagin","pene","nalg","teta","boob",
    "maric","joto","zorr","asesin","suicid","degoll","ptm","vrg"
]

def contiene_lenguaje_ofensivo(texto: str) -> bool:
    t = normalizar_texto(texto)
    return any(p in t for p in PATRONES) or re.search(r"[a-z]{1,3}\*{2,}[a-z]*", texto.lower())

def validar_mensaje_con_openai(mensaje: str) -> bool:
    if not client:
        return False
    try:
        result = client.moderations.create(model="omni-moderation-latest", input=mensaje)
        return result.results[0].flagged
    except:
        return False

# ==============================
# CLASIFICADOR LOCAL
# ==============================
def clasificador_local(mensaje: str):

    msg_original = mensaje.strip()
    msg = eliminar_acentos(msg_original.lower())

    palabras_ingreso = [
        "recibi","gane","me depositaron","ingreso","ingresaron","me transfirieron",
        "cobre","me pagaron","obtuve","premio","venta","vendi"
    ]
    palabras_gasto = [
        "gaste","pague","compre","inverti","saque","deposite",
        "consumi","use","donacion","pagado","gastado"
    ]

    tipo = "income" if any(p in msg for p in palabras_ingreso) else "expense"

    categorias = {
        "Comida": [
            "comida", "restaurante", "taco", "hamburguesa", "pizza", "pollo",
            "pescado", "cena", "almuerzo", "desayuno", "antojito", "refresco",
            "bebida", "cafe", "te", "pan", "pastel", "postre", "lonche",
            "snack", "botana", "super", "mercado"
        ],
        "Transporte": [
            "uber", "taxi", "camion", "autobus", "metro", "gasolina", "pasaje",
            "peaje", "transporte", "carro", "vehiculo", "auto", "camioneta",
            "bicicleta", "moto", "combustible", "estacionamiento", "boleto",
            "metrobus", "tren"
        ],
        "Entretenimiento": [
            "cine", "pelicula", "concierto", "fiesta", "juego", "videojuego",
            "netflix", "spotify", "disney", "hbo", "series", "musica", "deporte",
            "futbol", "teatro", "bar", "discoteca", "parque", "evento", "torneo"
        ],
        "Salud": [
            "doctor", "medicina", "farmacia", "dentista", "hospital", "clinica",
            "consulta", "operacion", "cirugia", "terapia", "fisioterapia",
            "gimnasio", "nutriologo", "psicologo", "optica", "laboratorio",
            "analisis", "examen", "vacuna"
        ],
        "Educacion": [
            "libro", "escuela", "colegiatura", "universidad", "curso", "taller",
            "clase", "seminario", "capacitacion", "maestria", "diplomado"
        ],
        "Hogar": [
            "renta", "luz", "agua", "internet", "telefono", "gas", "hogar",
            "limpieza", "muebles", "decoracion", "plomeria", "reparacion",
            "electrodomestico"
        ],
        "Otros": [
            "donacion", "impuesto", "seguro", "credito", "banco", "deuda",
            "efectivo", "retiro", "transferencia", "ahorro", "prestamo"
        ]
    }

    # ===== CONTADOR DE PALABRAS =====
    coincidencias = {}
    for cat, palabras in categorias.items():
        coincidencias[cat] = sum(1 for p in palabras if re.search(rf"\b{p}\b", msg))

    categoria = max(coincidencias, key=coincidencias.get)

    # ⚠ Si no coincidió nada → error que Android sabe manejar
    if coincidencias[categoria] == 0:
        raise HTTPException(
            status_code=400,
            detail="No se pudo identificar la categoría. Menciona en qué gastaste o de dónde proviene el ingreso."
        )

    # ===== EXTRACCIÓN DE MONTO ROBUSTA =====
    match = re.search(r"\$?\s*([\d.,]+)", mensaje)
    if not match:
        raise HTTPException(
            status_code=400,
            detail="No se pudo identificar un monto válido. Por favor, menciona claramente la cantidad."
        )

    monto_str = match.group(1).replace(",", "").replace(".", "")
    monto = float(monto_str)

    if monto <= 0:
        raise HTTPException(
            status_code=400,
            detail="No se pudo identificar un monto válido. Por favor, menciona claramente la cantidad."
        )

    return tipo, categoria, monto, msg_original.capitalize()

# ==============================
# FUNCIÓN PRINCIPAL
# ==============================
def clasificar_gasto(mensaje: str, token: str):

    ahora = datetime.now()
    tipo, categoria, monto, descripcion = clasificador_local(mensaje)

    data = {
        "type": tipo,
        "amount": monto,
        "category": categoria,
        "descripcion": descripcion,
        "date": ahora.strftime("%Y-%m-%dT%H:%M:%S")
    }

    # Enviar al backend real
    headers = {"Authorization": token, "Content-Type": "application/json"}
    try:
        requests.post(BACKEND_URL, json=data, headers=headers, timeout=10)
    except:
        pass

    return data

# ==============================
# ENDPOINT
# ==============================
@app.post("/clasificar_gasto", response_model=ClasificacionRespuesta)
async def clasificar_endpoint(payload: MensajeUsuario, authorization: str = Header(...)):

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token inválido o ausente")

    if contiene_lenguaje_ofensivo(payload.mensaje):
        raise HTTPException(status_code=400, detail="El mensaje contiene lenguaje ofensivo o no permitido.")

    if validar_mensaje_con_openai(payload.mensaje):
        raise HTTPException(status_code=400, detail="El mensaje contiene contenido inapropiado (Moderation).")

    return clasificar_gasto(payload.mensaje, authorization)

@app.get("/")
async def root():
    return {"status": "ok", "message": "IA lista y sincronizada con Android 🚀"}
