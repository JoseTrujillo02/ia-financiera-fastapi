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
# CONFIGURACIÓN DE FASTAPI
# ==============================
app = FastAPI(title="IA Financiera - Clasificador con categorías ampliadas y filtro extremo")

# ==============================
# MODELOS
# ==============================
class MensajeUsuario(BaseModel):
    mensaje: str

class ClasificacionRespuesta(BaseModel):
    type: str
    amount: float
    category: str
    descripcion: str
    date: str

# ==============================
# UTILIDADES DE TEXTO
# ==============================
def eliminar_acentos(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )

def normalizar_texto(texto: str) -> str:
    texto = eliminar_acentos(texto.lower())
    reemplazos = {
        "@": "a", "$": "s", "1": "i", "!": "i", "3": "e", "0": "o",
        "4": "a", "7": "t", "5": "s", "8": "b", "*": "", "-": "", ".": ""
    }
    for simb, letra in reemplazos.items():
        texto = texto.replace(simb, letra)
    return re.sub(r'[^a-zñ]+', '', texto)

def registrar_bloqueo(mensaje: str, motivo: str):
    os.makedirs("logs", exist_ok=True)
    ruta = os.path.join("logs", "filtro.log")
    with open(ruta, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] Bloqueado ({motivo}): {mensaje}\n")
    print(f"🚫 Bloqueado ({motivo})")

# ==============================
# FILTRO DE LENGUAJE OFENSIVO
# ==============================
PATRONES_OFENSIVOS = [
    "pendej", "idiot", "imbecil", "estupid", "put", "ching", "verga", "mamad",
    "cabron", "culer", "mierd", "sexo", "porn", "follar", "coger", "chupar",
    "vagin", "pene", "nalg", "teta", "boob", "pito", "maric", "joto", "zorr",
    "perr", "violac", "asesin", "suicid", "degoll", "ahorc", "ptm", "vrg", "hpta"
]

def contiene_lenguaje_ofensivo(texto: str) -> bool:
    texto_normal = normalizar_texto(texto)
    for patron in PATRONES_OFENSIVOS:
        if patron in texto_normal:
            registrar_bloqueo(texto, f"Coincidencia: {patron}")
            return True
    if re.search(r"[a-z]{1,3}\*{2,}[a-z]*", texto.lower()):
        registrar_bloqueo(texto, "Censura con asteriscos")
        return True
    return False

def validar_mensaje_con_openai(mensaje: str) -> bool:
    if not client:
        return False
    try:
        result = client.moderations.create(model="omni-moderation-latest", input=mensaje)
        flagged = result.results[0].flagged
        if flagged:
            registrar_bloqueo(mensaje, "OpenAI Moderation")
        return flagged
    except Exception as e:
        print("⚠ Error al usar OpenAI Moderation:", e)
        return False

# ==============================
# CLASIFICADOR LOCAL CON CATEGORÍAS EXTENDIDAS
# ==============================
def clasificador_local(mensaje: str):
    msg_original = mensaje.strip()
    msg = eliminar_acentos(msg_original.lower())

    # Detectar tipo (ingreso o gasto)
    palabras_ingreso = [
        "recibi", "gane", "me depositaron", "ingreso", "ingresaron", "me transfirieron",
        "cobre", "me pagaron", "obtuve", "entrada", "premio", "venta", "vendi", "me dieron"
    ]
    palabras_gasto = [
        "gaste", "pague", "compre", "inverti", "saque", "deposite", "consumi", "use",
        "donacion", "done", "pagado", "realice un pago", "invertido", "gastado"
    ]

    tipo = "income" if any(p in msg for p in palabras_ingreso) else "expense" if any(p in msg for p in palabras_gasto) else "expense"

    # CATEGORÍAS AMPLIADAS
    categorias = {
        "Comida": [
            "comida", "restaurante", "taco", "hamburguesa", "pizza", "pollo", "pescado",
            "cena", "almuerzo", "desayuno", "antojito", "refresco", "bebida", "cafe", "te",
            "pan", "pastel", "postre", "lonche", "antojito", "snack", "botana", "super", "mercado"
        ],
        "Transporte": [
            "transporte", "uber", "taxi", "camion", "autobus", "metro", "gasolina", "pasaje", "peaje",
            "carro", "vehiculo", "auto", "camioneta", "bicicleta", "moto",
            "combustible", "estacionamiento", "boleto", "metrobus"
        ],
        "Entretenimiento": [
            "cine", "pelicula", "concierto", "fiesta", "juego", "videojuego", "netflix",
            "spotify", "disney", "hbo", "series", "musica", "deporte", "futbol", "baloncesto",
            "teatro", "bar", "discoteca", "parque", "evento", "torneo", "show"
        ],
        "Salud": [
            "doctor", "medicina", "farmacia", "dentista", "hospital", "clinica", "consulta",
            "operacion", "cirugia", "terapia", "fisioterapia", "gimnasio", "entrenamiento",
            "nutriologo", "psicologo", "optica", "laboratorio", "analisis", "examen", "vacuna"
        ],
        "Educacion": [
            "libro", "escuela", "colegiatura", "universidad", "curso", "taller", "clase",
            "seminario", "capacitacion", "maestria", "diplomado", "tutorial", "coaching",
            "plataforma educativa", "suscripcion", "beca", "educacion"
        ],
        "Hogar": [
            "renta", "luz", "agua", "internet", "telefono", "gas", "hogar", "limpieza",
            "muebles", "decoracion", "plomeria", "electricidad", "reparacion", "electrodomestico",
            "lavanderia", "jardin", "vivienda", "supermercado", "herramienta"
        ],
        "Ropa": [
            "ropa", "camisa", "pantalon", "zapato", "tenis", "vestido", "abrigo", "blusa",
            "falda", "bufanda", "accesorio", "sombrero", "reloj", "moda", "jeans", "playera"
        ],
        "Mascotas": [
            "perro", "gato", "mascota", "veterinario", "croquetas", "alimento", "correa",
            "adopcion", "jaula", "juguete para perro", "juguete para gato", "limpieza animal"
        ],
        "Trabajo": [
            "oficina", "computadora", "laptop", "teclado", "monitor", "papeleria", "impresora",
            "software", "licencia", "suscripcion", "proyecto", "cliente", "material", "herramienta",
            "nomina", "empleo", "jornada", "sueldo", "salario", "servicio profesional"
        ],
        "Otros": [
            "donacion", "impuesto", "seguro", "credito", "banco", "deuda", "efectivo",
            "retiro", "transferencia", "ahorro", "prestamo", "gasto", "cuenta", "servicio"
        ]
    }

    # Contar coincidencias por categoría
    coincidencias = {}
    for cat, palabras in categorias.items():
        coincidencias[cat] = sum(
            1 for p in palabras
            if re.search(rf'\b{re.escape(p)}\b', msg)  # Coincidencia exacta de palabra
        )

    # Determinar categoría con mayor coincidencia
    categoria = max(coincidencias, key=coincidencias.get)
    if coincidencias[categoria] == 0:
        # Fallback por tiendas
        if any(tienda in msg for tienda in ["oxxo", "seven", "7eleven", "gasolinera"]):
            categoria = "Transporte"
        elif any(tienda in msg for tienda in ["walmart", "soriana", "chedraui", "bodega", "super"]):
            categoria = "Hogar"
        elif any(tienda in msg for tienda in ["netflix", "spotify", "disney", "hbo", "youtube"]):
            categoria = "Entretenimiento"
        else:
            categoria = "Otros"

    # Extraer monto
    match = re.search(r'\$?\s*(\d+(?:[.,]\d{1,2})?)', mensaje)
    monto = float(match.group(1).replace(',', '.')) if match else 0.0

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

    headers = {"Authorization": token, "Content-Type": "application/json"}

    response = requests.post(BACKEND_URL, json=data, headers=headers, timeout=10)
    print(f"📤 Enviado al backend: {response.status_code}")
    return data

# ==============================
# ENDPOINT
# ==============================
@app.post("/clasificar_gasto", response_model=ClasificacionRespuesta)
async def clasificar_endpoint(payload: MensajeUsuario, authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token inválido o ausente")

    if contiene_lenguaje_ofensivo(payload.mensaje):
        raise HTTPException(status_code=400, detail="El mensaje contiene lenguaje ofensivo o censurado.")
    if validar_mensaje_con_openai(payload.mensaje):
        raise HTTPException(status_code=400, detail="El mensaje contiene contenido inapropiado (Moderation).")

    return clasificar_gasto(payload.mensaje, authorization)

@app.get("/")
async def root():
    return {"status": "ok", "message": "IA Financiera con filtro y categorías ampliadas funcionando 🚀"}
