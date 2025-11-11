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
app = FastAPI(title="IA Financiera - Clasificador (con filtro avanzado)")

# ==============================
# MODELOS DE DATOS
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
# UTILIDADES
# ==============================
def eliminar_acentos(texto: str) -> str:
    """Normaliza texto eliminando tildes y diacríticos."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def registrar_bloqueo(mensaje: str, motivo: str):
    """Guarda en logs los mensajes bloqueados."""
    os.makedirs("logs", exist_ok=True)
    ruta = os.path.join("logs", "filtro.log")
    with open(ruta, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] Bloqueado ({motivo}): {mensaje}\n")
    print(f"🚫 [Bloqueado] ({motivo})")


# ==============================
# FILTRO DE LENGUAJE OFENSIVO
# ==============================
PALABRAS_PROHIBIDAS = [
    # Insultos comunes
    "pendejo", "pendeja", "idiota", "imbecil", "imbécil", "estupido", "estúpido", 
    "tonto", "tonta", "tarado", "baboso", "bruto", "bruta",
    "culero", "culera", "payaso", "ridiculo", "cretino", "menso",

    # Vulgaridades / groserías
    "puta", "puto", "put@", "chingar", "chingada", "chingado", "chingona", "chingon",
    "verga", "vrga", "v3rga", "mamada", "cabron", "cabrona", "chinga", "ptm", "ptmr",
    "pinche", "mierda", "mrd", "mierd@", "culiao", "culia@",

    # Términos sexuales o explícitos
    "sexo", "sexual", "porn", "porno", "cojer", "coger", "follar", "masturbar", 
    "orgasmo", "anal", "oral", "vagina", "pene", "chupar", "chupame", "tragar",
    "nalgas", "chichi", "tetas", "boobs", "pito", "vergon", "correrse", "semen",

    # Ofensas sociales
    "marica", "marico", "marikon", "putazo", "gay", "lesbiana", "travesti",
    "negro", "negra", "sidoso", "mongol", "retrasado", "naco", "indio", "zorra",
    "perra", "cerda", "cerdo", "prostituta", "prosti", "golfa", "ramera", "joto",

    # Violencia / amenazas
    "matar", "asesinar", "disparar", "violar", "degollar", "apuñalar",
    "golpear", "torturar", "suicidio", "suicidarme", "mátate", "muerte",
    "matalo", "ahorcar", "colgarme", "pegar", "tiro", "fusilar",

    # Variantes disfrazadas
    "hijo de puta", "p3ndejo", "m1erda", "ching4", "vrg4", "chng", "pnch",
    "hpt", "hpta", "qlo", "qlia", "p0rn", "put4", "imb3cil", "imbesil",
    "maldito", "maldita", "basura", "lacra", "escoria", "asqueroso", "repugnante"
]

def contiene_lenguaje_ofensivo(texto: str) -> bool:
    """
    Detecta lenguaje ofensivo, incluyendo censura con asteriscos o símbolos.
    """
    texto_lower = texto.lower()
    texto_sin_acentos = eliminar_acentos(texto_lower)

    # 1️⃣ Detectar palabras ofensivas directas
    for palabra in PALABRAS_PROHIBIDAS:
        if palabra in texto_sin_acentos:
            registrar_bloqueo(texto, f"Palabra directa: {palabra}")
            return True

    # 2️⃣ Detectar censura parcial (p***, m**da, ch***)
    censura_patterns = [
        r"\b([a-zA-ZñÑ])\*{2,}([a-zA-ZñÑ]*)\b",
        r"p\*{1,3}t", r"p\*{1,3}nd", r"ch\*{1,3}ng",
        r"m\*{1,3}rd", r"v\*{1,3}rg", r"ptm", r"hpta", r"vrg"
    ]
    for patron in censura_patterns:
        if re.search(patron, texto_lower):
            registrar_bloqueo(texto, f"Censura parcial: {patron}")
            return True

    return False


def validar_mensaje_con_openai(mensaje: str) -> bool:
    """Usa la API de moderación de OpenAI para detectar lenguaje inapropiado."""
    if not client:
        return False
    try:
        result = client.moderations.create(
            model="omni-moderation-latest",
            input=mensaje
        )
        flagged = result.results[0].flagged
        if flagged:
            registrar_bloqueo(mensaje, "OpenAI Moderation")
        return flagged
    except Exception as e:
        print("⚠️ Error al usar OpenAI Moderation:", e)
        return False


# ==============================
# CLASIFICADOR LOCAL
# ==============================
def clasificador_local(mensaje: str):
    mensaje_original = mensaje.strip()
    mensaje_sin_acentos = eliminar_acentos(mensaje_original.lower())

    # --- Detectar tipo ---
    palabras_ingreso = [
        "recibi", "me pagaron", "me depositaron", "gane", "ingreso", "ingresaron",
        "entrada", "vendi", "obtuve", "cobre", "me transfirieron", "me enviaron", "deposito"
    ]
    palabras_gasto = [
        "gaste", "pague", "compre", "inverti", "deposite", "saque", "transfiri",
        "done", "consumi", "pagado", "adquiri", "use", "realice un pago"
    ]

    tipo = "expense"
    if any(p in mensaje_sin_acentos for p in palabras_ingreso):
        tipo = "income"
    elif any(p in mensaje_sin_acentos for p in palabras_gasto):
        tipo = "expense"

    # --- Categorías ---
    categorias = {
        "Comida": ["comida", "restaurante", "cafe", "taco", "hamburguesa", "almuerzo", "cena", "desayuno", "super", "mercado", "bebida", "snack", "postre", "helado"],
        "Transporte": ["gasolina", "uber", "taxi", "camion", "metro", "pasaje", "auto", "vehiculo", "transporte", "carro", "moto", "peaje"],
        "Entretenimiento": ["cine", "pelicula", "concierto", "juego", "netflix", "spotify", "evento", "fiesta", "parque", "videojuego", "deporte"],
        "Salud": ["medicina", "doctor", "farmacia", "dentista", "consulta", "terapia", "hospital", "gimnasio", "psicologo", "analisis"],
        "Educacion": ["libro", "curso", "escuela", "educacion", "universidad", "clase", "colegiatura", "taller", "capacitacion", "seminario"],
        "Hogar": ["renta", "luz", "agua", "internet", "gas", "casa", "hogar", "muebles", "reparacion", "limpieza"],
        "Ropa": ["ropa", "camisa", "pantalon", "zapato", "tenis", "vestido", "blusa", "moda"],
        "Mascotas": ["perro", "gato", "veterinario", "croquetas", "mascota", "alimento para perro", "adopcion"],
        "Trabajo": ["oficina", "computadora", "herramienta", "software", "papeleria", "proyecto", "nomina", "cliente", "salario"],
        "Otros": ["donacion", "regalo", "impuesto", "banco", "seguro", "credito", "deuda", "ahorro"]
    }

    categoria = "Otros"
    for cat, palabras in categorias.items():
        if any(p in mensaje_sin_acentos for p in palabras):
            categoria = cat
            break

    # --- Extraer monto ---
    patrones = [
        r'\$\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
        r'(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*pesos?',
        r'(\d+(?:,\d{3})*(?:\.\d{1,2})?)'
    ]
    monto = 0.0
    for patron in patrones:
        match = re.search(patron, mensaje)
        if match:
            try:
                monto_str = match.group(1).replace(',', '')
                monto = float(monto_str)
            except:
                monto = 0.0
            break

    return tipo, categoria, monto, mensaje_original.capitalize()


# ==============================
# FUNCIÓN PRINCIPAL
# ==============================
def clasificar_gasto(mensaje: str, token: str):
    ahora = datetime.now()
    tipo, categoria, monto, descripcion = clasificador_local(mensaje)
    data = {
        "type": tipo,
        "category": categoria,
        "amount": monto,
        "descripcion": descripcion,
        "date": ahora.strftime("%Y-%m-%dT%H:%M:%S")
    }

    headers = {"Authorization": token, "Content-Type": "application/json"}

    try:
        response = requests.post(BACKEND_URL, json=data, headers=headers, timeout=10)
        print(f"📥 Status: {response.status_code} | 📝 Body: {response.text}")
    except Exception as ex:
        print(f"❌ Error enviando al backend: {ex}")

    return data


# ==============================
# ENDPOINT PRINCIPAL
# ==============================
@app.post("/clasificar_gasto", response_model=ClasificacionRespuesta)
async def clasificar_endpoint(payload: MensajeUsuario, authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token inválido o ausente")

    # 🚫 Verificar lenguaje ofensivo o censurado
    if contiene_lenguaje_ofensivo(payload.mensaje):
        raise HTTPException(status_code=400, detail="El mensaje contiene lenguaje ofensivo o censurado.")
    if validar_mensaje_con_openai(payload.mensaje):
        raise HTTPException(status_code=400, detail="El mensaje contiene lenguaje inapropiado (Moderation).")

    return clasificar_gasto(payload.mensaje, authorization)


@app.get("/")
async def root():
    return {"status": "ok", "message": "IA Financiera con filtro extremo funcionando 🚫"}
