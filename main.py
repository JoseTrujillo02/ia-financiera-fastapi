from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from datetime import datetime
import requests
import os
import re
import unicodedata
from dotenv import load_dotenv

# ==============================
# CARGAR VARIABLES DE ENTORNO
# ==============================
load_dotenv()

# ==============================
# CONFIGURACIÓN DE GROQ
# ==============================
try:
    from groq import Groq
    GROQ_KEY = os.getenv("GROQ_API_KEY")

    if GROQ_KEY:
        client = Groq(api_key=GROQ_KEY)
        print("✅ Groq cargado correctamente")
    else:
        print("❌ ERROR: GROQ_API_KEY no existe como variable de entorno")
        client = None

except Exception as e:
    print("❌ No se pudo inicializar Groq:", e)
    client = None


# ==============================
# CONFIGURACIÓN DEL BACKEND
# ==============================
BACKEND_URL = os.getenv("BACKEND_URL")

app = FastAPI(title="IA Financiera - Groq Edition")

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


# ==============================
# DETECCIÓN DE GROCERÍAS CON IA
# ==============================
def contiene_groserias_IA(texto: str) -> bool:
    if not client:
        return False  # si no hay IA, no bloquee

    prompt = f"""
    Analiza el siguiente mensaje y responde SOLO "SI" o "NO".

    La pregunta es:
    ¿El mensaje contiene lenguaje ofensivo, groserías, vulgaridades, insultos,
    contenido sexual explícito, amenazas, odio, acoso o expresiones inapropiadas?

    Mensaje:
    "{texto}"

    Responde SOLO "SI" o SOLO "NO".
    """

    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2
        )

        respuesta = res.choices[0].message.content.strip().lower()
        print("🟥 Filtro ofensivo IA:", respuesta)

        return respuesta == "si"

    except Exception as e:
        print("❌ ERROR filtro IA:", e)
        return False


# ==============================
# CATEGORÍAS
# ==============================
CATEGORIAS = [
    "Comida", "Transporte", "Entretenimiento", "Salud", "Educacion",
    "Hogar", "Ropa", "Mascotas", "Trabajo", "Ingresos", "Finanzas",
    "Tecnologia", "Servicios personales", "Otros"
]


# ==============================
# CLASIFICADOR
# ==============================
def clasificador_local(mensaje: str):

    msg_original = mensaje.strip().lower()

    # === MONTO ===
    match = re.search(r"\$?\s*([\d.,]+)", mensaje)
    if not match:
        raise HTTPException(400, "No se pudo identificar un monto válido")

    monto = float(match.group(1).replace(",", "").replace(".", ""))
    if monto <= 0:
        raise HTTPException(400, "Monto inválido")

    # === TIPO ===
    palabras_ingreso = ["recibi", "gane", "depositaron", "ingreso", "venta", "vendi", "pago", "me pagaron"]
    tipo = "income" if any(p in msg_original for p in palabras_ingreso) else "expense"

    categoria = "Otros"

    if client:
        prompt = f"""
        Clasifica este gasto EXACTAMENTE en una categoría:

        {", ".join(CATEGORIAS)}

        NO inventes categorías.
        Si no coincide con ninguna, responde "Otros".

        Mensaje: "{mensaje}"

        Responde solo una categoría.
        """

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10
            )

            respuesta = response.choices[0].message.content.strip()
            print("🟦 RESPUESTA GROQ:", respuesta)

            if respuesta in CATEGORIAS:
                categoria = respuesta

        except Exception as e:
            print("❌ ERROR GROQ:", e)
            categoria = "Otros"

    return tipo, categoria, monto, mensaje.capitalize()


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

    print("📤 ENVIANDO AL BACKEND:", data)

    try:
        headers = {"Authorization": token, "Content-Type": "application/json"}
        requests.post(BACKEND_URL, json=data, headers=headers, timeout=10)
    except Exception as e:
        print("⚠ Error enviando al backend:", e)

    return data


# ==============================
# ENDPOINT
# ==============================
@app.post("/clasificar_gasto", response_model=ClasificacionRespuesta)
async def clasificar_endpoint(payload: MensajeUsuario, authorization: str = Header(...)):

    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Token inválido o ausente")

    # 🔥 FILTRO DE GROCERÍAS CON IA
    if contiene_groserias_IA(payload.mensaje):
        raise HTTPException(
            400,
            "El mensaje contiene lenguaje ofensivo, groserías o contenido inapropiado."
        )

    return clasificar_gasto(payload.mensaje, authorization)


@app.get("/")
async def root():
    return {"status": "ok", "message": "IA Financiera con Groq funcionando 🚀"}
