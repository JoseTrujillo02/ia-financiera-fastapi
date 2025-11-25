from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from datetime import datetime
import requests
import os
import re
import json
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
# UTILIDADES
# ==============================
def eliminar_acentos(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


# =====================================================
# 🔥 DETECCIÓN DE GROCERÍAS / CONTENIDO OFENSIVO (IA)
# =====================================================
def contiene_groserias_IA(texto: str) -> bool:
    """
    Usa Groq para detectar lenguaje ofensivo, vulgaridades,
    contenido sexual, amenazas, etc.
    Responde SOLO con JSON:
    { "ofensivo": true/false }
    """
    if not client:
        return False

    prompt = f"""
    Analiza el siguiente mensaje y responde únicamente en JSON válido:
    {{
        "ofensivo": true/false
    }}

    Considera ofensivo:
    - groserías
    - insultos
    - vulgaridades
    - lenguaje explícito sexual
    - amenazas
    - acoso
    - odio
    - violencia

    Mensaje: "{texto}"
    """

    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )

        contenido = res.choices[0].message.content.strip()
        print("🟥 Filtro ofensivo IA (JSON):", contenido)

        data = json.loads(contenido)

        return data.get("ofensivo", False)

    except Exception as e:
        print("❌ ERROR filtro IA:", e)
        return False



# =====================================================
# 🔥 CLASIFICACIÓN DE CATEGORÍA CON IA (JSON)
# =====================================================
CATEGORIAS = [
    "Comida", "Transporte", "Entretenimiento", "Salud", "Educacion",
    "Hogar", "Ropa", "Mascotas", "Trabajo", "Ingresos", "Finanzas",
    "Tecnologia", "Servicios personales", "Otros"
]

def clasificar_categoria_IA(mensaje: str) -> str:
    """
    Usa Groq para elegir la mejor categoría.
    Siempre responde JSON:
    { "categoria": "string" }
    """
    if not client:
        return "Otros"

    prompt = f"""
    Determina la CATEGORÍA del siguiente gasto.

    Categorías válidas:
    {", ".join(CATEGORIAS)}

    Regla:
    - Si el mensaje describe "croquetas", "alimento de perro", "veterinario", → Mascotas
    - Si describe comida general → Comida
    - Si describe gasolina, uber, taxi → Transporte
    - Si no encaja con ninguna → "Otros"

    FORMATO OBLIGATORIO:
    {{
        "categoria": "string"
    }}

    Mensaje: "{mensaje}"
    """

    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20
        )

        data = json.loads(res.choices[0].message.content.strip())

        categoria = data.get("categoria", "Otros")

        if categoria not in CATEGORIAS:
            return "Otros"

        return categoria

    except Exception as e:
        print("❌ Error categorización IA:", e)
        return "Otros"



# =====================================================
# 🔥 CLASIFICADOR PRINCIPAL
# =====================================================
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
    palabras_ingreso = [
        "recibi", "gane", "depositaron", "ingreso",
        "venta", "vendi", "pago", "me pagaron"
    ]
    tipo = "income" if any(p in msg_original for p in palabras_ingreso) else "expense"

    # === CATEGORÍA IA ===
    categoria = clasificar_categoria_IA(mensaje)

    return tipo, categoria, monto, mensaje.capitalize()



# =====================================================
# 🔥 FUNCIÓN PRINCIPAL
# =====================================================
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

    # envío al backend real
    try:
        headers = {"Authorization": token, "Content-Type": "application/json"}
        requests.post(BACKEND_URL, json=data, headers=headers, timeout=10)
    except Exception as e:
        print("⚠ Error enviando al backend:", e)

    return data



# =====================================================
# 🔥 ENDPOINT
# =====================================================
@app.post("/clasificar_gasto", response_model=ClasificacionRespuesta)
async def clasificar_endpoint(payload: MensajeUsuario, authorization: str = Header(...)):

    # Validar token
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Token inválido o ausente")

    # Validar groserías con IA
    if contiene_groserias_IA(payload.mensaje):
        raise HTTPException(
            400,
            "El mensaje contiene lenguaje ofensivo, groserías o contenido inapropiado."
        )

    return clasificar_gasto(payload.mensaje, authorization)



@app.get("/")
async def root():
    return {"status": "ok", "message": "IA Financiera con Groq funcionando 🚀"}
