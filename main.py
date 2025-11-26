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
    advertencia: str | None = None


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
    if not client:
        return False

    prompt = f"""
    Determina si este mensaje contiene groserías, vulgaridades, insultos,
    lenguaje ofensivo, contenido sexual explícito o palabras inapropiadas.

    Responde SOLO con JSON estricto:
    {{
        "ofensivo": true or false
    }}

    Mensaje: "{texto}"
    """

    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20
        )

        raw = res.choices[0].message.content.strip()
        print("🟥 Filtro ofensivo IA RAW:", raw)

        data = json.loads(raw)
        return data.get("ofensivo", False)

    except:
        return False


# =====================================================
# 🔥 DETECCIÓN DOBLE SENTIDO (SOLO ADVERTENCIA)
# =====================================================
def contiene_doble_sentido_IA(texto: str) -> bool:
    if not client:
        return False

    prompt = f"""
    Determina si el mensaje contiene doble sentido, frases con
    insinuación sexual indirecta, albures mexicanos o lenguaje ambiguo.

    Responde SOLO JSON:
    {{
        "doble_sentido": true or false
    }}

    Mensaje: "{texto}"
    """

    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20
        )

        raw = res.choices[0].message.content.strip()
        print("🟨 Doble sentido IA RAW:", raw)

        data = json.loads(raw)
        return data.get("doble_sentido", False)

    except:
        return False


# =====================================================
# 🔥 DETECCIÓN INGRESO/GASTO
# =====================================================
def clasificar_tipo_IA(mensaje: str) -> str:
    if not client:
        return "expense"

    prompt = f"""
    Determina si esta transacción es ingreso o gasto.

    Responde SOLO JSON:
    {{
        "type": "income" | "expense"
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
        tipo = data.get("type", "expense")

        print("🟩 Tipo IA:", tipo)

        return tipo

    except:
        return "expense"


# =====================================================
# 🔥 CREACIÓN DE CATEGORÍAS (LIBRE)
# =====================================================
def clasificar_categoria_IA(mensaje: str) -> str:
    """
    La IA crea UNA categoría nueva basada solo en el mensaje.
    Sin listas, sin ejemplos, sin categorías existentes.
    """

    if not client:
        return "SinCategoria"

    prompt = f"""
    Crea una categoría de UNA sola palabra que describa el gasto o ingreso.
    No uses listas existentes.
    No inventes frases largas.
    No uses "Otros".
    La categoría debe ser concreta y relacionada al mensaje.

    Responde SOLO JSON:
    {{
        "categoria": "UnaPalabra"
    }}

    Mensaje: "{mensaje}"
    """

    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=25
        )

        data = json.loads(res.choices[0].message.content.strip())
        categoria = data.get("categoria", "SinCategoria")

        categoria = categoria.replace(" ", "")

        print("🟦 Categoría IA:", categoria)

        return categoria

    except:
        return "SinCategoria"


# =====================================================
# 🔥 CLASIFICADOR LOCAL
# =====================================================
def clasificador_local(mensaje: str):

    match = re.search(r"\$?\s*([\d.,]+)", mensaje)
    if not match:
        raise HTTPException(400, "No se encontró monto")

    monto = float(match.group(1).replace(",", "").replace(".", ""))
    if monto <= 0:
        raise HTTPException(400, "Monto inválido")

    tipo = clasificar_tipo_IA(mensaje)
    categoria = clasificar_categoria_IA(mensaje)

    return tipo, categoria, monto, mensaje.capitalize()


# =====================================================
# 🔥 FUNCIÓN PRINCIPAL
# =====================================================
def clasificar_gasto(mensaje: str, token: str, advertencia=None):

    ahora = datetime.now()
    tipo, categoria, monto, descripcion = clasificador_local(mensaje)

    data = {
        "type": tipo,
        "amount": monto,
        "category": categoria,
        "descripcion": descripcion,
        "date": ahora.strftime("%Y-%m-%dT%H:%M:%S"),
        "advertencia": advertencia
    }

    print("📤 ENVIANDO:", data)

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

    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Token inválido")

    # ❶ SI HAY GROCERÍAS → BLOQUEA
    if contiene_groserias_IA(payload.mensaje):
        raise HTTPException(400, "El mensaje contiene lenguaje ofensivo")

    # ❷ SI HAY DOBLE SENTIDO → ADVIERTE PERO **SÍ GUARDA**
    advertencia = None
    if contiene_doble_sentido_IA(payload.mensaje):
        advertencia = "El mensaje contiene doble sentido. Por favor, exprésate con claridad."

    return clasificar_gasto(payload.mensaje, authorization, advertencia)


@app.get("/")
async def root():
    return {"status": "ok", "message": "IA Financiera con Groq funcionando 🚀"}
