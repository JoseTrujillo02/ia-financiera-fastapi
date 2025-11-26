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
    if not client:
        return False

    prompt = f"""
    Analiza si el mensaje contiene groserías, vulgaridades, insultos,
    lenguaje ofensivo o expresiones inapropiadas.

    Responde SOLO en JSON:
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

    except Exception as e:
        print("❌ ERROR filtro IA:", e)
        return False


# =====================================================
# 🔥 DETECCIÓN DOBLE SENTIDO (IA)
# =====================================================
def contiene_doble_sentido_IA(texto: str) -> bool:
    if not client:
        return False

    prompt = f"""
    Analiza si el mensaje contiene doble sentido, insinuaciones sexuales
    indirectas o albures mexicanos.

    Responde SOLO en JSON:
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
        print("🟨 Filtro doble sentido IA RAW:", raw)

        data = json.loads(raw)
        return data.get("doble_sentido", False)

    except Exception as e:
        print("❌ ERROR doble sentido IA:", e)
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

        print("🟩 TIPO IA:", tipo)

        return tipo

    except:
        return "expense"


# =====================================================
# 🔥 CLASIFICACIÓN DE CATEGORÍAS
# =====================================================
CATEGORIAS = [
    "Comida", "Transporte", "Entretenimiento", "Salud", "Educacion",
    "Hogar", "Ropa", "Mascotas", "Trabajo", "Ingresos", "Finanzas",
    "Tecnologia", "Servicios personales"
]


def clasificar_categoria_IA(mensaje: str) -> str:
    if not client:
        return "Sin categoría"

    prompt = f"""
    Determina la mejor categoría.  
    Si no existe, CREA UNA NUEVA.

    Responde JSON:
    {{
        "categoria": "string"
    }}

    Mensaje: "{mensaje}"
    """

    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=20
        )

        data = json.loads(res.choices[0].message.content.strip())
        categoria = data.get("categoria", "Sin categoria")

        print("🟦 Categoría IA:", categoria)

        if categoria not in CATEGORIAS:
            CATEGORIAS.append(categoria)
            print("🆕 Nueva categoría creada:", categoria)

        return categoria

    except:
        return "Sin categoria"


# =====================================================
# 🔥 CLASIFICADOR LOCAL
# =====================================================
def clasificador_local(mensaje: str):

    # === MONTO ===
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
# 🔥 FUNCIÓN PRINCIPAL (FALTABA AQUÍ)
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

    if contiene_groserias_IA(payload.mensaje):
        raise HTTPException(400, "El mensaje contiene lenguaje ofensivo")

    if contiene_doble_sentido_IA(payload.mensaje):
        raise HTTPException(400, "El mensaje contiene doble sentido")

    return clasificar_gasto(payload.mensaje, authorization)


@app.get("/")
async def root():
    return {"status": "ok", "message": "IA Financiera con Groq funcionando 🚀"}
