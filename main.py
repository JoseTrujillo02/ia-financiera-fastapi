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
    Analiza el siguiente mensaje y determina si contiene lenguaje ofensivo,
    insultos, vulgaridades, groserías, modismos ofensivos o expresiones
    inapropiadas EN CUALQUIERA DE ESTAS FORMAS:

    🔥 Considera ofensivo si está:
    - escrito normal: "pendejo", "puta", "culero"
    - abreviado: "ptm", "vrg", "mdr"
    - censurado: "p***", "m****", "ching*d*"
    - con símbolos: "p#to", "mi3rd@", "v3rg@"
    - con números: "p3ndejo", "put0"
    - disfrazado: "pndjo", "vrg", "chng", "cbrn"
    - modismos ofensivos: "vale madre", "vale madres", "me vale madre",
      "valió madre", "me vale verga", "vale pito"

    EJEMPLOS DE EXPRESIONES OFENSIVAS:
    - "vale madre"
    - "vale madres"
    - "me vale madre"
    - "me vale madres"
    - "valió madre"
    - "valió madres"
    - "vale mdr"
    - "valio mdr"

    INSTRUCCIÓN:
    Responde ÚNICAMENTE en JSON ESTRICTO:
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
# 🔥 DETECCIÓN INTELIGENTE DE INGRESOS/GASTOS (IA)
# =====================================================
def clasificar_tipo_IA(mensaje: str) -> str:
    """
    Devuelve: "income" o "expense" usando IA
    """
    if not client:
        return "expense"

    prompt = f"""
    Determina si esta transacción es ingreso o gasto.
    Responde SOLO en JSON:
    {{
        "type": "income" | "expense"
    }}

    Si la persona dice:
    - me encontré
    - me llegó
    - me depositaron
    - gané
    - recibí
    - salario/sueldo
    → eso es ingreso.

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
# 🔥 CLASIFICACIÓN DE CATEGORÍA (SIN OTROS)
# =====================================================
CATEGORIAS = [
    "Comida", "Transporte", "Entretenimiento", "Salud", "Educacion",
    "Hogar", "Ropa", "Mascotas", "Trabajo", "Ingresos", "Finanzas",
    "Tecnologia", "Servicios personales"
]


def clasificar_categoria_IA(mensaje: str) -> str:
    """
    Selecciona una categoría o crea una nueva si la IA genera una.
    Ya NO usamos "Otros".
    """
    if not client:
        return "Sin categoría"

    prompt = f"""
    Determina la mejor categoría para este gasto o ingreso.

    Categorías actuales:
    {", ".join(CATEGORIAS)}

    Si la categoría no existe o no coincide, CREA UNA NUEVA categoría 
    (una palabra clara, legible y relacionada).

    Responde SOLO en JSON válido:
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

        # Si la categoría NO existe → CREARLA
        if categoria not in CATEGORIAS:
            CATEGORIAS.append(categoria)
            print("🆕 Nueva categoría creada:", categoria)

        return categoria

    except Exception as e:
        print("❌ Error categorización IA:", e)
        return "Sin categoria"



# =====================================================
# 🔥 CLASIFICADOR COMPLETO
# =====================================================
def clasificador_local(mensaje: str):

    # === MONTO ===
    match = re.search(r"\$?\s*([\d.,]+)", mensaje)
    if not match:
        raise HTTPException(400, "No se encontró monto")

    monto = float(match.group(1).replace(",", "").replace(".", ""))
    if monto <= 0:
        raise HTTPException(400, "Monto inválido")

    # === TIPO CON IA ===
    tipo = clasificar_tipo_IA(mensaje)

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

    return clasificar_gasto(payload.mensaje, authorization)



@app.get("/")
async def root():
    return {"status": "ok", "message": "IA Financiera con Groq funcionando 🚀"}
