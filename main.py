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
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")

    if OPENAI_KEY:
        client = OpenAI(api_key=OPENAI_KEY)
        print("✅ OpenAI cargado correctamente desde variable de entorno")
    else:
        print("❌ ERROR: OPENAI_API_KEY no existe como variable de entorno.")
        client = None

except Exception as e:
    print("❌ No se pudo inicializar OpenAI:", e)
    client = None


# ==============================
# CONFIGURACIÓN DEL BACKEND
# ==============================
BACKEND_URL = os.getenv("BACKEND_URL")

# ==============================
# FASTAPI
# ==============================
app = FastAPI(title="IA Financiera - Clasificación Inteligente")

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
    except Exception as e:
        print("⚠ Moderation error:", e)
        return False

# ==============================
# CATEGORÍAS FIJAS
# ==============================
CATEGORIAS = [
    "Comida",
    "Transporte",
    "Entretenimiento",
    "Salud",
    "Educacion",
    "Hogar",
    "Ropa",
    "Mascotas",
    "Trabajo",
    "Ingresos",
    "Finanzas",
    "Tecnologia",
    "Servicios personales",
    "Otros"
]

# ==============================
# CLASIFICADOR (CORREGIDO + LOGS)
# ==============================
def clasificador_local(mensaje: str):

    msg_original = mensaje.strip().lower()

    # ===== 1. MONTO =====
    match = re.search(r"\$?\s*([\d.,]+)", mensaje)
    if not match:
        raise HTTPException(400, "No se pudo identificar un monto válido")

    monto_str = match.group(1).replace(",", "").replace(".", "")
    monto = float(monto_str)
    if monto <= 0:
        raise HTTPException(400, "Monto inválido")

    # ===== 2. TIPO =====
    palabras_ingreso = ["recibi","gane","depositaron","ingreso","venta","vendi","pago","me pagaron"]
    tipo = "income" if any(p in msg_original for p in palabras_ingreso) else "expense"

    # ===== 3. CLASIFICACIÓN OPENAI =====
    categoria = "Otros"

    if client:

        prompt = f"""
        Clasifica este gasto EXACTAMENTE en una categoría:

        {", ".join(CATEGORIAS)}

        REGLAS:
        - “tacos”, “dulces”, “hamburguesa”, “pizza”, etc. → Comida
        - “gasolina”, “camion”, “uber”, etc. → Transporte
        - “cine”, “netflix”, “concierto” → Entretenimiento
        - “doctor”, “dentista”, “farmacia” → Salud
        - “comida de mi perro”, “croquetas”, “alimento para gato”, “veterinario” → Mascotas
        - No inventes categorías nuevas.
        - Si no estás seguro, responde “Otros”.

        Mensaje: "{mensaje}"

        Responde solo con la categoría, sin texto adicional.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # si falla usa gpt-4o
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
            )

            respuesta = response.choices[0].message.content.strip()
            print("🟦 RESPUESTA DE OPENAI:", respuesta)

            if respuesta in CATEGORIAS:
                categoria = respuesta
            else:
                categoria = "Otros"

        except Exception as e:
            print("❌ ERROR LLAMANDO A OPENAI:", e)
            categoria = "Otros"

    else:
        print("⚠ No hay cliente OpenAI — clasificando como Otros")
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

    # Enviar al backend
    headers = {"Authorization": token, "Content-Type": "application/json"}
    try:
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

    if contiene_lenguaje_ofensivo(payload.mensaje):
        raise HTTPException(400, "El mensaje contiene lenguaje ofensivo o no permitido.")

    if validar_mensaje_con_openai(payload.mensaje):
        raise HTTPException(400, "El mensaje contiene contenido inapropiado (Moderation).")

    return clasificar_gasto(payload.mensaje, authorization)

@app.get("/")
async def root():
    return {"status": "ok", "message": "IA financiera funcionando 🚀"}
