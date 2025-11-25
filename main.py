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
# CONFIGURACIÓN DE GROQ (GRATIS)
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

# ==============================
# FASTAPI
# ==============================
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
# CLASIFICADOR CON GROQ
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

    # ===== 3. CLASIFICACIÓN GROQ =====
    categoria = "Otros"

    if client:
        prompt = f"""
        Clasifica este gasto EXACTAMENTE en una categoría:

        {", ".join(CATEGORIAS)}

        REGLAS IMPORTANTES:

        Mascotas →
        - comida de mi perro
        - croquetas
        - comida de mi gato
        - alimento para mascota
        - veterinario

        Comida →
        - tacos
        - dulces
        - pizza
        - hamburguesa

        Transporte →
        - gasolina
        - uber
        - camión
        - taxi

        NO inventes categorías nuevas.
        Si no estás seguro, responde "Otros".

        Mensaje: "{mensaje}"

        Responde solo con la categoría.
        """

        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10
            )

            respuesta = response.choices[0].message.content.strip()
            print("🟦 RESPUESTA GROQ:", respuesta)

            if respuesta in CATEGORIAS:
                categoria = respuesta
            else:
                categoria = "Otros"

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
        raise HTTPException(400, "El mensaje contiene lenguaje ofensivo.")

    return clasificar_gasto(payload.mensaje, authorization)

@app.get("/")
async def root():
    return {"status": "ok", "message": "IA Financiera con Groq funcionando 🚀"}
