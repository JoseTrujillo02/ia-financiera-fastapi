from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import requests
import json
import os
from dotenv import load_dotenv

# ==============================
# CARGAR VARIABLES DE ENTORNO
# ==============================
load_dotenv()  # Carga .env automáticamente

# ==============================
# CONFIGURACIÓN DE OPENAI
# ==============================
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    client = None
    print("⚠️ No se pudo cargar OpenAI:", e)

# ==============================
# CONFIGURACIÓN DEL BACKEND
# ==============================
BACKEND_URL = os.getenv("BACKEND_URL", "https://api-firebase-auth.onrender.com/api/transactions")

# ==============================
# CONFIGURACIÓN DE FASTAPI
# ==============================
app = FastAPI(title="IA Financiera - Clasificador de Gastos")

# ==============================
# MODELOS DE DATOS
# ==============================
class MensajeUsuario(BaseModel):
    user_id: str
    mensaje: str

class ClasificacionRespuesta(BaseModel):
    user_id: str
    categoria: str
    monto: float
    descripcion: str
    fecha: str
    hora: str

# ==============================
# CLASIFICADOR LOCAL DE RESPALDO
# ==============================
def clasificador_local(mensaje: str):
    mensaje = mensaje.lower()
    categorias = {
        "alimentación": ["comida", "restaurante", "café", "taco", "hamburguesa"],
        "transporte": ["gasolina", "uber", "taxi", "camión", "pasaje", "auto"],
        "entretenimiento": ["cine", "película", "concierto", "juego", "netflix"],
        "salud": ["medicina", "doctor", "farmacia", "dentista"],
        "educación": ["libro", "colegiatura", "curso", "escuela"],
        "hogar": ["renta", "luz", "agua", "internet", "super"],
    }

    categoria = "Otros"
    for cat, palabras in categorias.items():
        if any(p in mensaje for p in palabras):
            categoria = cat
            break

    monto = 0
    for palabra in mensaje.split():
        if palabra.isdigit():
            monto = float(palabra)
        elif palabra.replace(".", "", 1).isdigit():
            monto = float(palabra)

    descripcion = mensaje.capitalize()
    return categoria, monto, descripcion


# ==============================
# FUNCIÓN PRINCIPAL
# ==============================
def clasificar_gasto(user_id: str, mensaje: str):
    ahora = datetime.now()

    if client:
        prompt = f"""
        Analiza este texto: "{mensaje}".
        Devuelve un JSON con: categoria, monto y descripcion.
        Categorías posibles: Alimentación, Transporte, Entretenimiento, Salud, Educación, Hogar, Otros.
        """

        try:
            respuesta = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            texto = respuesta.choices[0].message.content.strip()

            if "{" in texto:
                inicio = texto.find("{")
                fin = texto.rfind("}") + 1
                texto = texto[inicio:fin]

            data = json.loads(texto)
        except Exception as e:
            print(f"⚠️ Error con OpenAI ({e}). Usando clasificador local.")
            categoria, monto, descripcion = clasificador_local(mensaje)
            data = {"categoria": categoria, "monto": monto, "descripcion": descripcion}
    else:
        print("⚠️ OpenAI no está disponible. Usando clasificador local.")
        categoria, monto, descripcion = clasificador_local(mensaje)
        data = {"categoria": categoria, "monto": monto, "descripcion": descripcion}

    data["user_id"] = user_id
    data["fecha"] = ahora.strftime("%Y-%m-%d")
    data["hora"] = ahora.strftime("%H:%M:%S")

    try:
        response = requests.post(BACKEND_URL, json=data)
        if response.status_code in [200, 201]:
            print(f"✅ JSON enviado al backend ({response.status_code})")
        else:
            print(f"⚠️ Backend respondió error: {response.status_code} - {response.text}")
    except Exception as ex:
        print(f"❌ No se pudo conectar al backend: {ex}")

    return data


# ==============================
# ENDPOINT
# ==============================
@app.post("/clasificar_gasto", response_model=ClasificacionRespuesta)
async def clasificar_endpoint(payload: MensajeUsuario):
    resultado = clasificar_gasto(payload.user_id, payload.mensaje)
    return resultado
