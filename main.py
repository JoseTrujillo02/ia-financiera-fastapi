from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from datetime import datetime
import requests
import json
import os
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
    print("⚠️ No se pudo cargar OpenAI:", e)

# ==============================
# CONFIGURACIÓN DEL BACKEND
# ==============================
BACKEND_URL = os.getenv("BACKEND_URL")

# ==============================
# CONFIGURACIÓN DE FASTAPI
# ==============================
app = FastAPI(title="IA Financiera - Clasificador de Gastos")

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
# CLASIFICADOR LOCAL DE RESPALDO
# ==============================
def clasificador_local(mensaje: str):
    mensaje = mensaje.lower()
    categorias = {
        "Food": ["comida", "restaurante", "café", "taco", "hamburguesa"],
        "Transport": ["gasolina", "uber", "taxi", "camión", "pasaje", "auto"],
        "Entertainment": ["cine", "película", "concierto", "juego", "netflix"],
        "Health": ["medicina", "doctor", "farmacia", "dentista"],
        "Education": ["libro", "colegiatura", "curso", "escuela"],
        "Home": ["renta", "luz", "agua", "internet", "super"]
    }

    categoria = "Other"
    for cat, palabras in categorias.items():
        if any(p in mensaje for p in palabras):
            categoria = cat
            break

    monto = 0.0
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
def clasificar_gasto(mensaje: str, token: str):
    ahora = datetime.now()

    if client:
        prompt = f"""
        Analiza este texto: "{mensaje}".
        Devuelve un JSON con: category, amount y description.
        Las categorías posibles son: Food, Transport, Entertainment, Health, Education, Home, Other.
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
            data = {"category": categoria, "amount": monto, "descripcion": descripcion}
    else:
        print("⚠️ OpenAI no disponible. Usando clasificador local.")
        categoria, monto, descripcion = clasificador_local(mensaje)
        data = {"category": categoria, "amount": monto, "descripcion": descripcion}

    # Estructura que espera el backend Node
    data["type"] = "expense"
    data["date"] = ahora.strftime("%Y-%m-%dT%H:%M:%S")

    # Envío al backend
    headers = {
        "Authorization": token,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(BACKEND_URL, json=data, headers=headers)
        if response.status_code in [200, 201]:
            print(f"✅ JSON enviado al backend ({response.status_code})")
        else:
            print(f"⚠️ Error del backend: {response.status_code} - {response.text}")
    except Exception as ex:
        print(f"❌ No se pudo conectar al backend: {ex}")

    return data


# ==============================
# ENDPOINT
# ==============================
@app.post("/clasificar_gasto", response_model=ClasificacionRespuesta)
async def clasificar_endpoint(payload: MensajeUsuario, authorization: str = Header(...)):
    """
    Recibe el token Bearer por cabecera y el mensaje en el body.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token inválido o ausente en Authorization header")

    resultado = clasificar_gasto(payload.mensaje, authorization)
    return resultado
