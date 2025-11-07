from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from datetime import datetime
import requests
import json
import os
import re
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
    mensaje_lower = mensaje.lower()
    
    categorias = {
        "Food": ["comida", "restaurante", "café", "taco", "hamburguesa", "comer", "almuerzo", "cena", "desayuno", "alimento"],
        "Transport": ["gasolina", "uber", "taxi", "camión", "pasaje", "auto", "transporte"],
        "Entertainment": ["cine", "película", "concierto", "juego", "netflix", "entretenimiento"],
        "Health": ["medicina", "doctor", "farmacia", "dentista", "salud"],
        "Education": ["libro", "colegiatura", "curso", "escuela", "educación"],
        "Home": ["renta", "luz", "agua", "internet", "super", "casa", "hogar"]
    }

    categoria = "Other"
    for cat, palabras in categorias.items():
        if any(p in mensaje_lower for p in palabras):
            categoria = cat
            break

    # Extraer monto con regex mejorado
    patrones = [
        r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*pesos',
        r'(\d+(?:,\d{3})*(?:\.\d{2})?)',
    ]
    
    monto = 0.0
    for patron in patrones:
        match = re.search(patron, mensaje)
        if match:
            monto_str = match.group(1).replace(',', '')
            monto = float(monto_str)
            break

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
        Extrae el monto numérico (sin símbolos), la categoría y una descripción.
        Devuelve SOLO un JSON válido con: "category", "amount", "descripcion".
        
        Categorías posibles: Food, Transport, Entertainment, Health, Education, Home, Other.
        
        Ejemplo: {{"category": "Food", "amount": 30.0, "descripcion": "Gasté en comida"}}
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
            print(f"⚠ Error con OpenAI ({e}). Usando clasificador local.")
            categoria, monto, descripcion = clasificador_local(mensaje)
            data = {"category": categoria, "amount": monto, "descripcion": descripcion}
    else:
        print("⚠ OpenAI no disponible. Usando clasificador local.")
        categoria, monto, descripcion = clasificador_local(mensaje)
        data = {"category": categoria, "amount": monto, "descripcion": descripcion}

    # Estructura que espera el backend Node
    data["type"] = "expense"
    data["date"] = ahora.strftime("%Y-%m-%dT%H:%M:%S")

    # 🔍 LOGS DETALLADOS
    print("="*60)
    print(f"📱 Mensaje recibido: {mensaje}")
    print(f"🔑 Token recibido: {token[:50]}...")
    print(f"📤 JSON a enviar al backend:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"🌐 URL del backend: {BACKEND_URL}")
    print("="*60)

    # Envío al backend
    headers = {
        "Authorization": token,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(BACKEND_URL, json=data, headers=headers, timeout=10)
        
        print(f"📥 Status Code: {response.status_code}")
        print(f"📄 Response Headers: {dict(response.headers)}")
        print(f"📝 Response Body: {response.text}")
        print("="*60)
        
        if response.status_code in [200, 201]:
            print(f"✅ JSON enviado al backend ({response.status_code})")
            # Intentar parsear la respuesta del backend
            try:
                backend_data = response.json()
                print(f"✅ Backend respondió con: {json.dumps(backend_data, indent=2)}")
            except:
                print(f"⚠ Backend no devolvió JSON válido")
        else:
            print(f"❌ Error del backend: {response.status_code}")
            print(f"❌ Respuesta: {response.text}")
            
    except requests.Timeout:
        print(f"⏱ Timeout al conectar con el backend")
    except requests.ConnectionError as ex:
        print(f"❌ Error de conexión con el backend: {ex}")
    except Exception as ex:
        print(f"❌ Error inesperado: {ex}")

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


# ==============================
# ENDPOINT DE PRUEBA
# ==============================
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "API de clasificación de gastos funcionando",
        "backend_url": BACKEND_URL
    }