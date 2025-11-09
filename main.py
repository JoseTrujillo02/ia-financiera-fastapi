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
app = FastAPI(title="IA Financiera - Clasificador de Gastos e Ingresos")

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

    # --- Detectar tipo (gasto o ingreso) ---
    palabras_ingreso = [
        "recibí", "me pagaron", "me depositaron", "gané", "ingresó", "ingreso", "entrada", "vendí", "obtuve", "cobré"
    ]
    palabras_gasto = [
        "gasté", "pagué", "compré", "invertí", "deposité", "saqué", "transferí", "doné", "consumí", "pagado", "adquirí"
    ]

    tipo = "expense"
    if any(p in mensaje_lower for p in palabras_ingreso):
        tipo = "income"
    elif any(p in mensaje_lower for p in palabras_gasto):
        tipo = "expense"

    # --- Categorías ampliadas ---
    categorias = {
        "Comida": [
            "comida", "restaurante", "café", "taco", "hamburguesa", "almuerzo", "cena", "desayuno",
            "pan", "super", "mercado", "bebida", "antojito", "snack", "almuerzos", "lunch"
        ],
        "Transporte": [
            "gasolina", "uber", "taxi", "camión", "metro", "pasaje", "auto", "vehículo",
            "estacionamiento", "peaje", "transporte", "camioneta", "bicicleta", "bus", "carro"
        ],
        "Entretenimiento": [
            "cine", "película", "concierto", "juego", "netflix", "spotify", "evento",
            "teatro", "música", "fiesta", "parque", "discoteca", "diversión", "ocio"
        ],
        "Salud": [
            "medicina", "doctor", "farmacia", "dentista", "consulta", "terapia", "gimnasio",
            "hospital", "análisis", "examen", "vacuna", "cirugía", "oftalmólogo"
        ],
        "Educacion": [
            "libro", "colegiatura", "curso", "escuela", "educación", "universidad", "clase",
            "seminario", "capacitacion", "maestría", "diplomado", "taller", "tutorial"
        ],
        "Hogar": [
            "renta", "luz", "agua", "internet", "super", "casa", "hogar", "gas", "muebles",
            "electrodoméstico", "reparación", "plomería", "decoración", "limpieza", "electricidad"
        ],
        "Ropa": [
            "ropa", "camisa", "pantalón", "zapato", "tenis", "vestido", "abrigo", "accesorio",
            "sombrero", "reloj", "moda", "blusa", "jeans"
        ],
        "Mascotas": [
            "perro", "gato", "veterinario", "alimento para perro", "croquetas", "mascota", "juguete para gato"
        ],
        "Trabajo": [
            "oficina", "computadora", "herramienta", "software", "suscripción", "material de trabajo",
            "impresora", "papelería", "licencia", "servicio profesional"
        ],
        "Otros": [
            "donación", "regalo", "impuesto", "banco", "seguro", "crédito", "deuda", "otro"
        ]
    }

    categoria = "Otros"
    for cat, palabras in categorias.items():
        if any(p in mensaje_lower for p in palabras):
            categoria = cat
            break

    # --- Extraer monto con regex mejorado ---
    patrones = [
        r'\$\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
        r'(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*pesos?',
        r'(\d+(?:,\d{3})*(?:\.\d{1,2})?)'
    ]

    monto = 0.0
    for patron in patrones:
        match = re.search(patron, mensaje)
        if match:
            monto_str = match.group(1).replace(',', '')
            try:
                monto = float(monto_str)
            except:
                monto = 0.0
            break

    descripcion = mensaje.capitalize()
    return tipo, categoria, monto, descripcion


# ==============================
# FUNCIÓN PRINCIPAL
# ==============================
def clasificar_gasto(mensaje: str, token: str):
    ahora = datetime.now()

    # --- Si hay OpenAI disponible, usarla ---
    if client:
        prompt = f"""
        Analiza este texto financiero: "{mensaje}".
        Devuelve un JSON con los campos:
        "type" (expense o income), "category", "amount" y "descripcion".
        Categorías posibles: Comida, Transporte, Entretenimiento, Salud, Educación, Hogar, Ropa, Mascotas, Trabajo, Otros.
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
            tipo, categoria, monto, descripcion = clasificador_local(mensaje)
            data = {"type": tipo, "category": categoria, "amount": monto, "descripcion": descripcion}
    else:
        print("⚠ OpenAI no disponible. Usando clasificador local.")
        tipo, categoria, monto, descripcion = clasificador_local(mensaje)
        data = {"type": tipo, "category": categoria, "amount": monto, "descripcion": descripcion}

    # --- Agregar fecha ---
    data["date"] = ahora.strftime("%Y-%m-%dT%H:%M:%S")

    # --- Envío al backend ---
    headers = {
        "Authorization": token,
        "Content-Type": "application/json"
    }

    print("="*60)
    print(f"📱 Mensaje recibido: {mensaje}")
    print(f"🔑 Token recibido: {token[:50]}...")
    print(f"📤 JSON a enviar al backend:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"🌐 URL del backend: {BACKEND_URL}")
    print("="*60)

    try:
        response = requests.post(BACKEND_URL, json=data, headers=headers, timeout=10)
        print(f"📥 Status Code: {response.status_code}")
        print(f"📝 Body: {response.text}")
    except Exception as ex:
        print(f"❌ Error enviando al backend: {ex}")

    return data


# ==============================
# ENDPOINT
# ==============================
@app.post("/clasificar_gasto", response_model=ClasificacionRespuesta)
async def clasificar_endpoint(payload: MensajeUsuario, authorization: str = Header(...)):
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
        "message": "API de clasificación de gastos e ingresos funcionando",
        "backend_url": BACKEND_URL
    }
