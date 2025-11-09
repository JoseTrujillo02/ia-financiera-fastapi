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
app = FastAPI(title="IA Financiera - Clasificador de Gastos e Ingresos (acentos tolerantes)")

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
# FUNCIONES AUXILIARES
# ==============================
def eliminar_acentos(texto: str) -> str:
    """Normaliza texto eliminando tildes y diacríticos."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


# ==============================
# CLASIFICADOR LOCAL DE RESPALDO
# ==============================
def clasificador_local(mensaje: str):
    # Texto original y normalizado
    mensaje_original = mensaje.strip()
    mensaje_sin_acentos = eliminar_acentos(mensaje_original.lower())

    # --- Detectar tipo (gasto o ingreso) ---
    palabras_ingreso = [
        "recibi", "me pagaron", "me depositaron", "gane", "ingreso", "ingresaron",
        "entrada", "vendi", "obtuve", "cobre", "me transfirieron", "me enviaron", "deposito"
    ]
    palabras_gasto = [
        "gaste", "pague", "compre", "inverti", "deposite", "saque", "transfiri",
        "done", "consumi", "pagado", "adquiri", "use", "realice un pago"
    ]

    tipo = "expense"
    if any(p in mensaje_sin_acentos for p in palabras_ingreso):
        tipo = "income"
    elif any(p in mensaje_sin_acentos for p in palabras_gasto):
        tipo = "expense"

    # --- Categorías ampliadas ---
    categorias = {
        "Comida": [
            "comida", "restaurante", "cafe", "taco", "hamburguesa", "almuerzo",
            "cena", "desayuno", "pan", "super", "mercado", "bebida", "antojito",
            "snack", "almuerzos", "lunch", "pizzas", "pollo", "carne", "postre"
        ],
        "Transporte": [
            "gasolina", "uber", "taxi", "camion", "metro", "pasaje", "auto",
            "vehiculo", "estacionamiento", "peaje", "transporte", "camioneta",
            "bicicleta", "bus", "carro", "moto", "combustible"
        ],
        "Entretenimiento": [
            "cine", "pelicula", "concierto", "juego", "netflix", "spotify",
            "evento", "teatro", "musica", "fiesta", "parque", "discoteca",
            "diversion", "ocio", "videojuego", "deporte", "show"
        ],
        "Salud": [
            "medicina", "doctor", "farmacia", "dentista", "consulta", "terapia",
            "gimnasio", "hospital", "analisis", "examen", "vacuna", "cirugia",
            "oftalmologo", "psicologo", "nutriologo", "optica"
        ],
        "Educacion": [
            "libro", "colegiatura", "curso", "escuela", "educacion", "universidad",
            "clase", "seminario", "capacitacion", "maestria", "diplomado",
            "taller", "tutorial", "coaching"
        ],
        "Hogar": [
            "renta", "luz", "agua", "internet", "super", "casa", "hogar", "gas",
            "muebles", "electrodomestico", "reparacion", "plomeria", "decoracion",
            "limpieza", "electricidad", "servicio", "lavanderia"
        ],
        "Ropa": [
            "ropa", "camisa", "pantalon", "zapato", "tenis", "vestido", "abrigo",
            "accesorio", "sombrero", "reloj", "moda", "blusa", "jeans", "calcetin"
        ],
        "Mascotas": [
            "perro", "gato", "veterinario", "alimento para perro", "croquetas",
            "mascota", "juguete para gato", "adopcion"
        ],
        "Trabajo": [
            "oficina", "computadora", "herramienta", "software", "suscripcion",
            "material de trabajo", "impresora", "papeleria", "licencia", "servicio profesional",
            "cliente", "proyecto", "nomina", "pago de salario"
        ],
        "Otros": [
            "donacion", "regalo", "impuesto", "banco", "seguro", "credito",
            "deuda", "otro", "efectivo", "retiro", "transferencia", "ahorro"
        ]
    }

    categoria = "Otros"
    for cat, palabras in categorias.items():
        if any(p in mensaje_sin_acentos for p in palabras):
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

    descripcion = mensaje_original.capitalize()
    return tipo, categoria, monto, descripcion


# ==============================
# FUNCIÓN PRINCIPAL
# ==============================
def clasificar_gasto(mensaje: str, token: str):
    ahora = datetime.now()

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

    data["date"] = ahora.strftime("%Y-%m-%dT%H:%M:%S")

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
        "message": "API de clasificación de gastos e ingresos (soporta acentos y sin acentos)",
        "backend_url": BACKEND_URL
    }
