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
app = FastAPI(title="IA Financiera - Clasificador (con filtro avanzado)")

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
# UTILIDADES
# ==============================
def eliminar_acentos(texto: str) -> str:
    """Normaliza texto eliminando tildes y diacríticos."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )

# 🧩 Lista ampliada de palabras ofensivas
PALABRAS_PROHIBIDAS = [
    # Insultos generales
    "pendejo", "pendeja", "idiota", "imbecil", "imbécil", "estupido", "estúpido", 
    "tonto", "tonta", "tarado", "baboso", "babosa", "bruto", "bruta",
    "payaso", "payasa", "ridiculo", "ridícula", "cretino", "menso", "mens@", "tard@",

    # Vulgaridades / groserías
    "puta", "puto", "put@", "chingar", "chingada", "chingado", "chingona", "chingon",
    "verga", "vrga", "v3rga", "mamada", "mamado", "cabrón", "cabron", "cabrona", "chinga",
    "chingate", "chingues", "chinguen", "chinguenasumadre", "chingatumadre", "ptm", "ptmr",
    "p1nche", "pinche", "pinches", "culero", "culera", "culer@", "mierda", "mrd", "mierd@",

    # Términos sexuales o explícitos
    "sexo", "sexual", "porn", "porno", "pornografia", "pornografía", "pornografico", 
    "pornográfico", "cojer", "coger", "cogio", "cogió", "follar", "follando", "masturbar",
    "masturbacion", "masturbación", "orgasmo", "anal", "oral", "penetracion", "penetración",
    "vagina", "pene", "vergon", "vergon@", "vergonaso", "chupar", "chupamela", "chupame",
    "tragala", "trágala", "tragar", "culo", "trasero", "nalgas", "chichi", "chichis", "tetas",
    "boobs", "teta", "semen", "corrida", "correrse", "eyacular", "pito", "falo", "verga", "vrg",

    # Ofensas sociales o discriminatorias
    "marica", "marico", "maricón", "marikon", "putazo", "gay", "lesbiana", "travesti",
    "transexual", "negro", "negra", "negrata", "sidoso", "mongol", "retrasado", "down",
    "naco", "indio", "zorra", "perra", "perro", "cerda", "cerdo", "prostituta", "prosti",
    "golfa", "ramera", "puta barata", "joto", "loca", "loc@", "culiado", "culia@", "idiot@",

    # Violencia o amenazas
    "matar", "asesinar", "disparar", "violacion", "violación", "violar", "degollar",
    "apuñalar", "golpear", "torturar", "suicidio", "suicidarme", "mátate", "matate", 
    "muerte", "mátalo", "matalo", "desangrar", "ahorcar", "colgarme", "pegartiro", "tiro",

    # Variantes disfrazadas
    "hdp", "hijo de puta", "hija de puta", "perrazo", "p3ndejo", "m1erda", "ching4", "vrg4",
    "chng", "pnch", "chngd", "hpt", "hpta", "qlo", "qlia", "p0rn", "put4", "put@", "put4s",
    "idi0ta", "imb3cil", "imbesil", "cabro", "maldito", "maldita", "basura", "lacra", "escoria",
    "enfermo", "asqueroso", "asquerosa", "repugnante", "mierdero", "inutil", "inútil"
]


def contiene_lenguaje_ofensivo(texto: str) -> bool:
    """Detecta si el texto contiene lenguaje ofensivo."""
    texto_normalizado = eliminar_acentos(texto.lower())
    for palabra in PALABRAS_PROHIBIDAS:
        if palabra in texto_normalizado:
            print(f"🚫 [Filtro local] Mensaje bloqueado por contener: '{palabra}'")
            return True
    return False


def validar_mensaje_con_openai(mensaje: str) -> bool:
    """Usa la API de moderación de OpenAI para detectar lenguaje inapropiado."""
    if not client:
        return False
    try:
        result = client.moderations.create(
            model="omni-moderation-latest",
            input=mensaje
        )
        flagged = result.results[0].flagged
        if flagged:
            print("🚫 [OpenAI Moderation] Mensaje bloqueado por contenido inapropiado.")
        return flagged
    except Exception as e:
        print("⚠️ Error al usar OpenAI Moderation:", e)
        return False


# ==============================
# CLASIFICADOR LOCAL
# ==============================
def clasificador_local(mensaje: str):
    mensaje_original = mensaje.strip()
    mensaje_sin_acentos = eliminar_acentos(mensaje_original.lower())

    # --- Detectar tipo ---
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

    # --- Categorías ---
    categorias = {
        "Comida": [
            "comida", "restaurante", "cafe", "taco", "hamburguesa", "almuerzo",
            "cena", "desayuno", "pan", "super", "mercado", "bebida",
            "antojito", "snack", "pizzas", "pollo", "carne", "postre", "helado"
        ],
        "Transporte": [
            "gasolina", "uber", "taxi", "camion", "metro", "pasaje", "auto",
            "vehiculo", "estacionamiento", "peaje", "transporte", "camioneta",
            "bicicleta", "bus", "carro", "moto", "combustible"
        ],
        "Entretenimiento": [
            "cine", "pelicula", "concierto", "juego", "netflix", "spotify",
            "evento", "teatro", "musica", "fiesta", "parque", "discoteca",
            "ocio", "videojuego", "deporte", "show"
        ],
        "Salud": [
            "medicina", "doctor", "farmacia", "dentista", "consulta", "terapia",
            "gimnasio", "hospital", "analisis", "examen", "vacuna", "cirugia",
            "psicologo", "nutriologo", "optica"
        ],
        "Educacion": [
            "libro", "colegiatura", "curso", "escuela", "educacion", "universidad",
            "clase", "seminario", "capacitacion", "maestria", "diplomado",
            "taller", "tutorial", "coaching"
        ],
        "Hogar": [
            "renta", "luz", "agua", "internet", "super", "casa", "hogar", "gas",
            "muebles", "reparacion", "plomeria", "decoracion", "limpieza",
            "electricidad", "servicio", "lavanderia"
        ],
        "Ropa": [
            "ropa", "camisa", "pantalon", "zapato", "tenis", "vestido", "abrigo",
            "accesorio", "sombrero", "reloj", "moda", "blusa", "jeans"
        ],
        "Mascotas": [
            "perro", "gato", "veterinario", "alimento para perro", "croquetas",
            "mascota", "juguete para gato", "adopcion"
        ],
        "Trabajo": [
            "oficina", "computadora", "herramienta", "software", "suscripcion",
            "material de trabajo", "impresora", "papeleria", "licencia",
            "servicio profesional", "cliente", "proyecto", "nomina", "salario"
        ],
        "Otros": [
            "donacion", "regalo", "impuesto", "banco", "seguro", "credito",
            "deuda", "efectivo", "retiro", "transferencia", "ahorro"
        ]
    }

    categoria = "Otros"
    for cat, palabras in categorias.items():
        if any(p in mensaje_sin_acentos for p in palabras):
            categoria = cat
            break

    # --- Extraer monto ---
    patrones = [
        r'\$\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)',
        r'(\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*pesos?',
        r'(\d+(?:,\d{3})*(?:\.\d{1,2})?)'
    ]

    monto = 0.0
    for patron in patrones:
        match = re.search(patron, mensaje)
        if match:
            try:
                monto_str = match.group(1).replace(',', '')
                monto = float(monto_str)
            except:
                monto = 0.0
            break

    return tipo, categoria, monto, mensaje_original.capitalize()


# ==============================
# FUNCIÓN PRINCIPAL
# ==============================
def clasificar_gasto(mensaje: str, token: str):
    ahora = datetime.now()
    tipo, categoria, monto, descripcion = clasificador_local(mensaje)
    data = {"type": tipo, "category": categoria, "amount": monto, "descripcion": descripcion, "date": ahora.strftime("%Y-%m-%dT%H:%M:%S")}

    headers = {"Authorization": token, "Content-Type": "application/json"}

    try:
        response = requests.post(BACKEND_URL, json=data, headers=headers, timeout=10)
        print(f"📥 Status: {response.status_code} | 📝 Body: {response.text}")
    except Exception as ex:
        print(f"❌ Error enviando al backend: {ex}")

    return data


# ==============================
# ENDPOINT PRINCIPAL
# ==============================
@app.post("/clasificar_gasto", response_model=ClasificacionRespuesta)
async def clasificar_endpoint(payload: MensajeUsuario, authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token inválido o ausente")

    if contiene_lenguaje_ofensivo(payload.mensaje) or validar_mensaje_con_openai(payload.mensaje):
        print(f"🚨 Intento bloqueado ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        raise HTTPException(status_code=400, detail="El mensaje contiene lenguaje inapropiado o no permitido.")

    return clasificar_gasto(payload.mensaje, authorization)


@app.get("/")
async def root():
    return {"status": "ok", "message": "IA Financiera con filtro avanzado de lenguaje ofensivo"}
