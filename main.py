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

def normalizar_texto_para_filtro(texto: str) -> str:
    """
    Normaliza el texto para detectar variaciones ofensivas:
    - Elimina asteriscos, guiones, espacios entre letras
    - Reemplaza números por letras (l33t speak)
    - Elimina acentos
    """
    texto = texto.lower()
    texto = eliminar_acentos(texto)
    
    # Eliminar caracteres especiales comunes en censura
    texto = texto.replace('*', '').replace('-', '').replace('_', '')
    texto = texto.replace('@', 'a').replace('4', 'a').replace('3', 'e')
    texto = texto.replace('1', 'i').replace('0', 'o').replace('5', 's')
    
    # Eliminar espacios entre letras (ej: "p u t a" -> "puta")
    texto_sin_espacios = ''.join(texto.split())
    
    return texto_sin_espacios

# 🧩 Lista ampliada de palabras ofensivas (base)
PALABRAS_PROHIBIDAS_BASE = [
    # Insultos generales
    "pendejo", "pendeja", "idiota", "imbecil", "estupido", 
    "tonto", "tonta", "tarado", "baboso", "babosa", "bruto", "bruta",
    "payaso", "payasa", "ridiculo", "cretino", "menso",

    # Vulgaridades / groserías
    "puta", "puto", "chingar", "chingada", "chingado", "chingona", "chingon",
    "verga", "vrga", "mamada", "mamado", "cabron", "cabrona", "chinga",
    "chingate", "chingues", "chinguen", "chinguenasumadre", "chingatumadre",
    "pinche", "pinches", "culero", "culera", "mierda",

    # Términos sexuales o explícitos
    "sexo", "sexual", "porn", "porno", "pornografia", "pornografico", 
    "cojer", "coger", "cogio", "follar", "follando", "masturbar",
    "masturbacion", "orgasmo", "anal", "oral", "penetracion",
    "vagina", "pene", "vergon", "vergonaso", "chupar", "chupamela", "chupame",
    "tragala", "tragar", "culo", "trasero", "nalgas", "chichi", "chichis", "tetas",
    "boobs", "teta", "semen", "corrida", "correrse", "eyacular", "pito", "falo",

    # Ofensas sociales o discriminatorias
    "marica", "marico", "maricon", "marikon", "putazo", "travesti",
    "transexual", "negrata", "sidoso", "mongol", "retrasado",
    "naco", "zorra", "perra", "perro", "cerda", "cerdo", "prostituta", "prosti",
    "golfa", "ramera", "joto", "loca", "culiado",

    # Violencia o amenazas
    "matar", "asesinar", "disparar", "violacion", "violar", "degollar",
    "apunalar", "golpear", "torturar", "suicidio", "suicidarme", "matate", 
    "muerte", "matalo", "desangrar", "ahorcar", "colgarme",

    # Variantes comunes
    "hdp", "hijo de puta", "hija de puta", "perrazo",
    "hpt", "hpta", "qlo", "qlia",
    "maldito", "maldita", "basura", "lacra", "escoria",
    "enfermo", "asqueroso", "asquerosa", "repugnante", "inutil"
]


def contiene_lenguaje_ofensivo(texto: str) -> tuple[bool, str]:
    """
    Detecta si el texto contiene lenguaje ofensivo, incluso con censura.
    Retorna (es_ofensivo, palabra_detectada)
    """
    texto_normalizado = normalizar_texto_para_filtro(texto)
    
    # También normalizamos el texto original sin quitar espacios para detectar palabras completas
    texto_original_normalizado = eliminar_acentos(texto.lower())
    
    for palabra in PALABRAS_PROHIBIDAS_BASE:
        palabra_normalizada = normalizar_texto_para_filtro(palabra)
        
        # Detectar en texto sin espacios (para casos como "p*")
        if palabra_normalizada in texto_normalizado:
            print(f"🚫 [Filtro local] Mensaje bloqueado por contener: '{palabra}' (detectado como: '{palabra_normalizada}')")
            return True, palabra
        
        # Detectar en texto original (palabras completas)
        if palabra in texto_original_normalizado:
            print(f"🚫 [Filtro local] Mensaje bloqueado por contener: '{palabra}'")
            return True, palabra
        
        # Detectar variaciones con asteriscos entre letras
        patron_asterisco = ''.join([f"{letra}[_\\-\\s]" for letra in palabra])
        if re.search(patron_asterisco, texto_original_normalizado):
            print(f"🚫 [Filtro local] Mensaje bloqueado por variación censurada de: '{palabra}'")
            return True, palabra
    
    return False, ""


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
        print("⚠ Error al usar OpenAI Moderation:", e)
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

    # Validar con filtro mejorado
    es_ofensivo, palabra_detectada = contiene_lenguaje_ofensivo(payload.mensaje)
    
    if es_ofensivo or validar_mensaje_con_openai(payload.mensaje):
        print(f"🚨 Intento bloqueado ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}): {payload.mensaje}")
        raise HTTPException(
            status_code=400, 
            detail="El mensaje contiene lenguaje inapropiado. Por favor, usa un lenguaje respetuoso para registrar tus transacciones."
        )

    return clasificar_gasto(payload.mensaje, authorization)


@app.get("/")
async def root():
    return {"status": "ok", "message": "IA Financiera con filtro avanzado de lenguaje ofensivo"}