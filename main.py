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
# FASTAPI
# ==============================
app = FastAPI(title="IA Financiera - Ajustada para Android")

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
    except:
        return False

# ==============================
# CLASIFICADOR LOCAL
# ==============================
def clasificador_local(mensaje: str):

    msg_original = mensaje.strip()
    msg = eliminar_acentos(msg_original.lower())

    palabras_ingreso = [
        "recibi","gane","me depositaron","ingreso","ingresaron","me transfirieron",
        "cobre","me pagaron","obtuve","premio","venta","vendi"
    ]
    palabras_gasto = [
        "gaste","pague","compre","inverti","saque","deposite",
        "consumi","use","donacion","pagado","gastado"
    ]

    tipo = "income" if any(p in msg for p in palabras_ingreso) else "expense"

    categorias = {
    # ============================================================
    # 🥘 COMIDA (MUY AMPLIADA)
    # ============================================================
    "Comida": [
        # Tipos de comida
        "comida", "restaurante", "antojito", "antojitos", "cena", "almuerzo",
        "desayuno", "snack", "botana", "dulces", "postre", "panaderia",
        "panadería", "pasteleria", "dulcería",

        # Comida Mexicana
        "taco", "tacos", "torta", "tamales", "pozole",
        "sopes", "gorditas", "enchiladas", "chilaquiles", "quesadilla", "burrito",
        "birria", "barbacoa", "menudo", "pambazo", "mollete", "huarache",

        # Comida rápida
"hamburguesa", "hamburguesas", "hamburguesita",
"hotdog", "hotdogs", "jocho", "jochos",
"papas", "papitas", "papas fritas", "papas a la francesa",
"boneless", "alitas", "wings",
"tenders", "nuggets", "fingers",
"pizza", "pizzas", "rebanada de pizza",
"cheetos", "doritos", "sabritas", "ruffles", "takis", "churrumais",
"tostitos", "pringles", "camaronazo", "cacahuates", "cacahuate enchilado",

# Sandwiches / tortas calientes / wraps
"torta", "tortas", "baguette", "sándwich", "sandwich",
"wrap", "wraps", "burrito", "burritos", "gringa", "quesadilla frita",

# Pollo frito
"pollo frito", "bucket de pollo",
"kfc", "kentucky", "churchs", "pollo loco",

# Hamburgueserías y combos
"combo", "combo grande", "combo mediano", "combo chico",
"mcdonalds", "burger king", "sonic", "carls jr", "wendys",
"baconator", "big mac", "whopper", "doble whopper", "cuarto de libra",

# Comida callejera
"tacos de canasta", "tacos dorados", "flautas", "huarache frito",
"tacos árabes", "tacos de suadero", "tacos de pastor", "tacos de tripa",
"tacos dorados",

# Locales de comida rápida México / LATAM
"little caesars", "dominos", "pizzahut", "papa johns",
"subway", "wingstop", "buffalo wild wings", "pollo feliz",
"vips", "toks", "el portón",

# Hot snacks y frituras
"nachos", "nachos con queso", "pretzel", "palomitas",
"churros", "banderillas", "corn dog", "piruleta",

# Maruchan / ramen rápido
"maruchan", "instant ramen", "sopa instantánea",

# Bebidas asociadas a comida rápida
"malteada", "milkshake", "refresco grande", "refresco mediano",
"frappe", "smoothie",

# Postres rápidos
"helado soft", "sundae", "mcdouble", "conito", "oreo frappe",

# Food trucks
"foodtruck", "food truck", "camioncito de comida"


        # Internacional
        "sushi", "ramen", "pasta", "lasagna", "curry", "wrap", "ensalada",
        "shawarma", "kebab", "pita", "falafel", "dumplings", "poke",

        # Carnes
        "pollo", "carne", "res", "cerdo", "pescado", "mariscos", "atun",
        "salmon", "shrimp", "camarones", "ceviche",

        # Bebidas
        "bebida", "refresco", "coca", "pepsi", "cola", "jugo", "agua",
        "té", "te", "cafe", "licuado", "malteada",

        # Alcohol
        "cerveza", "cheve", "vino", "whisky", "ron", "vodka", "tequila",
        "mezcal", "michelada",

        # Postres
        "pastel", "galleta", "donas", "helado", "nieve", "pay", "chocolate",

        # Supermercados
        "super", "supermercado", "mercado", "tienda", "bodega", "abarrotes",
        "walmart", "soriana", "chedraui", "sam’s", "costco", "oxxo",
        "seven", "7eleven", "farmacia guadalajara", "bodega aurrera",

        # Franquicias
        "mcdonalds", "burger king", "starbucks", "kfc", "dominos", "little caesars",
        "subway", "carls jr", "wingstop", "panda express", "toks"
    ],

    # ============================================================
    # 🚗 TRANSPORTE (MUY AMPLIADO)
    # ============================================================
    "Transporte": [
        # Apps
        "uber", "didi", "cabify", "inDriver", "beat",

        # Vehículos
        "carro", "vehiculo", "auto", "camioneta", "moto",
        "bicicleta", "e-bike", "scooter", "patin", "motocicleta",

        # Transportes públicos
        "camion", "autobus", "bus", "metro", "suburbano", "tren",
        "metrobus", "trolebús", "cablebus", "taxi", "combi", "colectivo",

        # Combustible
        "gasolina", "diesel", "combustible", "gasolinera", "pemex",
        "premium", "magna",

        # Refacciones
        "llanta", "llantas", "balatas", "aceite", "filtro", "bujia",
        "amortiguadores", "escape", "radiador", "faros",

        # Servicios mecánicos
        "taller", "mecanico", "alineacion", "balanceo", "verificacion",
        "lavado de coche", "detallado automotriz",

        # Pagos de transporte
        "peaje", "caseta", "estacionamiento", "parking", "boletos"
    ],

    # ============================================================
    # 🎮 ENTRETENIMIENTO (MEGA)
    # ============================================================
    "Entretenimiento": [
        # Eventos
        "cine", "pelicula", "boletos", "concierto", "festival", "evento",
        "torneo", "show", "teatro", "circo", "convención",

        # Streaming
        "netflix", "spotify", "hbo", "max", "disney", "apple music",
        "youtube premium", "amazon prime", "crunchyroll", "paramount",

        # Videojuegos
        "juego", "videojuego", "steam", "epic games", "riot", "league of legends",
        "valorant", "roblox", "fortnite", "minecraft", "playstation",
        "xbox", "nintendo switch",

        # Actividades
        "bar", "antro", "boliche", "billar", "karaoke", "casino", "escape room",

        # Deportes
        "gimnasio", "gym", "clase deportiva", "futbol", "basquetbol",
        "natacion", "box", "yoga", "zumba"
    ],

    # ============================================================
    # 🏥 SALUD (MEGA)
    # ============================================================
    "Salud": [
        # Profesionales
        "doctor", "doctora", "medico", "consulta", "farmacia",
        "dentista", "odontologo", "psicologo", "nutriologo",

        # Servicios
        "hospital", "clinica", "consultorio", "laboratorio",
        "ultrasonido", "radiografia", "tomografia", "resonancia",
        "inyeccion", "vacuna", "suturas",

        # Medicamentos
        "medicina", "analgesico", "antibiotico", "antigripal",
        "vitaminas", "suero", "jarabe",

        # Cuidado personal
        "spa", "masaje", "terapia", "fisioterapia",

        # Seguros
        "seguro medico", "gastos medicos", "aseguradora"
    ],

    # ============================================================
    # 📚 EDUCACIÓN (MUY AMPLIADA)
    # ============================================================
    "Educacion": [
        # Niveles
        "escuela", "primaria", "secundaria", "prepa", "universidad",
        "maestria", "doctorado", "colegiatura",

        # Material
        "libro", "libros", "cuadernos", "papeleria", "mochila",
        "pluma", "lapiz", "marcadores",

        # Plataformas
        "udemy", "platzi", "domestika", "coursera", "khan academy",
        "crehana", "skillshare",

        # Cursos
        "curso", "taller", "clase", "certificacion", "examen",
        "seminario", "capacitacion"
    ],

    # ============================================================
    # 🏠 HOGAR (MUY COMPLETA)
    # ============================================================
    "Hogar": [
        # Servicios
        "renta", "hipoteca", "agua", "luz", "cfe", "internet", "telmex",
        "izzi", "totalplay", "gas", "propano",

        # Artículos del hogar
        "muebles", "cama", "colchon", "sillon", "mesa", "silla",
        "electrodomestico", "refrigerador", "estufa", "lavadora",
        "microondas", "licuadora", "ventilador", "aire acondicionado",

        # Limpieza
        "limpieza", "detergente", "cloro", "escoba", "trapeador",
        "jabón", "desinfectante",

        # Reparaciones
        "plomeria", "electricidad", "pintura", "carpinteria",
        "albanil", "herramienta", "reparacion",

        # Tiendas
        "home depot", "lowes", "liverpool", "sears", "bodega"
    ],

    # ============================================================
    # 👕 ROPA Y MODA
    # ============================================================
    "Ropa": [
        "ropa", "playera", "camisa", "pantalon", "jeans", "shorts",
        "tenis", "zapatos", "sandalias", "vestido", "blusa",
        "accesorio", "gorra", "reloj", "pulsera", "collar", "aretes",

        # Marcas
        "nike", "adidas", "puma", "zara", "bershka", "h&m",
        "aeropostale", "pull and bear", "shein", "gucci"
    ],

    # ============================================================
    # 🐶 MASCOTAS
    # ============================================================
    "Mascotas": [
        "perro", "gato", "mascota", "veterinario", "croquetas",
        "alimento", "juguete", "correa", "baño", "corte",
        "camita", "jaula", "arena para gato"
    ],

    # ============================================================
    # 💼 TRABAJO
    # ============================================================
    "Trabajo": [
        "oficina", "papeleria", "computadora", "laptop", "monitor",
        "impresora", "teclado", "mouse", "software", "licencia",
        "suscripcion", "herramienta", "proyecto", "cliente"
    ],

    # ============================================================
    # 💰 INGRESOS
    # ============================================================
    "Ingresos": [
        "aguinaldo", "bono", "salario", "sueldo", "quincena",
        "deposito recibido", "comision", "propina", "venta",
        "ganancia", "ingreso extra", "premio"
    ],

    # ============================================================
    # 💳 FINANZAS / TRÁMITES
    # ============================================================
    "Finanzas": [
        "impuesto", "iva", "isr", "multas", "comision", "banco",
        "prestamo", "credito", "tarjeta", "intereses", "seguro",
        "transferencia", "retiro", "deposito"
    ],

    # ============================================================
    # 📱 TECNOLOGÍA
    # ============================================================
    "Tecnologia": [
        "celular", "telefono", "iphone", "samsung", "xiaomi",
        "tablet", "laptop", "computadora", "audifonos",
        "bocina", "cargador", "usb", "memoria", "router"
    ],

    # ============================================================
    # 🧍 SERVICIOS PERSONALES
    # ============================================================
    "Servicios personales": [
        "corte de cabello", "barbería", "peluqueria", "uñas",
        "spa", "masaje", "depilacion", "cejas", "tinte",
        "maquillaje", "peinado"
    ],

    # ============================================================
    # 🔄 OTROS
    # ============================================================
    "Otros": [
        "donacion", "regalo", "servicio", "pago", "gasto",
        "imprevisto", "tramite", "cita", "compras"
    ]
}



    # ===== CONTADOR DE PALABRAS =====
    coincidencias = {}
    for cat, palabras in categorias.items():
        coincidencias[cat] = sum(1 for p in palabras if re.search(rf"\b{p}\b", msg))

    categoria = max(coincidencias, key=coincidencias.get)

    # ⚠ Si no coincidió nada → error que Android sabe manejar
    if coincidencias[categoria] == 0:
        raise HTTPException(
            status_code=400,
            detail="No se pudo identificar la categoría. Menciona en qué gastaste o de dónde proviene el ingreso."
        )

    # ===== EXTRACCIÓN DE MONTO ROBUSTA =====
    match = re.search(r"\$?\s*([\d.,]+)", mensaje)
    if not match:
        raise HTTPException(
            status_code=400,
            detail="No se pudo identificar un monto válido. Por favor, menciona claramente la cantidad."
        )

    monto_str = match.group(1).replace(",", "").replace(".", "")
    monto = float(monto_str)

    if monto <= 0:
        raise HTTPException(
            status_code=400,
            detail="No se pudo identificar un monto válido. Por favor, menciona claramente la cantidad."
        )

    return tipo, categoria, monto, msg_original.capitalize()

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

    # Enviar al backend real
    headers = {"Authorization": token, "Content-Type": "application/json"}
    try:
        requests.post(BACKEND_URL, json=data, headers=headers, timeout=10)
    except:
        pass

    return data

# ==============================
# ENDPOINT
# ==============================
@app.post("/clasificar_gasto", response_model=ClasificacionRespuesta)
async def clasificar_endpoint(payload: MensajeUsuario, authorization: str = Header(...)):

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token inválido o ausente")

    if contiene_lenguaje_ofensivo(payload.mensaje):
        raise HTTPException(status_code=400, detail="El mensaje contiene lenguaje ofensivo o no permitido.")

    if validar_mensaje_con_openai(payload.mensaje):
        raise HTTPException(status_code=400, detail="El mensaje contiene contenido inapropiado (Moderation).")

    return clasificar_gasto(payload.mensaje, authorization)

@app.get("/")
async def root():
    return {"status": "ok", "message": "IA lista y sincronizada con Android 🚀"}
