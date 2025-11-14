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
    "Comida": [
        # Lugares y tipos
        "comida", "restaurante", "cocina", "fondita", "antojito", "antojitos",
        "taco", "tacos", "hamburguesa", "pizza", "pasta", "mariscos", "sushi",
        "pollo", "pescado", "carne", "barbacoa", "birria", "pozole", "enchilada",
        "torta", "tortas", "lonche", "tamales", "gorditas", "sopes", "quesadilla",
        "hotdog", "boneless", "alitas",

        # Bebidas
        "refresco", "coca", "pepsi", "bebida", "cafe", "té", "te", "agua",
        "jugo", "cerveza", "cheve", "vino", "licor",

        # Dulces / postres
        "pan", "panaderia", "pastel", "postre", "galleta", "chocolate", "dulces",
        "helado", "nieve", "donas",

        # Supermercados
        "super", "mercado", "abarrotes", "tienda", "bodega", "walmart", "chedraui",
        "soriana", "sam’s", "costco", "oxxo", "seven", "7eleven"
    ],

    "Transporte": [
        "uber", "didi", "cabify", "taxi", "transporte", "colectivo", "camion",
        "autobus", "bus", "metro", "metrobus", "suburbano", "tren", "peaje",
        "caseta", "estacionamiento", "parking",

        # Vehículos
        "gasolina", "diesel", "combustible", "carro", "auto", "vehiculo",
        "camioneta", "moto", "bicicleta", "uber moto",

        # Refacciones
        "llanta", "aceite", "bujia", "refaccion", "taller", "mecanico",
        "alineacion", "balanceo", "lavado de coche", "verificacion"
    ],

    "Entretenimiento": [
        "cine", "pelicula", "estreno", "concierto", "fiesta", "evento", "festival",
        "conferencia", "show", "torneo", "karaoke",

        # Streaming
        "netflix", "hbo", "max", "disney", "prime", "spotify", "youtube premium",
        "apple music", "paramount", "crunchyroll",

        # Videojuegos
        "juego", "videojuego", "steam", "epic games", "playstation", "xbox",
        "nintendo", "roblox", "fortnite", "fifa",

        # Parques y actividades
        "parque", "boliche", "billar", "bar", "antro", "discoteca"
    ],

    "Salud": [
        "doctor", "doctora", "consulta", "medico", "enfermera",
        "hospital", "clinica", "consultorio", "farmacia", "medicina",

        # Servicios
        "dentista", "odontologo", "ortodoncia", "limpieza dental",
        "terapia", "fisioterapia", "psicologo", "nutriologo", "gimnasio",
        "spa", "masaje",

        # Gastos médicos
        "analisis", "estudio", "examen", "laboratorio", "ultrasonido",
        "radiografia", "receta", "inyeccion", "vacuna",

        # Seguros
        "seguro medico", "seguro de gastos medicos", "aseguradora"
    ],

    "Educacion": [
        "escuela", "clase", "universidad", "prepa", "colegiatura", "pago escolar",
        "primaria", "secundaria", "kinder", "maestria", "doctorado",

        # Libros y materiales
        "libro", "cuaderno", "papeleria", "mochila", "lapiz",

        # Cursos
        "curso", "taller", "seminario", "capacitacion", "certificacion",
        "plataforma educativa", "udemy", "coursera", "platzi", "domestika",

        # Apps educativas
        "duolingo", "babbel"
    ],

    "Hogar": [
        "renta", "hipoteca", "agua", "luz", "cfe", "internet", "izzi", "telmex",
        "telefono", "gas", "propano", "hogar",

        # Artículos
        "lavadora", "refrigerador", "estufa", "microondas", "licuadora",
        "muebles", "sillon", "colchon", "cama", "cobija", "ropa de cama",
        "decoracion", "planta",

        # Reparaciones
        "plomeria", "electricidad", "pintura", "albanil", "herramienta",
        "reparacion", "carpinteria",

        # Tiendas
        "home depot", "lowes", "liverpool", "sears"
    ],

    "Ropa": [
        "ropa", "camisa", "playera", "pantalon", "jeans", "short", "vestido",
        "falda", "blusa", "sueter", "abrigo", "tenis", "zapato", "sandalia",
        "calcetin", "gorra", "ropa interior",

        # Accesorios
        "collar", "anillo", "reloj", "pulsera", "aretes", "lentes",

        # Marcas
        "nike", "adidas", "puma", "zara", "pull and bear", "bershka",
        "h&m", "aeropostale", "liverpool", "gucci"
    ],

    "Mascotas": [
        "perro", "gato", "mascota", "veterinario", "alimento", "croquetas",
        "juguete", "correa", "camita", "baño para perro", "corte",
    ],

    "Trabajo": [
        "oficina", "papeleria", "computadora", "laptop", "impresora", "teclado",
        "mouse", "monitor", "software", "licencia", "suscripcion", "herramienta",

        "material", "proyecto", "cliente", "servicio profesional"
    ],

    "Finanzas": [
        "impuesto", "iva", "isr", "multas", "recargo", "banco", "comision",
        "retiro", "deposito", "transferencia", "ahorro", "prestamo", "credito",
        "tarjeta", "intereses"
    ],

    "Ingresos": [
        "aguinaldo", "bono", "salario", "sueldo", "comision", "propina",
        "prestacion", "ingreso extra", "venta", "pago recibido"
    ],

    "Tecnologia": [
        "celular", "telefono", "iphone", "samsung", "xiaomi", "tablet",
        "computadora", "laptop", "monitor", "audifonos", "bocina",
        "cargador", "usb", "memoria", "tarjeta sd", "router"
    ],

    "Servicios personales": [
        "corte de cabello", "peluqueria", "barberia", "maquillaje", "uñas",
        "spa", "masaje", "depilacion", "cejas", "tinte"
    ],

    "Otros": [
        "donacion", "regalo", "servicio", "compra", "gasto", "imprevisto",
        "evento", "cita", "trámite"
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
