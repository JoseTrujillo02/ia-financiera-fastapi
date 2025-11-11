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
app = FastAPI(title="IA Financiera - Filtro Extendido Antilenguaje Ofensivo")

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
# NORMALIZACIÓN DE TEXTO
# ==============================
def eliminar_acentos(texto: str) -> str:
    """Elimina acentos y diacríticos."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )

def normalizar_texto(texto: str) -> str:
    """Elimina símbolos, acentos y reemplaza caracteres similares."""
    texto = eliminar_acentos(texto.lower())
    reemplazos = {
        "@": "a", "$": "s", "1": "i", "!": "i", "3": "e", "0": "o",
        "4": "a", "7": "t", "5": "s", "8": "b", "*": "", "-": "", ".": ""
    }
    for simb, letra in reemplazos.items():
        texto = texto.replace(simb, letra)
    texto = re.sub(r'[\W_]+', '', texto)
    return texto


# ==============================
# FILTRO DE PALABRAS OFENSIVAS
# ==============================

# 🔥 Súper lista de groserías, insultos, sexuales, discriminatorios, disfrazados
PATRONES_OFENSIVOS = [
    # Vulgaridades comunes
    r"p+u+t+[ao@]+", r"p+u+t+a+s+", r"p+u+t+@", r"p+1nch+e+", r"pinch[e3]+",
    r"ch+i+n+g+[ao]+", r"ch+1ng+[ao]+", r"v+e+r+g+[a4]+", r"vrg[a4]+", r"m+e+r+d+a+",
    r"m+i+e+r+d+[ao]+", r"p+e+n+d+e+j+[ao]+", r"p3nd[e3]j[oa]", r"p+e+n+d+[io]jo+s*",
    r"c+u+l+[ao]+", r"c+u+1+[ao]+", r"k+u+l+[ao]+", r"m+a+m+a+d+[ao]+", r"m+a+m+[ao]+n+",
    r"c+a+b+r+o+n+", r"h+d+p", r"h+ijo+d+[ep]+u+t+[ao]+", r"h+dp", r"hdp+", r"hijoputa",
    r"m+a+r+i+c+[ao]+", r"i+d+i+o+t+[ao]+", r"i+m+b+e+c+i+l+", r"t+o+n+t+[ao]+",
    r"t+ar+a+d+[ao]+", r"r+e+t+r+a+s+[ao]+", r"b+a+b+[ao]+s+[ao]+", r"m+e+n+s+[ao]+",
    r"g+u+e+y+", r"wey", r"pinshi+", r"pnch[e3]+", r"p1nchi+", r"put[oi]n+",
    r"p+t+m+", r"ptmr", r"chngd", r"chng4", r"chingatumadre", r"chinguesumadre",

    # Sexuales explícitas o sugerentes
    r"c+o+j+[ei]+r+", r"f+o+l+l+a+r+", r"m+a+s+t+u+r+b+[ao]+", r"p+o+r+n+", r"s+e+x+[ao]+",
    r"s+e+g+s+", r"v+a+g+[i1]+n+[ao]+", r"p+i+t+[ao]+", r"v+e+r+g+o+n+", r"v+e+r+g+u+d+o+",
    r"n+a+l+g+[ao]+", r"t+e+t+[ao]+", r"ch+i+c+h+i+", r"b+o+o+b+s+", r"p+e+c+h+[oa]+",
    r"o+r+a+l+", r"a+n+a+l+", r"t+r+a+g+a+", r"c+u+l+[oa]+", r"p+e+r+r+[ao]+", r"z+o+r+r+[ao]+",
    r"m+a+m+[ao]+n+", r"ch+u+p+a+[rm]+", r"c+h+i+c+h+[ao]+", r"f+a+l+[oa]+", r"e+y+a+c+u+l+a+",
    r"corrida", r"f+u+c+k+", r"s+u+c+k+", r"69", r"p+o+l+l+[ao]+",

    # Discriminación / odio
    r"n+e+g+r+[ao]+", r"p+u+t+[o@]+", r"m+a+r+i+k+[ao]+", r"m+a+r+i+c+[ao]+", r"l+e+s+b+i+a+n+",
    r"t+r+a+v+[ei]+s+[ti]+", r"t+r+a+n+s+", r"s+i+d+o+s+[ao]+", r"m+o+n+g+o+l+", r"d+o+w+n+",
    r"n+a+c+[ao]+", r"i+n+d+i+[ao]+", r"z+o+r+r+[ao]+", r"p+r+o+s+t+i+t+[uo]+", r"p+u+t+[ao]+",
    r"p+e+r+r+[ao]+", r"m+o+r+e+n+[ao]+", r"g+o+r+d+[ao]+", r"f+e+[oa]+", r"h+o+m+o+f+o+b+[oa]+",
    r"r+a+c+i+s+t+[ao]+",

    # Violencia / amenazas
    r"m+a+t+[ao]+", r"a+s+e+s+i+n+[ao]+", r"v+i+o+l+a+[ao]+", r"d+e+g+o+l+[ao]+", r"a+h+o+r+c+[ao]+",
    r"m+u+e+r+[te]+", r"s+u+i+c+i+d+[ao]+", r"s+u+i+c+i+d+a+r+", r"a+t+a+c+a+r+", r"t+i+r+[oa]+",
    r"d+i+s+p+a+r+[oa]+", r"b+a+l+[ao]+", r"p+e+g+a+r+", r"g+o+l+p+[ea]+r+", r"t+o+r+t+u+r+[ao]+",

    # Variantes disfrazadas
    r"p3nd3j[o0]+", r"m13rd[a4]+", r"vrg[a4]+", r"ch1ng[ao]+", r"p1nch[e3]+", r"put[a@]+",
    r"f0llar", r"s3x0", r"s3xo", r"p0rn", r"hpt+", r"qlo", r"qlia", r"vrg4", r"vrgon",
    r"ptm+", r"hpta", r"marik", r"imb3cil", r"idi0ta", r"malparid", r"mierd@", r"loc@", r"mar1ca",
    r"vergon", r"c4bron", r"put@", r"pendej@", r"culer@", r"pndj", r"pr0st", r"suicid@", r"violaci[ao]+n"
]

def contiene_lenguaje_ofensivo(texto: str) -> bool:
    """Detecta lenguaje ofensivo aunque esté disfrazado con símbolos."""
    texto_normalizado = normalizar_texto(texto)
    for patron in PATRONES_OFENSIVOS:
        if re.search(patron, texto_normalizado):
            print(f"🚫 [Filtro] Bloqueado por patrón: {patron}")
            return True
    return False


def validar_mensaje_con_openai(mensaje: str) -> bool:
    """Valida con IA de OpenAI si hay lenguaje inapropiado."""
    if not client:
        return False
    try:
        result = client.moderations.create(
            model="omni-moderation-latest",
            input=mensaje
        )
        flagged = result.results[0].flagged
        if flagged:
            print("🚫 [OpenAI Moderation] Contenido inapropiado detectado.")
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

    # Detectar tipo (ingreso o gasto)
    palabras_ingreso = [
        "recibi", "me pagaron", "depositaron", "gane", "ingreso", "entrada",
        "vendi", "obtuve", "cobre", "me transfirieron", "me enviaron", "deposito", "venta", "salario"
    ]
    palabras_gasto = [
        "gaste", "pague", "compre", "inverti", "saque", "transfiri", "done", "consumi",
        "adquiri", "pago", "gastado", "comprado", "realice un pago", "use"
    ]

    tipo = "expense"
    if any(p in mensaje_sin_acentos for p in palabras_ingreso):
        tipo = "income"
    elif any(p in mensaje_sin_acentos for p in palabras_gasto):
        tipo = "expense"

    # Categorías
    categorias = {
        "Comida": ["comida", "restaurante", "cafe", "hamburguesa", "super", "snack", "cena", "almuerzo"],
        "Transporte": ["gasolina", "uber", "taxi", "camion", "metro", "auto", "vehiculo", "moto", "transporte"],
        "Entretenimiento": ["cine", "pelicula", "netflix", "concierto", "fiesta", "juego", "evento"],
        "Salud": ["medicina", "doctor", "farmacia", "dentista", "terapia", "hospital", "gimnasio"],
        "Educacion": ["libro", "curso", "escuela", "colegiatura", "universidad", "taller", "clase"],
        "Hogar": ["renta", "luz", "agua", "internet", "gas", "electricidad", "limpieza", "hogar"],
        "Ropa": ["ropa", "zapato", "camisa", "pantalon", "blusa", "vestido"],
        "Trabajo": ["oficina", "computadora", "papeleria", "herramienta", "proyecto", "nomina"],
        "Mascotas": ["perro", "gato", "veterinario", "croquetas", "mascota"],
        "Otros": ["banco", "impuesto", "seguro", "credito", "donacion", "regalo", "ahorro"]
    }

    categoria = "Otros"
    for cat, palabras in categorias.items():
        if any(p in mensaje_sin_acentos for p in palabras):
            categoria = cat
            break

    # Extraer monto
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

    descripcion = mensaje_original.capitalize()
    return tipo, categoria, monto, descripcion


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

    headers = {"Authorization": token, "Content-Type": "application/json"}

    print(f"📤 Enviando al backend:\n{json.dumps(data, indent=2, ensure_ascii=False)}")

    try:
        response = requests.post(BACKEND_URL, json=data, headers=headers, timeout=10)
        print(f"📥 Backend status: {response.status_code}")
        print(f"📄 Respuesta: {response.text}")
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

    # 🧠 Filtro avanzado
    if contiene_lenguaje_ofensivo(payload.mensaje):
        raise HTTPException(status_code=400, detail="El mensaje contiene lenguaje ofensivo o disfrazado con símbolos.")
    if validar_mensaje_con_openai(payload.mensaje):
        raise HTTPException(status_code=400, detail="El mensaje contiene lenguaje inapropiado según moderación IA.")

    resultado = clasificar_gasto(payload.mensaje, authorization)
    return resultado


# ==============================
# ENDPOINT DE PRUEBA
# ==============================
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "IA Financiera con filtro extremo de lenguaje ofensivo lista 🚫",
        "backend_url": BACKEND_URL
    }
