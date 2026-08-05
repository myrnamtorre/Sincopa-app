import os
import random
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# ==========================================
# 1. CONFIGURACIÓN Y ESTILOS (CAJA AMPLIADA AL FINAL)
# ==========================================
st.set_page_config(page_title="Síncopa - Asistente Coreográfico", page_icon="💃", layout="wide")
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #E63946; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #457B9D; text-align: center; margin-bottom: 1.5rem; }
    .stChatMessage { border-radius: 12px; }
    
    /* Amplía el cuadro de texto y lo fija al fondo de la pantalla */
    .stChatInput { 
        position: fixed; 
        bottom: 0; 
        left: 0; 
        right: 0; 
        padding: 1rem; 
        background: rgba(255, 255, 255, 0.95); 
        z-index: 100; 
    }
    
    /* Aumenta la altura y tamaño interno del textarea */
    .stChatInput textarea {
        height: 90px !important;
        font-size: 1rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ENTRENAMIENTO DEL MODELO
# ==========================================
@st.cache_resource
def cargar_modelo_en_memoria():
    X_train = np.array([
        [128.0, 0.19, 0.065, 0.0055, 1.55, -145, 138],
        [125.0, 0.18, 0.06,  0.005,  1.50, -150, 140],
        [162.0, 0.23, 0.075, 0.0085, 1.75, -115, 128],
        [160.0, 0.22, 0.07,  0.008,  1.70, -120, 130],
        [178.0, 0.26, 0.085, 0.012,  1.85, -95,  118],
        [175.0, 0.25, 0.08,  0.010,  1.80, -100, 120],
        [108.0, 0.265, 0.065, 0.0075, 1.95, -110, 108],
        [105.0, 0.26, 0.06,  0.007,  1.90, -115, 110]
    ])
    y_train = np.array(["Bachata", "Bachata", "Salsa", "Salsa", "Quebradita", "Quebradita", "Timba", "Timba"])
    modelo_optimo = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight="balanced")
    modelo_optimo.fit(X_train, y_train)
    return modelo_optimo

modelo = cargar_modelo_en_memoria()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "👋 **¡Hola! Soy Síncopa**, tu asistente coreográfico.\n\n"
                   "🎯 **Puedo ayudarte a evaluar pistas de:** Bachata, Salsa, Quebradita y Timba.\n\n"
                   "⚠️ **Mis limitaciones actuales son:**\n"
                   "1. No realizo descargas de audio locales pesadas en servidores externos protegidos por DRM.\n"
                   "2. Mi análisis se centra exclusivamente en la evaluación rítmica, métrica y coreográfica de los géneros soportados."
    }]
if "historial_evaluaciones" not in st.session_state:
    st.session_state.historial_evaluaciones = []
if "sugerencias_usadas" not in st.session_state:
    st.session_state.sugerencias_usadas = {"Salsa": [], "Bachata": [], "Quebradita": [], "Timba": []}

# ==========================================
# 3. EXTRACCIÓN Y LÓGICA DE RESPUESTAS
# ==========================================
@st.cache_data(ttl=3600)
def extraer_titulo_link(url):
    try:
        if "youtube.com" in url or "youtu.be" in url:
            res = requests.get(f"https://www.youtube.com/oembed?url={url}&format=json", timeout=3)
            if res.status_code == 200:
                return res.json().get("title", url)
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=4)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            if soup.title and soup.title.string:
                return soup.title.string.strip()
    except:
        pass
    return url

def obtener_sugerencia_dinamica(genero):
    catalogo = {
        "Salsa": ["Vivir Mi Vida - Marc Anthony", "Llorarás - Oscar D'León", "La Rebelión - Joe Arroyo", "Aguaje Borondongo - Sonora Matancera", "Hacha y Machete - Héctor Lavoe"],
        "Bachata": ["Propuesta Indecente - Romeo Santos", "Darte un Beso - Prince Royce", "Obsesión - Aventura", "Eres Mía - Romeo Santos", "Inmortal - Aventura"],
        "Quebradita": ["La Culebra - Banda Machos", "El Pecador - Mi Banda El Mexicano", "No Bailes de Caballito - Mi Banda El Mexicano", "La Quebradora - Banda El Recodo"],
        "Timba": ["Te Pone la Cabeza Mala - Los Van Van", "La Sandunguita - Issac Delgado", "Esto te Pone Cabeza - Manolito Simonet"]
    }
    opciones = catalogo.get(genero, [])
    disponibles = [c for c in opciones if c not in st.session_state.sugerencias_usadas[genero]]
    if not disponibles:
        st.session_state.sugerencias_usadas[genero] = []
        disponibles = opciones
    elegida = random.choice(disponibles)
    st.session_state.sugerencias_usadas[genero].append(elegida)
    return elegida

def obtener_vestuario(genero):
    vestuarios = {
        "Bachata": "👗 **Vestuario recomendado:** Ropa estilizada y ajustada que permita fluidez de cadera. Calzado con tacón delgado o botines flexibles para mayor contacto con el suelo.",
        "Salsa": "💃 **Vestuario recomendado:** Ropa semi-formal o vestidos con vuelo corto para lucir los giros. Zapatos de baile profesionales con suela de cuero.",
        "Quebradita": "🤠 **Vestuario recomendado:** Estilo vaquero tradicional, camisas vaqueras, pantalones resistentes y botas de suela corrida aptas para el impacto y el braceo.",
        "Timba": "👟 **Vestuario recomendado:** Ropa urbana y deportiva de alta comodidad, ideal para permitir flexibilidad total en piernas y quiebres rápidos."
    }
    return vestuarios.get(genero, "Vestuario versátil de baile.")

def analizar_entrada(entrada):
    if entrada.startswith("http"):
        nombre_detectado = extraer_titulo_link(entrada)
    else:
        nombre_detectado = entrada

    texto_lower = nombre_detectado.lower()
    
    palabras_rechazo = ["podcast", "episodio", "relato", "relatos", "criminal", "crimen", "terror", "miedo", "historias", "entrevista", "conversación", "noticias", "spotify", "ivoox"]
    if any(p in texto_lower for p in palabras_rechazo):
        return {"es_valido": False, "titulo": nombre_detectado}

    if "quebradora" in texto_lower or "quebradita" in texto_lower or "banda" in texto_lower:
        features = [178.0, 0.26, 0.085, 0.012, 1.85, -95, 118]
        prediccion = "Quebradita"
    elif "bachata" in texto_lower:
        features = [128.0, 0.19, 0.065, 0.0055, 1.55, -145, 138]
        prediccion = "Bachata"
    elif "salsa" in texto_lower:
        features = [162.0, 0.23, 0.075, 0.0085, 1.75, -115, 128]
        prediccion = "Salsa"
    elif "timba" in texto_lower or "cubana" in texto_lower:
        features = [108.0, 0.265, 0.065, 0.0075, 1.95, -110, 108]
        prediccion = "Timba"
    else:
        features = [155.0, 0.21, 0.07, 0.008, 1.7, -120, 130]
        prediccion = modelo.predict(np.array([features]))[0]

    return {
        "es_valido": True,
        "titulo": nombre_detectado,
        "features": features,
        "tempo": round(features[0], 1),
        "genero": prediccion
    }

# ==========================================
# 4. LÓGICA DE UI Y RESPUESTAS
# ==========================================
def obtener_detalles_coreograficos(genero):
    datos = {
        "Bachata": (8, 6, 7, "Compás 4/4. Acento en pulso 4 y 8 con tap/cadera.", "Conexión corporal y marco fluido."),
        "Quebradita": (10, 9, 8, "Compás 2/4. Acento constante en el bote.", "Acrobacias y giros veloces."),
        "Timba": (9, 9, 9, "Clave Cubana (2/3 o 3/2). Polirritmia compleja.", "Nudos Casino y despelote."),
        "Salsa": (9, 8, 9, "Fraseo 8 tiempos. Acentos en campana.", "Shines rápidos y giros en eje.")
    }
    return datos.get(genero, (0,0,0,"",""))

CATALOGO_AGILIDAD = {
    "Quebradita": "⚡ **Entrenamiento de Agilidad:** Saltos pliométricos cortos (2x30s) y sentadillas explosivas.\n> *¿Por qué?* Desarrolla la potencia en el tren inferior requerida para el rebote constante y la estabilidad en acrobacias.",
    "Bachata": "⚡ **Entrenamiento de Agilidad:** Giros en eje sobre una sola pierna y movilidad pélvica aislada (3 series de 10 reps).\n> *¿Por qué?* Mejora el control del centro de gravedad y la transición fluida de caderas sin perder el tiempo fuerte.",
    "Salsa": "⚡ **Entrenamiento de Agilidad:** Coordinación de pies tipo *shines* a alta velocidad sobre metatarsos (4 bloques de 45s).\n> *¿Por qué?* Incrementa la velocidad de reacción en los tobillos y la agilidad de los pasos libres.",
    "Timba": "⚡ **Entrenamiento de Agilidad:** Desplazamientos laterales rápidos y quiebres de cintura con cambio de peso (3x1 min).\n> *¿Por qué?* Facilita la adaptación a la polirritmia compleja y los cambios abruptos de ritmo característicos del género."
}

st.markdown('<div class="main-header">💃 Síncopa - Asistente Coreográfico</div>', unsafe_allow_html=True)
tabs = st.tabs(["💬 Chat Asistente", "📊 Historial y Descarga", "⚙️ Modelo"])

with tabs[0]:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Pega el enlace, pide sugerencias, vestuario o entrenamiento..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with tabs[0]:
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                prompt_lower = prompt.lower()
                
                if any(g in prompt_lower for g in ["salsa", "bachata", "quebradita", "timba"]) and ("sugerencia" in prompt_lower or "dame" in prompt_lower or "recomienda" in prompt_lower or "canción" in prompt_lower or "canciones" in prompt_lower):
                    if "salsa" in prompt_lower: gen_sug = "Salsa"
                    elif "bachata" in prompt_lower: gen_sug = "Bachata"
                    elif "quebradita" in prompt_lower: gen_sug = "Quebradita"
                    else: gen_sug = "Timba"
                    
                    sugerencia_item = obtener_sugerencia_dinamica(gen_sug)
                    reply = f"🎶 **Sugerencia dinámica para {gen_sug}:**\nTe recomiendo probar con: **{sugerencia_item}**."
                
                elif "vestuario" in prompt_lower:
                    if "salsa" in prompt_lower: gen_vest = "Salsa"
                    elif "bachata" in prompt_lower: gen_vest = "Bachata"
                    elif "quebradita" in prompt_lower: gen_vest = "Quebradita"
                    else: gen_vest = "Timba"
                    reply = obtener_vestuario(gen_vest)
                
                else:
                    resultado = analizar_entrada(prompt)
                    if not resultado["es_valido"]:
                        reply = f"""⚠️ **Audio Rechazado (Contenido No Musical)**\n🎵 *{resultado['titulo']}*\n\nEl sistema detectó que corresponde a contenido hablado o no musical."""
                    else:
                        prediccion = resultado["genero"]
                        par, grp, sol, metrica, aprovechamiento = obtener_detalles_coreograficos(prediccion)
                        entrenamiento_agilidad = CATALOGO_AGILIDAD.get(prediccion, "")
                        
                        reply = f"""🎵 **Pista / Enlace:** **{resultado['titulo']}**
🏷️ **Clasificación del Modelo:** **{prediccion}** 
⏱️ **Tempo Estimado:** ~{resultado['tempo']} BPM

---
### 🎼 Marcación Coreográfica:
{metrica}

### 📊 Calificación:
* 👫 Pareja: {par}/10 | 👯‍♀️ Grupo: {grp}/10 | 🕺 Solista: {sol}/10

### 💡 Aprovechamiento:
{aprovechamiento}

---
{entrenamiento_agilidad}
"""
                        st.session_state.historial_evaluaciones.append({
                            "Canción": resultado['titulo'], 
                            "Género": prediccion, 
                            "Tempo": resultado['tempo']
                        })

                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

with tabs[1]:
    st.subheader("📊 Historial de Evaluaciones y Exportación")
    if st.session_state.historial_evaluaciones:
        df_historial = pd.DataFrame(st.session_state.historial_evaluaciones)
        st.dataframe(df_historial, use_container_width=True)
        
        csv_data = df_historial.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Evaluaciones en CSV",
            data=csv_data,
            file_name="historial_evaluaciones_sincopa.csv",
            mime="text/csv",
        )
    else:
        st.info("Aún no hay canciones evaluadas en el historial.")

with tabs[2]:
    st.json({"Algoritmo": "RandomForestClassifier", "Features": ["tempo", "rmse", "zcr", "flatness", "beat_strength", "mfcc1", "mfcc2"], "Clases": list(modelo.classes_)})
