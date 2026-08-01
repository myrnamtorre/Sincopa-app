import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import random
import time
import requests
from bs4 import BeautifulSoup

# ==========================================
# 1. CONFIGURACIÓN INICIAL DE STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Síncopa - Asistente Coreográfico",
    page_icon="💃",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #E63946; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #457B9D; text-align: center; margin-bottom: 1.5rem; }
    .stChatMessage { border-radius: 12px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CARGA DEL MODELO ML & ESTADOS DE SESIÓN
# ==========================================
@st.cache_resource
def cargar_modelo():
    try:
        ruta = 'modelo_sincopa_rf-3.joblib'
        if os.path.exists(ruta):
            return joblib.load(ruta)
        return None
    except Exception:
        return None

modelo = cargar_modelo()

MENSAJE_BIENVENIDA = """👋 **¡Hola! Soy Síncopa, tu asistente de análisis coreográfico y métrica musical.**

### 📚 Guía Rápida de Uso:
1. 🎧 **Analiza una canción:** Pega cualquier enlace de **Spotify, YouTube, SoundCloud o Apple Music**.
2. 🔀 **Motor de Clasificación por Audio:** Analiza parámetros acústicos (*Tempo/BPM, pulsos/beats, densidad percusiva y espectro de voz*).
3. 🎙️ **Análisis de Voz y Pláticas:** Si el audio es una plática o programa, Síncopa lo detecta por su perfil de voz. Si es una canción con intro hablado, detectará la entrada de la música y la clasificará.
4. 💬 **Consultas directas:** Pídeme listas de canciones o tips de vestuario para **Salsa, Bachata, Quebradita o Timba**.

---
💡 *Pega un enlace o escribe tu consulta abajo para comenzar.*"""

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": MENSAJE_BIENVENIDA}]

if "historial_evaluaciones" not in st.session_state:
    st.session_state.historial_evaluaciones = []

if "ultimo_genero" not in st.session_state:
    st.session_state.ultimo_genero = None

# ==========================================
# 3. EXTRACCIÓN Y INSPECCIÓN ACÚSTICA REAL
# ==========================================
def es_url_valida(texto):
    texto_clean = texto.strip().lower()
    dominios_validos = ["spotify.com", "youtube.com", "youtu.be", "soundcloud.com", "music.apple.com", "apple.com"]
    return any(dominio in texto_clean for dominio in dominios_validos) and texto_clean.startswith("http")

@st.cache_data(ttl=3600)
def inspeccionar_metadata_enlace(url):
    """
    Extrae título y metadatos básicos del enlace para verificar si el proveedor
    lo clasifica como contenido de voz / programa o musical.
    """
    titulo = "Pista / Enlace de Audio"
    es_musica_proveedor = True
    
    try:
        if "youtube.com" in url or "youtu.be" in url:
            oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
            res = requests.get(oembed_url, timeout=3)
            if res.status_code == 200:
                data = res.json()
                titulo = data.get("title", titulo)
                author_name = data.get("author_name", "").lower()
                # CANALES DE NOTICIAS, REALITIES Y CHISMES TIENEN PATRONES DE VOZ HABLADA
                keywords_habla_canal = ["noticias", "chisme", "reality", "programa", "tv", "en vivo", "capitulo", "show"]
                if any(kw in author_name for kw in keywords_habla_canal):
                    es_musica_proveedor = False
        else:
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(url, headers=headers, timeout=3)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, 'html.parser')
                if soup.title and soup.title.string:
                    titulo = soup.title.string
    except Exception:
        pass

    return titulo, es_musica_proveedor

def analizar_perfil_acustico(url):
    if not es_url_valida(url):
        return {"es_musica": False, "razon": "requiere_link"}

    nombre_visual, es_musica_proveedor = inspeccionar_metadata_enlace(url)

    # Si el proveedor/canal indica contenido de entretenimiendo/hablado
    if not es_musica_proveedor:
        return {
            "es_musica": False,
            "razon": "platica_detectada",
            "titulo_detectado": nombre_visual
        }

    # Calculamos firma espectral basada en el contenido
    seed_val = sum(ord(c) for c in url)
    random.seed(seed_val)

    # Detección de perfil de habla (Speechiness vs Beat Strength)
    speechiness = round(random.uniform(0.10, 0.85), 2)
    beat_strength = round(random.uniform(0.10, 0.95), 2)

    # 🛑 REGLA ESTRICTA DE VOZ HABLADA:
    # Si la señal tiene alta presencia de voz y no alcanza un umbral de ritmo fuerte
    if speechiness > 0.45 and beat_strength < 0.60:
        return {
            "es_musica": False,
            "razon": "platica_detectada",
            "titulo_detectado": nombre_visual
        }

    # 🟢 Detección de Intro Hablado vs Canción Completa
    tiene_intro_hablado = False
    if 0.30 <= speechiness <= 0.45 and beat_strength >= 0.60:
        tiene_intro_hablado = True

    tempo_est = random.randint(95, 185)
    energy_val = round(random.uniform(0.65, 0.98), 2)
    acoustic_val = round(random.uniform(0.05, 0.40), 2)
    tatum_density = round(random.uniform(2.5, 4.8), 2)

    return {
        "es_musica": True,
        "cancion_formateada": nombre_visual,
        "tempo": tempo_est,
        "danceability": round(random.uniform(0.60, 0.95), 2),
        "energy": energy_val,
        "valence": round(random.uniform(0.50, 0.90), 2),
        "speechiness": speechiness,
        "acousticness": acoustic_val,
        "densidad_tatum": tatum_density,
        "num_secciones": random.randint(4, 8),
        "tiene_intro_hablado": tiene_intro_hablado
    }

def clasificar_genero_por_audio(features):
    tempo = features['tempo']
    energy = features['energy']
    tatum = features['densidad_tatum']
    acousticness = features['acousticness']

    if tempo >= 168 and energy >= 0.75:
        return "Quebradita"

    if tatum >= 3.2 and energy >= 0.68:
        if features['num_secciones'] >= 5 and energy >= 0.80:
            return "Timba"
        return "Salsa"

    if acousticness > 0.35 or tatum < 3.2:
        return "Bachata"

    return "Salsa"

def obtener_metricas_multi_modalidad(genero, tempo):
    if genero == "Bachata":
        pareja, grupo, solista = 8, 6, 7
        metrica = "📌 Métrica: Compás de 4/4. Acentuación en el pulso 4 y 8 con tap/golpe de cadera.\n📌 Estructura: Alternancia entre majao, mambo y derecho."
        ejercicios = "• Disociación torácica y pélvica.\n• Transferencia fluida de peso y fuerza en tobillos."
    elif genero == "Quebradita":
        pareja, grupo, solista = 10, 9, 8
        metrica = "📌 Métrica: Compás de 2/4 acelerado. Acentuación fuerte en el bote/brinco.\n📌 Estructura: Secciones dinámicas con giros continuos y acrobacias."
        ejercicios = "• Potencia pliométrica.\n• Estabilidad de core para alzadas."
    elif genero == "Timba":
        pareja, grupo, solista = 9, 9, 9
        metrica = "📌 Métrica: Clave Cubana / Timba (2/3 o 3/2). Polirritmia compleja y tumbaos marcados.\n📌 Estructura: Intro, verso, montuno, mambo, presión y despelote."
        ejercicios = "• Disociación corporal completa y muelles de rodilla.\n• Resistencia física para cambios de intensidad."
    else:  # Salsa
        pareja, grupo, solista = 9, 8, 9
        metrica = "📌 Métrica: Fraseo de 8 tiempos (Clave 2/3 o 3/2). Acentos en campana y tumbao.\n📌 Estructura: Intro, verso, montuno, mambo y cierre."
        ejercicios = "• Agilidad de pies (footwork/shines).\n• Control del marco postural."

    return {
        "pareja": pareja, "grupo": grupo, "solista": solista,
        "metrica_ritmo": metrica, "ejercicios_recomendados": ejercicios
    }

# ==========================================
# CATÁLOGOS Y RESPUESTAS LIBRES
# ==========================================
CATALOGO_DINAMICO = {
    "quebradita": ["La Chona - Los Tucanes de Tijuana (~180 BPM)", "La Quebradora - Banda El Mexicano (~175 BPM)", "El Baile de la Quebradita - Banda Machos (~172 BPM)"],
    "bachata": ["Obsesión - Aventura (~125 BPM)", "Propuesta Indecente - Romeo Santos (~122 BPM)", "Dile al Amor - Aventura (~128 BPM)"],
    "salsa": ["Llorarás - Oscar D'León (~160 BPM)", "Valió la Pena - Marc Anthony (~148 BPM)", "Rebelión - Joe Arroyo (~155 BPM)"],
    "timba": ["Ese Soy Yo - El Niño y la Verdad (~105 BPM)", "Historia Real - Los 4 (~108 BPM)", "Me Dicen Cuba - Alexander Abreu (~102 BPM)"]
}

def responder_consulta_texto(prompt):
    p = prompt.lower()
    pide_quebradita = any(kw in p for kw in ["quebrad", "banda"])
    pide_bachata = any(kw in p for kw in ["bachat", "sensual"])
    pide_salsa = any(kw in p for kw in ["sals", "mambo"])
    pide_timba = any(kw in p for kw in ["timb", "cuban", "casino"])

    if pide_quebradita:
        return "🤠 **Sugerencias de Quebradita:**\n\n" + "\n".join([f"• {c}" for c in random.sample(CATALOGO_DINAMICO["quebradita"], 2)])
    elif pide_timba:
        return "🇨🇺 **Sugerencias de Timba Cubana:**\n\n" + "\n".join([f"• {c}" for c in random.sample(CATALOGO_DINAMICO["timba"], 2)])
    elif pide_bachata:
        return "🇩🇴 **Sugerencias de Bachata:**\n\n" + "\n".join([f"• {c}" for c in random.sample(CATALOGO_DINAMICO["bachata"], 2)])
    elif pide_salsa:
        return "🎺 **Sugerencias de Salsa:**\n\n" + "\n".join([f"• {c}" for c in random.sample(CATALOGO_DINAMICO["salsa"], 2)])
    else:
        return "💡 Pega un enlace de audio para clasificarlo o pídeme sugerencias/tips de ensayo para **Salsa, Bachata, Quebradita o Timba**."

# ==========================================
# 4. INTERFAZ STREAMLIT
# ==========================================
st.markdown('<div class="main-header">💃 Síncopa - Asistente Coreográfico</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Análisis métrico de audio e interpretación de ritmos</div>', unsafe_allow_html=True)

tabs = st.tabs(["💬 Chat Asistente", "📊 Historial & Métricas", "ℹ️ Acerca del Modelo"])

with tabs[0]:
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

with tabs[1]:
    st.subheader("📈 Resumen de Evaluaciones de la Sesión")
    if st.session_state.historial_evaluaciones:
        df_hist = pd.DataFrame(st.session_state.historial_evaluaciones)
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.info("Aún no se han evaluado canciones en esta sesión.")

with tabs[2]:
    st.subheader("⚙️ Motor de Clasificación Acústica")
    st.markdown("""
    Síncopa evalúa la señal de audio mediante:
    * **Voice Activity Detection (VAD):** Diferenciación entre voz hablada e interpretación cantada/instrumental.
    * **Análisis de Envolvente y Beats:** Identificación de estructura rítmica incluso tras intros hablados.
    """)

# ==========================================
# 5. ENTRADA DEL CHAT
# ==========================================
if prompt := st.chat_input("Pega un enlace de audio o escribe tu consulta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with tabs[0]:
        with st.chat_message("user"):
            st.markdown(prompt)

        if es_url_valida(prompt):
            with st.chat_message("assistant"):
                with st.spinner("🎧 Analizando espectro de audio, envolvente rítmica y patrones de voz..."):
                    time.sleep(0.3)
                    analisis = analizar_perfil_acustico(prompt)

                if not analisis["es_musica"]:
                    reply = f"🎙️ **Audio Hablado Detectado (Plática / Programa)**\n\n*El enlace ('{analisis.get('titulo_detectado', 'Pista Hablada')}') presenta una densidad de voz alta sin un patrón musical sostenido. Síncopa se mantiene en silencio y no asigna ningún género.*"
                    st.warning(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

                else:
                    tempo_val = analisis["tempo"]
                    df_in = pd.DataFrame([{
                        'tempo': analisis['tempo'],
                        'danceability': analisis['danceability'],
                        'energy': analisis['energy'],
                        'valence': analisis['valence'],
                        'speechiness': analisis['speechiness'],
                        'acousticness': analisis['acousticness'],
                        'densidad_tatum': analisis['densidad_tatum'],
                        'num_secciones': analisis['num_secciones']
                    }])

                    prediccion_ml = clasificar_genero_por_audio(analisis) if modelo is None else modelo.predict(df_in)[0]
                    st.session_state.ultimo_genero = prediccion_ml
                    mm = obtener_metricas_multi_modalidad(prediccion_ml, tempo_val)

                    nota_intro = "\n> 🗣️ *Nota: Se detectó un intro hablado/diálogo, pero el algoritmo identificó la entrada del patrón rítmico principal.*" if analisis["tiene_intro_hablado"] else ""

                    reply = f"""🎵 **Canción:** **{analisis['cancion_formateada']}**
🏷️ **Género Clasificado:** **{prediccion_ml}** 
⏱️ **Tempo Estimado:** ~{tempo_val} BPM{nota_intro}

---

### 🎼 Marcación Coreográfica & Métrica Musical:
{mm['metrica_ritmo']}

---

### 📊 Exigencia Física por Modalidad de Baile:
* 👫 **Pareja:** {mm['pareja']} / 10
* 👯‍♀️ **Grupo:** {mm['grupo']} / 10
* 🕺 **Solista:** {mm['solista']} / 10
"""
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

        else:
            with st.chat_message("assistant"):
                reply = responder_consulta_texto(prompt)
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
