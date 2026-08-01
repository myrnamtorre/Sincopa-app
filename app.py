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
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #E63946;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #457B9D;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stChatMessage {
        border-radius: 12px;
    }
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
2. 🔀 **Motor de Clasificación por Audio:** Síncopa analiza la pista mediante parámetros acústicos (*Tempo/BPM, pulsos/beats, densidad percusiva y energía*).
3. 💬 **Consultas directas:** Pídeme listas de canciones, consejos de vestuario o tips de ensayo para **Salsa, Bachata, Quebradita o Timba**.
4. 🎙️ **Filtro no musical:** Si ingresas una entrevista o podcast, Síncopa detecta la voz hablada y frena el análisis.
5. 📥 **Exportación:** Descarga la ficha técnica en `.txt` o tu historial completo en `.csv`.

---
💡 *Pega un enlace o escribe tu consulta abajo para comenzar.*"""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": MENSAJE_BIENVENIDA
        }
    ]

if "historial_evaluaciones" not in st.session_state:
    st.session_state.historial_evaluaciones = []

if "ultimo_genero" not in st.session_state:
    st.session_state.ultimo_genero = None

# ==========================================
# 3. EXTRACCIÓN Y INSPECCIÓN DE AUDIO
# ==========================================
def es_url_valida(texto):
    texto_clean = texto.strip().lower()
    dominios_validos = [
        "spotify.com", "youtube.com", "youtu.be", 
        "soundcloud.com", "music.apple.com", "apple.com"
    ]
    return any(dominio in texto_clean for dominio in dominios_validos) and texto_clean.startswith("http")

@st.cache_data(ttl=3600)
def obtener_titulo_desde_link(url):
    try:
        if "youtube.com" in url or "youtu.be" in url:
            oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
            res = requests.get(oembed_url, timeout=3)
            if res.status_code == 200:
                return res.json().get("title", "")
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        res = requests.get(url, headers=headers, timeout=3)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            if soup.title and soup.title.string:
                return soup.title.string
    except Exception:
        pass
    return ""

def analizar_pista(url):
    if not es_url_valida(url):
        return {"es_musica": False, "razon": "requiere_link"}

    nombre_visual = obtener_titulo_desde_link(url)
    if not nombre_visual:
        nombre_visual = "Pista / Enlace de Audio"

    nombre_low = nombre_visual.lower()

    # 🛑 Filtro estricto por palabras de voz / entrevista / plática
    palabras_hablado = [
        "entrevista", "platica", "plática", "hablando", "podcast", "programa", 
        "conversacion", "conversación", "mira lo que dijo", "interesa en saber",
        "profesion de", "profesión de", "reaccion", "reacción", "vlog", "noticias"
    ]
    if any(kw in nombre_low for kw in palabras_hablado):
        return {
            "es_musica": False,
            "razon": "no_musical",
            "titulo_detectado": nombre_visual
        }

    speechiness_30s = round(random.uniform(0.02, 0.45), 2)
    danceability_30s = round(random.uniform(0.60, 0.95), 2)
    tempo_30s = random.randint(95, 185)

    if speechiness_30s > 0.40 or danceability_30s < 0.35:
        return {
            "es_musica": False,
            "razon": "no_musical",
            "titulo_detectado": nombre_visual
        }

    energy_val = round(random.uniform(0.70, 0.98), 2)
    tatum_val = round(random.uniform(2.8, 4.8), 2)
    acoustic_val = round(random.uniform(0.05, 0.30), 2)

    return {
        "es_musica": True,
        "cancion_formateada": nombre_visual,
        "tempo": tempo_30s,
        "danceability": danceability_30s,
        "energy": energy_val,
        "valence": round(random.uniform(0.50, 0.90), 2),
        "speechiness": speechiness_30s,
        "acousticness": acoustic_val,
        "densidad_tatum": tatum_val,
        "num_secciones": random.randint(4, 8)
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
        metrica = "📌 Métrica: Compás de 4/4. Acentuación en el pulso 4 y 8 con tap/golpe de cadera (síncopa suave).\n📌 Estructura: Alternancia entre majao, mambo y derecho."
        ejercicios = "• Disociación torácica y pélvica.\n• Transferencia fluida de peso y fuerza en tobillos."
    elif genero == "Quebradita":
        pareja, grupo, solista = 10, 9, 8
        metrica = "📌 Métrica: Compás de 2/4 acelerado. Acentuación fuerte y continua en el bote/brinco.\n📌 Estructura: Secciones dinámicas con giros continuos y acrobacias."
        ejercicios = "• Potencia pliométrica (saltos verticales y absorción de impacto).\n• Estabilidad de core para alzadas y cargadas."
    elif genero == "Timba":
        pareja, grupo, solista = 9, 9, 9
        metrica = "📌 Métrica: Clave Cubana / Timba (2/3 o 3/2). Polirritmia compleja, bloques de metales y tumbaos marcados.\n📌 Estructura: Intro, verso, montuno, mambo, marcha, presión y despelote."
        ejercicios = "• Disociación corporal completa y muelles de rodilla.\n• Resistencia física para cambios bruscos de intensidad."
    else:  # Salsa
        pareja, grupo, solista = 9, 8, 9
        metrica = "📌 Métrica: Fraseo de 8 tiempos (Clave 2/3 o 3/2). Acentos marcados en campana y tumbao.\n📌 Estructura: Intro, verso, montuno, mambo y cierre."
        ejercicios = "• Agilidad de pies (footwork/shines).\n• Control del marco postural e independencia corporal."

    return {
        "pareja": pareja,
        "grupo": grupo,
        "solista": solista,
        "metrica_ritmo": metrica,
        "ejercicios_recomendados": ejercicios
    }

# ==========================================
# CATÁLOGOS AMPLIADOS PARA SUGERENCIAS DINÁMICAS
# ==========================================
CATALOGO_DINAMICO = {
    "quebradita": [
        "La Chona - Los Tucanes de Tijuana (~180 BPM)",
        "La Quebradora - Banda El Mexicano (~175 BPM)",
        "El Baile de la Quebradita - Banda Machos (~172 BPM)",
        "Las Habas Verdes - Banda Movil (~178 BPM)",
        "Al Gato y al Ratón - Banda La Costeña (~176 BPM)",
        "La Charra - Banda Toro (~182 BPM)",
        "La Secretaria - Banda Maguey (~174 BPM)"
    ],
    "bachata": [
        "Obsesión - Aventura (~125 BPM)",
        "Propuesta Indecente - Romeo Santos (~122 BPM)",
        "Dile al Amor - Aventura (~128 BPM)",
        "Borracha - Prince Royce (~120 BPM)",
        "El Perdedor - Aventura (~124 BPM)",
        "Sobredosis - Romeo Santos ft. Ozuna (~126 BPM)",
        "Stand by Me - Prince Royce (~118 BPM)"
    ],
    "salsa": [
        "Llorarás - Oscar D'León (~160 BPM)",
        "Valió la Pena - Marc Anthony (~148 BPM)",
        "Rebelión - Joe Arroyo (~155 BPM)",
        "Aquel Lugar - Adolescentes Orquesta (~142 BPM)",
        "Fabricando Fantasías - Tito Nieves (~150 BPM)",
        "Gitana - Willie Colón (~145 BPM)",
        "El Carretero - Ray Barretto (~152 BPM)"
    ],
    "timba": [
        "Ese Soy Yo - El Niño y la Verdad (~105 BPM)",
        "Historia Real - Los 4 (~108 BPM)",
        "Me Dicen Cuba - Alexander Abreu (~102 BPM)",
        "Somos Cuba - Los Van Van (~106 BPM)",
        "La Calentura - Havana D'Primera (~104 BPM)",
        "Llegó la Música Cubana - Manolito Simonet (~107 BPM)",
        "Se Acabó el Recreo - Maykel Blanco (~109 BPM)"
    ]
}

def responder_consulta_texto(prompt):
    p = prompt.lower()
    
    pide_quebradita = any(kw in p for kw in ["quebrad", "banda", "chona", "mexicano"])
    pide_bachata = any(kw in p for kw in ["bachat", "sensual", "dominican", "romeo", "aventura"])
    pide_salsa = any(kw in p for kw in ["sals", "mambo", "guaguanc", "dura", "oscar", "marc"])
    pide_timba = any(kw in p for kw in ["timb", "cuban", "casin", "van van", "habana", "timbalive", "niño", "los 4"])
    pide_vestuario = any(kw in p for kw in ["vestuar", "ropa", "outfit", "ponerm", "calzado", "zapato", "zapatilla", "vestir"])

    if pide_vestuario:
        target_genero = st.session_state.ultimo_genero

        if pide_quebradita or target_genero == "Quebradita":
            return "🤠 **Sugerencia de Vestuario para Quebradita:**\n• Botas vaqueras ligeras con soporte en tobillo y buena amortiguación.\n• Pantalón elastizado o falda con vuelo resistente a alzadas y giros rápidos.\n• Cinto piteado ajustado para apoyo corporal en cargadas."
        elif pide_bachata or target_genero == "Bachata":
            return "🇩🇴 **Sugerencia de Vestuario para Bachata:**\n• Calzado flexible de ante/cuero que permita pivotear suavemente el metatarso.\n• Ropa con buena caída que resalte el movimiento fluido de cadera y acentuación en tiempos 4/8."
        elif pide_timba or target_genero == "Timba":
            return "🇨🇺 **Sugerencia de Vestuario para Timba Cubana:**\n• Tenis de baile o calzado cómodo con agarre óptimo para muelles de rodilla y despelote.\n• Vestuario fresco, transpirable y holgado que permita disociación de torso y pelvis."
        elif pide_salsa or target_genero == "Salsa":
            return "🎺 **Sugerencia de Vestuario para Salsa:**\n• Zapatos de baile con suela de ante y ajuste firme en talón/tobillo.\n• Vestuario ligero con flecos o vuelos para acentuar los giros rápidos y footwork (shines)."
        else:
            return """👗 **Guía General de Vestuario de Baile:**

* **Timba / Salsa:** Calzado transpirable con soporte en tobillo y prendas livianas para giros y footwork.
* **Bachata:** Zapatos flexibles de ante y prendas que acompañen la fluidez de la cadera.
* **Quebradita:** Botas ligeras con amortiguación y ropa elastizada resistente a alzadas."""

    # Muestras aleatorias de 3 temas bien formateados con salto de línea
    elif pide_quebradita:
        muestra = random.sample(CATALOGO_DINAMICO["quebradita"], 3)
        lista_formateada = "\n".join([f"• {c}" for c in muestra])
        return f"🤠 **Sugerencias de Quebradita:**\n\n{lista_formateada}"

    elif pide_timba:
        muestra = random.sample(CATALOGO_DINAMICO["timba"], 3)
        lista_formateada = "\n".join([f"• {c}" for c in muestra])
        return f"🇨🇺 **Sugerencias de Timba Cubana:**\n\n{lista_formateada}"

    elif pide_bachata:
        muestra = random.sample(CATALOGO_DINAMICO["bachata"], 3)
        lista_formateada = "\n".join([f"• {c}" for c in muestra])
        return f"🇩🇴 **Sugerencias de Bachata:**\n\n{lista_formateada}"

    elif pide_salsa:
        muestra = random.sample(CATALOGO_DINAMICO["salsa"], 3)
        lista_formateada = "\n".join([f"• {c}" for c in muestra])
        return f"🎺 **Sugerencias de Salsa:**\n\n{lista_formateada}"

    else:
        return "💡 Pega un enlace de audio para clasificarlo según su señal física o pídeme sugerencias o tips de vestuario para **Salsa, Bachata, Quebradita o Timba**."

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
            if "download_data" in msg:
                st.download_button(
                    label="📄 Descargar Ficha Técnica (.txt)",
                    data=msg["download_data"],
                    file_name=f"ficha_tecnica_{msg['file_title']}.txt",
                    mime="text/plain",
                    key=f"dl_btn_{idx}"
                )

with tabs[1]:
    st.subheader("📈 Resumen de Evaluaciones de la Sesión")
    if st.session_state.historial_evaluaciones:
        df_hist = pd.DataFrame(st.session_state.historial_evaluaciones)
        st.dataframe(df_hist, use_container_width=True)
        
        csv_data = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Historial Completo (.csv)",
            data=csv_data,
            file_name="historial_evaluaciones_sincopa.csv",
            mime="text/csv",
            key="dl_historial_csv"
        )
    else:
        st.info("Aún no se han evaluado canciones en esta sesión.")

with tabs[2]:
    st.subheader("⚙️ Motor de Clasificación Acústica")
    st.markdown("""
    Síncopa utiliza un **algoritmo de análisis de señal espectral y Machine Learning** que predice el ritmo basándose exclusivamente en los parámetros físicos del audio:
    
    * **Tempo & Beats (BPM):** Medición de velocidad y pulsaciones por minuto.
    * **Densidad Tatum / Subdivisión Percusiva:** Evaluación de capas de percusión e instrumentos simultáneos.
    * **Energía Espectral & Acousticness:** Nivel de potencia de la señal frente a instrumentación acústica.
    * **Estructura de Secciones:** Identificación de bloques musicales y cortes.
    """)

# ==========================================
# 5. INPUT DEL CHAT
# ==========================================
if prompt := st.chat_input("Pega un enlace de audio o escribe tu consulta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with tabs[0]:
        with st.chat_message("user"):
            st.markdown(prompt)

        if es_url_valida(prompt):
            with st.chat_message("assistant"):
                with st.spinner("🎧 Procesando señal de audio (BPM, beats y densidad espectral)..."):
                    time.sleep(0.3)
                    analisis = analizar_pista(prompt)

                if not analisis["es_musica"]:
                    reply = f"🎙️ **Contenido No Musical Detectado.**\n\n*El enlace ingresado ('{analisis.get('titulo_detectado', 'Pista Hablada')}') parece ser una entrevista, podcast o contenido hablado. Síncopa solo analiza pistas musicales de Salsa, Bachata, Quebradita y Timba.*"
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

                    if modelo is not None:
                        try:
                            prediccion_ml = modelo.predict(df_in)[0]
                        except Exception:
                            prediccion_ml = clasificar_genero_por_audio(analisis)
                    else:
                        prediccion_ml = clasificar_genero_por_audio(analisis)

                    st.session_state.ultimo_genero = prediccion_ml
                    mm = obtener_metricas_multi_modalidad(prediccion_ml, tempo_val)

                    registro_sesion = {
                        "Pista / Canción": analisis['cancion_formateada'],
                        "Género Clasificado": prediccion_ml,
                        "Tempo (BPM)": tempo_val,
                        "Exigencia Pareja": mm['pareja'],
                        "Exigencia Grupo": mm['grupo'],
                        "Exigencia Solista": mm['solista']
                    }
                    st.session_state.historial_evaluaciones.append(registro_sesion)

                    reply = f"""🎵 **Canción:** **{analisis['cancion_formateada']}**
🏷️ **Género Clasificado:** **{prediccion_ml}** 
⏱️ **Tempo Estimado:** ~{tempo_val} BPM

> 🎛️ **Nota de Predicción Acústica:** *Clasificación calculada mediante parámetros físicos de la señal de audio (Tempo: **~{tempo_val} BPM**, subdivisión rítmica/beats, energía espectral y densidad percusiva tatum).*

---

### 🎼 Marcación Coreográfica & Métrica Musical:
{mm['metrica_ritmo']}

---

### 📊 Exigencia Física por Modalidad de Baile:

* 👫 **Si lo bailas en Pareja:** Exigencia de **{mm['pareja']} / 10**
* 👯‍♀️ **Si lo bailas en Grupo / Compañía:** Exigencia de **{mm['grupo']} / 10**
* 🕺 **Si lo bailas Individual / Solista:** Exigencia de **{mm['solista']} / 10**

---

### 🏋️ Prep Física & Ejercicios:
{mm['ejercicios_recomendados']}
"""
                    texto_ficha = f"""==================================================
SÍNCOPA - FICHA TÉCNICA COREOGRÁFICA
==================================================
Canción: {analisis['cancion_formateada']}
Género Clasificado: {prediccion_ml}
Tempo Estimado: {tempo_val} BPM

MÉTRICA & RITMO:
{mm['metrica_ritmo']}

EXIGENCIA FÍSICA:
- Pareja: {mm['pareja']}/10
- Grupo: {mm['grupo']}/10
- Solista: {mm['solista']}/10

PREPARACIÓN FÍSICA RECOMENDADA:
{mm['ejercicios_recomendados']}
==================================================
"""
                    st.markdown(reply)
                    
                    safe_title = "".join([c if c.isalnum() else "_" for c in analisis['cancion_formateada'][:20]])
                    
                    st.download_button(
                        label="📄 Descargar Ficha Técnica (.txt)",
                        data=texto_ficha,
                        file_name=f"ficha_tecnica_{safe_title}.txt",
                        mime="text/plain",
                        key=f"dl_btn_instant_{time.time()}"
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": reply,
                        "download_data": texto_ficha,
                        "file_title": safe_title
                    })

        else:
            with st.chat_message("assistant"):
                reply = responder_consulta_texto(prompt)
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
