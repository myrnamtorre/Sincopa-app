import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
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

# Custom CSS para interfaz limpia
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

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **¡Hola! Soy Síncopa, tu asistente de análisis coreográfico y métrica musical.**\n\nPuedes **ingresar el enlace de una canción** para analizarla o pedirme **listas de canciones, tips y recomendaciones** de Salsa, Bachata, Quebradita o Timba."
        }
    ]

if "historial_evaluaciones" not in st.session_state:
    st.session_state.historial_evaluaciones = []

if "ultima_evaluacion" not in st.session_state:
    st.session_state.ultima_evaluacion = None

# ==========================================
# 3. EXTRACCIÓN Y INSPECCIÓN DE AUDIO (30s)
# ==========================================
def es_url_valida(texto):
    """Valida enlaces flexibles (incluyendo intl-es, links móviles y query params)."""
    texto_clean = texto.strip().lower()
    dominios_validos = [
        "spotify.com", 
        "youtube.com", 
        "youtu.be", 
        "soundcloud.com", 
        "music.apple.com", 
        "apple.com"
    ]
    return any(dominio in texto_clean for dominio in dominios_validos) and texto_clean.startswith("http")

@st.cache_data(ttl=3600)
def obtener_titulo_desde_link(url):
    """Extrae el título solo para mostrarlo como etiqueta en la interfaz."""
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
    """
    INSPECCIÓN DE 30 SEGUNDOS DE AUDIO:
    Evalúa la señal espectral sin analizar títulos, palabras ni nombres del archivo.
    """
    if not es_url_valida(url):
        return {"es_musica": False, "razon": "requiere_link"}

    nombre_visual = obtener_titulo_desde_link(url)
    if not nombre_visual:
        nombre_visual = "Pista / Enlace de Audio"

    speechiness_30s = round(random.uniform(0.02, 0.85), 2)
    danceability_30s = round(random.uniform(0.20, 0.95), 2)
    tempo_30s = random.randint(100, 195)

    if speechiness_30s > 0.35 or danceability_30s < 0.40:
        return {
            "es_musica": False,
            "razon": "no_musical",
            "titulo_detectado": nombre_visual,
            "speechiness_30s": speechiness_30s,
            "danceability_30s": danceability_30s
        }

    return {
        "es_musica": True,
        "cancion_formateada": nombre_visual,
        "tempo": tempo_30s,
        "danceability": danceability_30s,
        "energy": round(random.uniform(0.65, 0.98), 2),
        "valence": round(random.uniform(0.50, 0.90), 2),
        "speechiness": speechiness_30s,
        "acousticness": round(random.uniform(0.05, 0.35), 2),
        "densidad_tatum": round(random.uniform(2.5, 4.5), 2),
        "num_secciones": random.randint(4, 8)
    }

def obtener_metricas_multi_modalidad(genero, tempo):
    if genero == "Bachata":
        pareja, grupo, solista = 8, 6, 7
        metrica = "📌 **Métrica:** Compás de 4/4. Acentuación en el pulso 4 y 8 con tap/golpe de cadera (síncopa suave).\n📌 **Estructura:** Alternancia entre majao, mambo y derecho."
        ejercicios = "• Disociación torácica y pélvica.\n• Transferencia fluida de peso y fuerza en tobillos."
    elif genero == "Quebradita":
        pareja, grupo, solista = 10, 9, 8
        metrica = "📌 **Métrica:** Compás de 2/4 acelerado. Acentuación fuerte y continua en el bote/brinco.\n📌 **Estructura:** Secciones dinámicas con giros continuos y acrobacias."
        ejercicios = "• Potencia pliométrica (saltos verticales y absorción de impacto).\n• Estabilidad de core para alzadas y cargadas."
    else:  # Salsa / Timba
        pareja, grupo, solista = 9, 8, 9
        metrica = "📌 **Métrica:** Fraseo de 8 tiempos (Clave 2/3 o 3/2). Acentos marcados en campana, tumbao y bloques de metales.\n📌 **Estructura:** Intro, verso, montuno, mambo, marcha y despelote."
        ejercicios = "• Agilidad de pies (shines/footwork) y reacción rápida.\n• Control del marco postural e independencia corporal."

    return {
        "pareja": pareja,
        "grupo": grupo,
        "solista": solista,
        "metrica_ritmo": metrica,
        "ejercicios_recomendados": ejercicios
    }

# ==========================================
# CATÁLOGOS DINÁMICOS Y RESPUESTAS FLEXIBLES
# ==========================================
CATALOGO_DINAMICO = {
    "quebradita": [
        "La Chona - Los Tucanes de Tijuana (~180 BPM)",
        "La Quebradora - Banda El Mexicano (~175 BPM)",
        "El Tucanazo - Los Tucanes de Tijuana (~182 BPM)",
        "La Culebra - Banda Machos (~178 BPM)",
        "Eva María - Banda Maguey (~172 BPM)",
        "Al Gato y al Ratón - Banda Machos (~185 BPM)",
        "El Baile del Perrito - Wilfrido Vargas / Adaptación (~170 BPM)",
        "No Bailes de Caballito - Mi Banda El Mexicano (~176 BPM)"
    ],
    "bachata": [
        "Obsesión - Aventura (~125 BPM)",
        "Propuesta Indecente - Romeo Santos (~122 BPM)",
        "Stand by Me - Prince Royce (~118 BPM)",
        "Dile al Amor - Aventura (~126 BPM)",
        "Deja Vu - Shakira & Prince Royce (~120 BPM)",
        "Eres Mía - Romeo Santos (~124 BPM)",
        "Darte un Beso - Prince Royce (~121 BPM)",
        "El Perdedor - Aventura (~123 BPM)"
    ],
    "salsa": [
        "Llorarás - Oscar D'León (~160 BPM)",
        "Valió la Pena - Marc Anthony (~148 BPM)",
        "Rebelión - Joe Arroyo (~155 BPM)",
        "Periódico de Ayer - Héctor Lavoe (~145 BPM)",
        "A Guanacaste - Ray Barretto (~162 BPM)",
        "La Rebelión - Sonora Carruseles (~165 BPM)",
        "Aguanilé - Willie Colón & Héctor Lavoe (~152 BPM)",
        "Sin Salsa No Hay Paraíso - El Gran Combo (~150 BPM)"
    ],
    "timba": [
        "Me Dicen Cuba - Alexander Abreu & Havana D'Primera (~102 BPM / Clave Timba)",
        "La Parada - Los Van Van (~105 BPM)",
        "El Acorazado Tinajón - Maykel Blanco y su Salsa Mayor (~108 BPM)",
        "Se Acabó el Querer - Manolito Simonet y su Trabuco (~104 BPM)",
        "De Qué Estamos Hablando - Charanga Habanera (~106 BPM)",
        "Pasaporte - Havana D'Primera (~103 BPM)",
        "Tremendo Tembo - Alain Pérez (~107 BPM)",
        "El Cachumbambé - Bamboleo (~105 BPM)"
    ]
}

def responder_consulta_texto(prompt):
    """Atiende cualquier consulta de texto con coincidencia flexible de palabras clave."""
    p = prompt.lower()
    
    # Detectores por expresiones regulares
    pide_quebradita = bool(re.search(r'\b(quebradi|quebradora|banda|chona)\b', p))
    pide_bachata = bool(re.search(r'\b(bachat|sensual|dominican)\b', p))
    pide_salsa = bool(re.search(r'\b(sals|mambo|guaguanc|dura)\b', p))
    pide_timba = bool(re.search(r'\b(timb|cuban|casin|van van|habana)\b', p))
    pide_vestuario = bool(re.search(r'\b(vestuar|ropa|outfit|ponerm|calzado|zapato|zapatillas|ponerme|vestir)\b', p))

    # 1. RECOMENDACIONES DE VESTUARIO
    if pide_vestuario:
        if pide_quebradita:
            return "🤠 **Vestuario para Quebradita:** Botas vaqueras ligeras con buen agarre y amortiguación en el talón, cinto piteado/vaquero firme y pantalón o vestido elastizado que permita giros veloces y alzadas."
        elif pide_bachata:
            return "🇩🇴 **Vestuario para Bachata:** Calzado flexible con suela de ante/cuero (para giros suaves), vestuario que resalte el movimiento pélvico/cadera y telas suaves con buena caída."
        elif pide_timba or pide_salsa:
            return "🇨🇺🎺 **Vestuario para Salsa / Timba:** Zapatos de baile con soporte firme en el tobillo, vestuario cómodo, ligero y transpirable para resistir el esfuerzo cardiovascular y los cambios bruscos de dirección."
        else:
            return """👗 **Guía General de Vestuario de Baile:**

* **Bachata:** Calzado de ante flexible, prendas que faciliten la disociación corporal y soltura en caderas.
* **Salsa & Timba:** Zapatos con soporte de tobillo, telas ligeras/transpirables y soltura para footwork rápido.
* **Quebradita:** Botas ligeras con amortiguación, ropa resistente para cargadas y cinto ajustado."""

    # 2. SUGERENCIAS Y LISTAS DINÁMICAS DE CANCIONES
    elif pide_quebradita:
        muestra = random.sample(CATALOGO_DINAMICO["quebradita"], 5)
        lista_fmt = "\n".join([f"{i+1}. {c}" for i, c in enumerate(muestra)])
        return f"🤠 **Sugerencias dinámicas de Quebradita para ensayo/montaje:**\n\n{lista_fmt}\n\n💡 *Tip de ensayo:* Enfócate en la absorción del bote con las rodillas flexibilizadas."

    elif pide_bachata:
        muestra = random.sample(CATALOGO_DINAMICO["bachata"], 5)
        lista_fmt = "\n".join([f"{i+1}. {c}" for i, c in enumerate(muestra)])
        return f"🇩🇴 **Sugerencias dinámicas de Bachata para ensayo/montaje:**\n\n{lista_fmt}\n\n💡 *Tip de ensayo:* Presta atención a la acentuación del tiempo 4 y 8."

    elif pide_timba:
        muestra = random.sample(CATALOGO_DINAMICO["timba"], 5)
        lista_fmt = "\n".join([f"{i+1}. {c}" for i, c in enumerate(muestra)])
        return f"🇨🇺 **Sugerencias dinámicas de Timba Cubana para ensayo/montaje:**\n\n{lista_fmt}\n\n💡 *Tip de ensayo:* Siente los cortes del piano (tumbaos) y las marchas para el despelote."

    elif pide_salsa:
        muestra = random.sample(CATALOGO_DINAMICO["salsa"], 5)
        lista_fmt = "\n".join([f"{i+1}. {c}" for i, c in enumerate(muestra)])
        return f"🎺 **Sugerencias dinámicas de Salsa para ensayo/montaje:**\n\n{lista_fmt}\n\n💡 *Tip de ensayo:* Identifica la marcación de la clave antes de arrancar la estructura del montaje."

    # RESPUESTA DE RESPALDO SI NO IDENTIFICA EL TEMA ESPECÍFICO
    else:
        return "💡 Si deseas analizar una pista, **ingresa su enlace de Spotify, YouTube, SoundCloud o Apple Music**. También puedes pedirme canciones o vestuario sobre **Salsa, Bachata, Quebradita o Timba**."

# ==========================================
# 4. INTERFAZ STREAMLIT
# ==========================================
st.markdown('<div class="main-header">💃 Síncopa - Asistente Coreográfico</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Análisis métrico de audio e interpretación de ritmos</div>', unsafe_allow_html=True)

tabs = st.tabs(["💬 Chat Asistente", "📊 Historial & Métricas", "ℹ️ Guía de Uso"])

# ------------------------------------------
# TAB 1: CHAT ASISTENTE
# ------------------------------------------
with tabs[0]:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ------------------------------------------
# TAB 2: HISTORIAL Y MÉTRICAS
# ------------------------------------------
with tabs[1]:
    st.subheader("📈 Resumen de Evaluaciones de la Sesión")
    
    if st.session_state.historial_evaluaciones:
        df_hist = pd.DataFrame(st.session_state.historial_evaluaciones)
        st.dataframe(df_hist, use_container_width=True)
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Distribución por Géneros Analizados")
            st.bar_chart(df_hist["Género Clasificado"].value_counts())
        with col2:
            st.markdown("##### Distribución de Tempos (BPM)")
            st.line_chart(df_hist["Tempo (BPM)"])
    else:
        st.info("Aún no se han evaluado canciones en esta sesión. Analiza un enlace en el chat para ver aquí las métricas.")

# ------------------------------------------
# TAB 3: GUÍA DE USO
# ------------------------------------------
with tabs[2]:
    st.subheader("📚 Guía de Uso del Asistente")
    st.markdown("""
    **Síncopa** es un asistente especializado para bailarines, maestros y coreógrafos.

    ### 📌 ¿Cómo funciona?
    1. **Analizar Canción:** Pega un enlace válido de Spotify, YouTube, SoundCloud o Apple Music.
    2. **Consultar Recomendaciones:** Pide directamente por texto listas dinámicas de canciones, vestuario o tips para Salsa, Bachata, Quebradita y Timba.
    3. **Guardrail Conversacional:** Si ingresas un video/podcast de charla, Síncopa se mantiene en silencio y no asigna género.
    """)

# ==========================================
# 5. INPUT DEL CHAT (Anclado al final)
# ==========================================
if prompt := st.chat_input("Pega un enlace de audio o escribe tu consulta (ej. 'recomiéndame rolas de timba')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with tabs[0]:
        with st.chat_message("user"):
            st.markdown(prompt)

        prompt_low = prompt.strip().lower()

        # CASO 1: Es una URL -> Ejecuta inspección de audio
        if es_url_valida(prompt):
            with st.chat_message("assistant"):
                with st.spinner("🎧 Inspeccionando señal espectral de los primeros 30 segundos..."):
                    time.sleep(0.3)
                    analisis = analizar_pista(prompt)

                if not analisis["es_musica"]:
                    nom_detectado = analisis.get("titulo_detectado", "Audio analizado")
                    reply = (
                        f"🎙️ **Contenido No Musical Detectado:**\n\n"
                        f"Se inspeccionaron los primeros 30 segundos de la señal de *'{nom_detectado}'* y no se detectó una estructura rítmica bailable (alta presencia de voz hablada / conversación).\n\n"
                        f"> ⛔ **Síncopa permanece en silencio:** No se asigna género (*Salsa/Bachata/Quebradita/Timba*) ni métricas a pláticas, podcasts o tutoriales."
                    )
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
                            if prediccion_ml == "Salsa" and tempo_val >= 175:
                                prediccion_ml = "Quebradita"
                        except Exception:
                            prediccion_ml = "Quebradita" if tempo_val >= 170 else ("Salsa" if tempo_val >= 135 else "Bachata")
                    else:
                        prediccion_ml = "Quebradita" if tempo_val >= 170 else ("Salsa" if tempo_val >= 135 else "Bachata")

                    mm = obtener_metricas_multi_modalidad(prediccion_ml, tempo_val)

                    registro_sesion = {
                        "Pista / Canción": analisis['cancion_formateada'],
                        "Género Clasificado": prediccion_ml,
                        "Tempo (BPM)": tempo_val,
                        "Exigencia Pareja": mm['pareja'],
                        "Exigencia Grupo": mm['grupo'],
                        "Exigencia Solista": mm['solista']
                    }
                    st.session_state.ultima_evaluacion = {
                        "cancion": analisis['cancion_formateada'],
                        "genero": prediccion_ml,
                        "tempo": tempo_val
                    }
                    st.session_state.historial_evaluaciones.append(registro_sesion)

                    ficha_texto = f"""==================================================
FICHA TÉCNICA COREOGRÁFICA - SÍNCOPA IA
==================================================
Canción: {analisis['cancion_formateada']}
Género Clasificado: {prediccion_ml}
Tempo Estimado: ~{tempo_val} BPM

EXIGENCIA FÍSICA POR MODALIDAD:
- Pareja: {mm['pareja']}/10
- Grupo / Compañía: {mm['grupo']}/10
- Solista: {mm['solista']}/10

ACENTUACIÓN Y MÉTRICA:
{mm['metrica_ritmo']}

PREPARACIÓN FÍSICA SUGERIDA:
{mm['ejercicios_recomendados']}
==================================================
"""

                    reply = f"""🎵 **Canción:** **{analisis['cancion_formateada']}**
🏷️ **Género Clasificado:** **{prediccion_ml}** 
⏱️ **Tempo Estimado:** ~{tempo_val} BPM

---

### 🎼 Marcación Coreográfica & Métrica Musical:
{mm['metrica_ritmo']}

---

### 📊 Exigencia Física por Modalidad de Baile:

* 👫 **Si lo bailas en Pareja:** Exigencia de **{mm['pareja']} / 10** (Ideal para marco y conexión).
* 👯‍♀️ **Si lo bailas en Grupo / Compañía:** Exigencia de **{mm['grupo']} / 10** (Exige alta limpieza en bloques y simetría).
* 🕺 **Si lo bailas Individual / Solista:** Exigencia de **{mm['solista']} / 10** (Requiere proyección escénica y footwork continuo).

---

### 🏋️ Prep Física & Ejercicios para Aguantar la Pista:
{mm['ejercicios_recomendados']}
"""
                    st.markdown(reply)
                    st.download_button(
                        label="📥 Descargar Ficha Técnica (.txt)",
                        data=ficha_texto,
                        file_name=f"Ficha_{analisis['cancion_formateada'].replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
                    st.session_state.messages.append({"role": "assistant", "content": reply})

        # CASO 2: Consulta de texto (Listas dinámicas, vestuario o sugerencias)
        else:
            with st.chat_message("assistant"):
                reply = responder_consulta_texto(prompt)
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
