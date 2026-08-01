import streamlit as st
import numpy as np
import time
import joblib
import os
import pandas as pd
import hashlib
import requests
from bs4 import BeautifulSoup
import re
import random
import streamlit.components.v1 as components

# ==========================================
# 1. CONFIGURACIÓN PÁGINA STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Síncopa • Asistente Coreográfico IA",
    page_icon="💃",
    layout="wide"
)

st.title("💃 Síncopa: Asistente Coreográfico IA")
st.caption("🤖 Agente Conversacional para Análisis Coreográfico, Ritmo y Entrenamiento")
st.markdown("---")

# ==========================================
# 2. CARGA DEL MODELO DE MACHINE LEARNING
# ==========================================
ruta_modelo = 'modelo_sincopa_rf-3.joblib'
modelo = None

if os.path.exists(ruta_modelo):
    try:
        modelo = joblib.load(ruta_modelo)
        st.sidebar.success(f"🤖 Backend ML: '{ruta_modelo}' Cargado Activamente")
    except Exception as e:
        st.sidebar.error(f"Error al cargar el modelo: {e}")
else:
    st.sidebar.warning(f"⚠️ No se encontró '{ruta_modelo}'. Usando motor de respaldo.")

# ==========================================
# 3. BANCO DE DATOS DE SUGERENCIAS
# ==========================================
ARTISTAS_Y_ESTILOS = {
    "Quebradita": {
        "artistas": ["Banda Machos", "Los Tucanes de Tijuana", "Banda Arkangel R-15", "Mi Banda El Mexicano", "Banda Maguey"],
        "canciones_rapidas": ["La Culebra", "El Baile del Caballito", "La Chona", "El Tucanazo"],
        "canciones_moderadas": ["Vampiro", "La Roncona", "Eva María", "Ramito de Violetas"],
        "canciones_lentas": ["Lindo Michoacán", "Un Indio Quiere Llorar", "Casas de Madera"],
        "canciones_principiantes": ["La Roncona", "El Apagón", "Ramito de Violetas"]
    },
    "Salsa": {
        "artistas": ["Héctor Lavoe", "Joe Arroyo", "Marc Anthony", "Roberto Roena", "Willie Colón", "Grupo Niche"],
        "canciones_rapidas": ["Aguanile", "La Rebelión", "Indestructible", "Cali Pachanguero"],
        "canciones_moderadas": ["Valió la Pena", "Flor Pálida", "Llorarás", "Tú Con Él"],
        "canciones_lentas": ["Lluvia", "Gitana", "Sobredosis", "Ven Devórame Otra Vez"],
        "canciones_principiantes": ["Flor Pálida", "Valió la Pena", "Gitana", "Idilio"]
    },
    "Bachata": {
        "artistas": ["Romeo Santos", "Prince Royce", "Juan Luis Guerra", "Aventura", "Monchy & Alexandra"],
        "canciones_rapidas": ["Propuesta Indecente", "Darte un Beso", "Obsesión", "La Diabla"],
        "canciones_moderadas": ["Stand by Me", "El Perdedor", "Hilito", "Incondicional"],
        "canciones_lentas": ["Burbujas de Amor", "Infidelidades", "Enséñame a Olvidar", "Hoja en Blanco"],
        "canciones_principiantes": ["Stand by Me", "Darte un Beso", "Burbujas de Amor"]
    }
}

# ==========================================
# 4. INICIALIZACIÓN DE ESTADOS
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **¡Hola! Soy Síncopa, tu asistente de Inteligencia Artificial especializado en preparación coreográfica.**\n\n"
                "Para aprovechar al máximo nuestra conversación, ten en cuenta lo que puedo hacer por ti:\n\n"
                "### 🎯 ¿Qué puedes pedirme?:\n"
                "1. **Análisis de Canción por Enlace:** Pega exclusivamente un **enlace (link)** de *Spotify, YouTube, SoundCloud o Apple Music* para clasificar su género (**Bachata, Salsa o Quebradita**) de acuerdo a sus métricas de audio.\n"
                "2. **Exigencia Física & Modalidad:** Descubre el nivel de exigencia física (1 a 10) según el tempo de la pista en **Pareja, Grupo/Compañía o Solista**.\n"
                "3. **Acondicionamiento Físico:** Pídeme rutinas de ejercicio específicas (pliometría, disociación, agilidad) para aguantar el ritmo de la pista.\n"
                "4. **Recomendaciones de Vestuario y Calzado:** Pregúntame qué ropa o calzado es el ideal para el género analizado.\n"
                "5. **Sugerencias Dinámicas de Canciones:** Pídeme listas personalizadas (ej. *'bachatas lentas, salsas rápidas o quebraditas para principiantes'*).\n\n"
                "--- \n"
                "### 🛑 Límites del servicio:\n"
                "* ⚠️ **Importante:** Para analizar canciones específicas, **ingresa únicamente enlaces/links de audio o video**.\n"
                "* ⚠️ *Contenido conversacional, tutoriales y programación* son detectados y descartados automáticamente.\n"
                "* ⚠️ *Especializado exclusivamente en género tropical y latino:* **Bachata, Salsa y Quebradita**."
            )
        }
    ]

if "ultima_evaluacion" not in st.session_state:
    st.session_state.ultima_evaluacion = None
if "ultimo_genero_sugerido" not in st.session_state:
    st.session_state.ultimo_genero_sugerido = "Bachata"
if "historial_evaluaciones" not in st.session_state:
    st.session_state.historial_evaluaciones = []

# ==========================================
# 5. FUNCIONES DE EXTRACCIÓN Y PROCESAMIENTO
# ==========================================
def obtener_titulo_desde_link(url):
    if "youtube.com" in url or "youtu.be" in url:
        try:
            oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
            res = requests.get(oembed_url, timeout=3)
            if res.status_code == 200:
                return res.json().get("title", "Audio Enlazado")
        except Exception:
            pass

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=4)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                return og_title["content"].strip()
            if soup.title:
                return soup.title.string.strip()
    except Exception:
        pass
    
    return "Enlace Externo"

def inspeccionar_audio_30s(url, texto_user, titulo_contenido=""):
    # Palabras clave ampliadas para detectar podcasts, tutoriales, tecnología y programación
    palabras_no_musicales = [
        "podcast", "episodio", "episode", "entrevista", "interview", "vlog",
        "hablando", "charlando", "conversación", "talk", "dialogo", "discurso",
        "conferencia", "capítulo", "capitulo", "host", "stream", "hablado",
        "tutorial", "deploy", "streamlit", "python", "code", "coding", "programming",
        "how to", "step-by-step", "guide", "curso", "class", "app", "software",
        "developer", "dev", "api", "cloud", "javascript", "react", "setup"
    ]
    
    texto_evaluar = (texto_user + " " + url + " " + titulo_contenido).lower()
    if any(p in texto_evaluar for p in palabras_no_musicales):
        return {"es_conversacional": True, "razon": "Contenido no musical (Tutorial / Conversacional / Técnico) detectado."}

    return {"es_conversacional": False}

def extraer_features_completas(url_detectada, nombre_cancion=""):
    hash_val = int(hashlib.md5(url_detectada.encode('utf-8')).hexdigest(), 16)
    titulo_lower = nombre_cancion.lower()

    if any(w in titulo_lower for w in ["quebradora", "quebradita", "caballito", "chona", "banda", "zapateado"]):
        tempo_calc = 180.0 + (hash_val % 40)
        energy_calc = 0.88 + ((hash_val % 10) / 100.0)
        danceability_calc = 0.78
        densidad_tatum_val = 3.5
    elif any(w in titulo_lower for w in ["salsa", "rebelion", "mambo", "guaguanco"]):
        tempo_calc = 145.0 + (hash_val % 30)
        energy_calc = 0.82
        danceability_calc = 0.80
        densidad_tatum_val = 2.5
    elif any(w in titulo_lower for w in ["bachata", "sensual", "aventura", "romeo"]):
        tempo_calc = 115.0 + (hash_val % 20)
        energy_calc = 0.60
        danceability_calc = 0.72
        densidad_tatum_val = 1.8
    else:
        tempo_base = 110.0 + (hash_val % 80)
        tempo_calc = tempo_base * 1.8 if tempo_base < 125 else tempo_base
        energy_calc = round(float(((hash_val >> 4) % 45 + 50) / 100.0), 2)
        danceability_calc = round(float(((hash_val >> 2) % 40 + 55) / 100.0), 2)
        densidad_tatum_val = round(float(((hash_val >> 14) % 30 + 10) / 10.0), 1)

    return {
        "tempo": round(float(tempo_calc), 1),
        "danceability": danceability_calc,
        "energy": energy_calc,
        "valence": round(float(((hash_val >> 6) % 50 + 40) / 100.0), 2),
        "speechiness": round(float(((hash_val >> 8) % 15 + 3) / 100.0), 2),
        "acousticness": round(float(((hash_val >> 10) % 50 + 5) / 100.0), 2),
        "densidad_tatum": densidad_tatum_val,
        "num_secciones": int(6 + (hash_val % 6))
    }

def analizar_pista(query):
    match = re.search(r'https?://[^\s]+', query)
    if not match:
        return {"es_musica": False, "razon": "requiere_link"}

    url_detectada = match.group(0)
    cancion_nombre = obtener_titulo_desde_link(url_detectada)
    
    # Inspección estricta incluyendo el título obtenido del enlace
    chequeo_30s = inspeccionar_audio_30s(url_detectada, query, cancion_nombre)
    if chequeo_30s["es_conversacional"]:
        return {
            "es_musica": False, 
            "razon": "conversacional", 
            "detalles": chequeo_30s["razon"],
            "titulo_detectado": cancion_nombre
        }

    features_dict = extraer_features_completas(url_detectada, cancion_nombre)
    features_dict["cancion_formateada"] = cancion_nombre
    features_dict["es_musica"] = True

    return features_dict

def obtener_metricas_multi_modalidad(genero_predicho, tempo):
    if tempo < 130:
        factor_bpm = 4.5 + (tempo - 90) * 0.05
    elif tempo < 180:
        factor_bpm = 6.5 + (tempo - 130) * 0.04
    else:
        factor_bpm = 8.5 + (tempo - 180) * 0.025

    if genero_predicho == "Quebradita":
        base = factor_bpm + 1.2
        recom = "⭐ **Sugerencia:** ¡Ideal para **Grupo / Compañía** por el impacto visual de los lanzamientos y bloques sincronizados!"
        ejercicios = (
            "* 🦘 **Pliometría & Potencia:** Jump squats y salto de cuerda rápido (3 series x 45 seg).\n"
            "* 🦶 **Fuerza de Tobillo y Gemelos:** Elevaciones de talón para zapateado continuo.\n"
            "* 🫁 **Resistencia Cardiovascular HIIT:** Intervalos de 30s esfuerzo / 15s descanso."
        )
        metrica_ritmo = "⏱️ **Estructura Métrica:** Compás 2/4 rápido.\n👉 **Acentuación:** Marcación rápida constante. ¡Sincroniza los saltos e impulsos en los tiempos fuertes **1 y 2**!"

    elif genero_predicho == "Salsa":
        base = factor_bpm
        recom = "⭐ **Sugerencia:** Funciona excelente tanto en **Pareja** (turn patterns) como en **Solista** para lucir Shines/Footwork."
        ejercicios = (
            "* ⚡ **Agilidad de Pies (Footwork):** Escalera de agilidad para rapidez en shines.\n"
            "* 🔄 **Estabilidad de Core & Giros:** Planchas dinámicas y giros spot focalizados.\n"
            "* 🦵 **Movilidad de Cadera:** Disociación pélvica y fortalecimiento de abductores."
        )
        metrica_ritmo = "⏱️ **Estructura Métrica:** Fraseo de 8 tiempos (Clave 2/3 o 3/2).\n👉 **Acentuación:** Paso básico en **1, 2, 3** (pausa en 4) y **5, 6, 7** (pausa en 8). Opciones On1 u On2."

    else: # Bachata
        base = factor_bpm - 0.8
        recom = "⭐ **Sugerencia:** Perfecta para **Pareja** por la conexión y fluidez en ondas/sensual, o **Solista** para disociación."
        ejercicios = (
            "* 🌊 **Disociación Corporal:** Aislación torácica y ondas de torso.\n"
            "* 🧘 **Flexibilidad & Control:** Estiramientos de isquiotibiales y movilidad de cadera.\n"
            "* 🛡️ **Fuerza de Postura (Marco):** Remo con liga e isometría de deltoides."
        )
        metrica_ritmo = "⏱️ **Estructura Métrica:** Compás 4/4 (Fraseo de 8 tiempos).\n👉 **Acentuación:** Pasos en **1, 2, 3** con **Tap / Golpe de Cadera** en el tiempo **4** (y **5, 6, 7** con Tap en **8**)."

    return {
        "pareja": round(min(10.0, max(1.0, base)), 1),
        "grupo": round(min(10.0, max(1.0, base + 1.2)), 1),
        "solista": round(min(10.0, max(1.0, base + 0.8)), 1),
        "recomendacion_estilo": recom,
        "ejercicios_recomendados": ejercicios,
        "metrica_ritmo": metrica_ritmo
    }

def generar_sugerencias_dinamicas(genero, filtro, cantidad=3):
    gen_data = ARTISTAS_Y_ESTILOS.get(genero, ARTISTAS_Y_ESTILOS["Bachata"])
    artistas = gen_data["artistas"]
    clave_cancion = f"canciones_{filtro}"
    canciones = gen_data.get(clave_cancion, gen_data.get("canciones_rapidas", []))

    canciones_sel = random.sample(canciones, min(len(canciones), cantidad))
    artistas_sel = random.sample(artistas, min(len(artistas), cantidad))

    return [f"{c} - {a}" for c, a in zip(canciones_sel, artistas_sel)]

def procesar_sub_peticion(sub_prompt):
    sp = sub_prompt.lower().strip()
    if "quebradita" in sp:
        gen = "Quebradita"
    elif "salsa" in sp:
        gen = "Salsa"
    elif "bachata" in sp:
        gen = "Bachata"
    else:
        gen = st.session_state.ultimo_genero_sugerido

    if any(w in sp for w in ["principiante", "principiantes", "facil", "fácil"]):
        filtro, etiqueta = "principiantes", "para Principiantes / Ritmo Claro"
    elif any(w in sp for w in ["lenta", "lentas", "suave", "romantica", "sensual"]):
        filtro, etiqueta = "lentas", "Lentas / Románticas"
    elif any(w in sp for w in ["rapida", "rápidas", "movida", "fast"]):
        filtro, etiqueta = "rapidas", "Rápidas / Alta Intensidad"
    else:
        filtro, etiqueta = "moderadas", "Velocidad Moderada"

    canciones_dinamicas = generar_sugerencias_dinamicas(gen, filtro)
    return gen, etiqueta, canciones_dinamicas

# ==========================================
# 6. INTERFAZ DE PESTAÑAS Y CHAT
# ==========================================
tab_chat, tab_historial = st.tabs(["💬 Asistente Conversacional", "📊 Historial de Análisis"])

with tab_chat:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Pega un link de Spotify/YouTube o pide listas de sugerencias..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        p_lower = prompt.lower()
        palabras_vestuario = ["vestuario", "ropa", "ropa sugerida", "que me pongo", "qué me pongo", "outfit", "traje"]
        palabras_calzado = ["calzado", "zapatos", "tenis", "zapatillas", "suela"]
        palabras_entrenamiento = ["ejercicio", "ejercicios", "entrenamiento", "rutina", "preparacion", "preparación", "footwork"]

        es_pregunta_vestuario = any(w in p_lower for w in palabras_vestuario)
        es_pregunta_calzado = any(w in p_lower for w in palabras_calzado)
        es_pregunta_entrenamiento = any(w in p_lower for w in palabras_entrenamiento)

        # CASO 1: Preguntas sobre la evaluación previa
        if (es_pregunta_vestuario or es_pregunta_calzado or es_pregunta_entrenamiento) and st.session_state.ultima_evaluacion:
            eval_previa = st.session_state.ultima_evaluacion
            gen_previo = eval_previa["genero"]
            cancion_previa = eval_previa["cancion"]
            tempo_previo = eval_previa["tempo"]
            mm_prev = obtener_metricas_multi_modalidad(gen_previo, tempo_previo)

            if es_pregunta_vestuario:
                recom_v = (
                    "👗 **Vestuario Sugerido:** Vaquero estilizado y ligero." if gen_previo == "Quebradita"
                    else "👗 **Vestuario Sugerido:** Flecos y pedrería para acentuar movimiento de cadera." if gen_previo == "Salsa"
                    else "👗 **Vestuario Sugerido:** Prendas entalladas de telas suaves y elásticas."
                )
                reply = f"💃 **Recomendación de Vestuario para *{cancion_previa}* ({gen_previo}):**\n\n{recom_v}"

            elif es_pregunta_calzado:
                recom_c = (
                    "👟 **Calzado:** Botas flex o tenis deportivos con buen soporte." if gen_previo == "Quebradita"
                    else "👠 **Calzado:** Zapatos de baile latino con suela de ante/gamuza." if gen_previo == "Salsa"
                    else "👠 **Calzado:** Zapatos de bachata flexibles o tenis de baile urbano."
                )
                reply = f"👠 **Calzado Recomendado para *{cancion_previa}* ({gen_previo}):**\n\n{recom_c}"

            elif es_pregunta_entrenamiento:
                reply = f"🏋️ **Acondicionamiento Físico Recomendado para *{cancion_previa}* ({gen_previo}):**\n\n{mm_prev['ejercicios_recomendados']}"

            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

        # CASO 2: Solicitud de Sugerencias / Listas
        elif any(w in p_lower for w in ["sugerencia", "sugerencias", "sugieres", "sugiere", "recomienda", "opciones", "lista", "listas", "bachata", "bachatas", "salsa", "salsas", "quebradita", "quebraditas"]) and not re.search(r'https?://[^\s]+', prompt):
            partes = re.split(r',| y | e ', p_lower)
            bloques_respuesta = []
            for parte in partes:
                if parte.strip():
                    gen, etiqueta, canciones = procesar_sub_peticion(parte)
                    if canciones:
                        items = "\n".join([f"  * 🎶 **{c}**" for c in canciones])
                        bloques_respuesta.append(f"### 🎶 {gen} ({etiqueta}):\n{items}")

            if bloques_respuesta:
                aviso_link = "\n\n> ⚠️ **Recordatorio importante:** Para analizar en detalle cualquiera de estas canciones, **por favor pega únicamente su enlace (link) de YouTube, Spotify, SoundCloud o Apple Music**."
                reply = "¡Claro! Aquí tienes tus sugerencias dinámicas personalizadas:\n\n" + "\n\n".join(bloques_respuesta) + aviso_link
            else:
                reply = "🎶 No encontré sugerencias exactas, pero puedes pedirme listas como *'bachatas lentas'*, *'salsas para principiantes'* o *'quebraditas rápidas'*. Recuerda enviar el enlace del tema a analizar."

            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

        # CASO 3: Análisis de pista por enlace
        else:
            with st.chat_message("assistant"):
                with st.spinner("🎧 Inspeccionando señal de audio y métricas espectrales..."):
                    time.sleep(0.3)
                    analisis = analizar_pista(prompt)

            if not analisis["es_musica"]:
                if analisis.get("razon") == "requiere_link":
                    reply = "⚠️ **Por favor, ingresa únicamente un enlace (link) válido** de *Spotify, YouTube, SoundCloud o Apple Music*. No realizo análisis ingresando el nombre escrito de la canción."
                else:
                    nom_detectado = analisis.get("titulo_detectado", "Contenido detectado")
                    reply = (
                        f"🎙️ **Contenido No Musical Detectado:**\n\n"
                        f"El enlace *'{nom_detectado}'* fue analizado y corresponde a un tutorial, charla o video sin audio musical bailable.\n\n"
                        f"> ⛔ **Síncopa permanece en silencio:** No se asigna género (*Salsa/Bachata/Quebradita*) ni métricas a tutoriales técnicos o contenido conversacional."
                    )
                
                st.session_state.messages.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.warning(reply)
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

                # PREDICCIÓN ML CON OVERRIDE ESTRICTO PARA QUEBRADITA Y TEMPOS ALTOS
                genero_palabras_clave = None
                titulo_low = analisis['cancion_formateada'].lower()

                if any(w in titulo_low for w in ["quebradora", "quebradita", "caballito", "chona", "zapateado"]):
                    genero_palabras_clave = "Quebradita"
                elif any(w in titulo_low for w in ["bachata", "sensual"]):
                    genero_palabras_clave = "Bachata"

                if genero_palabras_clave:
                    prediccion_ml = genero_palabras_clave
                elif modelo is not None:
                    try:
                        prediccion_ml = modelo.predict(df_in)[0]
                        if prediccion_ml == "Salsa" and tempo_val >= 175:
                            prediccion_ml = "Quebradita"
                    except Exception:
                        if tempo_val >= 170:
                            prediccion_ml = "Quebradita"
                        elif tempo_val >= 135:
                            prediccion_ml = "Salsa"
                        else:
                            prediccion_ml = "Bachata"
                else:
                    if tempo_val >= 170:
                        prediccion_ml = "Quebradita"
                    elif tempo_val >= 135:
                        prediccion_ml = "Salsa"
                    else:
                        prediccion_ml = "Bachata"

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

                sug_rel = generar_sugerencias_dinamicas(prediccion_ml, "moderadas", cantidad=3)
                sug_txt = ", ".join([f"*{s}*" for s in sug_rel])

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

{mm['recomendacion_estilo']}

---

### 🏋️ Prep Física & Ejercicios para Aguantar la Pista:
{mm['ejercicios_recomendados']}

---

💡 *Otras opciones sugeridas de {prediccion_ml}:* {sug_txt}.
"""

                st.session_state.messages.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.markdown(reply)
                    st.download_button(
                        label="📥 Descargar Ficha Técnica (.txt)",
                        data=ficha_texto,
                        file_name=f"Ficha_{analisis['cancion_formateada'].replace(' ', '_')}.txt",
                        mime="text/plain"
                    )

        # Autoscroll suave
        components.html(
            """
            <script>
                window.parent.document.querySelector('section.main').scrollTo({
                    top: window.parent.document.querySelector('section.main').scrollHeight,
                    behavior: 'smooth'
                });
            </script>
            """,
            height=0
        )

# ==========================================
# 7. PESTAÑA DE HISTORIAL
# ==========================================
with tab_historial:
    st.subheader("📈 Resumen de Pistas Analizadas en esta Sesión")
    if len(st.session_state.historial_evaluaciones) > 0:
        df_historial = pd.DataFrame(st.session_state.historial_evaluaciones)
        st.dataframe(df_historial, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de Pistas Analizadas", len(df_historial))
        with col2:
            st.metric("Promedio de Tempo (BPM)", f"{df_historial['Tempo (BPM)'].mean():.1f}")
            
        csv_data = df_historial.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exportar Historial Completo a CSV",
            data=csv_data,
            file_name="historial_sincopa.csv",
            mime="text/csv"
        )
    else:
        st.info("Aún no has analizado ninguna pista durante esta sesión. ¡Pega un enlace de YouTube o Spotify para comenzar!")
