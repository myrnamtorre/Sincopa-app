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

st.set_page_config(
    page_title="Síncopa • Asistente Coreográfico IA",
    page_icon="💃",
    layout="centered"
)

st.title("💃 Síncopa: Asistente Coreográfico IA")
st.caption("🤖 Agente Conversacional para Análisis Coreográfico, Ritmo y Entrenamiento")
st.markdown("---")

# 1. CARGA DEL MODELO ML
ruta_modelo = 'modelo_sincopa_rf.joblib'
modelo = None
if os.path.exists(ruta_modelo):
    try:
        modelo = joblib.load(ruta_modelo)
        st.sidebar.success("🤖 Backend ML: Random Forest Activo")
    except Exception as e:
        st.sidebar.error(f"Error al cargar el modelo: {e}")

# 2. BASE DE DATOS EXTENDIDA DE SUGERENCIAS POR RITMO Y VELOCIDAD
SUGERENCIAS_GENERO = {
    "Quebradita": {
        "rapidas": ["La Culebra - Banda Machos", "El Baile del Caballito - Banda Machos", "Al Gato y al Ratón - Banda Machos", "La Chona - Los Tucanes de Tijuana"],
        "moderadas": ["La Roncona - Banda Arkangel R-15", "El Apagón - Banda Yuri", "Vampiro - Banda Machos"]
    },
    "Salsa": {
        "rapidas": ["Aguanile - Héctor Lavoe", "La Rebelión - Joe Arroyo", "Aguanile - Marc Anthony", "Que Se Sepa - Roberto Roena"],
        "lentas": ["Lluvia - Eddie Santiago", "Gitana - Willie Colón", "Sobredosis - Los Hermanos Lebrón", "Con Conciencia - Gilberto Santa Rosa"],
        "moderadas": ["Valió la Pena - Marc Anthony", "Flor Pálida - Marc Anthony"]
    },
    "Bachata": {
        "lentas": ["Burbujas de Amor - Juan Luis Guerra", "Infidelidades - Aventura", "Perdidos - Monchy & Alexandra", "Dile al Amor - Aventura"],
        "rapidas": ["Propuesta Indecente - Romeo Santos", "Darte un Beso - Prince Royce", "Eres Mía - Romeo Santos", "Obsesión - Aventura"],
        "moderadas": ["Stand by Me - Prince Royce", "El Perdedor - Aventura"]
    }
}

# 3. INICIALIZACIÓN DE ESTADOS DE SESIÓN
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **¡Hola! Soy Síncopa, tu asistente para ayudarte a analizar tus canciones y estructurar tus rutinas de baile.**\n\n"
                "Escribe el nombre de una canción, pega un enlace o pídeme sugerencias (ej. *'sugiere bachatas lentas'*, *'salsas rápidas'*, etc.).\n\n"
                "> ⚠️ *Recuerda que estoy en entrenamiento continuo y mis estimaciones métricas/BPM pueden contener margen de error.*"
            )
        }
    ]

if "ultima_evaluacion" not in st.session_state:
    st.session_state.ultima_evaluacion = None
if "ultimo_genero_sugerido" not in st.session_state:
    st.session_state.ultimo_genero_sugerido = "Bachata"
if "historial_evaluaciones" not in st.session_state:
    st.session_state.historial_evaluaciones = []

# Mostrar historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. FUNCIONES DE EXTRACCIÓN Y ANÁLISIS
def obtener_titulo_desde_link(url):
    """Extrae el título real de la canción desde enlaces de Spotify, YouTube, Apple Music, SoundCloud, etc."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=4)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                titulo = og_title["content"]
            elif soup.title and soup.title.string:
                titulo = soup.title.string
            else:
                titulo = "Enlace de Audio/Video"

            sufijos = [" | Spotify", " - YouTube", " on SoundCloud", " en Apple Music", " - Apple Music", " - Topic"]
            for s in sufijos:
                titulo = titulo.replace(s, "")

            return titulo.strip()
    except Exception:
        pass
    
    return "Canción por Enlace Externo"

def analizar_pista(query):
    q = query.lower().strip()
    
    tokens_no_musica = ["podcast", "entrevista", "interview", "vlog", "hablado", "conferencia", "noticias", "discurso"]
    if any(t in q for t in tokens_no_musica):
        return {"es_musica": False, "razon": "Contenido No Musical / Voz Hablada"}

    es_link = any(domain in q for domain in ["spotify.com", "youtube.com", "youtu.be", "soundcloud.com", "apple.com", "drive.google", ".mp3", ".wav"])
    
    if es_link:
        match = re.search(r'https?://[^\s]+', query)
        url_detectada = match.group(0) if match else query
        cancion_nombre = obtener_titulo_desde_link(url_detectada)
    else:
        cancion_nombre = query.title()

    hash_val = int(hashlib.md5(q.encode('utf-8')).hexdigest(), 16)

    tokens_quebradita = ["quebradita", "quebraditas", "banda", "zapateado", "brinco", "fast", "roncona", "culebra", "caballito", "vaquero", "machos", "arkangel", "tucanes", "vampiro"]
    tokens_bachata = ["bachata", "bachatas", "sensual", "bolero", "slow", "suave", "romantica", "romeo", "aventura", "prince", "royce", "guerra"]
    tokens_salsa = ["salsa", "salsas", "mambo", "guaguanco", "son", "timba", "marc anthony", "lavoe", "colon", "arroyo"]

    cadena_eval = (query + " " + cancion_nombre).lower()

    if any(w in cadena_eval for w in tokens_quebradita):
        tempo_base = 240.0 + (hash_val % 15)
        secciones_base = 12
    elif any(w in cadena_eval for w in tokens_bachata):
        tempo_base = 122.0 + (hash_val % 12)
        secciones_base = 7
    elif any(w in cadena_eval for w in tokens_salsa):
        tempo_base = 178.0 + (hash_val % 20)
        secciones_base = 9
    else:
        tempo_base = 135.0 + (hash_val % 50)
        secciones_base = 8

    return {
        "es_musica": True,
        "es_link": es_link,
        "tempo": round(tempo_base, 1),
        "secciones": secciones_base,
        "cancion_formateada": cancion_nombre
    }

def obtener_metricas_multi_modalidad(genero_predicho):
    """Genera recomendaciones simultáneas para Pareja, Grupo y Solista."""
    if genero_predicho == "Quebradita":
        base = 8.5
        recom = "⭐ **Sugerencia:** ¡Ideal para **Grupo / Compañía** por el impacto visual de los lanzamientos y bloques sincronizados!"
        ejercicios = (
            "* 🦘 **Pliometría & Potencia:** Jump squats y salto de cuerda rápido (3 series x 45 seg) para resistir el rebote alto.\n"
            "* 🦶 **Fuerza de Tobillo y Gemelos:** Elevaciones de talón en borde de escalón y trabajo de estabilidad de tobillos para zapateado continuo.\n"
            "* 🫁 **Resistencia Cardiovascular HIIT:** Intervalos de alta intensidad de 30s esfuerzo / 15s descanso para aguantar el ritmo vertiginoso."
        )
    elif genero_predicho == "Salsa":
        base = 7.0
        recom = "⭐ **Sugerencia:** Funciona excelente tanto en **Pareja** (turn patterns) como en **Solista** para lucir Shines/Footwork."
        ejercicios = (
            "* ⚡ **Agilidad de Pies (Footwork):** Trabajo en escalera de agilidad (in-out rápidos) para velocidad en shines y cambios de peso.\n"
            "* 🔄 **Estabilidad de Core & Giros:** Planchas dinámicas con rotación y giros spot focalizados para mantener el eje en secuencias rápidas.\n"
            "* 🦵 **Movilidad de Cadera:** Ejercicios de disociación pélvica y fortalecimiento de abductores/aductores."
        )
    else: # Bachata
        base = 5.0
        recom = "⭐ **Sugerencia:** Perfecta para **Pareja** por la conexión y fluidez en ondas/sensual, o **Solista** para disociación."
        ejercicios = (
            "* 🌊 **Disociación Corporal:** Trabajo de movilidad de columna, aislación torácica y ondas de torso frente al espejo.\n"
            "* 🧘 **Flexibilidad & Control:** Estiramientos profundos de isquiotibiales y movilidad de cadera para ondas suaves sin tensión muscular.\n"
            "* 🛡️ **Fuerza de Postura (Marco):** Remo con liga/mancuerna e isometría de deltoides para sostener el marco de pareja sin fatigar hombros."
        )

    return {
        "pareja": round(base, 1),
        "grupo": round(min(10.0, base + 1.5), 1),
        "solista": round(min(10.0, base + 1.0), 1),
        "recomendacion_estilo": recom,
        "ejercicios_recomendados": ejercicios
    }

# 5. ATENCIÓN DE INTERACCIONES EN EL CHAT
if prompt := st.chat_input("Escribe una canción, pega un link o pide sugerencias..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    p_lower = prompt.lower()

    # A) DETECCIÓN DE PREGUNTAS DE SEGUIMIENTO (VESTUARIO / CALZADO / ENTRENAMIENTO)
    palabras_vestuario = ["vestuario", "ropa", "ropa sugerida", "que me pongo", "qué me pongo", "outfit", "traje"]
    palabras_calzado = ["calzado", "zapatos", "tenis", "zapatillas", "suela"]
    palabras_entrenamiento = ["ejercicio", "ejercicios", "entrenamiento", "rutina", "preparacion", "preparación", "footwork", "pasos", "aguantar", "fisico", "físico"]

    es_pregunta_vestuario = any(w in p_lower for w in palabras_vestuario)
    es_pregunta_calzado = any(w in p_lower for w in palabras_calzado)
    es_pregunta_entrenamiento = any(w in p_lower for w in palabras_entrenamiento)

    if (es_pregunta_vestuario or es_pregunta_calzado or es_pregunta_entrenamiento) and st.session_state.ultima_evaluacion:
        eval_previa = st.session_state.ultima_evaluacion
        gen_previo = eval_previa["genero"]
        cancion_previa = eval_previa["cancion"]
        mm_prev = obtener_metricas_multi_modalidad(gen_previo)

        if es_pregunta_vestuario:
            if gen_previo == "Quebradita":
                recom_v = "👗 **Vestuario Sugerido para Quebradita:**\n* **Compañía / Pareja:** Vestuario vaquero estilizado pero ligero (pantalones/faldas con strech flexible, camisas de tela respirable y flecos con buen movimiento). Evitar faldas demasiado largas que interfieran en las acrobacias."
            elif gen_previo == "Salsa":
                recom_v = "👗 **Vestuario Sugerido para Salsa:**\n* **Pareja / Solista:** Flecos, pedrería o trajes de corte estilizado que acentúen la rotación de cadera y hombros. Para solistas, pantalones de caída fluida que resalten el footwork."
            else: # Bachata
                recom_v = "👗 **Vestuario Sugerido para Bachata:**\n* **Sensual / Pareja:** Prendas entalladas de tela suave/elástica (lycra, aterciopelados ligeros) que permitan apreciar la disociación corporal y las ondas del torso sin fricción."

            reply = f"💃 **Recomendación de Vestuario para *{cancion_previa}* ({gen_previo}):**\n\n{recom_v}"

        elif es_pregunta_calzado:
            if gen_previo == "Quebradita":
                recom_c = "👟 **Calzado:** Botas flex con suela de amortiguación o tenis deportivos con buen soporte en tobillos para absorber el impacto de los brincos y zapateados."
            elif gen_previo == "Salsa":
                recom_c = "👠 **Calzado:** Zapatos de baile latino con suela de ante/gamuza para giros rápidos. Para solistas/shines, tenis de baile con punto de pivote."
            else:
                recom_c = "👠 **Calzado:** Zapatos de bachata flexible de flexión completa o tenis de baile urbano/sensual con suela deslizante."

            reply = f"👠 **Calzado Recomendado para *{cancion_previa}* ({gen_previo}):**\n\n{recom_c}"

        elif es_pregunta_entrenamiento:
            reply = f"🏋️ **Acondicionamiento Físico Recomendado para *{cancion_previa}* ({gen_previo}):**\n\n{mm_prev['ejercicios_recomendados']}"

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    # B) DETECCIÓN DE SOLICITUD DE SUGERENCIAS CON FILTROS
    elif any(w in p_lower for w in ["sugerencia", "sugerencias", "sugieres", "sugiere", "recomienda", "opciones", "bachata", "bachatas", "salsa", "salsas", "quebradita", "quebraditas", "ideas", "otras", "mas", "más"]):
        if "quebradita" in p_lower or "quebraditas" in p_lower:
            gen = "Quebradita"
        elif "salsa" in p_lower or "salsas" in p_lower:
            gen = "Salsa"
        elif "bachata" in p_lower or "bachatas" in p_lower:
            gen = "Bachata"
        else:
            gen = st.session_state.ultimo_genero_sugerido

        st.session_state.ultimo_genero_sugerido = gen

        es_lenta = any(w in p_lower for w in ["lenta", "lentas", "suave", "suaves", "romantica", "románticas", "sensual", "despacio"])
        es_rapida = any(w in p_lower for w in ["rapida", "rápidas", "rapidas", "movida", "movidas", "prendida", "prendidas", "fast", "fuerte"])

        cat_dict = SUGERENCIAS_GENERO.get(gen, {})

        if es_lenta and "lentas" in cat_dict:
            sug_base = cat_dict["lentas"].copy()
            tag_vel = "Lentas / Románticas"
        elif es_rapida and "rapidas" in cat_dict:
            sug_base = cat_dict["rapidas"].copy()
            tag_vel = "Rápidas / Intensa Métrica"
        else:
            sug_base = []
            for k, lista in cat_dict.items():
                sug_base.extend(lista)
            tag_vel = "Variadas"

        if st.session_state.ultima_evaluacion:
            cancion_previa = st.session_state.ultima_evaluacion["cancion"].lower()
            sug_base = [s for s in sug_base if cancion_previa not in s.lower()]

        if any(w in p_lower for w in ["mas", "más", "otras", "diferentes", "nuevas"]):
            random.seed(len(p_lower) + int(time.time() % 100))
            random.shuffle(sug_base)

        items_txt = "\n".join([f"* 🎶 **{s}**" for s in sug_base])
        
        reply = f"🎶 **Sugerencias de {gen} ({tag_vel}):**\n\n{items_txt}\n\n*¿Te gustaría evaluar alguna de estas? Escribe su nombre o pega un enlace.*"
        
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    # C) EVALUACIÓN DIRECTA E INMEDIATA DE UNA NUEVA CANCIÓN O LINK
    else:
        with st.chat_message("assistant"):
            with st.spinner("🤖 Analizando pista y clasificando género..."):
                time.sleep(0.3)
                analisis = analizar_pista(prompt)

        if not analisis["es_musica"]:
            reply = "⚠️ La entrada parece ser un contenido hablado o no musical. Por favor, ingresa el nombre de una canción o pega un enlace de audio/video."
        else:
            tempo_val = analisis["tempo"]
            secciones_val = analisis["secciones"]

            if modelo is not None:
                df_in = pd.DataFrame({'tempo': [tempo_val], 'num_secciones': [secciones_val]})
                prediccion_ml = modelo.predict(df_in)[0]
            else:
                prediccion_ml = "Quebradita" if tempo_val > 220 else ("Bachata" if tempo_val < 140 else "Salsa")

            mm = obtener_metricas_multi_modalidad(prediccion_ml)

            st.session_state.ultima_evaluacion = {
                "cancion": analisis['cancion_formateada'],
                "genero": prediccion_ml,
                "tempo": tempo_val
            }
            st.session_state.historial_evaluaciones.append(st.session_state.ultima_evaluacion)

            cat_prev = SUGERENCIAS_GENERO.get(prediccion_ml, {})
            sug_planas = []
            for k, v in cat_prev.items():
                sug_planas.extend(v)

            sug_rel = [s for s in sug_planas if analisis['cancion_formateada'].lower() not in s.lower()][:3]
            sug_txt = ", ".join([f"*{s}*" for s in sug_rel])

            reply = f"""🎵 **Canción:** **{analisis['cancion_formateada']}**
🏷️ **Género Clasificado:** **{prediccion_ml}** 
⏱️ **Tempo Estimado:** ~{tempo_val} BPM

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
*(Puedes preguntarme sobre calzado, vestuario o pedir listas como "salsas lentas")*.
"""

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
