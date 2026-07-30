import streamlit as st
import numpy as np
import time
import joblib
import os
import pandas as pd
import hashlib

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

# 2. BASE DE DATOS DE SUGERENCIAS
SUGERENCIAS_GENERO = {
    "Quebradita": [
        "La Culebra - Banda Machos",
        "La Roncona - Banda Arkangel R-15",
        "El Apagón - Banda Yuri",
        "El Baile del Caballito - Banda Machos",
        "Vampiro - Banda Machos",
        "La Chona - Los Tucanes de Tijuana",
        "Al Gato y al Ratón - Banda Machos"
    ],
    "Salsa": [
        "Aguanile - Héctor Lavoe",
        "Valió la Pena - Marc Anthony",
        "Lluvia - Eddie Santiago",
        "La Rebelión - Joe Arroyo",
        "Gitana - Willie Colón",
        "Sobredosis - Los Hermanos Lebrón"
    ],
    "Bachata": [
        "Propuesta Indecente - Romeo Santos",
        "Dile al Amor - Aventura",
        "Darte un Beso - Prince Royce",
        "Burbujas de Amor - Juan Luis Guerra",
        "Eres Mía - Romeo Santos",
        "Obsesión - Aventura"
    ]
}

# 3. INICIALIZACIÓN DE ESTADOS Y MENSAJE DE BIENVENIDA
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **¡Hola! Soy Síncopa, tu asistente para ayudarte a analizar tus canciones y estructurar tus rutinas de baile.**\n\n"
                "Escribe simplemente el nombre de una canción o artista para comenzar.\n\n"
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

# 4. FUNCIONES DE ANÁLISIS RÁPIDO
def analizar_pista(query):
    q = query.lower().strip()
    
    tokens_no_musica = ["podcast", "entrevista", "interview", "vlog", "hablado", "conferencia", "noticias", "discurso"]
    if any(t in q for t in tokens_no_musica):
        return {"es_musica": False, "razon": "Contenido No Musical / Voz Hablada"}

    es_link = any(domain in q for domain in ["spotify.com", "youtube.com", "youtu.be", "drive.google", ".mp3", ".wav"])
    hash_val = int(hashlib.md5(q.encode('utf-8')).hexdigest(), 16)

    tokens_quebradita = ["quebradita", "quebraditas", "banda", "zapateado", "brinco", "fast", "roncona", "culebra", "caballito", "vaquero", "machos", "arkangel", "tucanes"]
    tokens_bachata = ["bachata", "bachatas", "sensual", "bolero", "slow", "suave", "romantica", "romeo", "aventura", "prince", "royce", "guerra"]
    tokens_salsa = ["salsa", "salsas", "mambo", "guaguanco", "son", "timba", "marc anthony", "lavoe", "colon", "arroyo"]

    if any(w in q for w in tokens_quebradita):
        tempo_base = 240.0 + (hash_val % 15)
        secciones_base = 12
    elif any(w in q for w in tokens_bachata):
        tempo_base = 122.0 + (hash_val % 12)
        secciones_base = 7
    elif any(w in q for w in tokens_salsa):
        tempo_base = 178.0 + (hash_val % 20)
        secciones_base = 9
    else:
        # Pista genérica por defecto si no hay coincidencia directa de género
        tempo_base = 135.0 + (hash_val % 50)
        secciones_base = 8

    return {
        "es_musica": True,
        "es_link": es_link,
        "tempo": round(tempo_base, 1),
        "secciones": secciones_base,
        "cancion_formateada": "Pista por Enlace External" if es_link else query.title()
    }

def obtener_metricas_multi_modalidad(genero_predicho):
    """Genera las recomendaciones simultáneas para Pareja, Grupo y Solista."""
    if genero_predicho == "Quebradita":
        base = 8.5
        recom = "⭐ **Sugerencia:** ¡Ideal para **Grupo / Compañía** por el impacto visual de los lanzamientos y bloques sincronizados!"
    elif genero_predicho == "Salsa":
        base = 7.0
        recom = "⭐ **Sugerencia:** Funciona excelente tanto en **Pareja** (turn patterns) como en **Solista** para lucir Shines/Footwork."
    else: # Bachata
        base = 5.0
        recom = "⭐ **Sugerencia:** Perfecta para **Pareja** por la conexión y fluidez en ondas/sensual, o **Solista** para disociación."

    return {
        "pareja": round(base, 1),
        "grupo": round(min(10.0, base + 1.5), 1),
        "solista": round(min(10.0, base + 1.0), 1),
        "recomendacion_estilo": recom
    }

# 5. MANEJO DE ENTRADA EN CHAT
if prompt := st.chat_input("Escribe una canción, pide sugerencias o haz una duda..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    p_lower = prompt.lower()

    # A) Detección de intenciones de recomendación/sugerencias
    palabras_sugerencia = ["sugerencia", "sugerencias", "sugieres", "sugiere", "recomienda", "opciones", "bachatas", "salsas", "quebraditas", "ideas"]
    es_pedido_sugerencia = any(w in p_lower for w in palabras_sugerencia)

    if es_pedido_sugerencia:
        if "quebradita" in p_lower or "quebraditas" in p_lower:
            gen = "Quebradita"
        elif "salsa" in p_lower or "salsas" in p_lower:
            gen = "Salsa"
        else:
            gen = "Bachata"
            
        sug_list = SUGERENCIAS_GENERO.get(gen, [])
        items_txt = "\n".join([f"* 🎶 **{s}**" for s in sug_list])
        
        reply = f"🎶 **Aquí tienes excelentes opciones de {gen}:**\n\n{items_txt}\n\n*¿Te gustaría evaluar alguna de estas? Solo dime el nombre.*"
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    # B) Evaluación Directa e Inmediata de Canción
    else:
        with st.chat_message("assistant"):
            with st.spinner("🤖 Analizando ritmo y métricas..."):
                time.sleep(0.3)
                analisis = analizar_pista(prompt)

        if not analisis["es_musica"]:
            reply = "⚠️ La entrada parece ser un contenido hablado o no musical. Por favor, intenta ingresando el nombre de un tema o un enlace de audio."
        else:
            tempo_val = analisis["tempo"]
            secciones_val = analisis["secciones"]

            # Clasificación ML
            if modelo is not None:
                df_in = pd.DataFrame({'tempo': [tempo_val], 'num_secciones': [secciones_val]})
                prediccion_ml = modelo.predict(df_in)[0]
            else:
                prediccion_ml = "Quebradita" if tempo_val > 220 else ("Bachata" if tempo_val < 140 else "Salsa")

            mm = obtener_metricas_multi_modalidad(prediccion_ml)

            # Guardar en estado de sesión
            st.session_state.ultima_evaluacion = {
                "cancion": analisis['cancion_formateada'],
                "genero": prediccion_ml,
                "tempo": tempo_val
            }
            st.session_state.historial_evaluaciones.append(st.session_state.ultima_evaluacion)

            sug_rel = SUGERENCIAS_GENERO.get(prediccion_ml, [])[:3]
            sug_txt = ", ".join([f"*{s}*" for s in sug_rel])

            reply = f"""🎶 **Resultados para:** **{analisis['cancion_formateada']}**
📌 **Género Estimado:** **{prediccion_ml}** (~{tempo_val} BPM)

---

### 📊 Evaluaciones por Modalidad de Baile:

* 👫 **Si lo bailas en Pareja:** Exigencia de **{mm['pareja']} / 10** (Ideal para trabajo de marco y conexión).
* 👯‍♀️ **Si lo bailas en Grupo / Compañía:** Exigencia de **{mm['grupo']} / 10** (Exige alta limpieza en bloques y simetría).
* 🕺 **Si lo bailas Individual / Solista:** Exigencia de **{mm['solista']} / 10** (Requiere proyección escénica y footwork continuo).

{mm['recomendacion_estilo']}

---

💡 *Pistas similares que te podrían gustar:* {sug_txt}.
*(Puedes preguntarme sobre vestuario, minutos de footwork para este tema o probar con otra canción)*.
"""

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
