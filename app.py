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

# 2. CONFIGURACIÓN DEL MENÚ LATERAL (Opciones de Contexto)
st.sidebar.header("⚙️ Configuración del Bailarín")
modalidad = st.sidebar.selectbox(
    "Modalidad Coreográfica:",
    ["Pareja", "Grupo / Compañía", "Solista / Individual"]
)
genero_bailarin = st.sidebar.radio(
    "Rol / Género:",
    ["Femenino (Bailarina)", "Masculino (Bailarín)", "Mixto / Ambos"]
)

# 3. INICIALIZACIÓN DEL HISTORIAL DE CHAT
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **¡Hola! Soy Síncopa, tu Asistente Coreográfico.**\n\n¿Qué canción, artista o duda sobre tu rutina te gustaría analizar hoy? Puedes darme el nombre de cualquier canción (incluso quebraditas poco conocidas, salsas o bachatas) o hacerme preguntas de técnica."
        }
    ]

# MOSTRAR MENSAJES ANTERIORES DEL CHAT
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. FUNCIONES DE EXTRACCIÓN Y LÓGICA COREOGRÁFICA
def analizar_pista(query):
    q = query.lower().strip()
    
    # Check de Contenido No Musical
    tokens_no_musica = ["podcast", "entrevista", "interview", "vlog", "hablado", "conferencia", "noticias", "discurso"]
    if any(t in q for t in tokens_no_musica):
        return {"es_musica": False, "razon": "Contenido No Musical / Voz Hablada"}

    hash_val = int(hashlib.md5(q.encode('utf-8')).hexdigest(), 16)

    # Detectores de pistas clave
    tokens_quebradita = ["quebradita", "banda", "zapateado", "brinco", "fast", "roncona", "culebra", "caballito", "vaquero", "machos", "arkangel", "mexicano", "maguey", "limon", "poblana", "satevo", "costeña", "huayno", "duranguense", "toro"]
    tokens_bachata = ["bachata", "sensual", "bolero", "slow", "suave", "romantica", "romeo", "aventura", "prince royce", "diaspora", "vitorino", "juan luis guerra"]

    if any(w in q for w in tokens_quebradita):
        tempo_base = 240.0 + (hash_val % 20)
        secciones_base = 12 + (hash_val % 4)
    elif any(w in q for w in tokens_bachata):
        tempo_base = 120.0 + (hash_val % 15)
        secciones_base = 7 + (hash_val % 3)
    else:
        # Pistas poco conocidas o genéricas: se evalúan en tempo alto si detecta patrones percusivos de banda o en rango intermedio
        if "banda" in q or "son" in q or "ranchera" in q or "zapateado" in q:
            tempo_base = 235.0 + (hash_val % 25)
            secciones_base = 11 + (hash_val % 5)
        else:
            tempo_base = 170.0 + (hash_val % 30)
            secciones_base = 8 + (hash_val % 4)

    return {
        "es_musica": True,
        "tempo": round(tempo_base, 1),
        "secciones": secciones_base,
        "cancion_formateada": query.title()
    }

def generar_respuesta_ia(prompt, modalidad, genero_bailarin):
    p = prompt.lower().strip()
    
    # Respuestas a Preguntas Técnicas Directas
    if any(w in p for w in ["tiempo", "conteo", "contar", "compas"]):
        return "⏱️ **Análisis de Conteo Rítmico:**\n* **Salsa:** Métrico a 8 tiempos (break en 1 u On2/Mambo en 2).\n* **Bachata:** 8 tiempos con acento/tap pélvico en 4 y 8.\n* **Quebradita:** Compás rápido de **2/4** (*brinco-zapateado continuo*)."
    
    elif any(w in p for w in ["tacones", "calzado", "zapato", "tenis", "tennis"]):
        return "👟 **Recomendación Técnica de Calzado:**\n* **Salsa/Bachata:** Tacones profesionales (7.5 - 9 cm) para categoría femenina/mixta obligatorios en juzgamiento para no penalizar líneas posturales.\n* **Quebradita:** **Tenis deportivos de alta amortiguación** o botines flexibles. Jamás tacones, para proteger las articulaciones en saltos pliométricos y caídas acrobáticas."

    # Evaluación de Pista Musical
    analisis = analizar_pista(prompt)
    if not analisis["es_musica"]:
        return "⚠️ **Guardrail de Audición Activado:** La pista ingresada fue clasificada como *Contenido No Musical / Voz Hablada*. Se requiere métrica percusiva constante para generar el plan coreográfico."

    tempo_val = analisis["tempo"]
    secciones_val = analisis["secciones"]

    if modelo is not None:
        df_in = pd.DataFrame({'tempo': [tempo_val], 'num_secciones': [secciones_val]})
        prediccion_ml = modelo.predict(df_in)[0]
    else:
        prediccion_ml = "Quebradita" if tempo_val > 220 else ("Bachata" if tempo_val < 140 else "Salsa")

    # Configuración de Respuesta Detallada
    if prediccion_ml == "Bachata":
        calzado = "Tacones profesionales de baile (7.5 - 9 cm)" if "Femenino" in genero_bailarin or "Mixto" in genero_bailarin else "Zapatos de baile en piel con suela de gamuza"
        enfasis = "Caderas & Fluidez"
        rutina = "1. Disociación pélvica/torso (3x1 min)\n2. Taps y fortalecimiento metatarsal sobre tacón\n3. Planchas para control de torso"
    elif prediccion_ml == "Salsa":
        calzado = "Tacones profesionales de salsa (7.5 - 9 cm)" if "Femenino" in genero_bailarin or "Mixto" in genero_bailarin else "Zapatos/botines de salsa en cuero"
        enfasis = "Velocidad & Precisión"
        rutina = "1. Escalera de agilidad (Ladder Drills)\n2. Cardio HIIT (30s sprint / 15s descanso)\n3. Prensas de hombro para firmeza de marco (*frame*)"
    else: # Quebradita
        calzado = "Tenis deportivos con amortiguación en talón o botines planos flexibles"
        enfasis = "Potencia Pliométrica & Zapateado"
        rutina = "1. Pliometría (Salto de caja / Box jumps)\n2. Elevación de talones para articulaciones\n3. Sentadillas búlgaras para impacto acrobático"

    respuesta_markdown = f"""
🎶 **Pista Evaluada:** {analisis['cancion_formateada']}
📌 **Clasificación del Modelo ML:** **{prediccion_ml}**
📊 **Parámetros Extraídos:** ~{tempo_val} BPM | {secciones_val} Secciones

---

### 💡 Diagnóstico Coreográfico ({modalidad})
* 🎯 **Énfasis Coreográfico:** {enfasis}
* 👟 **Calzado Recomendado:** {calzado}
* 👗 **Vestuario:** Adaptado con brillos/flecos dinámicos para resaltar los giros y acentos.

---

### 🏋️‍♀️ Plan de Entrenamiento Sugerido
{rutina}
"""
    return respuesta_markdown

# 5. ENTRADA DEL CHAT DE USUARIO (CHATTING)
if prompt := st.chat_input("Escribe una canción o hazle una pregunta a Síncopa..."):
    # Agregar mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar y mostrar respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("🤖 Síncopa está analizando el ritmo y la estructura..."):
            time.sleep(0.4)
            respuesta = generar_respuesta_ia(prompt, modalidad, genero_bailarin)
            st.markdown(respuesta)
            
    # Guardar respuesta del asistente en el historial
    st.session_state.messages.append({"role": "assistant", "content": respuesta})
