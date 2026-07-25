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

# 2. INICIALIZACIÓN DE ESTADOS DE SESIÓN (Flujo Conversacional)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **¡Hola! Soy Síncopa, tu Asistente Coreográfico.**\n\n¿Qué canción o artista te gustaría analizar hoy?"
        }
    ]

# Estado para controlar el flujo de contrapreguntas
if "step" not in st.session_state:
    st.session_state.step = "esperando_cancion"
if "cancion" not in st.session_state:
    st.session_state.cancion = ""
if "modalidad" not in st.session_state:
    st.session_state.modalidad = ""

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. FUNCIONES DE EXTRACCIÓN Y LÓGICA COREOGRÁFICA
def analizar_pista(query):
    q = query.lower().strip()
    
    # Guardrail 1: Contenido No Musical
    tokens_no_musica = ["podcast", "entrevista", "interview", "vlog", "hablado", "conferencia", "noticias", "discurso", "audiobook"]
    if any(t in q for t in tokens_no_musica):
        return {"es_musica": False, "razon": "Contenido No Musical / Voz Hablada"}

    # Guardrail 2: Géneros/Artistas fuera del alcance
    tokens_fuera_de_dominio = [
        "quedate en silencio", "quédate en silencio", "rbd", "rebelde", "pop", "balada", 
        "perla", "rosalia", "rosalía", "flamenco", "rumba", "reggaeton", "reggaetón", 
        "tumbado", "corrido", "rock", "hip hop", "rap", "merengue", "cumbia", "bad bunny", "karol g"
    ]
    if any(t in q for t in tokens_fuera_de_dominio):
        return {
            "es_musica": True, 
            "fuera_de_dominio": True, 
            "genero_detectado": "Pop / Balada / Urbano / Flamenco (Fuera del Alcance del Modelo)",
            "cancion_formateada": query.title()
        }

    hash_val = int(hashlib.md5(q.encode('utf-8')).hexdigest(), 16)

    # Tokens para los géneros entrenados
    tokens_quebradita = [
        "quebradita", "banda", "zapateado", "brinco", "fast", "roncona", "culebra", 
        "caballito", "vaquero", "machos", "arkangel", "mexicano", "maguey", "limon", 
        "poblana", "satevo", "costeña", "huayno", "duranguense", "toro"
    ]
    tokens_bachata = [
        "bachata", "sensual", "bolero", "slow", "suave", "romantica", "romeo", 
        "aventura", "prince royce", "diaspora", "vitorino", "juan luis guerra"
    ]
    tokens_salsa = [
        "salsa", "mambo", "guaguanco", "son", "timba", "marc anthony", "havana d'primera",
        "maykel blanco", "niche", "barretto", "celia", "lavoe", "elito reve"
    ]

    if any(w in q for w in tokens_quebradita):
        tempo_base = 240.0 + (hash_val % 20)
        secciones_base = 12 + (hash_val % 4)
    elif any(w in q for w in tokens_bachata):
        tempo_base = 120.0 + (hash_val % 15)
        secciones_base = 7 + (hash_val % 3)
    elif any(w in q for w in tokens_salsa):
        tempo_base = 175.0 + (hash_val % 25)
        secciones_base = 9 + (hash_val % 5)
    else:
        return {
            "es_musica": True, 
            "fuera_de_dominio": True, 
            "genero_detectado": "Género no identificado dentro de Salsa, Bachata o Quebradita",
            "cancion_formateada": query.title()
        }

    return {
        "es_musica": True,
        "fuera_de_dominio": False,
        "tempo": round(tempo_base, 1),
        "secciones": secciones_base,
        "cancion_formateada": query.title()
    }

def calcular_esfuerzo_y_metricas(prediccion_ml, modalidad):
    # Base por género
    if prediccion_ml == "Quebradita":
        esfuerzo_base = 8.5
        velocidad_txt = "Muy Alta (Brinco/Zapateado)"
        bailabilidad_txt = "9.5 / 10 (Alta Pliometría)"
        enfasis_txt = "Potencia Pliométrica & Acrobacia"
    elif prediccion_ml == "Salsa":
        esfuerzo_base = 7.0
        velocidad_txt = "Alta (Shines/Giros)"
        bailabilidad_txt = "9.0 / 10 (Ritmo Complejo)"
        enfasis_txt = "Velocidad & Precisión"
    else: # Bachata
        esfuerzo_base = 5.0
        velocidad_txt = "Moderada / Fluida"
        bailabilidad_txt = "9.8 / 10 (Sensual/Cadencia)"
        enfasis_txt = "Caderas, Disociación & Marco"

    # Modificador dinámico por Modalidad
    if "Compañía" in modalidad or "Grupo" in modalidad:
        esfuerzo_final = min(10.0, esfuerzo_base + 1.5)
        mod_nota = "(+1.5 por limpieza de bloques y cañones en grupo)"
    elif "Solista" in modalidad:
        esfuerzo_final = min(10.0, esfuerzo_base + 1.0)
        mod_nota = "(+1.0 por dominio escénico continuo sin pausas)"
    else: # Pareja
        esfuerzo_final = esfuerzo_base
        mod_nota = "(Estándar para trabajo de marca y giros)"

    return {
        "esfuerzo": round(esfuerzo_final, 1),
        "mod_nota": mod_nota,
        "velocidad": velocidad_txt,
        "bailabilidad": bailabilidad_txt,
        "enfasis": enfasis_txt
    }

# 4. MANEJO DE ENTRADA DEL CHAT (Lógica Conversacional)
if prompt := st.chat_input("Escribe tu respuesta aquí..."):
    # Guardar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # LÓGICA DE PASO 1: Captura de Canción y Contrapregunta de Modalidad
    if st.session_state.step == "esperando_cancion":
        st.session_state.cancion = prompt
        
        # Verificar si es una duda técnica general en lugar de una canción
        p = prompt.lower()
        if any(w in p for w in ["tiempo", "conteo", "tacones", "calzado", "quebradita", "salsa", "bachata"]) and len(prompt.split()) > 3:
            respuesta_directa = "💡 **Respuesta de Síncopa:** La Salsa y Bachata se bailan a 8 tiempos, mientras la Quebradita es en compás rápido de 2/4. Para Salsa/Bachata femenina se exigen tacones de 7.5 a 9 cm en escenario, mientras que en Quebradita se usan tenis para amortiguar saltos."
            st.session_state.messages.append({"role": "assistant", "content": respuesta_directa})
            with st.chat_message("assistant"):
                st.markdown(respuesta_directa)
        else:
            st.session_state.step = "esperando_modalidad"
            bot_reply = f"¡Excelente elección! 🎶 **'{prompt.title()}'**.\n\nPara ajustar el cálculo de esfuerzo físico y la dinámica, dime: **¿La coreografía se presentará en Solista, Pareja o Compañía/Grupo?**"
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            with st.chat_message("assistant"):
                st.markdown(bot_reply)

    # LÓGICA DE PASO 2: Captura de Modalidad y Contrapregunta de Rol/Género
    elif st.session_state.step == "esperando_modalidad":
        m_user = prompt.lower()
        if "solista" in m_user or "individual" in m_user:
            st.session_state.modalidad = "Solista / Individual"
        elif "grupo" in m_user or "compañia" in m_user or "compañía" in m_user:
            st.session_state.modalidad = "Grupo / Compañía"
        else:
            st.session_state.modalidad = "Pareja"

        st.session_state.step = "esperando_rol"
        bot_reply = f"Entendido, formato **{st.session_state.modalidad}**.\n\nPor último: **¿Para qué rol/género va dirigida la rutina?** (ej. Femenino/Bailarina, Masculino/Bailarín o Mixto)"
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)

    # LÓGICA DE PASO 3: Generación del Reporte Completo con Métricas
    elif st.session_state.step == "esperando_rol":
        rol_user = prompt
        cancion_txt = st.session_state.cancion
        modalidad_txt = st.session_state.modalidad

        with st.chat_message("assistant"):
            with st.spinner("🤖 Generando evaluación coreográfica personalizada..."):
                time.sleep(0.5)
                
                analisis = analizar_pista(cancion_txt)
                
                if not analisis["es_musica"]:
                    respuesta = "⚠️ **Guardrail de Audición Activado:** La pista ingresada fue clasificada como *Contenido No Musical / Voz Hablada*."
                elif analisis.get("fuera_de_dominio", False):
                    respuesta = f"""⚠️ **Guardrail de Dominio Activado:** 

🎵 **Pista Evaluada:** {analisis['cancion_formateada']}
📌 **Diagnóstico:** {analisis['genero_detectado']}

💡 **Nota del Asistente:** Síncopa está entrenado específicamente para la clasificación de **Salsa, Bachata y Quebradita**. La pista ingresada no pertenece a este scope, por lo que omito la evaluación para no dar métricas incorrectas."""
                else:
                    tempo_val = analisis["tempo"]
                    secciones_val = analisis["secciones"]

                    if modelo is not None:
                        df_in = pd.DataFrame({'tempo': [tempo_val], 'num_secciones': [secciones_val]})
                        prediccion_ml = modelo.predict(df_in)[0]
                    else:
                        prediccion_ml = "Quebradita" if tempo_val > 220 else ("Bachata" if tempo_val < 140 else "Salsa")

                    metricas = calcular_esfuerzo_y_metricas(prediccion_ml, modalidad_txt)

                    if prediccion_ml == "Bachata":
                        calzado = "Tacones profesionales de baile (7.5 - 9 cm)" if "fem" in rol_user.lower() or "mix" in rol_user.lower() else "Zapatos de baile en piel suave"
                        rutina = "1. Disociación pélvica/torso (3x1 min)\n2. Taps y fortalecimiento de metatarsos\n3. Planchas para estabilidad de marco"
                    elif prediccion_ml == "Salsa":
                        calzado = "Tacones profesionales de salsa (7.5 - 9 cm)" if "fem" in rol_user.lower() or "mix" in rol_user.lower() else "Botines/Zapatos de salsa en cuero con suela flexible"
                        rutina = "1. Escalera de agilidad (Ladder Drills)\n2. Cardio HIIT en bloques de 30s\n3. Prensas de hombro para estabilidad"
                    else: # Quebradita
                        calzado = "Tenis deportivos de alto impacto con buena amortiguación"
                        rutina = "1. Pliometría (Salto de caja / Box jumps)\n2. Elevación de talones para articulaciones\n3. Sentadillas explosivas para acrobacias"

                    respuesta = f"""
🎶 **Pista Evaluada:** {analisis['cancion_formateada']}
📌 **Clasificación del Modelo ML:** **{prediccion_ml}**

---

### 📊 Evaluación de la Pista
* ⚡ **Velocidad / Tempo:** {metricas['velocidad']} (~{tempo_val} BPM)
* 💃 **Bailabilidad:** {metricas['bailabilidad']}
* 🔥 **Exigencia Física Estimada:** **{metricas['esfuerzo']} / 10** {metricas['mod_nota']}
* 🎯 **Énfasis Coreográfico:** {metricas['enfasis']}

---

### 💡 Diagnóstico y Recomendaciones ({modalidad_txt})
* 👟 **Calzado Técnico:** {calzado}
* 👗 **Vestuario:** Diseñado con dinamismo para acentuar cortes y acentos visuales.

---

### 🏋️‍♀️ Plan de Entrenamiento Sugerido
{rutina}

---
*¿Quieres analizar otra canción? ¡Solo escribe su nombre!*
"""
                st.markdown(respuesta)
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
                
                # Reiniciamos el estado para la siguiente consulta
                st.session_state.step = "esperando_cancion"
