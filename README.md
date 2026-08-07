# 💃 Síncopa — Asistente Coreográfico

Síncopa es un chatbot construido con Streamlit que evalúa canciones de **Bachata, Salsa (incluye Timba cubana) y Quebradita** a partir de un link de Spotify, YouTube o Apple Music, y devuelve:

- El género de baile clasificado por un modelo `RandomForestClassifier`, entrenado sobre un dataset real de 482 canciones.
- Un análisis de **aprovechamiento coreográfico** generado dinámicamente a partir de las métricas reales de esa canción (tempo, densidad rítmica, estructura, energía, ánimo) comparadas contra su género.
- Sugerencias de canciones, rutinas de entrenamiento físico y vestuario de competencia, específicas por género.
- Detección de contenido no musical (podcasts, tutoriales) y de contenido con música de fondo que no es una pista dedicada para bailar (tráilers, gameplay, resúmenes deportivos).

---

## Tabla de contenido

1. [Arquitectura del proyecto](#arquitectura-del-proyecto)
2. [Estructura de archivos](#estructura-de-archivos)
3. [Cómo funciona (flujo de una consulta)](#cómo-funciona-flujo-de-una-consulta)
4. [El dataset](#el-dataset)
5. [Instalación y ejecución local](#instalación-y-ejecución-local)
6. [Reentrenar el modelo](#reentrenar-el-modelo)
7. [Usar el notebook de entrenamiento](#usar-el-notebook-de-entrenamiento)
8. [Desplegar en Streamlit Community Cloud](#desplegar-en-streamlit-community-cloud)
9. [Limitaciones conocidas](#limitaciones-conocidas)
10. [Preguntas frecuentes / troubleshooting](#preguntas-frecuentes--troubleshooting)

---

## Arquitectura del proyecto

El diseño tiene un solo objetivo: que **el modelo y la app nunca puedan desincronizarse**. Para eso, toda la lógica de negocio (dataset, features, entrenamiento, clasificación de contenido, catálogos) vive en un único módulo, `sincopa_core.py`, del que dependen tanto el script de entrenamiento como la app y el notebook:

```
┌─────────────────────────┐
│   sincopa_core.py        │  ← única fuente de verdad
│   (features, dataset,    │
│   entrenamiento, catálogos)│
└───────────┬──────────────┘
            │
   ┌────────┴────────┬──────────────────┐
   │                  │                  │
entrenar_modelo.py   app.py     Sincopa_Entrenamiento_v4
   │                  │           _autocontenido.ipynb
   ▼                  ▼
modelo_sincopa_rf.joblib  ←── app.py lo carga directo
```

Si algún día necesitas cambiar un rango de tempo, agregar un género o ajustar un umbral de confianza, lo cambias **una sola vez** en `sincopa_core.py` y tanto el entrenamiento como la app quedan consistentes automáticamente.

---

## Estructura de archivos

| Archivo | Qué es |
|---|---|
| `sincopa_core.py` | Módulo central: carga del dataset, esquema de features, entrenamiento del modelo, búsqueda de canciones reales, clasificador de tipo de contenido, catálogos de sugerencias/entrenamiento/vestuario, generador de comentarios de aprovechamiento. |
| `app.py` | La app de Streamlit (interfaz de chat). |
| `entrenar_modelo.py` | Script standalone para (re)entrenar el modelo y generar el `.joblib`. |
| `modelo_sincopa_rf.joblib` | Modelo `RandomForestClassifier` ya entrenado, listo para usar. |
| `dataset_bachata_salsa_quebradita.csv` | Dataset real: 482 canciones (Spotify + SoundCloud), 3 clases. |
| `Sincopa_Entrenamiento_v4_autocontenido.ipynb` | Notebook de Colab/Jupyter, autocontenido (genera `sincopa_core.py` con `%%writefile`), documenta y reproduce el entrenamiento paso a paso. |

---

## Cómo funciona (flujo de una consulta)

Cuando alguien pega un link en el chat, la app sigue este pipeline:

1. **¿Es un link soportado?** Solo se aceptan links de `open.spotify.com`, `youtube.com`/`youtu.be` o `music.apple.com`. Cualquier otra entrada de texto se interpreta como comando (sugerencia / entrenamiento / vestuario), no como canción a evaluar.
2. **Extracción del título** vía oEmbed (YouTube) o scraping del `<title>` de la página (Spotify/Apple Music), con limpieza de sufijos típicos de cada plataforma.
3. **Clasificación de tipo de contenido** (`sincopa_core.clasificar_tipo_contenido`): decide si el título corresponde a
   - una canción reconocible (coincide con el dataset o con una palabra clave de género),
   - contenido con música pero **no dedicado** a bailar (tráiler, gameplay, resumen deportivo, anuncio...),
   - contenido **no musical** (podcast, tutorial, entrevista...), o
   - algo ambiguo que no parece título de canción.
4. **Si es una canción**, se busca coincidencia exacta en el dataset real (`buscar_en_dataset`). Si existe, se usan **sus features reales**. Si no, se estima un vector de features plausible muestreando dentro del rango real (percentiles 5-95) del género detectado por palabra clave.
5. El vector de features pasa por el `RandomForestClassifier`, que devuelve el género (`Bachata`, `Salsa` o `Quebradita`) y la probabilidad. Se aplica un umbral de confianza (más exigente si no hubo palabra clave de por medio) para rechazar predicciones poco fiables en vez de forzarlas.
6. Si el género predicho es `Salsa` y el título/artista coincide con palabras clave de Timba cubana (Los Van Van, Alexander Abreu, etc.), se etiqueta como **variante Timba** solo para elegir catálogo de contenido — el modelo en sí nunca predice "Timba" como clase, porque el dataset no la tiene etiquetada así.
7. Se genera el comentario de **aprovechamiento coreográfico** (`comentar_aprovechamiento`) comparando el tempo, la densidad de tatum, el número de secciones y la combinación energía/valence de la canción contra la mediana y los percentiles de su género.

---

## El dataset

`dataset_bachata_salsa_quebradita.csv` — 482 canciones reales, extraídas de Spotify y SoundCloud.

**Columnas relevantes usadas por el modelo** (definidas en `sincopa_core.FEATURES`):

```
tempo, danceability, energy, valence, speechiness, acousticness,
densidad_tatum, num_secciones, num_compases, num_tiempos_beats
```

**Distribución de clases:**

| Género | Canciones |
|---|---|
| Salsa | 204 |
| Bachata | 146 |
| Quebradita | 132 |

> **Nota importante:** el dataset no tiene una etiqueta "Timba" independiente. Las canciones de timba cubana (Los Van Van, Alexander Abreu, Issac Delgado, etc.) están etiquetadas como `Salsa`. Por eso el modelo entrena sobre **3 clases**, y "Timba" se maneja únicamente como una variante de despliegue (detectada por palabra clave) para elegir catálogos de sugerencias/entrenamiento/vestuario específicos — nunca como una salida del clasificador, ya que los datos no la respaldan como clase aparte.

Si quieres ampliar el dataset (más canciones, un género nuevo), solo necesitas:
1. Agregar filas al CSV con las mismas columnas.
2. Si agregas un género nuevo, agrégalo también a `sincopa_core.PALABRAS_GENERO_BASE`, `CATALOGO_SUGERENCIAS`, `CATALOGO_ENTRENAMIENTOS`, `CATALOGO_VESTUARIO` y `DETALLES_COREOGRAFICOS`.
3. Correr `entrenar_modelo.py` de nuevo.

---

## Instalación y ejecución local

### Requisitos
- Python 3.9 o superior
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone <URL-de-tu-repo>
cd <carpeta-del-repo>

# 2. (Recomendado) Crear un entorno virtual
python3 -m venv venv
source venv/bin/activate      # en Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install streamlit pandas numpy scikit-learn joblib requests beautifulsoup4

# 4. Correr la app
streamlit run app.py
```

Streamlit abrirá automáticamente `http://localhost:8501` en tu navegador. Si `modelo_sincopa_rf.joblib` está en la misma carpeta (viene incluido en el repo), la app lo carga directo — no necesitas entrenar nada para empezar a usarla.

> Si no tienes el `.joblib` (por ejemplo, lo borraste o lo excluiste del repo), la app lo detecta automáticamente y entrena el modelo en memoria al arrancar, usando el mismo dataset y la misma lógica — solo tardará unos segundos más en el primer arranque.

---

## Reentrenar el modelo

Si modificaste el dataset o quieres regenerar el `.joblib` desde cero:

```bash
python entrenar_modelo.py
```

Esto imprime el accuracy en un holdout del 20% y guarda `modelo_sincopa_rf.joblib` en la carpeta actual, listo para que `app.py` lo recoja en el siguiente arranque.

---

## Usar el notebook de entrenamiento

`Sincopa_Entrenamiento_v4_autocontenido.ipynb` está pensado para Google Colab o Jupyter, y es **autocontenido**: no necesitas subir `sincopa_core.py` por separado, porque la Celda 2 lo genera automáticamente con `%%writefile` (garantizando que es exactamente el mismo código que usa la app).

1. Sube el notebook a Colab (o ábrelo en Jupyter).
2. Corre las celdas en orden (`Entorno de ejecución → Ejecutar todas`).
3. Cuando la Celda 3 lo pida, sube `dataset_bachata_salsa_quebradita.csv`.
4. El notebook hace EDA, entrena el modelo, muestra matriz de confusión e importancia de features, y guarda `modelo_sincopa_rf.joblib` al final.
5. Descarga `sincopa_core.py` y `modelo_sincopa_rf.joblib` generados y colócalos junto a `app.py` para desplegar (o usa directamente los que ya vienen en el repo — son idénticos si no cambiaste el dataset).

---

## Desplegar en Streamlit Community Cloud

1. Sube este repositorio a GitHub (incluyendo `modelo_sincopa_rf.joblib` y el `.csv`).
2. Crea un archivo `requirements.txt` en la raíz con:
   ```
   streamlit
   pandas
   numpy
   scikit-learn
   joblib
   requests
   beautifulsoup4
   ```
3. Ve a [share.streamlit.io](https://share.streamlit.io), conecta tu cuenta de GitHub y selecciona el repositorio.
4. Indica `app.py` como archivo principal.
5. Deploy. La primera carga puede tardar un poco mientras Streamlit instala las dependencias.

---

## Limitaciones conocidas

- **No se analiza audio real.** Cuando una canción no está en el dataset, sus features se *estiman* dentro del rango real observado para el género detectado por palabra clave — no hay extracción acústica del link.
- **Solo 3 géneros de modelo** (Bachata, Salsa, Quebradita). Timba se maneja como variante de catálogo, no como clase predicha.
- **La detección de contenido no musical / música no dedicada es por reglas (regex sobre el título)**, no por análisis del contenido audiovisual real — un título ambiguo puede escapar a la clasificación correcta.
- **Solo se aceptan links de Spotify, YouTube y Apple Music.** Otras plataformas (SoundCloud, TikTok, etc.) son rechazadas explícitamente.
- La extracción de título depende de que la página exponga metadatos accesibles (oEmbed / `<title>`); cambios en esas plataformas pueden requerir ajustes en `extraer_titulo_link` (en `app.py`).

---

## Preguntas frecuentes / troubleshooting

**La app tarda mucho en arrancar la primera vez.**
Si no encuentra `modelo_sincopa_rf.joblib`, lo entrena en memoria al vuelo. Confirma que el archivo `.joblib` esté en la misma carpeta que `app.py`.

**`ModuleNotFoundError: No module named 'sincopa_core'`.**
`app.py` y `entrenar_modelo.py` deben estar en la **misma carpeta** que `sincopa_core.py`; Python los importa por ruta relativa.

**Quiero agregar un género nuevo.**
Necesitas: (1) canciones reales etiquetadas con ese género en el CSV, (2) agregar el género a los catálogos en `sincopa_core.py` (sugerencias, entrenamientos, vestuario, detalles coreográficos), y (3) correr `entrenar_modelo.py` de nuevo.

**El modelo predice con 100% de confianza casi siempre.**
Es esperable: los rangos de tempo entre Bachata, Salsa y Quebradita en el dataset real no se solapan, así que el bosque aleatorio separa las clases con muy poco margen de error. Esto puede cambiar si agregas géneros con rangos de tempo más parecidos entre sí.

**¿Por qué a veces dice "certeza insuficiente" en vez de dar un género?**
Ocurre cuando el título no coincide con ninguna canción del dataset ni con ninguna palabra clave de género — en ese caso se le exige más confianza al modelo (65% en vez de 55%) antes de arriesgar una clasificación, para evitar el problema original de "cualquier texto se detecta como canción".
