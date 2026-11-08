import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json


st.set_page_config(
    page_title="Clasificador de Mascotas IA",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================
# ESTILOS - MODO CLARO
# ==========================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 55%, #ffffff 100%);
    color: #1e293b;
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2.5rem;
    max-width: 980px;
}

/* Encabezado */
.hero {
    text-align: center;
    padding: 1.5rem 1rem 1rem;
    margin-bottom: 1.5rem;
}

.nombre {
    display: inline-block;
    background: #e0e7ff;
    color: #4338ca;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
}

.titulo {
    font-size: 2.7rem;
    font-weight: 800;
    color: #111827;
    line-height: 1.15;
    margin-bottom: 0.5rem;
}

.titulo span {
    background: linear-gradient(90deg, #6366f1, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitulo {
    color: #64748b;
    font-size: 1.05rem;
    max-width: 620px;
    margin: 0 auto;
}

/* Tarjetas */
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    padding: 1.6rem;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    margin-bottom: 1.2rem;
}

.card-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.8rem;
}

/* Botón */
div.stButton > button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 0.8rem 1rem;
    font-weight: 700;
    color: white;
    background: linear-gradient(90deg, #6366f1, #06b6d4);
    box-shadow: 0 10px 22px rgba(99, 102, 241, 0.25);
    transition: transform .18s ease, box-shadow .18s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 28px rgba(6, 182, 212, 0.28);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #f1f5f9;
    border-radius: 14px;
    padding: 5px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #475569;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: #ffffff;
    color: #4338ca !important;
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.08);
}

/* Imagen */
[data-testid="stImage"] img {
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.10);
}

/* Métricas */
[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1rem;
}

[data-testid="stMetricValue"] {
    font-weight: 800;
    color: #0f172a;
}

/* Progreso */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #6366f1, #06b6d4) !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 0.85rem;
    margin-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# MODELO
# ==========================
@st.cache_resource
def cargar_recursos():
    modelo = tf.keras.models.load_model(
        "modelo_perro_Gato/waste_mobilenet.keras",
        compile=False
    )

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    with open("modelo_perro_Gato/class_names.json", "r") as f:
        clases = json.load(f)

    return modelo, clases


modelo, class_names = cargar_recursos()

# ==========================
# ESTADO DE LA CÁMARA
# ==========================
if "camara_activa" not in st.session_state:
    st.session_state["camara_activa"] = False

# ==========================
# ENCABEZADO
# ==========================
st.markdown("""
<div class='hero'>
    <div class='nombre'>Aaron Isaac Lopez - Cta. 20211900269</div>
    <div class='titulo'>🐾 Clasificador de <span>Mascotas IA</span></div>
    <div class='subtitulo'>
        Sube una fotografía o toma una foto para identificar si corresponde a un perro 🐶 o un gato 🐱.
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================
# SELECCIÓN DE IMAGEN
# ==========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='card-title'>📷 Selecciona una imagen</div>", unsafe_allow_html=True)

opcion = st.radio(
    "Elige cómo ingresar la imagen:",
    ["📂 Subir desde el dispositivo", "📸 Tomar una foto"],
    horizontal=True
)

imagen = None

if opcion == "📂 Subir desde el dispositivo":

    archivo = st.file_uploader(
        "Arrastra una imagen o haz clic para seleccionarla",
        type=["jpg", "jpeg", "png"]
    )

    if archivo:
        imagen = Image.open(archivo)

else:
    col_cam_1, col_cam_2 = st.columns([1, 2])

    with col_cam_1:
        activar_camara = st.button(
            "📷 Tomar foto",
            use_container_width=True
        )

        if activar_camara:
            st.session_state["camara_activa"] = True

        if st.session_state["camara_activa"]:
            if st.button("❌ Cerrar cámara", use_container_width=True):
                st.session_state["camara_activa"] = False
                st.rerun()

    with col_cam_2:
        if st.session_state["camara_activa"]:
            camara = st.camera_input("Permite el acceso a la cámara y toma la fotografía")

            if camara:
                imagen = Image.open(camara)
        else:
            st.info("Presiona **📷 Tomar foto** para activar la cámara.")

st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# ANÁLISIS Y RESULTADO
# ==========================
if imagen is not None:

    imagen_rgb = imagen.convert("RGB")

    col1, col2 = st.columns([1.15, 1])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Imagen seleccionada</div>", unsafe_allow_html=True)

        st.image(imagen_rgb, use_container_width=True)

        analizar = st.button(
            "Analizar imagen",
            type="primary",
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📊 Resultado</div>", unsafe_allow_html=True)

        if analizar:
            with st.spinner("Analizando imagen..."):

                img = imagen_rgb.resize((224, 224))

                arr = np.expand_dims(
                    np.array(img, dtype=np.float32) / 255.0,
                    axis=0
                )

                pred = modelo.predict(arr, verbose=0)[0]
                prob = pred[0]

                if prob > 0.5:
                    resultado = "🐱 Gato"
                    confianza = float(prob)
                else:
                    resultado = "🐶 Perro"
                    confianza = float(1 - prob)

                st.session_state["resultado"] = resultado
                st.session_state["confianza"] = confianza

        if "resultado" in st.session_state:

            st.metric(
                "Predicción",
                st.session_state["resultado"]
            )

            st.metric(
                "Confianza",
                f"{st.session_state['confianza']:.2%}"
            )

            st.progress(st.session_state["confianza"])

            if "Perro" in st.session_state["resultado"]:
                st.success("La IA detecta un perro.")
            else:
                st.success("La IA detecta un gato.")

            if st.session_state["confianza"] >= 0.90:
                st.balloons()

        else:
            st.info("Presiona **Analizar imagen** para obtener la predicción.")

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("👆 Selecciona o toma una fotografía para comenzar.")

# ==========================
# FOOTER
# ==========================
st.markdown("""
<div class='footer'>
    ‍💻 Aaron Isaac Lopez · UNAH · IA 2026 · 20211900269
</div>
""", unsafe_allow_html=True)