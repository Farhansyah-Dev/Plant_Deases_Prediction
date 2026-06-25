import streamlit as st
import tensorflow as tf
import numpy as np

# --- KONFIGURASI HALAMAN (Harus di paling atas) ---
st.set_page_config(
    page_title="Smart PlantCare",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)


# --- FUNGSI PREDIKSI MODEL ---
@st.cache_resource  # Cache model biar nggak load ulang terus tiap interaksi
def load_model():
    return tf.keras.models.load_model("trained_model.keras")


def model_prediction(test_image):
    model = load_model()
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch

    predictions = model.predict(input_arr)
    predicted_index = np.argmax(predictions)
    confidence_score = np.max(
        predictions
    )  # Fitur Tambahan: Ambil nilai persentase keyakinan

    return predicted_index, confidence_score


# --- SIDEBAR DESIGN ---
st.sidebar.title("🌿 Smart PlantCare")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio(
    "Navigasi", ["Beranda", "Deteksi Penyakit", "Tentang Sistem"]
)
st.sidebar.markdown("---")
st.sidebar.info("Aplikasi pendeteksi penyakit tanaman berbasis Machine Learning.")

# --- HALAMAN: BERANDA ---
if app_mode == "Beranda":
    st.title("Welcome guys to PlantCare")

    # Hero Image
    try:
        st.image("daun_jambu.jpeg", use_container_width=True)
    except:
        st.info(
            "💡 Tip: Pastikan file 'daun_jambu.jpeg' ada di folder yang sama dengan kodemu ya."
        )

    st.markdown("""
    Selamat datang di **Smart PlantCare**. Misi kami simpel: membantu kamu mengidentifikasi penyakit tanaman secara instan dan akurat. 
    Unggah foto daun tanamanmu, dan biarkan sistem kami yang menganalisis kondisinya.
    """)

    # Menggunakan Expander agar UI tetap bersih ala Apple
    with st.expander("Bagaimana Cara Kerjanya?"):
        st.markdown("""
        1. Buka menu **Deteksi Penyakit** di panel sebelah kiri.
        2. Unggah foto daun tanaman yang kamu curigai sakit (pastikan fotonya jelas).
        3. Klik tombol **Analisis Daun**, dan AI kami akan memberikan diagnosis dalam hitungan detik.
        """)

    with st.expander("Kenapa Memilih Sistem Ini?"):
        st.markdown("""
        * **Akurat:** Menggunakan arsitektur Deep Learning modern.
        * **Cepat:** Hasil diagnosis instan tanpa perlu menunggu.
        * **Mudah:** Antarmuka yang didesain agar bisa digunakan oleh siapa saja, dari petani hingga penghobi tanaman keras.
        """)

# --- HALAMAN: TENTANG SISTEM ---
elif app_mode == "Tentang Sistem":
    st.title("Tentang Dataset")
    st.markdown("""
    Sistem ini dilatih menggunakan puluhan ribu citra daun untuk mengenali pola penyakit dengan presisi tinggi.
    """)

    # Menggunakan metrics untuk tampilan data yang elegan
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Total Citra Training", value="70.295")
    col2.metric(label="Total Citra Validasi", value="17.572")
    col3.metric(label="Total Kelas Penyakit", value="38")

    st.markdown("---")
    st.markdown("""
    Dataset ini merupakan hasil rekonstruksi menggunakan *offline augmentation* dari dataset orisinal. 
    Proporsi pembagian data adalah **80% Training** dan **20% Validation** untuk memastikan model belajar dengan optimal tanpa *overfitting*.
    """)

# --- HALAMAN: DETEKSI PENYAKIT ---
elif app_mode == "Deteksi Penyakit":
    st.title("Diagnostik Tanaman 🔍")
    st.markdown("Unggah foto daun untuk memulai analisis.")

    # Area Upload File
    test_image = st.file_uploader(
        "Format yang didukung: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"]
    )

    if test_image is not None:
        # Menampilkan gambar dengan kolom agar ukurannya proporsional dan tidak menutupi layar penuh
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(
                test_image, caption="Citra yang diunggah", use_container_width=True
            )

        # Tombol Analisis dengan styling
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(
            "🔬 Analisis Daun Sekarang", use_container_width=True, type="primary"
        ):

            # Loading state yang elegan
            with st.spinner("PlantCare sedang memindai pola daun..."):
                result_index, confidence = model_prediction(test_image)

            # Daftar Kelas
            class_name = [
                "Apple___Apple_scab",
                "Apple___Black_rot",
                "Apple___Cedar_apple_rust",
                "Apple___healthy",
                "Blueberry___healthy",
                "Cherry_(including_sour)___Powdery_mildew",
                "Cherry_(including_sour)___healthy",
                "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
                "Corn_(maize)___Common_rust_",
                "Corn_(maize)___Northern_Leaf_Blight",
                "Corn_(maize)___healthy",
                "Grape___Black_rot",
                "Grape___Esca_(Black_Measles)",
                "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                "Grape___healthy",
                "Orange___Haunglongbing_(Citrus_greening)",
                "Peach___Bacterial_spot",
                "Peach___healthy",
                "Pepper,_bell___Bacterial_spot",
                "Pepper,_bell___healthy",
                "Potato___Early_blight",
                "Potato___Late_blight",
                "Potato___healthy",
                "Raspberry___healthy",
                "Soybean___healthy",
                "Squash___Powdery_mildew",
                "Strawberry___Leaf_scorch",
                "Strawberry___healthy",
                "Tomato___Bacterial_spot",
                "Tomato___Early_blight",
                "Tomato___Late_blight",
                "Tomato___Leaf_Mold",
                "Tomato___Septoria_leaf_spot",
                "Tomato___Spider_mites Two-spotted_spider_mite",
                "Tomato___Target_Spot",
                "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
                "Tomato___Tomato_mosaic_virus",
                "Tomato___healthy",
            ]

            # Mempercantik format output (menghilangkan underscore)
            predicted_label = (
                class_name[result_index].replace("___", " - ").replace("_", " ")
            )
            confidence_percentage = confidence * 100

            # Menampilkan hasil dengan status box
            st.markdown("---")
            st.subheader("Hasil Diagnosis")

            if "healthy" in predicted_label.lower():
                st.success(
                    f"🌱 **Kabar Baik!** Tanamanmu terdeteksi **{predicted_label}**."
                )
            else:
                st.error(f"⚠️ **Peringatan!** Terdeteksi gejala **{predicted_label}**.")

            st.info(f"Tingkat Keyakinan PlantCare: **{confidence_percentage:.2f}%**")

# --- FOOTER COPYRIGHT ---
st.markdown(
    """
<div style="position: fixed; bottom: 10px; width: 100%; text-align: center;">
    <hr style="border: 0.5px solid #eaeaea; margin-bottom: 5px;">
    <p style="color: #888888; font-size: 12px; margin: 0;">
        &copy; 2026 Smart PlantCare. Designed with minimal elegance.
    </p>
</div>
""",
    unsafe_allow_html=True,
)
