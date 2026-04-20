import streamlit as st
from deepface import DeepFace
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import tempfile
import time
import pandas as pd
import os
import subprocess
import mediapipe as mp
import imageio_ffmpeg

# ========== НАСТРОЙКА СТРАНИЦЫ ==========
st.set_page_config(page_title="Распознавание эмоций", page_icon="🎭", layout="centered")

st.title("🎭 Распознавание эмоций")
st.write("Анализ эмоций на фото и видео с помощью двух разных моделей ИИ.")

# ========== ПЕРЕВОД ЭМОЦИЙ НА РУССКИЙ ==========
EMOTION_RU = {
    "happy":    "радость",
    "sad":      "грусть",
    "angry":    "злость",
    "fear":     "страх",
    "surprise": "удивление",
    "disgust":  "отвращение",
    "neutral":  "нейтральное",
}

def ru(emotion_en):
    return EMOTION_RU.get(emotion_en.lower(), emotion_en)

# ========== ЗАГРУЗКА ViT-МОДЕЛИ ==========
@st.cache_resource
def load_vit_model():
    return pipeline("image-classification", model="trpakov/vit-face-expression")

# ========== ДЕТЕКТОР ЛИЦ ==========
@st.cache_resource
def load_face_detector():
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    base_options = mp_python.BaseOptions(model_asset_path="blaze_face_short_range.tflite")
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5
    )
    return mp_vision.FaceDetector.create_from_options(options)

# ========== ЗАГРУЗКА ШРИФТА (параметризованный размер) ==========
@st.cache_resource
def load_font(size=28):
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def fit_font(draw, text, max_width, max_size=28, min_size=12):
    """
    Подбирает такой размер шрифта, чтобы текст влезал в max_width.
    Начинает с max_size и уменьшает до min_size.
    """
    for size in range(max_size, min_size - 1, -1):
        font = load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        if text_w <= max_width:
            return font
    return load_font(min_size)

# ========== ЦВЕТА ==========
EMOTION_COLORS = {
    "happy":    (0, 200, 0),
    "sad":      (50, 100, 220),
    "angry":    (220, 40, 40),
    "fear":     (150, 50, 180),
    "surprise": (240, 200, 0),
    "disgust":  (150, 130, 0),
    "neutral":  (140, 140, 140),
}

def get_color(emotion_key):
    return EMOTION_COLORS.get(emotion_key.lower(), (255, 255, 255))

# ========== ДЕТЕКЦИЯ ВСЕХ ЛИЦ ==========
def detect_all_faces(frame_bgr, detector):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    if not result.detections:
        return []

    h_img, w_img = frame_bgr.shape[:2]
    faces = []

    for detection in result.detections:
        bbox = detection.bounding_box
        x = max(0, bbox.origin_x)
        y = max(0, bbox.origin_y)
        w = min(bbox.width, w_img - x)
        h = min(bbox.height, h_img - y)
        if w > 0 and h > 0:
            faces.append((x, y, w, h))

    faces.sort(key=lambda f: f[2] * f[3], reverse=True)
    return faces

# ========== РИСОВАНИЕ РАМКИ И ПОДПИСИ ==========
def draw_overlay(frame_bgr, faces_with_emotions):
    """faces_with_emotions — список (face_box, emotion_en, score)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    for face_box, emotion_en, score in faces_with_emotions:
        if emotion_en is None:
            continue

        x, y, w, h = face_box
        color = get_color(emotion_en)
        label = f"{ru(emotion_en)} {score:.0f}%"

        # Подбираем шрифт под ширину рамки (оставляем немного запаса на padding)
        padding = 6
        max_text_width = max(30, w - padding * 2)
        font = fit_font(draw, label, max_text_width, max_size=26, min_size=12)

        # Рамка
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

        # Размеры подписи
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        label_y_top = max(0, y - text_h - padding * 2 - 4)

        # Фон под подпись
        draw.rectangle(
            [x, label_y_top, x + text_w + padding * 2, label_y_top + text_h + padding * 2],
            fill=color
        )
        # Текст
        draw.text((x + padding, label_y_top + padding - 2), label, fill=(255, 255, 255), font=font)

    result_rgb = np.array(pil_img)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

# ========== АНАЛИЗ КАДРА ==========
def analyze_frame(img_array, model_name):
    if model_name == "DeepFace":
        result = DeepFace.analyze(
            img_array, actions=["emotion"],
            enforce_detection=False, silent=True
        )
        return result[0]["emotion"]
    else:
        vit_model = load_vit_model()
        pil_img = Image.fromarray(img_array)
        results = vit_model(pil_img)
        return {item["label"]: item["score"] * 100 for item in results}

# ========== ПЕРЕКОДИРОВКА H.264 ==========
def convert_to_h264(input_path, output_path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "fast",
        "-pix_fmt", "yuv420p", output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

# ========== ВЫБОР МОДЕЛИ ==========
st.subheader("Выберите модель")
model_choice = st.radio(
    "Какой моделью анализировать?",
    options=["DeepFace (CNN, базовая)", "ViT (Transformer, улучшенная)"],
    horizontal=True
)
model_name = "DeepFace" if "DeepFace" in model_choice else "ViT"

st.caption("ℹ️ Модели обучены на фронтальных изображениях лиц. На лицах в профиль или сильно под углом точность заметно падает — это известное ограничение датасета FER2013.")

# ========== ВКЛАДКИ ==========
tab_photo, tab_video = st.tabs(["📷 Фото", "🎬 Видео"])

# ========== ВКЛАДКА ФОТО ==========
with tab_photo:
    uploaded_photo = st.file_uploader("Выберите фото", type=["jpg", "jpeg", "png"], key="photo_uploader")

    if uploaded_photo is not None:
        image = Image.open(uploaded_photo).convert("RGB")
        img_array = np.array(image)

        with st.spinner(f"Ищу лица и анализирую через {model_name}..."):
            try:
                detector = load_face_detector()
                frame_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                faces = detect_all_faces(frame_bgr, detector)

                if not faces:
                    st.warning("Лиц не обнаружено.")
                    st.image(image, caption="Загруженное фото", width="stretch")
                else:
                    faces_with_emotions = []
                    all_results = []

                    for i, (x, y, w, h) in enumerate(faces):
                        face_crop = img_array[y:y+h, x:x+w]
                        if face_crop.size == 0:
                            continue
                        try:
                            emotions = analyze_frame(face_crop, model_name)
                            dominant = max(emotions, key=emotions.get)
                            score = emotions[dominant]
                            faces_with_emotions.append(((x, y, w, h), dominant, score))
                            all_results.append({
                                "Лицо": f"#{i+1}",
                                "Эмоция": ru(dominant),
                                "Уверенность": f"{score:.1f}%"
                            })
                        except Exception:
                            pass

                    annotated = draw_overlay(frame_bgr, faces_with_emotions)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, caption=f"Найдено лиц: {len(faces)}", width="stretch")

                    if all_results:
                        st.subheader("Результаты по каждому лицу")
                        df = pd.DataFrame(all_results)
                        st.dataframe(df, width="stretch")

            except Exception as e:
                st.error(f"Ошибка: {e}")

# ========== ВКЛАДКА ВИДЕО ==========
with tab_video:
    uploaded_video = st.file_uploader("Выберите видео", type=["mp4", "avi", "mov", "mkv"], key="video_uploader")

    if uploaded_video is not None:
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile_in.write(uploaded_video.read())
        tfile_in.close()
        video_path_in = tfile_in.name

        cap = cv2.VideoCapture(video_path_in)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        st.info(f"📊 Видео: {duration:.1f} сек | {total_frames} кадров | {fps:.1f} FPS | {width}x{height}")
        st.video(uploaded_video)

        st.subheader("Настройки обработки")
        analysis_interval = st.slider(
            "Обновлять эмоцию раз в N кадров",
            min_value=1, max_value=30, value=5,
            help="Меньше = точнее, но медленнее. 5 — хороший баланс."
        )

        expected_faces = 1.5
        analyses_count = total_frames // analysis_interval
        sec_per_analysis = 0.4 if model_name == "DeepFace" else 1.0
        rendering_time = total_frames * 0.03
        estimated_time = analyses_count * sec_per_analysis * expected_faces + rendering_time

        col1, col2, col3 = st.columns(3)
        col1.metric("Всего кадров", total_frames)
        col2.metric("Анализов эмоции", analyses_count)
        col3.metric("Примерное время", f"{estimated_time:.0f} сек")
        st.caption("⚠️ Время зависит от количества лиц в кадре.")

        if st.button("🎬 Обработать видео", type="primary"):
            face_detector = load_face_detector()

            tfile_raw = tempfile.NamedTemporaryFile(delete=False, suffix="_raw.mp4")
            tfile_raw.close()
            video_path_raw = tfile_raw.name

            tfile_final = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile_final.close()
            video_path_final = tfile_final.name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_path_raw, fourcc, fps, (width, height))

            cap = cv2.VideoCapture(video_path_in)
            progress_bar = st.progress(0.0, text="Запуск...")
            start_time = time.time()

            frame_idx = 0
            last_results = []
            results_log = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                faces = detect_all_faces(frame, face_detector)

                if faces:
                    if frame_idx % analysis_interval == 0:
                        new_results = []
                        for i, (x, y, w, h) in enumerate(faces):
                            face_crop = frame[y:y+h, x:x+w]
                            if face_crop.size == 0:
                                continue
                            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            try:
                                emotions = analyze_frame(face_rgb, model_name)
                                dominant = max(emotions, key=emotions.get)
                                score = emotions[dominant]
                                new_results.append(((x, y, w, h), dominant, score))

                                if i == 0:
                                    timestamp = frame_idx / fps
                                    results_log.append({"time": timestamp, **emotions})
                            except Exception:
                                pass
                        last_results = new_results
                    else:
                        updated = []
                        for i, face_box in enumerate(faces):
                            if i < len(last_results):
                                _, em, sc = last_results[i]
                                updated.append((face_box, em, sc))
                        if updated:
                            last_results = updated

                    if last_results:
                        frame = draw_overlay(frame, last_results)
                else:
                    last_results = []

                out.write(frame)
                frame_idx += 1

                if frame_idx % 10 == 0 or frame_idx == total_frames:
                    progress = frame_idx / total_frames
                    elapsed = time.time() - start_time
                    progress_bar.progress(
                        min(progress, 1.0),
                        text=f"Обработано: {frame_idx}/{total_frames} | {elapsed:.1f} сек"
                    )

            cap.release()
            out.release()

            progress_bar.progress(1.0, text="Перекодирую в H.264...")

            try:
                convert_to_h264(video_path_raw, video_path_final)
                final_path = video_path_final
            except Exception as e:
                st.warning(f"Не удалось перекодировать: {e}")
                final_path = video_path_raw

            total_time = time.time() - start_time
            progress_bar.progress(1.0, text=f"✅ Готово за {total_time:.1f} сек")

            st.subheader("🎥 Результат")
            with open(final_path, "rb") as f:
                video_bytes = f.read()

            st.video(video_bytes)
            st.download_button(
                label="⬇️ Скачать обработанное видео",
                data=video_bytes,
                file_name="emotion_analysis_result.mp4",
                mime="video/mp4"
            )

            if len(results_log) > 0:
                df = pd.DataFrame(results_log)
                df = df.rename(columns={k: ru(k) for k in df.columns if k != "time"})
                df = df.set_index("time")

                st.subheader("📈 График эмоций (главное лицо)")
                st.line_chart(df)

                emotion_cols = [c for c in df.columns]
                dominants = df[emotion_cols].idxmax(axis=1)
                most_common = dominants.mode()[0]
                st.success(f"Преобладающая эмоция главного лица: **{most_common}**")

# ========== ИНФО ==========
with st.expander("ℹ️ О моделях и технологиях"):
    st.write("""
    **Детекция лиц:** MediaPipe BlazeFace (Google). Быстрый и точный детектор, находит все лица в кадре.

    **DeepFace (CNN):** сверточная нейросеть на основе FER2013. Базовая модель первой итерации.

    **ViT (Vision Transformer):** современная архитектура на основе трансформеров. Модель `trpakov/vit-face-expression`.

    **Обработка нескольких лиц:** каждое лицо анализируется отдельно своей моделью. Эмоции отображаются на русском.

    **Ограничения:** модели обучены на фронтальных лицах (датасет FER2013). На сильно повёрнутых лицах точность снижается.

    **Рендер видео:** OpenCV + перекодировка через ffmpeg в H.264 для веб-совместимости.
    """)