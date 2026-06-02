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
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import threading
from datetime import datetime
from fpdf import FPDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ========== НАСТРОЙКА СТРАНИЦЫ ==========
st.set_page_config(page_title="Распознавание эмоций", page_icon="🎭", layout="centered")

# ========== КАСТОМНЫЕ СТИЛИ ==========
st.markdown("""
<style>
    /* Общие отступы */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 800px;
    }

    /* Шапка приложения */
    .app-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 20px 24px;
        background: linear-gradient(135deg, #6C5CE7 0%, #4834D4 100%);
        border-radius: 16px;
        margin-bottom: 24px;
    }
    .app-header .icon {
        font-size: 40px;
        line-height: 1;
    }
    .app-header .title {
        font-size: 26px;
        font-weight: 700;
        color: white;
        margin: 0;
    }
    .app-header .subtitle {
        font-size: 14px;
        color: rgba(255,255,255,0.85);
        margin: 2px 0 0 0;
    }

    /* Кнопки */
    .stButton button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        transform: translateY(-1px);
    }

    /* Вкладки */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        font-weight: 500;
    }

    /* Метрики */
    [data-testid="stMetric"] {
        background: #F8F9FE;
        border: 1px solid #ECECF5;
        border-radius: 12px;
        padding: 14px 18px;
    }

    /* Радиокнопки модели */
    .stRadio > div {
        gap: 12px;
    }

    /* Карточки результатов лиц */
    .face-card {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        border: 1px solid #ECECF5;
        border-radius: 12px;
        margin-bottom: 8px;
        background: white;
    }
    .face-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .face-name {
        font-size: 15px;
        flex: 1;
    }
    .face-score {
        font-size: 14px;
        font-weight: 600;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Шапка
st.markdown("""
<div class="app-header">
    <div class="icon">🎭</div>
    <div>
        <p class="title">Распознавание эмоций</p>
        <p class="subtitle">Анализ медиапотока на основе ИИ</p>
    </div>
</div>
""", unsafe_allow_html=True)

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

EMOTION_COLORS_RGB = {
    "happy":    (0, 200, 0),
    "sad":      (50, 100, 220),
    "angry":    (220, 40, 40),
    "fear":     (150, 50, 180),
    "surprise": (240, 200, 0),
    "disgust":  (150, 130, 0),
    "neutral":  (140, 140, 140),
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
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    for face_box, emotion_en, score in faces_with_emotions:
        if emotion_en is None:
            continue

        x, y, w, h = face_box
        color = get_color(emotion_en)
        label = f"{ru(emotion_en)} {score:.0f}%"

        padding = 6
        max_text_width = max(30, w - padding * 2)
        font = fit_font(draw, label, max_text_width, max_size=26, min_size=12)

        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        label_y_top = max(0, y - text_h - padding * 2 - 4)

        draw.rectangle(
            [x, label_y_top, x + text_w + padding * 2, label_y_top + text_h + padding * 2],
            fill=color
        )
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

# ========== ГЕНЕРАЦИЯ ГРАФИКА ЭМОЦИЙ ДЛЯ PDF ==========
def generate_emotion_timeline_chart(results_log, output_path):
    """
    График: по горизонтали — время, по вертикали — эмоции.
    Цветные точки/полосы показывают доминантную эмоцию в каждый момент.
    """
    if not results_log:
        return False

    times = [r["time"] for r in results_log]

    # Определяем доминантную эмоцию для каждого момента
    emotion_keys = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
    dominant_emotions = []
    for r in results_log:
        scores = {k: r.get(k, 0) for k in emotion_keys}
        dominant = max(scores, key=scores.get)
        dominant_emotions.append(dominant)

    # Числовой индекс для каждой эмоции (для оси Y)
    emotion_to_idx = {e: i for i, e in enumerate(emotion_keys)}
    y_values = [emotion_to_idx[e] for e in dominant_emotions]

    # Цвета для каждой точки
    colors_for_plot = []
    for e in dominant_emotions:
        rgb = EMOTION_COLORS_RGB.get(e, (140, 140, 140))
        colors_for_plot.append((rgb[0]/255, rgb[1]/255, rgb[2]/255))

    # Русские названия для оси Y
    y_labels = [ru(e) for e in emotion_keys]

    fig, ax = plt.subplots(figsize=(10, 4))

    # Рисуем цветные полосы между точками
    for i in range(len(times) - 1):
        ax.barh(
            y_values[i],
            times[i+1] - times[i],
            left=times[i],
            height=0.7,
            color=colors_for_plot[i],
            edgecolor='none'
        )
    # Последняя точка — рисуем маленькую полоску
    if len(times) >= 1:
        bar_width = (times[-1] - times[0]) / max(len(times), 1) if len(times) > 1 else 1
        ax.barh(
            y_values[-1],
            bar_width,
            left=times[-1],
            height=0.7,
            color=colors_for_plot[-1],
            edgecolor='none'
        )

    ax.set_yticks(range(len(emotion_keys)))
    ax.set_yticklabels(y_labels, fontsize=12)
    ax.set_xlabel("Время (сек)", fontsize=12)
    ax.set_title("Эмоции по времени", fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True

# ========== ВЫБОР ЛУЧШИХ СКРИНШОТОВ ==========
def select_best_screenshots(screenshots_log, max_count=3):
    """
    Выбирает лучшие скриншоты — из середины самых длинных отрезков
    одной эмоции с наивысшей уверенностью.
    """
    if not screenshots_log:
        return []

    # Группируем последовательные кадры с одной эмоцией в сегменты
    segments = []
    current_segment = {
        "emotion": screenshots_log[0]["emotion"],
        "start_idx": 0,
        "entries": [screenshots_log[0]]
    }

    for i in range(1, len(screenshots_log)):
        entry = screenshots_log[i]
        if entry["emotion"] == current_segment["emotion"]:
            current_segment["entries"].append(entry)
        else:
            segments.append(current_segment)
            current_segment = {
                "emotion": entry["emotion"],
                "start_idx": i,
                "entries": [entry]
            }
    segments.append(current_segment)

    # Для каждого сегмента считаем "качество" = длина * средняя уверенность
    for seg in segments:
        scores = [e["score"] for e in seg["entries"]]
        seg["quality"] = len(seg["entries"]) * (sum(scores) / len(scores))
        seg["avg_score"] = sum(scores) / len(scores)

    # Сортируем по качеству, берём топ-N с разными эмоциями
    segments.sort(key=lambda s: s["quality"], reverse=True)

    selected = []
    used_emotions = set()

    for seg in segments:
        if len(selected) >= max_count:
            break
        # Предпочитаем разные эмоции
        if seg["emotion"] in used_emotions and len(selected) < max_count:
            continue
        # Берём кадр из середины сегмента
        mid_idx = len(seg["entries"]) // 2
        best_entry = seg["entries"][mid_idx]
        selected.append(best_entry)
        used_emotions.add(seg["emotion"])

    # Если не набрали max_count с разными эмоциями — добираем любыми
    if len(selected) < max_count:
        for seg in segments:
            if len(selected) >= max_count:
                break
            mid_idx = len(seg["entries"]) // 2
            candidate = seg["entries"][mid_idx]
            if candidate not in selected:
                selected.append(candidate)

    # Сортируем по времени
    selected.sort(key=lambda e: e["time"])
    return selected

# ========== ГЕНЕРАЦИЯ PDF-ОТЧЁТА ==========
def generate_pdf_report(video_name, model_name, results_log, screenshots_log, duration):
    """Генерирует PDF-отчёт и возвращает байты."""

    # Находим преобладающую эмоцию
    emotion_keys = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
    all_dominants = []
    for r in results_log:
        scores = {k: r.get(k, 0) for k in emotion_keys}
        dominant = max(scores, key=scores.get)
        all_dominants.append(dominant)

    from collections import Counter
    emotion_counter = Counter(all_dominants)
    most_common_en = emotion_counter.most_common(1)[0][0]
    most_common_ru = ru(most_common_en)

    # Генерируем график
    chart_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    chart_ok = generate_emotion_timeline_chart(results_log, chart_path)

    # Выбираем лучшие скриншоты
    best_screenshots = select_best_screenshots(screenshots_log, max_count=3)

    # Сохраняем скриншоты во временные файлы
    screenshot_paths = []
    for entry in best_screenshots:
        img_bgr = entry["frame"]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        pil_img.save(tmp.name, quality=85)
        screenshot_paths.append({
            "path": tmp.name,
            "time": entry["time"],
            "emotion": ru(entry["emotion"]),
            "score": entry["score"]
        })

    # Создаём PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Загружаем шрифт с поддержкой кириллицы ---
    font_path = None
    for fp in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/Arial.ttf", "C:/Windows/Fonts/segoeui.ttf"]:
        if os.path.exists(fp):
            font_path = fp
            break

    if font_path:
        pdf.add_font("CustomFont", "", font_path, uni=True)
        pdf.add_font("CustomFont", "B", font_path, uni=True)
        font_name = "CustomFont"
    else:
        font_name = "Helvetica"

    # --- Страница 1: Шапка + График ---
    pdf.add_page()

    # Заголовок
    pdf.set_font(font_name, "B", 20)
    pdf.cell(0, 12, "Отчёт по анализу эмоций", ln=True, align="C")
    pdf.ln(8)

    # Шапка
    pdf.set_font(font_name, "", 12)
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    pdf.cell(0, 8, f"Дата и время отчёта: {now}", ln=True)
    pdf.cell(0, 8, f"Видеофайл: {video_name}", ln=True)
    pdf.cell(0, 8, f"Длительность: {duration:.1f} сек", ln=True)
    pdf.cell(0, 8, f"Модель: {model_name}", ln=True)
    pdf.cell(0, 8, f"Преобладающая эмоция: {most_common_ru}", ln=True)
    pdf.ln(6)

    # График
    if chart_ok and os.path.exists(chart_path):
        pdf.set_font(font_name, "B", 14)
        pdf.cell(0, 10, "График эмоций по времени", ln=True)
        pdf.ln(2)
        # Вписываем график в ширину страницы
        page_width = pdf.w - pdf.l_margin - pdf.r_margin
        pdf.image(chart_path, x=pdf.l_margin, w=page_width)
        pdf.ln(4)

    # --- Скриншоты ---
    if screenshot_paths:
        # Проверяем хватит ли места на текущей странице
        if pdf.get_y() > 200:
            pdf.add_page()

        pdf.set_font(font_name, "B", 14)
        pdf.cell(0, 10, "Ключевые кадры", ln=True)
        pdf.ln(2)

        for sc in screenshot_paths:
            # Проверяем место на странице
            if pdf.get_y() > 190:
                pdf.add_page()

            pdf.set_font(font_name, "", 11)
            pdf.cell(0, 7, f"Время: {sc['time']:.1f} сек  |  Эмоция: {sc['emotion']}  |  Уверенность: {sc['score']:.0f}%", ln=True)
            pdf.ln(1)

            if os.path.exists(sc["path"]):
                page_width = pdf.w - pdf.l_margin - pdf.r_margin
                img_w = min(page_width, 160)
                pdf.image(sc["path"], x=pdf.l_margin + (page_width - img_w) / 2, w=img_w)
                pdf.ln(6)

    # Сохраняем в байты
    pdf_bytes = pdf.output()

    # Чистим временные файлы
    try:
        os.unlink(chart_path)
        for sc in screenshot_paths:
            os.unlink(sc["path"])
    except Exception:
        pass

    return bytes(pdf_bytes)

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
tab_photo, tab_video, tab_webcam = st.tabs(["📷 Фото", "🎬 Видео", "📹 Веб-камера"])

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
                        # Карточки-метрики
                        dominant_list = [r["Эмоция"] for r in all_results]
                        most_common_emotion = max(set(dominant_list), key=dominant_list.count)
                        avg_conf = sum(float(r["Уверенность"].replace("%", "")) for r in all_results) / len(all_results)

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Найдено лиц", len(all_results))
                        m2.metric("Преобладает", most_common_emotion)
                        m3.metric("Ср. уверенность", f"{avg_conf:.0f}%")

                        st.subheader("Результаты по каждому лицу")
                        # Карточки лиц с цветными точками
                        for fwe, res in zip(faces_with_emotions, all_results):
                            emotion_en = fwe[1]
                            color = get_color(emotion_en)
                            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                            st.markdown(f"""
                            <div class="face-card">
                                <div class="face-dot" style="background:{color_hex};"></div>
                                <span class="face-name">{res['Лицо']} — {res['Эмоция']}</span>
                                <span class="face-score">{res['Уверенность']}</span>
                            </div>
                            """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Ошибка: {e}")

# ========== ВКЛАДКА ВИДЕО ==========
with tab_video:
    uploaded_videos = st.file_uploader(
        "Выберите одно или несколько видео",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
        accept_multiple_files=True
    )

    if uploaded_videos:
        st.subheader("Настройки обработки")
        analysis_interval = st.slider(
            "Обновлять эмоцию раз в N кадров",
            min_value=1, max_value=30, value=5,
            help="Меньше = точнее, но медленнее. 5 — хороший баланс."
        )

        video_infos = []
        for i, vid in enumerate(uploaded_videos):
            tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile_in.write(vid.read())
            tfile_in.close()

            cap = cv2.VideoCapture(tfile_in.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()

            video_infos.append({
                "name": vid.name,
                "path": tfile_in.name,
                "fps": fps,
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "duration": duration,
            })

            st.info(f"📹 **{vid.name}** — {duration:.1f} сек | {total_frames} кадров | {fps:.1f} FPS | {width}x{height}")

        total_analyses = sum(vi["total_frames"] // analysis_interval for vi in video_infos)
        total_all_frames = sum(vi["total_frames"] for vi in video_infos)
        sec_per_analysis = 0.4 if model_name == "DeepFace" else 1.0
        estimated_total = total_analyses * sec_per_analysis * 1.5 + total_all_frames * 0.03

        col1, col2, col3 = st.columns(3)
        col1.metric("Видеофайлов", len(uploaded_videos))
        col2.metric("Всего анализов", total_analyses)
        col3.metric("Примерное время", f"{estimated_total:.0f} сек")
        st.caption("⚠️ Время зависит от количества лиц в кадре. Видео обрабатываются последовательно.")

        if st.button("🎬 Обработать все видео", type="primary"):
            face_detector = load_face_detector()

            for vi_idx, vi in enumerate(video_infos):
                st.divider()
                st.subheader(f"📹 {vi_idx + 1}/{len(video_infos)}: {vi['name']}")

                video_path_in = vi["path"]
                fps = vi["fps"]
                total_frames = vi["total_frames"]
                width = vi["width"]
                height = vi["height"]

                tfile_raw = tempfile.NamedTemporaryFile(delete=False, suffix="_raw.mp4")
                tfile_raw.close()
                video_path_raw = tfile_raw.name

                tfile_final = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile_final.close()
                video_path_final = tfile_final.name

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(video_path_raw, fourcc, fps, (width, height))

                cap = cv2.VideoCapture(video_path_in)
                progress_bar = st.progress(0.0, text=f"Запуск {vi['name']}...")
                start_time = time.time()

                frame_idx = 0
                last_results = []
                results_log = []
                screenshots_log = []  # Для PDF-отчёта

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

                                        # Сохраняем кадр с рамкой для скриншотов
                                        annotated_frame = draw_overlay(frame.copy(), new_results)
                                        screenshots_log.append({
                                            "time": timestamp,
                                            "emotion": dominant,
                                            "score": score,
                                            "frame": annotated_frame
                                        })
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
                            text=f"{vi['name']}: {frame_idx}/{total_frames} | {elapsed:.1f} сек"
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
                progress_bar.progress(1.0, text=f"✅ {vi['name']} — готово за {total_time:.1f} сек")

                st.subheader(f"🎥 Результат: {vi['name']}")
                with open(final_path, "rb") as f:
                    video_bytes = f.read()

                st.video(video_bytes)
                st.download_button(
                    label=f"⬇️ Скачать {vi['name']}",
                    data=video_bytes,
                    file_name=f"emotion_{vi['name']}",
                    mime="video/mp4",
                    key=f"download_{vi_idx}"
                )

                if len(results_log) > 0:
                    df = pd.DataFrame(results_log)
                    df = df.rename(columns={k: ru(k) for k in df.columns if k != "time"})
                    df = df.set_index("time")

                    st.subheader(f"📈 График эмоций: {vi['name']}")
                    st.line_chart(df)

                    emotion_cols = [c for c in df.columns]
                    dominants = df[emotion_cols].idxmax(axis=1)
                    most_common = dominants.mode()[0]
                    st.success(f"Преобладающая эмоция главного лица: **{most_common}**")

                    # PDF-отчёт
                    with st.spinner("Генерирую PDF-отчёт..."):
                        pdf_bytes = generate_pdf_report(
                            video_name=vi["name"],
                            model_name=model_name,
                            results_log=results_log,
                            screenshots_log=screenshots_log,
                            duration=vi["duration"]
                        )
                    import base64
                    b64 = base64.b64encode(pdf_bytes).decode()
                    pdf_filename = f"report_{vi['name']}.pdf"
                    href = f'<a href="data:application/pdf;base64,{b64}" download="{pdf_filename}" target="_blank" style="display:inline-block;padding:0.5em 1em;background-color:#FF4B4B;color:white;text-decoration:none;border-radius:8px;font-weight:bold;">📄 Скачать PDF-отчёт: {vi["name"]}</a>'
                    st.markdown(href, unsafe_allow_html=True)

            st.success(f"✅ Все {len(video_infos)} видео обработаны!")

# ========== ВКЛАДКА ВЕБ-КАМЕРА ==========
with tab_webcam:
    st.write("Анализ эмоций в реальном времени через веб-камеру.")

    # Прогрев моделей при открытии вкладки
    with st.spinner("Загружаю модели для веб-камеры..."):
        _ = load_face_detector()
        if model_name == "ViT":
            _ = load_vit_model()

    # Настройки
    webcam_interval = st.slider(
        "Анализировать каждый N-й кадр",
        min_value=1, max_value=30, value=10,
        help="Больше = быстрее видео, но реже обновляется эмоция. 10 — хороший баланс.",
        key="webcam_interval"
    )

    st.caption(f"🔧 Модель: **{model_name}** | Анализ каждого {webcam_interval}-го кадра")

    # Общее состояние между потоками
    class WebcamState:
        def __init__(self):
            self.lock = threading.Lock()
            self.frame_count = 0
            self.last_results = []
            self.detector = None
            self.model_name = "DeepFace"
            self.interval = 10

    if "webcam_state" not in st.session_state:
        st.session_state.webcam_state = WebcamState()

    state = st.session_state.webcam_state
    state.model_name = model_name
    state.interval = webcam_interval

    # Загружаем детектор заранее
    state.detector = load_face_detector()

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        with state.lock:
            state.frame_count += 1
            current_frame = state.frame_count
            interval = state.interval
            current_model = state.model_name
            detector = state.detector

        # Детекция лиц — каждый кадр (быстрая операция)
        faces = detect_all_faces(img, detector)

        if faces:
            # Анализ эмоций — только каждый N-й кадр
            if current_frame % interval == 0:
                new_results = []
                for (x, y, w, h) in faces:
                    face_crop = img[y:y+h, x:x+w]
                    if face_crop.size == 0:
                        continue
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    try:
                        emotions = analyze_frame(face_rgb, current_model)
                        dominant = max(emotions, key=emotions.get)
                        score = emotions[dominant]
                        new_results.append(((x, y, w, h), dominant, score))
                    except Exception:
                        pass
                with state.lock:
                    state.last_results = new_results
            else:
                # Между анализами — обновляем позиции лиц, сохраняем эмоции
                with state.lock:
                    old_results = state.last_results
                updated = []
                for i, face_box in enumerate(faces):
                    if i < len(old_results):
                        _, em, sc = old_results[i]
                        updated.append((face_box, em, sc))
                with state.lock:
                    if updated:
                        state.last_results = updated

            # Рисуем результат
            with state.lock:
                results_to_draw = list(state.last_results)
            if results_to_draw:
                img = draw_overlay(img, results_to_draw)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="emotion-webcam",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        sendback_audio=False,
    )

    st.info("💡 Нажмите **START** чтобы включить камеру. Эмоции определяются в реальном времени.")

# ========== ИНФО ==========
with st.expander("ℹ️ О моделях и технологиях"):
    st.write("""
    **Детекция лиц:** MediaPipe BlazeFace (Google). Быстрый и точный детектор, находит все лица в кадре.

    **DeepFace (CNN):** сверточная нейросеть на основе FER2013. Базовая модель первой итерации.

    **ViT (Vision Transformer):** современная архитектура на основе трансформеров. Модель `trpakov/vit-face-expression`.

    **Обработка нескольких лиц:** каждое лицо анализируется отдельно своей моделью. Эмоции отображаются на русском.

    **Несколько видеопотоков:** можно загрузить несколько видеофайлов и обработать их — каждый со своим графиком эмоций и результатом.

    **Веб-камера:** анализ эмоций в реальном времени через WebRTC. Детекция лиц каждый кадр, анализ эмоций — с настраиваемой частотой.

    **PDF-отчёты:** автоматическая генерация отчёта с графиком эмоций и ключевыми кадрами.

    **Ограничения:** модели обучены на фронтальных лицах (датасет FER2013). На сильно повёрнутых лицах точность снижается.

    **Рендер видео:** OpenCV + перекодировка через ffmpeg в H.264 для веб-совместимости.
    """)