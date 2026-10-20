# ==============================================================================
# 🚀 المنصة الهندسية والرياضية الشاملة - إصدار Streamlit الاحترافي الكامل
# الملف: Mohamed.py
# يدمج: الحل البرمجي والعددي الدقيق + الكاميرا المباشرة + OpenCV + Gemini Vision AI
# في جميع أنواع المسائل الهندسية والرياضية والبرمجية (10 أقسام متكاملة)
# ==============================================================================

import os
import io
import sys
import math
import time
import base64
import datetime
import traceback
import subprocess

# ==============================================================================
# 1. التثبيت التلقائي الآمن للمكتبات (بدون استخدام !pip لتجنب SyntaxError)
# ==============================================================================

REQUIRED_PACKAGES = [
    ("streamlit", "streamlit"),
    ("google.generativeai", "google-generativeai"),
    ("numpy", "numpy"),
    ("sympy", "sympy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("PIL", "pillow"),
    ("cv2", "opencv-python-headless"),
]


def ensure_packages_installed():
    """فحص وتثبيت المكتبات برمجياً بطريقة متوافقة 100% مع ملفات .py واستضافة Streamlit"""
    for module_name, pip_name in REQUIRED_PACKAGES:
        try:
            __import__(module_name)
        except ImportError:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", pip_name]
                )
            except Exception as err:
                print(f"Warning: Could not auto-install {pip_name}: {err}")


ensure_packages_installed()

import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import cv2
import google.generativeai as genai


# ==============================================================================
# 2. إعدادات صفحة Streamlit والتصميم الاحترافي باللغة العربية (RTL CSS)
# ==============================================================================

st.set_page_config(
    page_title="المنصة الهندسية والرياضية الشاملة | المهندس محمد خالد",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 96%;
    }
    .rtl-box {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .hero-banner {
        background: linear-gradient(135deg, #0D47A1 0%, #1565C0 50%, #00695C 100%);
        color: #FFFFFF;
        padding: 22px 28px;
        border-radius: 14px;
        margin-bottom: 20px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.22);
        direction: rtl;
        text-align: right;
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 26px;
        font-weight: 800;
        color: #FFFFFF;
    }
    .hero-banner p {
        margin: 8px 0 0 0;
        font-size: 15px;
        opacity: 0.95;
    }
    .section-card {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-right: 6px solid #1565C0;
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 14px;
        direction: rtl;
        text-align: right;
    }
    .camera-ai-card {
        background-color: #F0F9FF;
        border: 2px dashed #0284C7;
        padding: 16px;
        border-radius: 12px;
        margin-top: 16px;
        margin-bottom: 16px;
        direction: rtl;
        text-align: right;
    }
    .result-box {
        background-color: #FFFFFF;
        border: 1px solid #94A3B8;
        border-top: 4px solid #16A34A;
        padding: 16px;
        border-radius: 10px;
        margin-top: 12px;
        direction: rtl;
        text-align: right;
    }
    .formula-badge {
        display: inline-block;
        background-color: #E0F2FE;
        color: #0369A1;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        margin: 3px;
        font-size: 13px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #EFF6FF;
        border-radius: 8px 8px 0 0;
        padding: 8px 14px;
        font-weight: 700;
        color: #1E3A8A;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 3. تهيئة ذاكرة الجلسة (Streamlit Session State Initialization)
# ==============================================================================

if "history_logs" not in st.session_state:
    st.session_state.history_logs = []

if "extracted_texts" not in st.session_state:
    st.session_state.extracted_texts = {}

if "api_key_saved" not in st.session_state:
    st.session_state.api_key_saved = os.environ.get("GEMINI_API_KEY", "")


def log_operation_to_history(category_name, solver_type, problem_input, solution_output):
    """حفظ كل عملية حل (برمجية أو بالذكاء الاصطناعي) في سجل الجلسة"""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "id": len(st.session_state.history_logs) + 1,
        "timestamp": now_str,
        "category": category_name,
        "solver": solver_type,
        "input": str(problem_input),
        "output": str(solution_output),
    }
    st.session_state.history_logs.append(entry)


# ==============================================================================
# 4. القائمة الجانبية لإعدادات الذكاء الاصطناعي والرؤية الحاسوبية (Sidebar)
# ==============================================================================

with st.sidebar:
    st.markdown(
        "<div class='rtl-box'><h2>⚙️ لوحة التحكم المركزية</h2></div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    api_key_input = st.text_input(
        "🔑 مفتاح Gemini API Key:",
        value=st.session_state.api_key_saved,
        type="password",
        help="أدخل مفتاح Google Gemini API لتفعيل حل المسائل بالذكاء الاصطناعي وقراءة صور الكاميرا.",
    )
    st.session_state.api_key_saved = api_key_input.strip()

    if st.session_state.api_key_saved:
        genai.configure(api_key=st.session_state.api_key_saved)
        st.success("✅ الذكاء الاصطناعي والكاميرا الذكية مفعلان!")
    else:
        st.warning("⚠️ أدخل مفتاح API لتفعيل الـ AI (الحل البرمجي يعمل فوراً بدون مفتاح).")

    selected_ai_model = st.selectbox(
        "🤖 نموذج الذكاء الاصطناعي:",
        options=[
            "gemini-2.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
        index=0,
    )

    ai_explanation_style = st.selectbox(
        "📝 أسلوب شرح الذكاء الاصطناعي:",
        options=[
            "شرح هندسي تفصيلي خطوة بخطوة مع القوانين والوحدات",
            "شرح أكاديمي معمق مع إثبات القوانين والتحقق من النتيجة",
            "حل مختصر ومباشر مع الخطوات الأساسية فقط",
            "شرح مبسط جداً مع أمثلة توضيحية",
        ],
        index=0,
    )

    ai_temperature = st.slider(
        "🎯 درجة الدقة الإبداعية (Temperature):",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.05,
        help="القيم المنخفضة (0.1 - 0.2) تضمن أعلى دقة رياضية وهندسية.",
    )

    st.markdown("---")
    st.markdown(
        "<div class='rtl-box'><h4>📊 إحصائيات الجلسة الحالية</h4></div>",
        unsafe_allow_html=True,
    )
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("عدد المسائل المحلولة", len(st.session_state.history_logs))
    with col_s2:
        st.metric("الأقسام المتاحة", "10 أقسام")

    st.markdown("---")
    st.info(
        "💡 **ميزة الكاميرا والـ AI:** متوفرة داخل كل تبويب من التبويبات العشرة، وتدعم التصوير المباشر أو رفع الصور مع فلاتر OpenCV."
    )


# ==============================================================================
# 5. محرك الرؤية الحاسوبية (OpenCV Filters) والاتصال بـ Gemini Vision AI
# ==============================================================================

def process_image_with_opencv(pil_img, filter_name, rotation_deg=0, contrast_factor=1.5):
    """تطبيق فلاتر الرؤية الحاسوبية لتوضيح المعادلات المكتوبة بخط اليد والرسوم الهندسية"""
    if pil_img is None:
        return None
    try:
        img_rgb = pil_img.convert("RGB")
        if rotation_deg != 0:
            img_rgb = img_rgb.rotate(-rotation_deg, expand=True)

        img_np = np.array(img_rgb)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        if filter_name == "1. الصورة الأصلية (Original)":
            return img_rgb

        elif filter_name == "2. تصفية ذكية للمعادلات والنصوص (Otsu Thresholding)":
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return Image.fromarray(thresh)

        elif filter_name == "3. إزالة الظلال والإضاءة الضعيفة (Adaptive Threshold)":
            adapt = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                15,
                8,
            )
            return Image.fromarray(adapt)

        elif filter_name == "4. زيادة التباين والحدة للدوائر والرسوم (High Contrast & Sharp)":
            enh_c = ImageEnhance.Contrast(img_rgb).enhance(contrast_factor)
            enh_s = ImageEnhance.Sharpness(enh_c).enhance(2.2)
            return enh_s

        elif filter_name == "5. كشف الحواف الهندسية والأشكال (Canny Edges)":
            edges = cv2.Canny(gray, 50, 150)
            inverted = cv2.bitwise_not(edges)
            return Image.fromarray(inverted)

        elif filter_name == "6. تقليل التشويش وتحويل للرمادي (Denoised Grayscale)":
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            return Image.fromarray(denoised)

        return img_rgb
    except Exception:
        return pil_img


def analyze_problem_with_gemini_ai(
    category_title,
    domain_instructions,
    user_text_content,
    pil_image_processed=None,
    task_mode="full_solve",
):
    """إرسال المسألة (صورة من الكاميرا/مرفوعة + نص) إلى نموذج Gemini AI"""
    if not st.session_state.api_key_saved:
        return (
            "❌ **تنبيه:** لم يتم إدخال مفتاح `Gemini API Key` في القائمة الجانبية. "
            "يرجى إدخال المفتاح لتفعيل الحل بالذكاء الاصطناعي وقراءة الصور."
        )

    try:
        generation_config = {
            "temperature": ai_temperature,
            "top_p": 0.95,
            "max_output_tokens": 4096,
        }
        model = genai.GenerativeModel(
            model_name=selected_ai_model,
            generation_config=generation_config,
        )

        if task_mode == "ocr_extract":
            system_prompt = (
                f"أنت خبير قراءة بصرية (Math & Engineering OCR) لقسم ({category_title}).\n"
                "مهمتك قراءة الصورة المرفقة بدقة متناهية واستخراج:\n"
                "1. النص الكامل للمسألة كما يظهر في الصورة.\n"
                "2. جميع المعادلات الرياضية مكتوبة بصيغة بايثون (SymPy) جاهزة للنسخ، وكذلك بصيغة LaTeX.\n"
                "3. جميع الأرقام، الرموز، الزوايا، أو قيم عناصر الدائرة الكهربية الموجودة على الرسم.\n"
                "اعرض المخرجات بشكل منظم وواضح باللغة العربية."
            )
        elif task_mode == "verify_solution":
            system_prompt = (
                f"أنت مراجع هندسي وأكاديمي في قسم ({category_title}).\n"
                "قم بمراجعة المسألة والبيانات التالية، وتحقق من صحة الحل خطوة بخطوة، "
                "واذكر إن كان هناك أي خطأ في الإشارات أو الوحدات أو التعويض مع تقديم الحل الصحيح الكامل."
            )
        else:
            system_prompt = (
                f"أنت بروفيسور ومهندس خبير متخصص في قسم ({category_title}).\n"
                f"أسلوب الشرح المطلوب: ({ai_explanation_style}).\n"
                f"توجيهات التخصص: {domain_instructions}\n\n"
                "قم بحل المسألة المقدمة (سواء من الصورة الملتقطة بالكاميرا أو المرفوعة أو النص المكتوب) وفق الهيكل التالي:\n"
                "1. **📋 تفريغ المعطيات والمطلوب:** استخرج جميع القيم والرموز من الصورة أو النص مع وحداتها.\n"
                "2. **📐 القوانين والنظريات المستخدمة:** اكتب القوانين الرياضية أو الهندسية بتنسيق LaTeX واضح.\n"
                "3. **🧮 خطوات الحل التفصيلية:** اشرح كل خطوة تعويض وحساب بترتيب منطقي واضح.\n"
                "4. **🎯 الناتج النهائي والتحقق الهندسي:** اكتب النتيجة النهائية بوضوح مع الوحدات والتعليق الهندسي.\n"
            )

        if user_text_content and user_text_content.strip():
            system_prompt += f"\n---\nالبيانات النصية / المعادلات المدخلة في القسم:\n{user_text_content}\n"

        payload = [system_prompt]
        if pil_image_processed is not None:
            payload.append(pil_image_processed)

        response = model.generate_content(payload)
        return response.text

    except Exception as exc:
        return f"❌ **حدث خطأ أثناء الاتصال بالذكاء الاصطناعي:**\n`{str(exc)}`"


# ==============================================================================
# 6. المكون الموحد للكاميرا ورفع الصور والـ AI (يُدمج في كل قسم من الأقسام الـ 10)
# ==============================================================================

def render_camera_and_ai_hub(
    tab_key_prefix,
    category_title,
    domain_ai_instructions,
    current_text_data_getter,
):
    """
    يرسم وحدة الكاميرا + رفع الصور + معالجة OpenCV + أزرار الذكاء الاصطناعي
    داخل كل قسم من أقسام التطبيق بشكل مستقل تماماً عبر tab_key_prefix
    """
    st.markdown(
        f"""
        <div class='camera-ai-card'>
            <h4 style='margin:0 0 6px 0; color:#0369A1;'>
                📸🤖 وحدة الكاميرا الذكية، الرؤية الحاسوبية (OpenCV)، والذكاء الاصطناعي — ({category_title})
            </h4>
            <p style='margin:0; font-size:13.5px; color:#334155;'>
                يمكنك تصوير المسألة مباشرة بالكاميرا أو رفع صورتها من جهازك أو حل البيانات المكتوبة في هذا القسم بالذكاء الاصطناعي.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_src, col_flt, col_rot = st.columns([2, 2.5, 1.2])
    with col_src:
        img_source_mode = st.radio(
            "📷 مصدر صورة المسألة:",
            options=[
                "بدون صورة (نص فقط)",
                "📸 فتح الكاميرا المباشرة",
                "📁 رفع صورة من الجهاز",
            ],
            index=0,
            key=f"{tab_key_prefix}_img_src",
            horizontal=True,
        )
    with col_flt:
        selected_cv_filter = st.selectbox(
            "🔍 فلتر تحسين الصورة (OpenCV):",
            options=[
                "1. الصورة الأصلية (Original)",
                "2. تصفية ذكية للمعادلات والنصوص (Otsu Thresholding)",
                "3. إزالة الظلال والإضاءة الضعيفة (Adaptive Threshold)",
                "4. زيادة التباين والحدة للدوائر والرسوم (High Contrast & Sharp)",
                "5. كشف الحواف الهندسية والأشكال (Canny Edges)",
                "6. تقليل التشويش وتحويل للرمادي (Denoised Grayscale)",
            ],
            index=3,
            key=f"{tab_key_prefix}_cv_filter",
        )
    with col_rot:
        selected_rotation = st.selectbox(
            "🔄 تدوير الصورة:",
            options=[0, 90, 180, 270],
            format_func=lambda x: f"{x}° درجة",
            index=0,
            key=f"{tab_key_prefix}_rot",
        )

    raw_pil_image = None
    processed_pil_image = None

    if img_source_mode == "📸 فتح الكاميرا المباشرة":
        cam_buffer = st.camera_input(
            f"📸 التقط صورة مسألة ({category_title}) بالكاميرا:",
            key=f"{tab_key_prefix}_camera_widget",
        )
        if cam_buffer is not None:
            raw_pil_image = Image.open(cam_buffer)

    elif img_source_mode == "📁 رفع صورة من الجهاز":
        up_buffer = st.file_uploader(
            f"📁 اختر صورة مسألة ({category_title}) (PNG, JPG, JPEG, WEBP):",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            key=f"{tab_key_prefix}_uploader_widget",
        )
        if up_buffer is not None:
            raw_pil_image = Image.open(up_buffer)

    if raw_pil_image is not None:
        processed_pil_image = process_image_with_opencv(
            raw_pil_image, selected_cv_filter, selected_rotation
        )
        col_prev1, col_prev2 = st.columns(2)
        with col_prev1:
            st.image(
                raw_pil_image,
                caption="🖼️ الصورة الأصلية الملتقطة / المرفوعة",
                use_container_width=True,
            )
        with col_prev2:
            st.image(
                processed_pil_image,
                caption=f"✨ الصورة بعد معالجة OpenCV ({selected_cv_filter})",
                use_container_width=True,
            )

    custom_user_note = st.text_input(
        "💡 ملاحظة أو طلب إضافي للذكاء الاصطناعي في هذا القسم (اختياري):",
        placeholder="مثال: حل المسألة بطريقتين مختلفتين، أو احسب الناتج بالوحدات الدولية SI، أو اشرح الخطوة الثالثة بالتفصيل...",
        key=f"{tab_key_prefix}_extra_note",
    )

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        btn_ai_full = st.button(
            f"🤖 حل المسألة بالذكاء الاصطناعي ({category_title})",
            type="primary",
            use_container_width=True,
            key=f"{tab_key_prefix}_btn_ai_solve",
        )
    with col_btn2:
        btn_ai_ocr = st.button(
            "📝 استخراج المعادلات والنص من الصورة (OCR)",
            use_container_width=True,
            key=f"{tab_key_prefix}_btn_ai_ocr",
        )
    with col_btn3:
        btn_ai_verify = st.button(
            "✅ مراجعة الحل والتحقق من النتيجة بالـ AI",
            use_container_width=True,
            key=f"{tab_key_prefix}_btn_ai_verify",
        )

    if btn_ai_ocr:
        if processed_pil_image is None:
            st.warning("⚠️ يرجى التقاط صورة بالكاميرا أو رفع صورة أولاً لاستخراج المعادلات والنصوص منها.")
        else:
            with st.spinner("🔍 جاري قراءة الصورة واستخراج المعادلات والرموز عبر Gemini Vision OCR..."):
                ocr_res = analyze_problem_with_gemini_ai(
                    category_title=category_title,
                    domain_instructions=domain_ai_instructions,
                    user_text_content=custom_user_note,
                    pil_image_processed=processed_pil_image,
                    task_mode="ocr_extract",
                )
                st.session_state.extracted_texts[tab_key_prefix] = ocr_res
                st.markdown("### 📝 المعادلات والنصوص المستخرجة من الصورة:")
                st.markdown(ocr_res)
                log_operation_to_history(category_title, "AI Vision OCR", "صورة مرفقة", ocr_res)

    if btn_ai_full or btn_ai_verify:
        mode_to_run = "verify_solution" if btn_ai_verify else "full_solve"
        combined_text = current_text_data_getter()
        if custom_user_note.strip():
            combined_text += f"\n[توجيه إضافي من المستخدم]: {custom_user_note.strip()}"

        if processed_pil_image is None and not combined_text.strip():
            st.warning("⚠️ يرجى كتابة المسألة في الحقول المخصصة أو تصويرها بالكاميرا / رفع صورتها أولاً.")
        else:
            with st.spinner(f"🤖 جاري التحليل والحل الهندسي الشامل لقسم ({category_title})..."):
                ai_answer = analyze_problem_with_gemini_ai(
                    category_title=category_title,
                    domain_instructions=domain_ai_instructions,
                    user_text_content=combined_text,
                    pil_image_processed=processed_pil_image,
                    task_mode=mode_to_run,
                )
                st.markdown(f"### 🤖 تقرير الذكاء الاصطناعي — ({category_title}):")
                st.markdown(ai_answer)
                log_operation_to_history(
                    category_title,
                    "Gemini Vision & AI Solver",
                    combined_text,
                    ai_answer,
                )


# ==============================================================================
# 7. الواجهة الرئيسية والتبويبات الهندسية والرياضية العشرة (10 Mega Tabs)
# ==============================================================================

st.markdown(
    """
    <div class='hero-banner'>
        <h1>🚀 المنصة الهندسية والرياضية الشاملة (Enterprise AI & Camera Problem Solver)</h1>
        <p>
            نظام متكامل يجمع بين الحل البرمجي الدقيق (SymPy / NumPy / SciPy) + الكاميرا المباشرة + معالجة الصور (OpenCV) + الذكاء الاصطناعي (Gemini Vision) في جميع أنواع المسائل
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

TABS_LIST = st.tabs([
    "1️⃣ الجبر والأعداد المركبة",
    "2️⃣ المصفوفات و Gauss-Seidel",
    "3️⃣ الطرق العددية والاستكمال",
    "4️⃣ التفاضل والتكامل و Laplace",
    "5️⃣ الهندسة والمثلثات والمتجهات",
    "6️⃣ الدوائر الكهربية والمحولات",
    "7️⃣ الميكانيكا والاستاتيكا والديناميكا",
    "8️⃣ التحكم الآلي والـ PLC",
    "9️⃣ المسائل اللفظية والبرمجة",
    "🔟 السجل والمرجع والتقارير",
])


# ==============================================================================
# التبويب الأول: الجبر، المعادلات، الكسور الجزئية، والأعداد المركبة
# ==============================================================================
with TABS_LIST[0]:
    st.markdown(
        """
        <div class='section-card'>
            <h3>📐 القسم الأول: الجبر، حل المعادلات، الكسور الجزئية، والأعداد المركبة</h3>
            <p>حل المعادلات الخطية وغير الخطية، تحليل كثيرات الحدود، تفكيك الكسور الجزئية، وحسابات الأعداد المركبة مع الرسم البياني للجذور.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a1, col_a2 = st.columns([1.4, 1])
    with col_a1:
        alg_operation = st.selectbox(
            "🛠️ اختر نوع العملية الجبرية:",
            options=[
                "1. حل معادلة أو نظام معادلات (Solve Equations)",
                "2. تحليل المقدار الجبري إلى عوامل (Factorization)",
                "3. فك وتبسيط المقادير الجبرية (Expand & Simplify)",
                "4. تفكيك الكسور الجزئية (Partial Fractions Decomposition)",
                "5. العمليات على الأعداد المركبة (Complex Numbers Polar/Rectangular)",
                "6. حساب المتسلسلات والمجاميع (Symbolic Summation Σ)",
            ],
            key="alg_op_select",
        )
    with col_a2:
        alg_preset = st.selectbox(
            "📚 أمثلة جاهزة للتجربة السريعة:",
            options=[
                "-- اختر مثالاً جاهزاً --",
                "معادلة درجة ثالثة: x**3 - 6*x**2 + 11*x - 6 = 0",
                "نظام معادلتين غير خطي: x**2 + y = 5 و x - y = 1",
                "كسور جزئية: (3*x + 5) / (x**2 + 3*x + 2)",
                "عدد مركب: (3 + 4*I) * (2 - I) / (1 + I)",
            ],
            key="alg_preset_select",
        )

    default_alg_text = "x**3 - 6*x**2 + 11*x - 6 = 0"
    default_alg_vars = "x"
    if alg_preset == "نظام معادلتين غير خطي: x**2 + y = 5 و x - y = 1":
        default_alg_text = "x**2 + y = 5\nx - y = 1"
        default_alg_vars = "x, y"
    elif alg_preset == "كسور جزئية: (3*x + 5) / (x**2 + 3*x + 2)":
        default_alg_text = "(3*x + 5) / (x**2 + 3*x + 2)"
        default_alg_vars = "x"
    elif alg_preset == "عدد مركب: (3 + 4*I) * (2 - I) / (1 + I)":
        default_alg_text = "(3 + 4*I) * (2 - I) / (1 + I)"
        default_alg_vars = "x"

    alg_eq_input = st.text_area(
        "✍️ أدخل المعادلات أو المقدار الجبري (كل معادلة في سطر):",
        value=default_alg_text,
        height=110,
        key="alg_eq_textarea",
    )

    col_av1, col_av2 = st.columns([1, 1])
    with col_av1:
        alg_vars_input = st.text_input(
            "🔤 المتغيرات المطلوبة (مفصولة بفاصلة):",
            value=default_alg_vars,
            key="alg_vars_text",
        )
    with col_av2:
        alg_show_plot = st.checkbox(
            "📊 رسم منحنى الدالة وتحديد الجذور الحقيقية بيانياً",
            value=True,
            key="alg_plot_chk",
        )

    if st.button("⚙️ تنفيذ الحل الجبري الدقيق (SymPy Solver)", type="primary", key="btn_run_alg"):
        try:
            lines = [
                ln.strip()
                for ln in alg_eq_input.strip().split("\n")
                if ln.strip() and not ln.strip().startswith("#")
            ]
            sym_vars = [
                sp.Symbol(v.strip())
                for v in alg_vars_input.split(",")
                if v.strip()
            ]
            if not sym_vars:
                sym_vars = [sp.Symbol("x")]

            st.markdown("#### ✅ النتائج الجبرية الدقيقة:")

            if alg_operation.startswith("1."):
                equations = []
                first_expr = None
                for ln in lines:
                    if "=" in ln:
                        lhs_str, rhs_str = ln.split("=", 1)
                        lhs_e, rhs_e = sp.sympify(lhs_str), sp.sympify(rhs_str)
                        equations.append(sp.Eq(lhs_e, rhs_e))
                        if first_expr is None:
                            first_expr = lhs_e - rhs_e
                    else:
                        e_val = sp.sympify(ln)
                        equations.append(sp.Eq(e_val, 0))
                        if first_expr is None:
                            first_expr = e_val

                for idx, eq_obj in enumerate(equations, 1):
                    st.latex(f"\\text{{Equation ({idx}): }} {sp.latex(eq_obj)}")

                solutions = sp.solve(equations, sym_vars, dict=True)
                if solutions:
                    for idx, sol_dict in enumerate(solutions, 1):
                        parts_latex = []
                        for k_sym, v_sym in sol_dict.items():
                            approx_val = v_sym.evalf(6)
                            parts_latex.append(
                                f"{sp.latex(k_sym)} = {sp.latex(v_sym)} \\approx {sp.latex(approx_val)}"
                            )
                        st.latex(f"\\mathbf{{Solution\\ ({idx}):}}\\quad " + ",\\quad ".join(parts_latex))
                else:
                    st.warning("⚠️ لم يتم العثور على حلول جبرية صريحة.")

                if alg_show_plot and len(sym_vars) == 1 and len(equations) == 1 and first_expr is not None:
                    x_s = sym_vars[0]
                    f_lamb = sp.lambdify(x_s, first_expr, modules=["numpy"])
                    real_r = [
                        float(s[x_s].evalf())
                        for s in solutions
                        if x_s in s and s[x_s].is_real
                    ]
                    mid = float(np.mean(real_r)) if real_r else 0.0
                    span = max(5.0, (max(real_r) - min(real_r)) * 1.4) if len(real_r) > 1 else 6.0
                    xv = np.linspace(mid - span, mid + span, 400)
                    yv = f_lamb(xv)

                    fig, ax = plt.subplots(figsize=(7.5, 3.5))
                    ax.plot(xv, yv, color="#1D4ED8", linewidth=2.2, label=f"$f({x_s}) = {sp.latex(first_expr)}$")
                    ax.axhline(0, color="black", linestyle="--", linewidth=1)
                    if real_r:
                        ax.scatter(real_r, [0] * len(real_r), color="#DC2626", s=75, zorder=5, label="Real Roots")
                    ax.set_title("تمثيل الدالة ونقاط الجذور الحقيقية")
                    ax.grid(True, linestyle=":", alpha=0.7)
                    ax.legend()
                    st.pyplot(fig)

                log_operation_to_history("الجبر والمعادلات", "SymPy Solve", alg_eq_input, solutions)

            elif alg_operation.startswith("2."):
                expr = sp.sympify(lines[0].split("=")[0])
                res = sp.factor(expr)
                st.latex(f"{sp.latex(expr)} = {sp.latex(res)}")

            elif alg_operation.startswith("3."):
                expr = sp.sympify(lines[0].split("=")[0])
                res_exp = sp.expand(expr)
                res_simp = sp.simplify(expr)
                st.latex(f"\\text{{Expanded: }} {sp.latex(res_exp)}")
                st.latex(f"\\text{{Simplified: }} {sp.latex(res_simp)}")

            elif alg_operation.startswith("4."):
                expr = sp.sympify(lines[0].split("=")[0])
                res = sp.apart(expr, sym_vars[0])
                st.latex(f"{sp.latex(expr)} = {sp.latex(res)}")

            elif alg_operation.startswith("5."):
                z_expr = sp.sympify(lines[0])
                re_part = sp.re(z_expr)
                im_part = sp.im(z_expr)
                mag = sp.Abs(z_expr)
                arg_rad = sp.arg(z_expr)
                arg_deg = float(arg_rad.evalf()) * 180.0 / math.pi
                st.latex(f"Z = {sp.latex(sp.simplify(z_expr))} = {float(re_part.evalf()):.4f} + j({float(im_part.evalf()):.4f})")
                st.latex(f"|Z| = {sp.latex(mag)} \\approx {float(mag.evalf()):.4f}, \\quad \\theta = {arg_deg:.2f}^\\circ")

            elif alg_operation.startswith("6."):
                expr = sp.sympify(lines[0])
                res = sp.summation(expr, (sym_vars[0], 1, 10))
                st.latex(f"\\sum_{{{sp.latex(sym_vars[0])}=1}}^{{10}} \\left({sp.latex(expr)}\\right) = {sp.latex(res)}")

        except Exception as err:
            st.error(f"❌ خطأ في صياغة المعادلة: {err}")

    render_camera_and_ai_hub(
        tab_key_prefix="tab1_algebra",
        category_title="الجبر والمعادلات والأعداد المركبة",
        domain_ai_instructions="حل المعادلات الجبرية، أنظمة المعادلات، الكسور الجزئية، المتباينات، والأعداد المركبة بالصورة الديكارتية والقطبية.",
        current_text_data_getter=lambda: f"العملية: {alg_operation}\nالمعادلات:\n{alg_eq_input}\nالمتغيرات: {alg_vars_input}",
    )


# ==============================================================================
# التبويب الثاني: المصفوفات الموسعة [A|B]، Gauss-Seidel، Jacobi، وتفكيك LU
# ==============================================================================
with TABS_LIST[1]:
    st.markdown(
        """
        <div class='section-card'>
            <h3>🔢 القسم الثاني: المصفوفات الموسعة [A|B]، طريقة Gauss-Seidel، Jacobi، وتفكيك LU</h3>
            <p>حل أنظمة المعادلات الخطية مع الترتيب التلقائي للقطر المهيمن (Auto-Pivoting)، جدول التكرارات التفصيلي، ومنحنى سرعة التقارب.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_m1, col_m2, col_m3 = st.columns([1.5, 1, 1])
    with col_m1:
        mat_method = st.selectbox(
            "🧮 اختر خوارزمية الحل:",
            options=[
                "1. طريقة Gauss-Seidel التكرارية + الحل الدقيق",
                "2. طريقة Jacobi التكرارية + الحل الدقيق",
                "3. مقارنة شاملة (Gauss-Seidel vs Jacobi) + منحنى التقارب",
                "4. قاعدة كرامر وتفكيك المصفوفة (Cramer's Rule & LU Decomposition)",
                "5. خصائص المصفوفة A (المحدد، المعكوس، القيم الذاتية Eigenvalues)",
            ],
            key="mat_method_select",
        )
    with col_m2:
        mat_tol = st.number_input(
            "🎯 نسبة الخطأ المسموحة (Tolerance):",
            value=0.0001,
            format="%.6f",
            step=0.0001,
            key="mat_tol_num",
        )
    with col_m3:
        mat_max_iter = st.number_input(
            "🔄 أقصى عدد للتكرارات:",
            min_value=1,
            max_value=200,
            value=25,
            step=5,
            key="mat_iter_num",
        )

    col_mp1, col_mp2 = st.columns([1.2, 1])
    with col_mp1:
        mat_auto_pivot = st.checkbox(
            "✅ تفعيل الترتيب التلقائي للصفوف لتحقيق القطر المهيمن (Auto-Pivoting)",
            value=True,
            key="mat_pivot_chk",
        )
    with col_mp2:
        mat_omega = st.slider(
            "⚡ معامل الاسترخاء ω (SOR Relaxation Factor):",
            min_value=0.5,
            max_value=1.9,
            value=1.0,
            step=0.05,
            key="mat_omega_slider",
        )

    mat_aug_text = st.text_area(
        "✍️ أدخل المصفوفة الموسعة [A | B] (كل صف في سطر والأرقام مفصولة بمسافة):",
        value="4 1 2 4\n3 5 1 7\n1 1 3 3",
        height=115,
        key="mat_aug_textarea",
    )

    def run_iterative_linear_solver(A_mat, B_vec, mode_name, tol_val, max_it, omega_val=1.0):
        n_dim = len(B_vec)
        x_curr = np.zeros(n_dim, dtype=float)
        rows_log = []
        err_list = []

        for it_idx in range(1, max_it + 1):
            x_prev = x_curr.copy()
            for i in range(n_dim):
                if mode_name == "gauss_seidel":
                    s1 = float(np.dot(A_mat[i, :i], x_curr[:i]))
                    s2 = float(np.dot(A_mat[i, i + 1 :], x_prev[i + 1 :]))
                    x_gs = (B_vec[i] - s1 - s2) / A_mat[i, i]
                    x_curr[i] = (1.0 - omega_val) * x_prev[i] + omega_val * x_gs
                else:
                    s_all = float(np.dot(A_mat[i, :], x_prev) - A_mat[i, i] * x_prev[i])
                    x_curr[i] = (B_vec[i] - s_all) / A_mat[i, i]

            max_err = float(np.max(np.abs(x_curr - x_prev)))
            row_dict = {"التكرار (Iteration)": int(it_idx)}
            for j in range(n_dim):
                row_dict[f"x{j+1}"] = round(float(x_curr[j]), 6)
            row_dict["أقصى خطأ مطلق (Max Error)"] = round(max_err, 7)
            rows_log.append(row_dict)
            err_list.append(max_err)

            if max_err < tol_val:
                break

        return x_curr, pd.DataFrame(rows_log), err_list

    if st.button("⚙️ تشغيل محلل المصفوفات و Gauss-Seidel", type="primary", key="btn_run_matrix"):
        try:
            parsed_rows = [
                list(map(float, r.replace(",", " ").strip().split()))
                for r in mat_aug_text.strip().split("\n")
                if r.strip()
            ]
            M_aug = np.array(parsed_rows, dtype=float)
            n_r, n_c = M_aug.shape

            if n_c != n_r + 1:
                st.error(
                    f"❌ أبعاد المصفوفة الموسعة [A|B] يجب أن تكون (n × n+1). الأبعاد المدخلة حالياً: ({n_r} × {n_c})."
                )
            else:
                A = M_aug[:, :-1].copy()
                B = M_aug[:, -1].copy()

                if mat_auto_pivot:
                    for i in range(n_r):
                        best_row = i + int(np.argmax(np.abs(A[i:, i])))
                        if best_row != i:
                            A[[i, best_row]] = A[[best_row, i]]
                            B[[i, best_row]] = B[[best_row, i]]

                diag_abs = np.abs(np.diag(A))
                off_diag_sum = np.sum(np.abs(A), axis=1) - diag_abs
                is_dom = bool(np.all(diag_abs >= off_diag_sum))

                if is_dom:
                    st.success("✅ شرط القطر المهيمن (Diagonal Dominance) محقق بعد الترتيب — التقارب مضمون.")
                else:
                    st.warning("⚠️ المصفوفة لا تحقق شرط القطر المهيمن بالكامل؛ قد يتباطأ التقارب التكراري.")

                st.latex(
                    f"[A \\mid B]_{{\\text{{ordered}}}} = {sp.latex(sp.Matrix(np.round(np.column_stack((A, B)), 4)))}"
                )

                if mat_method.startswith("5."):
                    det_val = float(np.linalg.det(A))
                    rank_val = int(np.linalg.matrix_rank(A))
                    eig_vals = np.linalg.eigvals(A)
                    st.write(f"**محدد المصفوفة det(A):** `{det_val:.6f}` | **الرتبة Rank(A):** `{rank_val}`")
                    st.write(f"**القيم الذاتية (Eigenvalues):** `{np.round(eig_vals, 5)}`")
                    if abs(det_val) > 1e-12:
                        inv_A = np.linalg.inv(A)
                        st.latex(f"A^{{-1}} = {sp.latex(sp.Matrix(np.round(inv_A, 5)))}")

                elif mat_method.startswith("4."):
                    det_A = float(np.linalg.det(A))
                    st.write(f"**محدد المصفوفة الأساسي $\\Delta = \\det(A) =$** `{det_A:.6f}`")
                    if abs(det_A) > 1e-12:
                        cramer_sol = []
                        for j in range(n_r):
                            A_j = A.copy()
                            A_j[:, j] = B
                            det_j = float(np.linalg.det(A_j))
                            xj = det_j / det_A
                            cramer_sol.append(xj)
                            st.latex(f"\\Delta_{{{j+1}}} = {det_j:.4f} \\implies x_{{{j+1}}} = \\frac{{\\Delta_{{{j+1}}}}}{{\\Delta}} = {xj:.6f}")
                        L_mat, U_mat, _ = sp.Matrix(A).LUdecomposition()
                        st.latex(f"L = {sp.latex(L_mat.evalf(4))}, \\quad U = {sp.latex(U_mat.evalf(4))}")
                else:
                    chosen_algo = "jacobi" if mat_method.startswith("2.") else "gauss_seidel"
                    x_approx, df_iter, err_hist = run_iterative_linear_solver(
                        A, B, chosen_algo, mat_tol, int(mat_max_iter), mat_omega
                    )
                    st.markdown("#### 📋 جدول التكرارات التفصيلي:")
                    st.dataframe(df_iter, use_container_width=True)

                    exact_vec = np.linalg.solve(A, B)
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.success(f"🎯 **الحل التقريبي ({chosen_algo.upper()}):** `{np.round(x_approx, 6)}`")
                    with col_res2:
                        st.info(f"✅ **الحل الجبري الدقيق (Exact):** `{np.round(exact_vec, 6)}`")

                    fig, ax = plt.subplots(figsize=(7.5, 3.2))
                    ax.semilogy(
                        range(1, len(err_hist) + 1),
                        err_hist,
                        marker="o",
                        linewidth=2,
                        color="#1D4ED8",
                        label=f"{chosen_algo.upper()} Error",
                    )
                    if mat_method.startswith("3."):
                        _, _, err_jac = run_iterative_linear_solver(
                            A, B, "jacobi", mat_tol, int(mat_max_iter), 1.0
                        )
                        ax.semilogy(
                            range(1, len(err_jac) + 1),
                            err_jac,
                            marker="s",
                            linestyle="--",
                            linewidth=2,
                            color="#EA580C",
                            label="JACOBI Error",
                        )
                    ax.axhline(mat_tol, color="#16A34A", linestyle=":", label="Tolerance Limit")
                    ax.set_title("منحنى سرعة التقارب وتناقص الخطأ (Convergence Plot)")
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Max Absolute Error (Log Scale)")
                    ax.grid(True, which="both", linestyle="--", alpha=0.6)
                    ax.legend()
                    st.pyplot(fig)

                    log_operation_to_history(
                        "المصفوفات و Gauss-Seidel",
                        mat_method,
                        mat_aug_text,
                        f"Approx: {np.round(x_approx, 6)} | Exact: {np.round(exact_vec, 6)}",
                    )
        except Exception as err:
            st.error(f"❌ خطأ أثناء معالجة المصفوفة: {err}")

    render_camera_and_ai_hub(
        tab_key_prefix="tab2_matrices",
        category_title="المصفوفات والأنظمة الخطية و Gauss-Seidel",
        domain_ai_instructions="حل المصفوفات الموسعة، طريقة جاوس-سيدل، جاكوبي، الحذف لجاوس، تفكيك LU، قاعدة كرامر، والقيم والمتجهات الذاتية.",
        current_text_data_getter=lambda: f"الطريقة: {mat_method}\nالمصفوفة الموسعة [A|B]:\n{mat_aug_text}\nنسبة الخطأ: {mat_tol}",
    )


# ==============================================================================
# التبويب الثالث: الطرق العددية، إيجاد الجذور، والاستكمال والتكامل العددي
# ==============================================================================
with TABS_LIST[2]:
    st.markdown(
        """
        <div class='section-card'>
            <h3>🧮 القسم الثالث: الطرق العددية الهندسية (Numerical Analysis & Root Finding)</h3>
            <p>إيجاد جذور المعادلات غير الخطية (Bisection, Newton-Raphson, Secant)، الاستكمال الداخلي (Lagrange)، والتكامل العددي (Trapezoidal & Simpson).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    num_method_type = st.selectbox(
        "🛠️ اختر الطريقة العددية:",
        options=[
            "1. طريقة التنصيف لإيجاد الجذور (Bisection Method)",
            "2. طريقة نيوتن-رافسون السريعة (Newton-Raphson Method)",
            "3. طريقة القاطع (Secant Method)",
            "4. التكامل العددي بقاعدتي شبه المنحرف وسمبسون (Trapezoidal & Simpson's 1/3 Rule)",
            "5. الاستكمال الداخلي بكثيرات حدود لاجرانج وتوفيق المنحنيات (Lagrange Interpolation)",
        ],
        key="num_method_select",
    )

    col_n1, col_n2, col_n3, col_n4 = st.columns(4)
    with col_n1:
        num_func_str = st.text_input(
            "📈 الدالة f(x):",
            value="x**3 - x - 2",
            key="num_func_in",
        )
    with col_n2:
        num_a_val = st.number_input("النقطة a (أو x0):", value=1.0, format="%.4f", key="num_a_in")
    with col_n3:
        num_b_val = st.number_input("النقطة b (أو x1):", value=2.0, format="%.4f", key="num_b_in")
    with col_n4:
        num_sub_n = st.number_input("التقسيمات / التكرارات n:", min_value=2, max_value=100, value=12, step=2, key="num_n_in")

    num_points_data = st.text_input(
        "📊 نقاط الاستكمال (تُستخدم في خيار Lagrange فقط بصيغة x1,y1 ; x2,y2 ; ...):",
        value="0,1 ; 1,3 ; 2,7 ; 3,13",
        key="num_pts_in",
    )

    if st.button("⚙️ تنفيذ الخوارزمية العددية وعرض الجدول", type="primary", key="btn_run_numerical"):
        try:
            x_sym = sp.Symbol("x")
            f_sym = sp.sympify(num_func_str)
            df_sym = sp.diff(f_sym, x_sym)
            f_num = sp.lambdify(x_sym, f_sym, "numpy")
            df_num = sp.lambdify(x_sym, df_sym, "numpy")

            if num_method_type.startswith("1."):
                a_c, b_c = float(num_a_val), float(num_b_val)
                if f_num(a_c) * f_num(b_c) > 0:
                    st.warning("⚠️ تنبيه: f(a) و f(b) لهما نفس الإشارة، قد لا يوجد جذر وحيد في هذه الفترة.")
                rows_b = []
                for it in range(1, int(num_sub_n) + 1):
                    mid_c = (a_c + b_c) / 2.0
                    f_mid = float(f_num(mid_c))
                    err_b = abs(b_c - a_c) / 2.0
                    rows_b.append({
                        "Iteration": it,
                        "a": round(a_c, 6),
                        "b": round(b_c, 6),
                        "c (Midpoint)": round(mid_c, 6),
                        "f(c)": round(f_mid, 6),
                        "Error": round(err_b, 6),
                    })
                    if abs(f_mid) < 1e-7 or err_b < 1e-6:
                        break
                    if f_num(a_c) * f_mid < 0:
                        b_c = mid_c
                    else:
                        a_c = mid_c
                st.dataframe(pd.DataFrame(rows_b), use_container_width=True)
                st.success(f"🎯 **الجذر التقريبي بطريقة التنصيف:** `x ≈ {mid_c:.6f}`")

            elif num_method_type.startswith("2."):
                st.latex(f"f(x) = {sp.latex(f_sym)}, \\quad f'(x) = {sp.latex(df_sym)}")
                xk = float(num_a_val)
                rows_nr = []
                for it in range(1, int(num_sub_n) + 1):
                    fx = float(f_num(xk))
                    dfx = float(df_num(xk))
                    if abs(dfx) < 1e-12:
                        st.error("❌ المشتقة تساوي صفراً عند هذه النقطة!")
                        break
                    x_next = xk - fx / dfx
                    err_nr = abs(x_next - xk)
                    rows_nr.append({
                        "Iteration": it,
                        "x_k": round(xk, 6),
                        "f(x_k)": round(fx, 6),
                        "f'(x_k)": round(dfx, 6),
                        "x_{k+1}": round(x_next, 6),
                        "Error": round(err_nr, 7),
                    })
                    xk = x_next
                    if err_nr < 1e-7:
                        break
                st.dataframe(pd.DataFrame(rows_nr), use_container_width=True)
                st.success(f"🎯 **الجذر التقريبي بطريقة نيوتن-رافسون:** `x ≈ {xk:.7f}`")

            elif num_method_type.startswith("3."):
                x0, x1 = float(num_a_val), float(num_b_val)
                rows_sec = []
                for it in range(1, int(num_sub_n) + 1):
                    f0, f1 = float(f_num(x0)), float(f_num(x1))
                    if abs(f1 - f0) < 1e-12:
                        break
                    x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
                    err_s = abs(x2 - x1)
                    rows_sec.append({
                        "Iteration": it,
                        "x_{k-1}": round(x0, 6),
                        "x_k": round(x1, 6),
                        "x_{k+1}": round(x2, 6),
                        "f(x_{k+1})": round(float(f_num(x2)), 6),
                        "Error": round(err_s, 7),
                    })
                    x0, x1 = x1, x2
                    if err_s < 1e-7:
                        break
                st.dataframe(pd.DataFrame(rows_sec), use_container_width=True)
                st.success(f"🎯 **الجذر التقريبي بطريقة القاطع (Secant):** `x ≈ {x1:.7f}`")

            elif num_method_type.startswith("4."):
                n_int = int(num_sub_n)
                if n_int % 2 != 0:
                    n_int += 1
                x_nodes = np.linspace(float(num_a_val), float(num_b_val), n_int + 1)
                y_nodes = f_num(x_nodes)
                h_step = (float(num_b_val) - float(num_a_val)) / n_int

                trap_val = (h_step / 2.0) * (y_nodes[0] + 2.0 * np.sum(y_nodes[1:-1]) + y_nodes[-1])
                simp_val = (h_step / 3.0) * (
                    y_nodes[0]
                    + 4.0 * np.sum(y_nodes[1:-1:2])
                    + 2.0 * np.sum(y_nodes[2:-2:2])
                    + y_nodes[-1]
                )
                exact_int = float(sp.integrate(f_sym, (x_sym, num_a_val, num_b_val)).evalf())

                st.write(f"**خطوة التقسيم $h =$** `{h_step:.5f}` (عدد الفترات $n = {n_int}$)")
                st.success(f"📐 **نتيجة قاعدة شبه المنحرف (Trapezoidal):** `{trap_val:.6f}` (الخطأ: `{abs(exact_int - trap_val):.6f}`)")
                st.success(f"🎯 **نتيجة قاعدة سمبسون (Simpson's 1/3):** `{simp_val:.6f}` (الخطأ: `{abs(exact_int - simp_val):.6f}`)")
                st.info(f"✅ **القيمة الدقيقة للتكامل للمقارنة:** `{exact_int:.6f}`")

            elif num_method_type.startswith("5."):
                pairs = [p.strip().split(",") for p in num_points_data.split(";") if "," in p]
                xp = [float(p[0]) for p in pairs]
                yp = [float(p[1]) for p in pairs]
                poly_lag = 0
                for i in range(len(xp)):
                    term = yp[i]
                    for j in range(len(xp)):
                        if i != j:
                            term *= (x_sym - xp[j]) / (xp[i] - xp[j])
                    poly_lag += term
                poly_simp = sp.expand(poly_lag)
                st.latex(f"P(x) = {sp.latex(poly_simp)}")

            log_operation_to_history("الطرق العددية", num_method_type, num_func_str, "تم التنفيذ العددي بنجاح")
        except Exception as err:
            st.error(f"❌ خطأ في الحسابات العددية: {err}")

    render_camera_and_ai_hub(
        tab_key_prefix="tab3_numerical",
        category_title="الطرق العددية والتحليل العددي الهندسي",
        domain_ai_instructions="حل مسائل التحليل العددي: Bisection, False Position, Newton-Raphson, Secant, Fixed-Point Iteration, Lagrange & Newton Divided Differences Interpolation, Trapezoidal & Simpson Numerical Integration, Euler & Runge-Kutta ODEs.",
        current_text_data_getter=lambda: f"الطريقة العددية: {num_method_type}\nالدالة: {num_func_str}\nالفترة: [{num_a_val}, {num_b_val}]\nنقاط الاستكمال: {num_points_data}",
    )


# ==============================================================================
# التبويب الرابع: التفاضل، التكامل، النهايات، متسلسلات تايلور، والمعادلات التفاضلية و Laplace
# ==============================================================================
with TABS_LIST[3]:
    st.markdown(
        """
        <div class='section-card'>
            <h3>📈 القسم الرابع: التفاضل، التكامل، النهايات، المعادلات التفاضلية (ODEs)، وتحويل لابلاس</h3>
            <p>حساب المشتقات العليا، التكامل المحدود وغير المحدود، النهايات، مفكوك تايلور، حل المعادلات التفاضلية، وتحويلات لابلاس واللاپلاس العكسي.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    calc_operation = st.selectbox(
        "📈 اختر العملية الرياضية:",
        options=[
            "1. الاشتقاق الرمزي والمشتقات العليا (Derivative d^n/dx^n)",
            "2. التكامل غير المحدود (Indefinite Integral ∫f(x)dx)",
            "3. التكامل المحدود والمساحة تحت المنحنى (Definite Integral)",
            "4. حساب النهايات (Limits lim x->a)",
            "5. مفكوك متسلسلة تايلور وماكلورين (Taylor Series Expansion)",
            "6. تحويل لابلاس المباشر والعكسي (Laplace & Inverse Laplace Transform)",
            "7. حل المعادلات التفاضلية العادية (ODE Symbolic Solver)",
        ],
        key="calc_op_select",
    )

    calc_expr_in = st.text_input(
        "✍️ أدخل الدالة f(x) أو f(t) أو F(s) (أو معادلة تفاضلية مثل y(x).diff(x,2) + 4*y(x) - sin(x)):",
        value="x**3 * exp(-x) + sin(2*x)",
        key="calc_expr_input",
    )

    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        calc_var = st.text_input("المتغير:", value="x", key="calc_var_in")
    with col_c2:
        calc_order = st.number_input("رتبة المشتقة / الحدود:", min_value=1, max_value=10, value=1, key="calc_ord_in")
    with col_c3:
        calc_lim_a = st.text_input("الحد الأدنى / النقطة a:", value="0", key="calc_a_in")
    with col_c4:
        calc_lim_b = st.text_input("الحد الأعلى b:", value="pi", key="calc_b_in")

    if st.button("⚙️ تنفيذ حساب التفاضل والتكامل / لابلاس", type="primary", key="btn_run_calc"):
        try:
            x_v = sp.Symbol(calc_var.strip(), real=True)
            t_v = sp.Symbol("t", positive=True)
            s_v = sp.Symbol("s")

            if calc_operation.startswith("7."):
                y_fn = sp.Function("y")
                ode_parsed = eval(
                    calc_expr_in,
                    {"y": y_fn, "x": x_v, "sp": sp, "sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "Eq": sp.Eq},
                )
                sol_ode = sp.dsolve(ode_parsed, y_fn(x_v))
                st.latex(f"\\text{{ODE Solution: }} {sp.latex(sol_ode)}")
                log_operation_to_history("التفاضل والتكامل", "ODE Solver", calc_expr_in, sp.latex(sol_ode))

            elif calc_operation.startswith("6."):
                expr_lap = sp.sympify(calc_expr_in)
                if "s" in str(expr_lap) and "t" not in str(expr_lap):
                    inv_l = sp.inverse_laplace_transform(expr_lap, s_v, t_v)
                    st.latex(f"\\mathcal{{L}}^{{-1}}\\left\\{{{sp.latex(expr_lap)}\\right\\}} = {sp.latex(inv_l)}")
                else:
                    sub_expr = expr_lap.subs(x_v, t_v)
                    lap_res = sp.laplace_transform(sub_expr, t_v, s_v, noconds=True)
                    st.latex(f"\\mathcal{{L}}\\left\\{{{sp.latex(sub_expr)}\\right\\}} = {sp.latex(lap_res)}")

            else:
                f_c = sp.sympify(calc_expr_in)
                st.latex(f"f({sp.latex(x_v)}) = {sp.latex(f_c)}")

                if calc_operation.startswith("1."):
                    res_c = sp.diff(f_c, x_v, int(calc_order))
                    st.latex(f"\\frac{{d^{{{int(calc_order)}}}}}{{d{sp.latex(x_v)}^{{{int(calc_order)}}}}} f({sp.latex(x_v)}) = {sp.latex(res_c)}")

                elif calc_operation.startswith("2."):
                    res_c = sp.integrate(f_c, x_v)
                    st.latex(f"\\int \\left({sp.latex(f_c)}\\right) d{sp.latex(x_v)} = {sp.latex(res_c)} + C")

                elif calc_operation.startswith("3."):
                    a_sym = sp.sympify(calc_lim_a)
                    b_sym = sp.sympify(calc_lim_b)
                    res_c = sp.integrate(f_c, (x_v, a_sym, b_sym))
                    st.latex(
                        f"\\int_{{{sp.latex(a_sym)}}}^{{{sp.latex(b_sym)}}} \\left({sp.latex(f_c)}\\right) d{sp.latex(x_v)} = {sp.latex(res_c)} \\approx {sp.latex(res_c.evalf(6))}"
                    )

                elif calc_operation.startswith("4."):
                    pt_sym = sp.sympify(calc_lim_a)
                    res_c = sp.limit(f_c, x_v, pt_sym)
                    st.latex(f"\\lim_{{{sp.latex(x_v)} \\to {sp.latex(pt_sym)}}} \\left({sp.latex(f_c)}\\right) = {sp.latex(res_c)}")

                elif calc_operation.startswith("5."):
                    pt_sym = sp.sympify(calc_lim_a)
                    res_c = f_c.series(x_v, pt_sym, int(calc_order) + 2).removeO()
                    st.latex(f"\\text{{Taylor Series: }} {sp.latex(res_c)}")

                log_operation_to_history("التفاضل والتكامل", calc_operation, calc_expr_in, sp.latex(res_c))

        except Exception as err:
            st.error(f"❌ تعذر الحساب الرمزي المباشر: {err}")

    render_camera_and_ai_hub(
        tab_key_prefix="tab4_calculus",
        category_title="التفاضل والتكامل والمعادلات التفاضلية وتحويل لابلاس",
        domain_ai_instructions="حل مسائل الاشتقاق، التكامل بالتعويض والتجزيء والكسور الجزئية، التكامل الثنائي والثلاثي، النهايات، متسلسلات تايلور وفورييه، المعادلات التفاضلية، وتحويلات لابلاس.",
        current_text_data_getter=lambda: f"العملية: {calc_operation}\nالدالة: {calc_expr_in}\nالمتغير: {calc_var}, الحدود: [{calc_lim_a}, {calc_lim_b}]",
    )


# ==============================================================================
# التبويب الخامس: الهندسة المستوية، الفراغية، التحليلية، والمتجهات 3D
# ==============================================================================
with TABS_LIST[4]:
    st.markdown(
        """
        <div class='section-card'>
            <h3>📐 القسم الخامس: الهندسة المستوية، الفراغية، التحليلية، والمتجهات وحساب المثلثات</h3>
            <p>حل المثلثات (قانون الجيب وجيب التمام)، الهندسة التحليلية (المسافة، الميل، الدائرة)، والمتجهات الثلاثية الأبعاد (الضرب القياسي والاتجاهي).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    geom_sub_mode = st.selectbox(
        "📐 اختر الفرع الهندسي:",
        options=[
            "1. حل المثلث العام ورسمه (قانون الجيب وجيب التمام ومساحة هيرون)",
            "2. المتجهات في الفراغ الثلاثي 3D Vectors (الضرب القياسي، الاتجاهي، الزاوية، المسقط)",
            "3. الهندسة التحليلية المستوية (المسافة، المنتصف، معادلة الخط المستقيم والدائرة)",
            "4. مساحات وحجوم المجسمات الهندسية (أسطوانة، كرة، مخروط، هرم)",
        ],
        key="geom_sub_select",
    )

    if geom_sub_mode.startswith("1."):
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            tri_a = st.number_input("طول الضلع a:", value=8.0, min_value=0.1, key="tri_a_in")
        with col_g2:
            tri_b = st.number_input("طول الضلع b:", value=11.0, min_value=0.1, key="tri_b_in")
        with col_g3:
            tri_C_deg = st.number_input("الزاوية المحصورة C (بالدرجات):", value=60.0, min_value=1.0, max_value=179.0, key="tri_c_in")

        if st.button("⚙️ حل المثلث ورسمه هندسياً", type="primary", key="btn_solve_tri"):
            c_rad = math.radians(tri_C_deg)
            tri_c = math.sqrt(tri_a**2 + tri_b**2 - 2 * tri_a * tri_b * math.cos(c_rad))
            A_rad = math.acos((tri_b**2 + tri_c**2 - tri_a**2) / (2 * tri_b * tri_c))
            B_rad = math.pi - A_rad - c_rad
            s_p = (tri_a + tri_b + tri_c) / 2.0
            area_tri = math.sqrt(s_p * (s_p - tri_a) * (s_p - tri_b) * (s_p - tri_c))

            st.latex(f"c = \\sqrt{{a^2 + b^2 - 2ab\\cos(C)}} = {tri_c:.4f}")
            st.latex(f"\\angle A = {math.degrees(A_rad):.2f}^\\circ, \\quad \\angle B = {math.degrees(B_rad):.2f}^\\circ, \\quad \\angle C = {tri_C_deg:.2f}^\\circ")
            st.success(f"📏 **محيط المثلث:** `{2*s_p:.4f}` | **مساحة المثلث:** `{area_tri:.4f}` وحدة مربعة")

            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            px = [0, tri_c, tri_b * math.cos(A_rad), 0]
            py = [0, 0, tri_b * math.sin(A_rad), 0]
            ax.plot(px, py, marker="o", color="#1D4ED8", linewidth=2.5)
            ax.fill(px, py, color="#93C5FD", alpha=0.4)
            ax.set_aspect("equal")
            ax.grid(True, linestyle=":")
            st.pyplot(fig)

    elif geom_sub_mode.startswith("2."):
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            vec_u_str = st.text_input("المتجه الأول U (x, y, z):", value="3, -2, 6", key="vec_u_in")
        with col_v2:
            vec_v_str = st.text_input("المتجه الثاني V (x, y, z):", value="1, 4, -2", key="vec_v_in")

        if st.button("⚙️ حساب خصائص المتجهات الثلاثية", type="primary", key="btn_solve_vec"):
            u_arr = np.array(list(map(float, vec_u_str.split(","))))
            v_arr = np.array(list(map(float, vec_v_str.split(","))))
            dot_uv = float(np.dot(u_arr, v_arr))
            cross_uv = np.cross(u_arr, v_arr)
            nu, nv = float(np.linalg.norm(u_arr)), float(np.linalg.norm(v_arr))
            ang_deg = math.degrees(math.acos(np.clip(dot_uv / (nu * nv), -1.0, 1.0)))

            st.latex(f"\\|\\vec{{U}}\\| = {nu:.4f}, \\quad \\|\\vec{{V}}\\| = {nv:.4f}")
            st.latex(f"\\vec{{U}} \\cdot \\vec{{V}} = {dot_uv:.4f}, \\quad \\theta = {ang_deg:.2f}^\\circ")
            st.latex(f"\\vec{{U}} \\times \\vec{{V}} = ({cross_uv[0]:.3f})\\hat{{i}} + ({cross_uv[1]:.3f})\\hat{{j}} + ({cross_uv[2]:.3f})\\hat{{k}}")

    elif geom_sub_mode.startswith("3."):
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            pt1_str = st.text_input("النقطة الأولى P1 (x1, y1):", value="2, 3", key="pt1_in")
        with col_p2:
            pt2_str = st.text_input("النقطة الثانية P2 (x2, y2):", value="8, 11", key="pt2_in")
        if st.button("⚙️ حساب المسافة ومعادلة المستقيم", type="primary", key="btn_solve_analytic"):
            x1, y1 = map(float, pt1_str.split(","))
            x2, y2 = map(float, pt2_str.split(","))
            dist_p = math.hypot(x2 - x1, y2 - y1)
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            st.success(f"📏 **المسافة بين النقطتين:** `{dist_p:.4f}` | **نقطة المنتصف:** `({mx:.3f}, {my:.3f})`")
            if x2 != x1:
                m_sl = (y2 - y1) / (x2 - x1)
                c_in = y1 - m_sl * x1
                st.latex(f"y = {m_sl:.4f}x + ({c_in:.4f})")

    else:
        col_sh1, col_sh2 = st.columns(2)
        with col_sh1:
            sh_r = st.number_input("نصف القطر r:", value=5.0, min_value=0.1, key="sh_r_in")
        with col_sh2:
            sh_h = st.number_input("الارتفاع h:", value=12.0, min_value=0.1, key="sh_h_in")
        if st.button("⚙️ حساب المساحات والحجوم", type="primary", key="btn_solve_shapes"):
            st.write(f"- **الأسطوانة:** الحجم = `{math.pi * sh_r**2 * sh_h:.4f}` | المساحة الكلية = `{2 * math.pi * sh_r * (sh_r + sh_h):.4f}`")
            st.write(f"- **المخروط القائم:** الحجم = `{(1/3) * math.pi * sh_r**2 * sh_h:.4f}`")
            st.write(f"- **الكرة:** الحجم = `{(4/3) * math.pi * sh_r**3:.4f}` | مساحة السطح = `{4 * math.pi * sh_r**2:.4f}`")

    render_camera_and_ai_hub(
        tab_key_prefix="tab5_geometry",
        category_title="الهندسة المستوية والفراغية وحساب المثلثات والمتجهات",
        domain_ai_instructions="تحليل صور الأشكال الهندسية، الزوايا، الدوائر، المثلثات، البراهين الهندسية، المتجهات في الفراغ الثلاثي، والهندسة التحليلية.",
        current_text_data_getter=lambda: f"القسم الهندسي: {geom_sub_mode}",
    )


# ==============================================================================
# التبويب السادس: الدوائر الكهربية، التيار المتردد AC، والمحولات الكهربية
# ==============================================================================
with TABS_LIST[5]:
    st.markdown(
        """
        <div class='section-card'>
            <h3>⚡ القسم السادس: الدوائر الكهربية (DC / AC)، الممانعات، واختبارات المحولات الكهربية</h3>
            <p>تحليل دوائر التيار المستمر والمتردد RLC، حساب معامل القدرة، وتحليل بيانات اختبار الدائرة المفتوحة (Open-Circuit) والدائرة القصيرة (Short-Circuit) للمحولات.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    elec_mode = st.selectbox(
        "⚡ اختر نوع المسألة الكهربية:",
        options=[
            "1. تحليل دائرة تيار متردد RLC (الممانعة Z، زاوية الطور، معامل القدرة، والقدرات P, Q, S)",
            "2. حساب عناصر الدائرة المكافئة للمحول الكهربي (Open-Circuit & Short-Circuit Tests)",
            "3. حل معادلات كيرشوف والعقد للدوائر الكهربية العامة (Symbolic Circuit Equations)",
        ],
        key="elec_mode_select",
    )

    if elec_mode.startswith("1."):
        col_e1, col_e2, col_e3, col_e4, col_e5 = st.columns(5)
        with col_e1:
            ac_v = st.number_input("الجهد V (فولت):", value=220.0, key="ac_v_in")
        with col_e2:
            ac_f = st.number_input("التردد f (هرتز):", value=50.0, key="ac_f_in")
        with col_e3:
            ac_r = st.number_input("المقاومة R (أوم):", value=30.0, key="ac_r_in")
        with col_e4:
            ac_l = st.number_input("الملف L (مللي هنري):", value=120.0, key="ac_l_in")
        with col_e5:
            ac_c = st.number_input("المكثف C (ميكرو فاراد):", value=100.0, key="ac_c_in")

        if st.button("⚙️ حساب خصائص دائرة التيار المتردد RLC", type="primary", key="btn_solve_rlc"):
            w_ang = 2.0 * math.pi * ac_f
            xl = w_ang * (ac_l * 1e-3)
            xc = 1.0 / (w_ang * (ac_c * 1e-6)) if ac_c > 0 else 0.0
            z_mag = math.hypot(ac_r, xl - xc)
            phi_rad = math.atan2(xl - xc, ac_r)
            phi_deg = math.degrees(phi_rad)
            pf = math.cos(phi_rad)
            i_rms = ac_v / z_mag
            p_watt = ac_v * i_rms * pf
            q_var = ac_v * i_rms * math.sin(phi_rad)
            s_va = ac_v * i_rms

            st.latex(f"X_L = 2\\pi f L = {xl:.3f}\\,\\Omega, \\quad X_C = \\frac{{1}}{{2\\pi f C}} = {xc:.3f}\\,\\Omega")
            st.latex(f"Z = {ac_r:.2f} + j({xl - xc:.3f}) = {z_mag:.3f}\\angle {phi_deg:.2f}^\\circ\\,\\Omega")
            st.success(
                f"⚡ **التيار الكلي I =** `{i_rms:.4f} A` | **معامل القدرة PF =** `{pf:.4f}` | "
                f"**القدرة الفعالة P =** `{p_watt:.2f} W` | **غير الفعالة Q =** `{q_var:.2f} VAR` | **الظاهرية S =** `{s_va:.2f} VA`"
            )

    elif elec_mode.startswith("2."):
        st.markdown("##### 🔌 بيانات اختبار اللاحمل (Open-Circuit Test) واختبار القصر (Short-Circuit Test):")
        col_t1, col_t2, col_t3, col_t4, col_t5, col_t6 = st.columns(6)
        with col_t1:
            voc = st.number_input("V_oc (V):", value=220.0, key="tr_voc")
        with col_t2:
            ioc = st.number_input("I_oc (A):", value=1.5, key="tr_ioc")
        with col_t3:
            poc = st.number_input("P_oc (W):", value=120.0, key="tr_poc")
        with col_t4:
            vsc = st.number_input("V_sc (V):", value=45.0, key="tr_vsc")
        with col_t5:
            isc = st.number_input("I_sc (A):", value=10.0, key="tr_isc")
        with col_t6:
            psc = st.number_input("P_sc (W):", value=250.0, key="tr_psc")

        if st.button("⚙️ حساب ثوابت الدائرة المكافئة للمحول الكهربي", type="primary", key="btn_solve_transformer"):
            cos_phi0 = poc / (voc * ioc)
            sin_phi0 = math.sqrt(max(0.0, 1.0 - cos_phi0**2))
            iw = ioc * cos_phi0
            im = ioc * sin_phi0
            rc = voc / iw if iw > 0 else float("inf")
            xm = voc / im if im > 0 else float("inf")

            req = psc / (isc**2)
            zeq = vsc / isc
            xeq = math.sqrt(max(0.0, zeq**2 - req**2))

            st.latex(f"\\text{{Open-Circuit Branch: }} R_c = {rc:.2f}\\,\\Omega, \\quad X_m = {xm:.2f}\\,\\Omega, \\quad \\cos(\\phi_0) = {cos_phi0:.4f}")
            st.latex(f"\\text{{Short-Circuit Equivalent: }} R_{{eq}} = {req:.3f}\\,\\Omega, \\quad Z_{{eq}} = {zeq:.3f}\\,\\Omega, \\quad X_{{eq}} = {xeq:.3f}\\,\\Omega")

    else:
        circ_eqs = st.text_area(
            "✍️ أدخل معادلات كيرشوف للجهد والتيار (KVL / KCL):",
            value="10*I1 + 5*(I1 - I2) = 24\n5*(I2 - I1) + 15*I2 = 0",
            height=95,
            key="circ_kvl_area",
        )
        if st.button("⚙️ حل معادلات الدائرة الكهربية", type="primary", key="btn_solve_kvl"):
            eq_list_c = []
            for ln in circ_eqs.strip().split("\n"):
                if "=" in ln:
                    l_s, r_s = ln.split("=", 1)
                    eq_list_c.append(sp.Eq(sp.sympify(l_s), sp.sympify(r_s)))
            sol_c = sp.solve(eq_list_c, dict=True)
            st.write("✅ **قيم التيارات والجهود المستنتجة:**", sol_c)

    render_camera_and_ai_hub(
        tab_key_prefix="tab6_electrical",
        category_title="الدوائر الكهربية، التيار المتردد، والمحولات الكهربية",
        domain_ai_instructions="تحليل صور ومخططات الدوائر الكهربية (توالي، توازي، دلتا-ستار، قوانين كيرشوف، التحليل العقدي والشبكي، ثيفينين ونورتن، دوائر التيار المتردد ثلاثية الأوجه، والمحولات والآلات الكهربية).",
        current_text_data_getter=lambda: f"النظام الكهربي: {elec_mode}",
    )


# ==============================================================================
# التبويب السابع: الميكانيكا الهندسية، الاستاتيكا، الديناميكا، ومقاومة المواد
# ==============================================================================
with TABS_LIST[6]:
    st.markdown(
        """
        <div class='section-card'>
            <h3>⚙️ القسم السابع: الميكانيكا الهندسية (الاستاتيكا، الديناميكا، والإجهادات الميكانيكية)</h3>
            <p>تحليل محصلة القوى والاتزان، ردود أفعال الكمرات (Beams)، المقذوفات والحركة الديناميكية، وحسابات الإجهاد والانفعال (Stress & Strain).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mech_mode = st.selectbox(
        "⚙️ اختر نوع المسألة الميكانيكية:",
        options=[
            "1. محصلة القوى المستوية وزاوية المحصلة (2D Force Resultant ∑Fx, ∑Fy)",
            "2. حركة المقذوفات في مجال الجاذبية (Projectile Motion Kinematics)",
            "3. حسابات الإجهاد العمودي، إجهاد القص، ومعامل الأمان (Stress, Strain & Safety Factor)",
        ],
        key="mech_mode_select",
    )

    if mech_mode.startswith("1."):
        forces_input = st.text_area(
            "✍️ أدخل القوى بصيغة (مقدار القوة بالنيوتن , زاوية القوة بالدرجات مع محور x الموجب) كل قوة في سطر:",
            value="100, 30\n150, 120\n80, 270",
            height=95,
            key="forces_textarea",
        )
        if st.button("⚙️ حساب محصلة القوى واتجاهها", type="primary", key="btn_solve_forces"):
            rx, ry = 0.0, 0.0
            for ln in forces_input.strip().split("\n"):
                if "," in ln:
                    mag_f, ang_f = map(float, ln.split(","))
                    rx += mag_f * math.cos(math.radians(ang_f))
                    ry += mag_f * math.sin(math.radians(ang_f))
            r_mag = math.hypot(rx, ry)
            r_ang = math.degrees(math.atan2(ry, rx))
            st.latex(f"\\sum F_x = {rx:.3f}\\,\\text{{N}}, \\quad \\sum F_y = {ry:.3f}\\,\\text{{N}}")
            st.latex(f"R = \\sqrt{{(\\sum F_x)^2 + (\\sum F_y)^2}} = {r_mag:.3f}\\,\\text{{N}}, \\quad \\theta_R = {r_ang:.2f}^\\circ")

    elif mech_mode.startswith("2."):
        col_pr1, col_pr2, col_pr3 = st.columns(3)
        with col_pr1:
            v0_proj = st.number_input("السرعة الابتدائية v0 (m/s):", value=30.0, key="pr_v0")
        with col_pr2:
            ang_proj = st.number_input("زاوية الإطلاق θ (درجة):", value=45.0, key="pr_ang")
        with col_pr3:
            g_proj = st.number_input("عجلة الجاذبية g (m/s²):", value=9.81, key="pr_g")

        if st.button("⚙️ حساب مسار المقذوف ورسمه", type="primary", key="btn_solve_proj"):
            rad_p = math.radians(ang_proj)
            t_flight = (2.0 * v0_proj * math.sin(rad_p)) / g_proj
            h_max = (v0_proj**2 * (math.sin(rad_p) ** 2)) / (2.0 * g_proj)
            r_max = (v0_proj**2 * math.sin(2.0 * rad_p)) / g_proj
            st.success(f"⏱️ **زمن التحليق الكلي:** `{t_flight:.3f} s` | **أقصى ارتفاع:** `{h_max:.3f} m` | **المدى الأفقي:** `{r_max:.3f} m`")

            t_arr = np.linspace(0, t_flight, 200)
            x_tr = v0_proj * math.cos(rad_p) * t_arr
            y_tr = v0_proj * math.sin(rad_p) * t_arr - 0.5 * g_proj * (t_arr**2)
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.plot(x_tr, y_tr, color="#15803D", linewidth=2.2)
            ax.set_title("مسار حركة المقذوف (Projectile Trajectory)")
            ax.set_xlabel("Horizontal Distance x (m)")
            ax.set_ylabel("Height y (m)")
            ax.grid(True, linestyle=":")
            st.pyplot(fig)

    else:
        col_st1, col_st2, col_st3, col_st4 = st.columns(4)
        with col_st1:
            load_f = st.number_input("القوة المؤثرة F (N):", value=50000.0, key="st_f")
        with col_st2:
            area_mm = st.number_input("مساحة المقطع A (mm²):", value=250.0, key="st_a")
        with col_st3:
            len_mm = st.number_input("الطول الأصلي L (mm):", value=1000.0, key="st_l")
        with col_st4:
            e_gpa = st.number_input("معامل يونج E (GPa):", value=200.0, key="st_e")

        if st.button("⚙️ حساب الإجهاد والاستطالة", type="primary", key="btn_solve_stress"):
            sigma_mpa = load_f / area_mm
            delta_l = (load_f * len_mm) / (area_mm * (e_gpa * 1000.0))
            strain_val = delta_l / len_mm
            st.latex(f"\\sigma = \\frac{{F}}{{A}} = {sigma_mpa:.3f}\\,\\text{{MPa}}, \\quad \\epsilon = {strain_val:.6f}, \\quad \\Delta L = {delta_l:.4f}\\,\\text{{mm}}")

    render_camera_and_ai_hub(
        tab_key_prefix="tab7_mechanics",
        category_title="الميكانيكا الهندسية، الاستاتيكا، الديناميكا، ومقاومة المواد",
        domain_ai_instructions="قراءة صور مخططات الجسم الحر (Free Body Diagrams)، الجمالونات (Trusses)، الكمرات (Shear & Bending Moment Diagrams)، عزوم القصور الذاتي، الديناميكا، وتصميم الأجزاء الميكانيكية.",
        current_text_data_getter=lambda: f"الفرع الميكانيكي: {mech_mode}",
    )


# ==============================================================================
# التبويب الثامن: هندسة التحكم الآلي (Control Systems)، الـ PLC، والمتحكمات
# ==============================================================================
with TABS_LIST[7]:
    st.markdown(
        """
        <div class='section-card'>
            <h3>🎛️ القسم الثامن: هندسة التحكم الآلي الكلاسيكي (Control Systems)، الـ PLC، والميكاترونكس</h3>
            <p>تحليل دالة النقل (Transfer Function)، حساب الأقطاب والأصفار (Poles & Zeros)، خصائص استجابة النظام من الدرجة الثانية (Overshoot, Settling Time)، ومنطق الـ PLC والريلاي.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        tf_num_str = st.text_input(
            "🔢 معاملات بسط دالة النقل Numerator (من أعلى أس لـ s):",
            value="25",
            key="ctrl_num_in",
        )
    with col_ctrl2:
        tf_den_str = st.text_input(
            "🔢 معاملات مقام دالة النقل Denominator (مثال s^2 + 4s + 25 -> 1, 4, 25):",
            value="1, 4, 25",
            key="ctrl_den_in",
        )

    if st.button("⚙️ تحليل دالة النقل ورسم استجابة الخطوة (Step Response)", type="primary", key="btn_solve_control"):
        try:
            from scipy import signal

            num_c = [float(x.strip()) for x in tf_num_str.split(",") if x.strip()]
            den_c = [float(x.strip()) for x in tf_den_str.split(",") if x.strip()]

            sys_tf = signal.TransferFunction(num_c, den_c)
            zeros_tf = np.roots(num_c) if len(num_c) > 1 else []
            poles_tf = np.roots(den_c)

            st.write(f"**أصفار النظام (Zeros):** `{np.round(zeros_tf, 4)}`")
            st.write(f"**أقطاب النظام (Poles):** `{np.round(poles_tf, 4)}`")

            is_stable = bool(np.all(np.real(poles_tf) < 0))
            if is_stable:
                st.success("✅ **حالة الاستقرار:** النظام مستقر (جميع الأقطاب تقع في النصف الأيسر LHP).")
            else:
                st.error("⚠️ **حالة الاستقرار:** النظام غير مستقر أو حرج الاستقرار!")

            if len(den_c) == 3 and den_c[0] > 0:
                wn = math.sqrt(den_c[2] / den_c[0])
                zeta = (den_c[1] / den_c[0]) / (2.0 * wn)
                st.latex(f"\\omega_n = {wn:.4f}\\,\\text{{rad/s}}, \\quad \\zeta = {zeta:.4f}")
                if 0 < zeta < 1:
                    wd = wn * math.sqrt(1.0 - zeta**2)
                    mp_pct = 100.0 * math.exp((-zeta * math.pi) / math.sqrt(1.0 - zeta**2))
                    ts_4 = 4.0 / (zeta * wn)
                    tp_val = math.pi / wd
                    st.info(
                        f"📊 **أقصى تجاوز (Max Overshoot %Mp):** `{mp_pct:.2f}%` | "
                        f"**زمن الاستقرار (Settling Time 2%):** `{ts_4:.4f} s` | "
                        f"**زمن القمة (Peak Time Tp):** `{tp_val:.4f} s`"
                    )

            t_vals, y_vals = signal.step(sys_tf)
            fig, ax = plt.subplots(figsize=(7.5, 3.3))
            ax.plot(t_vals, y_vals, color="#1D4ED8", linewidth=2.2, label="Unit Step Response y(t)")
            ax.axhline(y_vals[-1], color="#DC2626", linestyle="--", linewidth=1, label=f"Steady-State = {y_vals[-1]:.3f}")
            ax.set_title("استجابة النظام لدخل الوحدة (Unit Step Response)")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, linestyle=":")
            ax.legend()
            st.pyplot(fig)

            log_operation_to_history("التحكم الآلي والميكاترونكس", "Transfer Function Analyzer", f"Num:{num_c}, Den:{den_c}", f"Poles:{poles_tf}")
        except Exception as err:
            st.error(f"❌ خطأ في تحليل نظام التحكم: {err}")

    render_camera_and_ai_hub(
        tab_key_prefix="tab8_control",
        category_title="هندسة التحكم الآلي، الـ PLC، والمتحكمات الدقيقة",
        domain_ai_instructions="تحليل المخططات الصندوقية (Block Diagrams)، معيار راوث للاستقرار (Routh-Hurwitz)، محلول الجذور (Root Locus)، مخططات بود (Bode Plots)، متحكمات PID، مخططات السلم (Ladder Logic PLC)، ودوائر الريلاي والكونتاكتور والأردوينو و AVR.",
        current_text_data_getter=lambda: f"بسط دالة النقل: {tf_num_str}\nمقام دالة النقل: {tf_den_str}",
    )


# ==============================================================================
# التبويب التاسع: المسائل اللفظية، البرمجة، الخوارزميات، وتصحيح الأكواد
# ==============================================================================
with TABS_LIST[8]:
    st.markdown(
        """
        <div class='section-card'>
            <h3>💻 القسم التاسع: المسائل اللفظية، البرمجة، الخوارزميات، وتصحيح الأكواد (Coding & Word Problems)</h3>
            <p>حل المسائل الكلامية وترجمتها لمعادلات، كتابة وتصحيح أكواد (Python, C++, MATLAB, Arduino, Embedded C)، وتنفيذ أكواد بايثون مباشرة.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    word_code_input = st.text_area(
        "✍️ اكتب المسألة اللفظية، أو الصق الكود البرمجي هنا:",
        value=(
            "# مثال كود بايثون لحساب مضروب الأعداد من 1 إلى 6:\n"
            "import math\n"
            "for n in range(1, 7):\n"
            "    print(f'{n}! = {math.factorial(n)}')"
        ),
        height=135,
        key="word_code_textarea",
    )

    if st.button("▶️ تشغيل كود بايثون مباشرة وعرض المخرجات", type="primary", key="btn_run_py_code"):
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        try:
            exec_env = {"np": np, "sp": sp, "pd": pd, "math": math}
            exec(word_code_input, exec_env)
            output_str = redirected_output.getvalue()
            st.code(output_str if output_str else "✅ تم تنفيذ الكود بنجاح (بدون مخرجات طباعة).", language="text")
        except Exception as err:
            st.error(f"❌ خطأ برمجي أثناء التنفيذ:\n{err}\n💡 اضغط على زر (حل المسألة بالذكاء الاصطناعي) بالأسفل ليقوم بتصحيح الخطأ فوراً.")
        finally:
            sys.stdout = old_stdout

    render_camera_and_ai_hub(
        tab_key_prefix="tab9_word_coding",
        category_title="المسائل اللفظية والبرمجة والخوارزميات",
        domain_ai_instructions="تحليل المسائل الكلامية واستنتاج المعادلات وحلها، أو قراءة صور الشاشات والأكواد البرمجية ومخططات التدفق (Flowcharts) بلغات Python, C, C++, MATLAB, Arduino واكتشاف الأخطاء وتصحيحها.",
        current_text_data_getter=lambda: word_code_input,
    )


# ==============================================================================
# التبويب العاشر: سجل العمليات، القوانين الهندسية المرجعية، وتصدير التقارير
# ==============================================================================
with TABS_LIST[9]:
    st.markdown(
        """
        <div class='section-card'>
            <h3>📑 القسم العاشر: سجل العمليات، ملخص القوانين الهندسية، وتصدير التقارير</h3>
            <p>مراجعة جميع المسائل التي قمت بحلها خلال الجلسة، مع إمكانية تحميلها كتقرير Markdown أو جدول CSV، بالإضافة إلى تصوير أي ورقة خارجية عامة.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.history_logs:
        df_hist = pd.DataFrame(st.session_state.history_logs)
        st.dataframe(df_hist[["id", "timestamp", "category", "solver", "input"]], use_container_width=True)

        md_report_lines = [
            "# 📑 التقرير الهندسي والرياضي الشامل",
            f"**تاريخ التصدير:** `{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n---\n",
        ]
        for item in st.session_state.history_logs:
            md_report_lines.append(f"## 🔹 العملية ({item['id']}) - {item['category']} ({item['solver']})")
            md_report_lines.append(f"- **التوقيت:** `{item['timestamp']}`")
            md_report_lines.append(f"### 📥 المدخلات:\n```\n{item['input']}\n```")
            md_report_lines.append(f"### 📤 النتيجة والحل:\n{item['output']}\n---\n")

        full_md_str = "\n".join(md_report_lines)

        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1:
            st.download_button(
                label="📥 تحميل التقرير الكامل (Markdown .md)",
                data=full_md_str.encode("utf-8"),
                file_name="Engineering_Solutions_Report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_dl2:
            st.download_button(
                label="📊 تحميل جدول السجل (Excel CSV)",
                data=df_hist.to_csv(index=False).encode("utf-8-sig"),
                file_name="Engineering_History.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_dl3:
            if st.button("🗑️ مسح السجل بالكامل", use_container_width=True):
                st.session_state.history_logs.clear()
                st.rerun()
    else:
        st.info("ℹ️ لا توجد عمليات مسجلة في السجل حتى الآن. قم بحل أي مسألة في الأقسام السابقة لتظهر هنا تلقائياً.")

    render_camera_and_ai_hub(
        tab_key_prefix="tab10_general",
        category_title="المسائل الهندسية والعلمية العامة والامتحانات الشاملة",
        domain_ai_instructions="حل أي ورقة امتحان كاملة أو مسألة متعددة الفروع تجمع بين الرياضيات والفيزياء والهندسة والبرمجة بدقة تفصيلية كاملة.",
        current_text_data_getter=lambda: "ورقة مسألة عامة شاملة",
    )
