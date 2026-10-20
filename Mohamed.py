# ==============================================================================
# 🚀 المنصة الهندسية الجامعية المتقدمة | Advanced College of Engineering Platform
# File: Mohamed.py (100% Streamlit Cloud Compatible - Zero SyntaxError)
# Modules: Project Management (AON/AOA/Lags/PERT/Gantt), Engineering Economy,
# Numerical Methods, ODEs/Laplace, Mechanics (Mohr/Beams), Power/Circuits, Control
# Integrated with Live Camera + OpenCV Vision Filters + Gemini AI in EVERY Tab
# ==============================================================================

import os
import io
import sys
import math
import datetime
import subprocess

REQUIRED_LIBS = [
    ("streamlit", "streamlit"),
    ("google.generativeai", "google-generativeai"),
    ("numpy", "numpy"),
    ("sympy", "sympy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("PIL", "pillow"),
    ("cv2", "opencv-python-headless"),
    ("networkx", "networkx"),
]

for mod_name, pip_pkg in REQUIRED_LIBS:
    try:
        __import__(mod_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_pkg])

import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy import signal, stats
from PIL import Image, ImageEnhance
import cv2
import google.generativeai as genai

# ==============================================================================
# 1. إعدادات الصفحة والتنسيق ثنائي اللغة (Page Config & RTL/LTR Styling)
# ==============================================================================
st.set_page_config(
    page_title="Engineering Solver Hub | المنصة الهندسية المتقدمة",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 97%; }
    .eng-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 55%, #0F766E 100%);
        color: white; padding: 20px 24px; border-radius: 12px; margin-bottom: 16px;
        direction: rtl; text-align: right; box-shadow: 0 4px 14px rgba(0,0,0,0.2);
    }
    .eng-banner h1 { margin: 0; font-size: 24px; color: #FFFFFF; }
    .eng-banner p { margin: 6px 0 0 0; font-size: 14px; opacity: 0.92; }
    .sec-header {
        background: #F8FAFC; border: 1px solid #CBD5E1; border-right: 6px solid #1D4ED8;
        padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; direction: rtl; text-align: right;
    }
    .ai-cam-box {
        background: #F0F9FF; border: 2px dashed #0284C7; padding: 14px;
        border-radius: 10px; margin-top: 14px; direction: rtl; text-align: right;
    }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("GEMINI_API_KEY", "")


def add_log(module_name, method_used, inputs_str, outputs_str):
    st.session_state.history.append({
        "ID": len(st.session_state.history) + 1,
        "Time | الوقت": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Module | القسم": module_name,
        "Method | الطريقة": method_used,
        "Inputs | المدخلات": str(inputs_str)[:300],
        "Result | النتيجة": str(outputs_str)[:500],
    })


# ==============================================================================
# 2. القائمة الجانبية وإعدادات الذكاء الاصطناعي (Sidebar & AI Configuration)
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ الإعدادات الهندسية | System Config")
    key_in = st.text_input(
        "🔑 Gemini API Key:",
        value=st.session_state.api_key,
        type="password",
        help="لتفعيل حل المسائل بالذكاء الاصطناعي وقراءة صور الكاميرا والرسومات الهندسية.",
    )
    st.session_state.api_key = key_in.strip()
    if st.session_state.api_key:
        genai.configure(api_key=st.session_state.api_key)
        st.success("✅ AI & Vision Active | الذكاء الاصطناعي مفعل")
    else:
        st.warning("⚠️ Enter API Key for AI | الحل البرمجي يعمل فوراً بدون مفتاح")

    ai_model_name = st.selectbox(
        "🤖 AI Model | النموذج:",
        ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
    )
    ai_detail_mode = st.selectbox(
        "📝 Solution Style | أسلوب الحل:",
        [
            "Academic Step-by-Step | حل أكاديمي مفصل بالقوانين والخطوات",
            "Exam Standard Solution | حل نموذجي للامتحانات الهندسية",
            "Concise Engineering Summary | ملخص هندسي سريع للنواتج",
        ],
    )
    ai_temp_val = st.slider("🎯 Precision (Temperature):", 0.0, 1.0, 0.1, 0.05)
    st.markdown("---")
    st.metric("📊 Solved Problems | المسائل المحلولة", len(st.session_state.history))


# ==============================================================================
# 3. محرك الكاميرا ومعالجة الصور والـ AI الموحد لكل الأقسام (Universal Camera & AI)
# ==============================================================================
def apply_cv_preprocessing(pil_img, mode_str, rot_angle=0):
    if pil_img is None:
        return None
    img = pil_img.convert("RGB")
    if rot_angle != 0:
        img = img.rotate(-rot_angle, expand=True)
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    if "Otsu" in mode_str:
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(th)
    elif "Adaptive" in mode_str:
        ad = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
        return Image.fromarray(ad)
    elif "Contrast" in mode_str:
        c_img = ImageEnhance.Contrast(img).enhance(1.85)
        return ImageEnhance.Sharpness(c_img).enhance(2.2)
    elif "Canny" in mode_str:
        edges = cv2. bitwise_not(cv2.Canny(gray, 50, 150))
        return Image.fromarray(edges)
    return img


def query_gemini_engine(domain_title, domain_guide, text_data, pil_img=None, mode="solve"):
    if not st.session_state.api_key:
        return "❌ **Error:** يرجى إدخال مفتاح `Gemini API Key` في القائمة الجانبية أولاً."
    try:
        model = genai.GenerativeModel(
            model_name=ai_model_name,
            generation_config={"temperature": ai_temp_val, "max_output_tokens": 4096},
        )
        if mode == "ocr":
            prompt = (
                f"You are an Engineering OCR Scanner for ({domain_title}). "
                "Extract all equations, tables, network nodes, numbers, and text accurately from the image. "
                "Provide both standard mathematical notation (LaTeX) and ready-to-use input format."
            )
        else:
            prompt = (
                f"أنت بروفيسور في كلية الهندسة متخصص في ({domain_title}).\n"
                f"أسلوب الإجابة المطلوب: {ai_detail_mode}.\n"
                f"توجيهات التخصص: {domain_guide}\n"
                "قم بحل المسألة الهندسية (من الصورة أو النص) باللغتين العربية والإنجليزية للمصطلحات، متضمناً:\n"
                "1. **المعطيات والمطلوب (Given & Required)** مع الوحدات القياسية.\n"
                "2. **القوانين الهندسية (Governing Equations)** بصيغة LaTeX واضحة.\n"
                "3. **خطوات الحل والجداول التفصيلية (Step-by-Step Calculation & Tables)**.\n"
                "4. **الناتج النهائي والتعليق الهندسي (Final Answer & Engineering Interpretation)**.\n"
                f"\nبيانات المسألة النصية:\n{text_data}"
            )
        content = [prompt]
        if pil_img is not None:
            content.append(pil_img)
        return model.generate_content(content).text
    except Exception as e:
        return f"❌ **AI Execution Error:** `{e}`"


def render_camera_ai_section(key_id, domain_title, domain_guide, get_context_fn):
    st.markdown(
        f"<div class='ai-cam-box'><b>📸🤖 وحدة الكاميرا الذكية والرؤية الحاسوبية والـ AI | Camera, OpenCV & AI Solver — ({domain_title})</b></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1.8, 2.2, 1])
    with c1:
        src = st.radio(
            "📷 Image Source | مصدر الصورة:",
            ["Text Only | نص فقط", "Camera | كاميرا مباشرة", "Upload | رفع صورة"],
            horizontal=True,
            key=f"{key_id}_src",
        )
    with c2:
        flt = st.selectbox(
            "🔍 OpenCV Filter | فلتر الرؤية:",
            [
                "Contrast & Sharpness | زيادة التباين والحدة للرسومات",
                "Otsu Thresholding | تصفية المعادلات والجداول",
                "Adaptive Threshold | إزالة الظلال والإضاءة الضعيفة",
                "Canny Edges | إبراز الحواف والشبكات الهندسية",
                "Original | الصورة الأصلية",
            ],
            key=f"{key_id}_flt",
        )
    with c3:
        rot = st.selectbox("🔄 Rotate | تدوير:", [0, 90, 180, 270], key=f"{key_id}_rot")

    raw_img, proc_img = None, None
    if "Camera" in src:
        cam_buf = st.camera_input(f"📸 Capture ({domain_title})", key=f"{key_id}_cam")
        if cam_buf:
            raw_img = Image.open(cam_buf)
    elif "Upload" in src:
        up_buf = st.file_uploader(
            f"📁 Upload Image ({domain_title})",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"{key_id}_up",
        )
        if up_buf:
            raw_img = Image.open(up_buf)

    if raw_img is not None:
        proc_img = apply_cv_preprocessing(raw_img, flt, rot)
        ic1, ic2 = st.columns(2)
        with ic1:
            st.image(raw_img, caption="Original Image | الصورة الأصلية", use_container_width=True)
        with ic2:
            st.image(proc_img, caption=f"OpenCV Processed ({flt})", use_container_width=True)

    extra_q = st.text_input(
        "💡 Specific AI Instruction | توجيه خاص للذكاء الاصطناعي (اختياري):",
        placeholder="مثال: احسب عند ضغط المشروع 3 أيام، أو حل باستخدام طريقة أخرى للتحقق...",
        key=f"{key_id}_note",
    )

    b1, b2 = st.columns(2)
    with b1:
        solve_ai_clicked = st.button(
            f"🤖 Solve with AI | حل شامل بالذكاء الاصطناعي ({domain_title})",
            type="primary",
            use_container_width=True,
            key=f"{key_id}_btn_solve",
        )
    with b2:
        ocr_ai_clicked = st.button(
            "📝 Extract Data/Equations from Image (OCR) | استخراج البيانات من الصورة",
            use_container_width=True,
            key=f"{key_id}_btn_ocr",
        )

    if ocr_ai_clicked:
        if proc_img is None:
            st.warning("⚠️ يرجى التقاط صورة بالكاميرا أو رفعها أولاً.")
        else:
            with st.spinner("🔍 Scanning image with Gemini Vision OCR..."):
                res_ocr = query_gemini_engine(domain_title, domain_guide, extra_q, proc_img, mode="ocr")
                st.markdown("#### 📝 المستخرج البصري | Extracted Problem Data:")
                st.markdown(res_ocr)
                add_log(domain_title, "Vision OCR", "Image Input", res_ocr)

    if solve_ai_clicked:
        ctx_str = get_context_fn()
        if extra_q.strip():
            ctx_str += f"\nUser Note: {extra_q.strip()}"
        if proc_img is None and not ctx_str.strip():
            st.warning("⚠️ أدخل معطيات المسألة أو التقط صورتها أولاً.")
        else:
            with st.spinner(f"🤖 Solving ({domain_title}) problem step-by-step..."):
                res_solve = query_gemini_engine(domain_title, domain_guide, ctx_str, proc_img, mode="solve")
                st.markdown(f"#### 🤖 الحل الهندسي التفصيلي | AI Engineering Solution ({domain_title}):")
                st.markdown(res_solve)
                add_log(domain_title, "Gemini AI Solver", ctx_str, res_solve)


# ==============================================================================
# 4. العنوان الرئيسي والتبويبات الهندسية الثمانية (8 Core Engineering Tabs)
# ==============================================================================
st.markdown("""
<div class='eng-banner'>
    <h1>⚙️ المنصة الهندسية الجامعية المتقدمة | Advanced Engineering Solver Platform</h1>
    <p>Project Management (AON/AOA/Lags/PERT) • Engineering Economy (Interest/CFD/IRR/Depreciation) • Numerical Methods • ODEs & Laplace • Mechanics & Mohr's Circle • 3-Phase & Transformers • Control Systems • Live Camera & AI</p>
</div>
""", unsafe_allow_html=True)

TABS = st.tabs([
    "1️⃣ إدارة المشروعات (AON / AOA / Lags / PERT)",
    "2️⃣ الاقتصاد الهندسي والفوائد (Engineering Economy)",
    "3️⃣ الطرق العددية المتقدمة (Numerical Methods)",
    "4️⃣ المعادلات التفاضلية ولابلاس (ODEs & Laplace)",
    "5️⃣ الميكانيكا ومقاومة المواد (Mohr & Beams)",
    "6️⃣ القوى والدوائر الكهربية (Circuits & Transformers)",
    "7️⃣ التحكم الآلي وحالة الفراغ (Control & State-Space)",
    "8️⃣ البرمجة الهندسية والتقارير (Code & Reports)",
])


# ==============================================================================
# TAB 1: إدارة المشروعات والشبكات (Project Management: AON, AOA, Lags, Gantt, PERT)
# ==============================================================================
with TABS[0]:
    st.markdown("""
    <div class='sec-header'>
        <h3>📊 1. إدارة المشروعات والشبكات الهندسية | Project Management (CPM, AON, AOA, Lags & PERT)</h3>
        <p>حساب المسار الحرج (Critical Path Method) مع دعم كامل لجميع العلاقات (<b>FS, SS, FF, SF</b>) وفترات التأخير والتداخل (<b>Lags / Leads</b>)، وحساب (<b>ES, EF, LS, LF, TF, FF</b>)، ورسم مخطط <b>AON</b>، ومخطط الأسهم <b>AOA</b>، ومخطط <b>Gantt Chart</b>، وتقييم <b>PERT</b> الاحتمالي.</p>
    </div>
    """, unsafe_allow_html=True)

    pm_sub = st.radio(
        "🛠️ Select Mode | اختر نظام الحل:",
        [
            "1. PDM / CPM Network with Lags (AON + AOA + Gantt Chart) | الشبكات والمسار الحرج والـ Lags",
            "2. PERT Probabilistic Analysis (to, tm, tp & Z-Probability) | تحليل بيرت الاحتمالي",
        ],
        horizontal=True,
    )

    if pm_sub.startswith("1."):
        st.caption(
            "📌 **طريقة الإدخال (Input Format):** في كل سطر اكتب: `Activity, Duration, Predecessors` "
            "| يمكنك كتابة الاعتمادية العادية مثل `A` (تلقائياً `FS+0`) أو تحديد نوع العلاقة والـ Lag مثل: "
            "`A:FS+2` أو `B:SS+3` أو `C:FF+4` أو `D:SF+1` (افصل بين الاعتماديات المتعددة بفاصلة `,`)."
        )
        default_pdm = (
            "A, 6, -\n"
            "B, 8, A:FS+0\n"
            "C, 10, A:SS+3\n"
            "D, 7, B:FS+2, C:FF+4\n"
            "E, 5, C:FS+0\n"
            "F, 4, D:FS+0, E:FF+2"
        )
        pdm_text = st.text_area(
            "✍️ جدول الأنشطة والعلاقات والـ Lags | Activities, Durations & Predecessor Relations (FS/SS/FF/SF ± Lag):",
            value=default_pdm,
            height=145,
        )

        if st.button("⚙️ Solve Network (ES, EF, LS, LF, TF, FF) & Draw AON / AOA / Gantt", type="primary"):
            try:
                tasks = {}
                order_keys = []
                for raw_line in pdm_text.strip().split("\n"):
                    if not raw_line.strip():
                        continue
                    parts = [p.strip() for p in raw_line.split(",")]
                    t_id = parts[0]
                    dur = float(parts[1])
                    rels = []
                    for p_tok in parts[2:]:
                        if not p_tok or p_tok == "-":
                            continue
                        if ":" in p_tok:
                            p_name, r_spec = p_tok.split(":", 1)
                            r_type = r_spec[:2].upper()
                            lag_v = float(r_spec[2:]) if len(r_spec) > 2 else 0.0
                        else:
                            p_name, r_type, lag_v = p_tok.strip(), "FS", 0.0
                        rels.append((p_name.strip(), r_type, lag_v))
                    tasks[t_id] = {
                        "dur": dur,
                        "preds": rels,
                        "ES": 0.0,
                        "EF": dur,
                        "LS": 0.0,
                        "LF": dur,
                        "TF": 0.0,
                        "FF": 0.0,
                    }
                    order_keys.append(t_id)

                # Forward Pass (Iterative relaxation to handle any topological order & lags)
                for _ in range(len(order_keys) + 2):
                    for t_id in order_keys:
                        d_curr = tasks[t_id]["dur"]
                        es_candidates = [0.0]
                        for p_id, r_type, lag in tasks[t_id]["preds"]:
                            if p_id not in tasks:
                                continue
                            p_es, p_ef = tasks[p_id]["ES"], tasks[p_id]["EF"]
                            if r_type == "FS":
                                es_candidates.append(p_ef + lag)
                            elif r_type == "SS":
                                es_candidates.append(p_es + lag)
                            elif r_type == "FF":
                                es_candidates.append(p_ef + lag - d_curr)
                            elif r_type == "SF":
                                es_candidates.append(p_es + lag - d_curr)
                        tasks[t_id]["ES"] = max(es_candidates)
                        tasks[t_id]["EF"] = tasks[t_id]["ES"] + d_curr

                proj_dur = max(tasks[k]["EF"] for k in order_keys)

                # Initialize Backward Pass
                for k in order_keys:
                    tasks[k]["LF"] = proj_dur
                    tasks[k]["LS"] = proj_dur - tasks[k]["dur"]

                # Backward Pass with FS, SS, FF, SF & Lags
                for _ in range(len(order_keys) + 2):
                    for t_id in reversed(order_keys):
                        d_curr = tasks[t_id]["dur"]
                        lf_candidates = [proj_dur]
                        for succ_id in order_keys:
                            for p_id, r_type, lag in tasks[succ_id]["preds"]:
                                if p_id == t_id:
                                    s_ls, s_lf = tasks[succ_id]["LS"], tasks[succ_id]["LF"]
                                    if r_type == "FS":
                                        lf_candidates.append(s_ls - lag)
                                    elif r_type == "SS":
                                        lf_candidates.append(s_ls - lag + d_curr)
                                    elif r_type == "FF":
                                        lf_candidates.append(s_lf - lag)
                                    elif r_type == "SF":
                                        lf_candidates.append(s_lf - lag + d_curr)
                        tasks[t_id]["LF"] = min(lf_candidates)
                        tasks[t_id]["LS"] = tasks[t_id]["LF"] - d_curr

                # Compute Total Float (TF) and Free Float (FF)
                crit_path = []
                table_rows = []
                for t_id in order_keys:
                    t_d = tasks[t_id]
                    tf_val = round(t_d["LS"] - t_d["ES"], 4)
                    t_d["TF"] = tf_val

                    ff_cands = []
                    for succ_id in order_keys:
                        for p_id, r_type, lag in tasks[succ_id]["preds"]:
                            if p_id == t_id:
                                s_es, s_ef = tasks[succ_id]["ES"], tasks[succ_id]["EF"]
                                if r_type == "FS":
                                    ff_cands.append(s_es - lag - t_d["EF"])
                                elif r_type == "SS":
                                    ff_cands.append(s_es - lag - t_d["ES"])
                                elif r_type == "FF":
                                    ff_cands.append(s_ef - lag - t_d["EF"])
                                elif r_type == "SF":
                                    ff_cands.append(s_ef - lag - t_d["ES"])
                    ff_val = round(min(ff_cands), 4) if ff_cands else round(proj_dur - t_d["EF"], 4)
                    t_d["FF"] = max(0.0, ff_val)
                    is_crit = abs(tf_val) < 1e-5
                    if is_crit:
                        crit_path.append(t_id)

                    pred_fmt = ", ".join([f"{p}({rt}{lg:+g})" for p, rt, lg in t_d["preds"]]) or "-"
                    table_rows.append({
                        "Activity | النشاط": t_id,
                        "Duration | المدة": t_d["dur"],
                        "Predecessors & Lags": pred_fmt,
                        "ES": t_d["ES"],
                        "EF": t_d["EF"],
                        "LS": t_d["LS"],
                        "LF": t_d["LF"],
                        "Total Float (TF)": t_d["TF"],
                        "Free Float (FF)": t_d["FF"],
                        "Critical | حرج؟": "🔴 YES" if is_crit else "🟢 NO",
                    })

                c_m1, c_m2 = st.columns(2)
                c_m1.success(f"⏱️ **Total Project Duration | مدة المشروع الكلية:** `{proj_dur}` وحدة زمنية")
                c_m2.error(f"🔥 **Critical Path | المسار الحرج:** `{' ➔ '.join(crit_path)}`")

                st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

                # Plots: 1) AON Diagram, 2) AOA Diagram, 3) Gantt Chart
                g_col1, g_col2 = st.columns(2)

                with g_col1:
                    G_aon = nx.DiGraph()
                    for t_id in order_keys:
                        d = tasks[t_id]
                        lbl = f"[{t_id}] Dur:{d['dur']:g}\nES:{d['ES']:g} | EF:{d['EF']:g}\nLS:{d['LS']:g} | LF:{d['LF']:g}\nTF:{d['TF']:g} | FF:{d['FF']:g}"
                        G_aon.add_node(t_id, label=lbl, crit=(abs(d["TF"]) < 1e-5))
                        for p_id, r_type, lag in d["preds"]:
                            if p_id in tasks:
                                G_aon.add_edge(p_id, t_id, rel=f"{r_type}{lag:+g}")

                    fig_aon, ax_aon = plt.subplots(figsize=(7, 4.5))
                    try:
                        for layer, nodes in enumerate(nx.topological_generations(G_aon)):
                            for n_item in nodes:
                                G_aon.nodes[n_item]["layer"] = layer
                        pos_aon = nx.multipartite_layout(G_aon, subset_key="layer")
                    except Exception:
                        pos_aon = nx.spring_layout(G_aon, seed=42)

                    n_colors = ["#FECACA" if G_aon.nodes[n].get("crit") else "#DBEAFE" for n in G_aon.nodes]
                    e_colors = [
                        "#DC2626" if (G_aon.nodes[u].get("crit") and G_aon.nodes[v].get("crit")) else "#475569"
                        for u, v in G_aon.edges
                    ]
                    nx.draw_networkx_nodes(G_aon, pos_aon, node_color=n_colors, node_size=2600, edgecolors="#1E293B", linewidths=1.5, ax=ax_aon)
                    nx.draw_networkx_labels(G_aon, pos_aon, labels=nx.get_node_attributes(G_aon, "label"), font_size=7.5, font_weight="bold", ax=ax_aon)
                    nx.draw_networkx_edges(G_aon, pos_aon, edge_color=e_colors, width=2, arrowsize=18, ax=ax_aon)
                    nx.draw_networkx_edge_labels(G_aon, pos_aon, edge_labels=nx.get_edge_attributes(G_aon, "rel"), font_size=8, font_color="#B91C1C", ax=ax_aon)
                    ax_aon.set_title("AON (Activity-on-Node) Network Diagram with Lags")
                    ax_aon.axis("off")
                    st.pyplot(fig_aon)

                with g_col2:
                    # AOA (Activity-on-Arrow) Event Graph Construction
                    G_aoa = nx.DiGraph()
                    event_id = 10
                    start_ev = {}
                    end_ev = {}
                    G_aoa.add_node(event_id, label=f"E{event_id}\n(t=0)")
                    for t_id in order_keys:
                        preds = [p for p, _, _ in tasks[t_id]["preds"] if p in tasks]
                        if not preds:
                            u_ev = 10
                        elif len(preds) == 1:
                            u_ev = end_ev[preds[0]]
                        else:
                            event_id += 10
                            u_ev = event_id
                            G_aoa.add_node(u_ev, label=f"E{u_ev}\n(t={tasks[t_id]['ES']:g})")
                            for p_idx, p_name in enumerate(preds):
                                G_aoa.add_edge(end_ev[p_name], u_ev, act=f"Dummy d{p_idx+1} (0)", style="dashed")
                        event_id += 10
                        v_ev = event_id
                        G_aoa.add_node(v_ev, label=f"E{v_ev}\n(t={tasks[t_id]['EF']:g})")
                        start_ev[t_id] = u_ev
                        end_ev[t_id] = v_ev
                        G_aoa.add_edge(u_ev, v_ev, act=f"{t_id} ({tasks[t_id]['dur']:g})", style="solid")

                    fig_aoa, ax_aoa = plt.subplots(figsize=(7, 4.5))
                    pos_aoa = nx.spring_layout(G_aoa, seed=24, k=1.1)
                    nx.draw_networkx_nodes(G_aoa, pos_aoa, node_color="#FEF08A", node_size=1300, edgecolors="#CA8A04", linewidths=1.5, ax=ax_aoa)
                    nx.draw_networkx_labels(G_aoa, pos_aoa, labels=nx.get_node_attributes(G_aoa, "label"), font_size=8, font_weight="bold", ax=ax_aoa)
                    nx.draw_networkx_edges(G_aoa, pos_aoa, edge_color="#0F766E", width=1.8, arrowsize=16, ax=ax_aoa)
                    nx.draw_networkx_edge_labels(G_aoa, pos_aoa, edge_labels=nx.get_edge_attributes(G_aoa, "act"), font_size=8, font_color="#0F172A", ax=ax_aoa)
                    ax_aoa.set_title("AOA (Activity-on-Arrow) Event Network & Dummies")
                    ax_aoa.axis("off")
                    st.pyplot(fig_aoa)

                # Gantt Chart
                fig_g, ax_g = plt.subplots(figsize=(10, 3.2))
                y_pos = np.arange(len(order_keys))
                for idx, t_id in enumerate(order_keys):
                    d = tasks[t_id]
                    c_bar = "#DC2626" if abs(d["TF"]) < 1e-5 else "#2563EB"
                    ax_g.barh(idx, d["dur"], left=d["ES"], height=0.5, color=c_bar, alpha=0.9, label="Critical" if idx == 0 else "")
                    if d["TF"] > 0:
                        ax_g.barh(idx, d["TF"], left=d["EF"], height=0.25, color="#94A3B8", alpha=0.6, hatch="//")
                ax_g.set_yticks(y_pos)
                ax_g.set_yticklabels(order_keys)
                ax_g.invert_yaxis()
                ax_g.set_xlabel("Timeline | الزمن")
                ax_g.set_title("Project Gantt Chart (Red = Critical Path, Gray Slashed = Total Float TF)")
                ax_g.grid(True, axis="x", linestyle="--", alpha=0.6)
                st.pyplot(fig_g)

                add_log("Project Management (CPM/Lags)", "PDM AON/AOA Solver", pdm_text, f"Duration={proj_dur}, CP={crit_path}")
            except Exception as e:
                st.error(f"❌ خطأ في تنسيق بيانات الشبكة: {e}")

    else:
        st.caption("📌 أدخل الأنشطة على المسار الحرج بصيغة: `Activity, to (Optimistic), tm (Most Likely), tp (Pessimistic)`")
        pert_in = st.text_area("✍️ PERT Activities (to, tm, tp):", "A, 3, 5, 13\nB, 4, 6, 8\nC, 5, 8, 17\nD, 2, 4, 6", height=110)
        target_deadline = st.number_input("🎯 Target Project Deadline (Td) | الزمن المستهدف لاحتمالية الإنجاز:", value=25.0)
        if st.button("⚙️ Calculate PERT Expected Time, Variance & Z-Probability", type="primary"):
            p_rows = []
            tot_te, tot_var = 0.0, 0.0
            for ln in pert_in.strip().split("\n"):
                if not ln.strip():
                    continue
                t_id, to, tm, tp = [x.strip() for x in ln.split(",")]
                to, tm, tp = float(to), float(tm), float(tp)
                te = (to + 4.0 * tm + tp) / 6.0
                sd = (tp - to) / 6.0
                var = sd ** 2
                tot_te += te
                tot_var += var
                p_rows.append({"Activity": t_id, "to": to, "tm": tm, "tp": tp, "Expected (te)": round(te, 3), "Std Dev (σ)": round(sd, 3), "Variance (σ²)": round(var, 4)})
            st.dataframe(pd.DataFrame(p_rows), use_container_width=True)
            proj_sd = math.sqrt(tot_var)
            z_val = (target_deadline - tot_te) / proj_sd if proj_sd > 0 else 0.0
            prob_pct = stats.norm.cdf(z_val) * 100.0
            st.latex(f"T_e = \\sum t_e = {tot_te:.3f}, \\quad \\sigma_T = \\sqrt{{\\sum \\sigma^2}} = {proj_sd:.4f}, \\quad Z = \\frac{{T_d - T_e}}{{\\sigma_T}} = {z_val:.4f}")
            st.success(f"📈 **احتمالية إنهاء المشروع خلال ({target_deadline}) وحدة زمنية | Probability P(T ≤ Td):** `{prob_pct:.2f}%`")

    render_camera_ai_section(
        "tab1_pm",
        "إدارة المشروعات والشبكات (Project Management: AON / AOA / Lags / PERT / Crashing)",
        "حل مسائل شبكات الأعمال AON و AOA، العلاقات المتقدمة FS, SS, FF, SF مع الـ Lags، ضغط المشروعات بأقل تكلفة (Project Crashing)، تسوية الموارد (Resource Leveling)، وتحليل PERT.",
        lambda: pdm_text if pm_sub.startswith("1.") else pert_in,
    )


# ==============================================================================
# TAB 2: الاقتصاد الهندسي وحسابات الفوائد (Engineering Economy & Interest)
# ==============================================================================
with TABS[1]:
    st.markdown("""
    <div class='sec-header'>
        <h3>💰 2. الاقتصاد الهندسي وحسابات الفوائد والتدفقات النقدية | Engineering Economy</h3>
        <p>معاملات الفائدة المركبة Discrete & Continuous Compounding (<b>F/P, P/F, P/A, A/P, F/A, A/F, P/G, A/G</b>)، التدرج الهندسي (Geometric Gradient)، الفائدة الاسمية والفعالة (<b>Effective Interest</b>)، تقييم المشروعات (<b>NPV, AW, FW, IRR, B/C, Payback</b>) مع رسم <b>Cash Flow Diagram</b>، وجداول الإهلاك (<b>Depreciation: SL, SYD, DDB</b>).</p>
    </div>
    """, unsafe_allow_html=True)

    eco_tab_mode = st.selectbox(
        "💰 Select Engineering Economy Module | اختر نوع مسألة الاقتصاد الهندسي:",
        [
            "1. Compound Interest Factors & Equivalence Calculator (P, F, A, G, g, i_eff) | معاملات الفوائد والتكافؤ المالي",
            "2. Cash Flow Analysis (NPV / PW, AW, FW, IRR, B/C, Payback) + CFD Plot | تحليل التدفقات النقدية وتقييم البدائل",
            "3. Depreciation Schedules (Straight-Line, SYD, Double Declining Balance) | طرق حساب الإهلاك والقيمة الدفترية",
        ],
    )

    if eco_tab_mode.startswith("1."):
        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            i_pct = st.number_input("Interest Rate i (%) | معدل الفائدة:", value=10.0, step=0.5)
        with ec2:
            n_yr = st.number_input("Periods n | عدد الفترات:", value=10, min_value=1, step=1)
        with ec3:
            g_pct = st.number_input("Geometric Gradient g (%) | التدرج الهندسي:", value=5.0, step=0.5)
        with ec4:
            m_comp = st.number_input("Compounding/Year m | مرات التركيب بالسنة:", value=12, min_value=1)

        ec5, ec6, ec7, ec8 = st.columns(4)
        with ec5:
            val_P = st.number_input("Present Worth (P) | القيمة الحالية:", value=10000.0)
        with ec6:
            val_F = st.number_input("Future Worth (F) | القيمة المستقبلية:", value=0.0)
        with ec7:
            val_A = st.number_input("Uniform Series (A) | الدفعة المنتظمة:", value=2000.0)
        with ec8:
            val_G = st.number_input("Arithmetic Gradient (G) | التدرج الحسابي:", value=500.0)

        if st.button("⚙️ Calculate All Interest Factors & Equivalent Values | احسب المعاملات والتكافؤ", type="primary"):
            i = i_pct / 100.0
            g = g_pct / 100.0
            n = int(n_yr)
            fp = (1.0 + i) ** n
            pf = 1.0 / fp
            fa = ((fp - 1.0) / i) if i > 0 else float(n)
            af = (i / (fp - 1.0)) if fp != 1.0 else (1.0 / n)
            pa = ((fp - 1.0) / (i * fp)) if i > 0 else float(n)
            ap = ((i * fp) / (fp - 1.0)) if fp != 1.0 else (1.0 / n)
            pg = ((1.0 / i) * (pa - (n / fp))) if i > 0 else 0.0
            ag = ((1.0 / i) - (n / (fp - 1.0))) if i > 0 else 0.0
            if abs(i - g) < 1e-9:
                pa_geom = n / (1.0 + i)
            else:
                pa_geom = (1.0 - ((1.0 + g) / (1.0 + i)) ** n) / (i - g)

            i_eff_m = ((1.0 + (i / m_comp)) ** m_comp - 1.0) * 100.0
            i_eff_cont = (math.exp(i) - 1.0) * 100.0

            st.markdown("#### 📋 جدول معاملات الفائدة المركبة القياسية | Standard Compound Interest Factors Table:")
            df_fac = pd.DataFrame([{
                "(F/P, i, n)": round(fp, 5),
                "(P/F, i, n)": round(pf, 5),
                "(P/A, i, n)": round(pa, 5),
                "(A/P, i, n)": round(ap, 5),
                "(F/A, i, n)": round(fa, 5),
                "(A/F, i, n)": round(af, 5),
                "(P/G, i, n)": round(pg, 5),
                "(A/G, i, n)": round(ag, 5),
                "(P/A1, g, i, n)": round(pa_geom, 5),
            }])
            st.dataframe(df_fac, use_container_width=True)

            p_total = val_P + val_F * pf + val_A * pa + val_G * pg
            f_total = p_total * fp
            a_total = p_total * ap
            st.latex(f"P_{{\\text{{total}}}} = P + F(P/F) + A(P/A) + G(P/G) = {p_total:,.2f}")
            st.latex(f"F_{{\\text{{total}}}} = {f_total:,.2f}, \\quad A_{{\\text{{equivalent}}}} = {a_total:,.2f}")
            st.info(
                f"🏦 **Effective Interest Rate ($m={m_comp}$):** $i_{{eff}} = (1 + r/m)^m - 1 = {i_eff_m:.4f}\\%$ | "
                f"**Continuous Compounding ($e^r - 1$):** `{i_eff_cont:.4f}%`"
            )

    elif eco_tab_mode.startswith("2."):
        cf_str = st.text_input(
            "✍️ Cash Flows (CF0, CF1, CF2, ... separated by comma) | التدفقات النقدية من السنة 0:",
            value="-50000, 15000, 18000, 20000, 22000, 25000",
        )
        marr_pct = st.number_input("🎯 MARR / Discount Rate i (%) | معدل العائد الأدنى الجاذب:", value=12.0)
        if st.button("⚙️ Evaluate Project (PW/NPV, FW, AW, IRR, B/C, Payback) & Plot CFD", type="primary"):
            cfs = [float(x.strip()) for x in cf_str.split(",") if x.strip()]
            r = marr_pct / 100.0
            n_cf = len(cfs) - 1
            npv_val = sum(cf / ((1.0 + r) ** t) for t, cf in enumerate(cfs))
            fw_val = npv_val * ((1.0 + r) ** n_cf)
            ap_fac = (r * ((1.0 + r) ** n_cf)) / (((1.0 + r) ** n_cf) - 1.0) if r > 0 and n_cf > 0 else 1.0
            aw_val = npv_val * ap_fac

            pv_benefits = sum(cf / ((1.0 + r) ** t) for t, cf in enumerate(cfs) if cf > 0)
            pv_costs = sum(-cf / ((1.0 + r) ** t) for t, cf in enumerate(cfs) if cf < 0)
            bc_ratio = (pv_benefits / pv_costs) if pv_costs > 0 else float("inf")

            # Secant / Newton IRR Finder
            irr_est = 0.15
            for _ in range(50):
                f_irr = sum(cf / ((1.0 + irr_est) ** t) for t, cf in enumerate(cfs))
                df_irr = sum(-t * cf / ((1.0 + irr_est) ** (t + 1)) for t, cf in enumerate(cfs) if t > 0)
                if abs(df_irr) < 1e-10:
                    break
                irr_est -= f_irr / df_irr

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PW / NPV (صافي القيمة الحالية)", f"{npv_val:,.2f}")
            c2.metric("AW (العائد السنوي المكافئ)", f"{aw_val:,.2f}")
            c3.metric("IRR (معدل العائد الداخلي)", f"{irr_est*100:.2f}%")
            c4.metric("B/C Ratio (نسبة المنافع للتكاليف)", f"{bc_ratio:.3f}")

            fig_cf, ax_cf = plt.subplots(figsize=(8, 3.2))
            times = list(range(len(cfs)))
            colors_cf = ["#DC2626" if c < 0 else "#16A34A" for c in cfs]
            ax_cf.bar(times, cfs, width=0.35, color=colors_cf)
            ax_cf.axhline(0, color="black", linewidth=1.2)
            ax_cf.set_title("Cash Flow Diagram (CFD) | مخطط التدفقات النقدية")
            ax_cf.set_xlabel("Year / Period (n)")
            ax_cf.set_ylabel("Net Cash Flow")
            ax_cf.grid(True, linestyle=":", alpha=0.6)
            st.pyplot(fig_cf)

    else:
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            cost_B = st.number_input("Initial Cost (B) | التكلفة الابتدائية للأصل:", value=100000.0)
        with dc2:
            salv_S = st.number_input("Salvage Value (S) | قيمة الخردة النهائية:", value=10000.0)
        with dc3:
            life_N = st.number_input("Useful Life (N) | العمر الإنتاجي بالسنوات:", value=5, min_value=1)

        if st.button("⚙️ Generate Depreciation Schedule (SL, SYD, DDB)", type="primary"):
            N_int = int(life_N)
            syd_denom = N_int * (N_int + 1) / 2.0
            bv_sl, bv_syd, bv_ddb = cost_B, cost_B, cost_B
            dep_rows = []
            for yr in range(1, N_int + 1):
                d_sl = (cost_B - salv_S) / N_int
                bv_sl -= d_sl
                d_syd = ((N_int - yr + 1) / syd_denom) * (cost_B - salv_S)
                bv_syd -= d_syd
                d_ddb = min((2.0 / N_int) * bv_ddb, max(0.0, bv_ddb - salv_S))
                bv_ddb -= d_ddb
                dep_rows.append({
                    "Year (t)": yr,
                    "SL Depr.": round(d_sl, 2),
                    "SL Book Value": round(bv_sl, 2),
                    "SYD Depr.": round(d_syd, 2),
                    "SYD Book Value": round(bv_syd, 2),
                    "DDB (200%) Depr.": round(d_ddb, 2),
                    "DDB Book Value": round(bv_ddb, 2),
                })
            df_dep = pd.DataFrame(dep_rows)
            st.dataframe(df_dep, use_container_width=True)

    render_camera_ai_section(
        "tab2_eco",
        "الاقتصاد الهندسي وحسابات الفوائد (Engineering Economy)",
        "حل مسائل الفوائد المركبة، مخططات التدفق النقدي (Cash Flow Diagrams)، المقارنة بين البدائل الهندسية بأعمار مختلفة (LCM, PW, AW, Incremental IRR, B/C)، تحليل الإحلال (Replacement Analysis)، نقطة التعادل، والتضخم والإهلاك.",
        lambda: eco_tab_mode,
    )


# ==============================================================================
# TAB 3: الطرق العددية الهندسية المتقدمة (Advanced Numerical Methods)
# ==============================================================================
with TABS[2]:
    st.markdown("""
    <div class='sec-header'>
        <h3>🧮 3. الطرق العددية والتحليل العددي الهندسي | Numerical Methods</h3>
        <p>جذور المعادلات غير الخطية (<b>Bisection, False Position, Newton-Raphson, Secant, Fixed-Point</b>)، الأنظمة الخطية (<b>Gauss-Seidel + SOR, Jacobi, LU, Eigenvalues</b>)، الاستكمال وتوفيق المنحنيات (<b>Lagrange, Newton Divided Differences, Least Squares</b>)، والتكامل والمعادلات التفاضلية عددياً (<b>Trapezoidal, Simpson 1/3 & 3/8, Euler, Heun, Runge-Kutta RK4</b>).</p>
    </div>
    """, unsafe_allow_html=True)

    num_cat = st.selectbox(
        "🧮 Select Numerical Category | اختر فئة الطرق العددية:",
        [
            "1. Non-Linear Equations Root Finding (Bisection, False Position, Newton-Raphson, Secant)",
            "2. Linear Systems Iterative & Direct (Gauss-Seidel with Auto-Pivoting & SOR, Jacobi, LU)",
            "3. Numerical ODEs Initial Value Problems (Euler, Heun's Modified Euler, Runge-Kutta 4th Order RK4)",
            "4. Numerical Integration & Interpolation (Trapezoidal, Simpson's 1/3 & 3/8, Lagrange & Newton Poly)",
        ],
    )

    if num_cat.startswith("1."):
        nc1, nc2, nc3, nc4 = st.columns(4)
        with nc1:
            root_algo = st.selectbox("Algorithm:", ["Newton-Raphson", "Bisection", "False Position (Regula Falsi)", "Secant"])
        with nc2:
            f_root_str = st.text_input("Function f(x):", value="x**3 - 2*x - 5")
        with nc3:
            r_x0 = st.number_input("a / x0:", value=2.0, format="%.4f")
        with nc4:
            r_x1 = st.number_input("b / x1:", value=3.0, format="%.4f")

        if st.button("⚙️ Run Root-Finding Iterations | تشغيل الجدول التكراري", type="primary"):
            x_s = sp.Symbol("x")
            f_s = sp.sympify(f_root_str)
            df_s = sp.diff(f_s, x_s)
            f_fn = sp.lambdify(x_s, f_s, "numpy")
            df_fn = sp.lambdify(x_s, df_s, "numpy")
            rows_r = []
            a, b = float(r_x0), float(r_x1)

            for it in range(1, 20):
                if root_algo == "Newton-Raphson":
                    fx, dfx = float(f_fn(a)), float(df_fn(a))
                    x_new = a - fx / dfx
                    err = abs(x_new - a)
                    rows_r.append({"Iter": it, "x_k": round(a, 6), "f(x_k)": round(fx, 6), "f'(x_k)": round(dfx, 6), "x_{k+1}": round(x_new, 6), "Error": round(err, 7)})
                    a = x_new
                elif root_algo in ["Bisection", "False Position (Regula Falsi)"]:
                    fa, fb = float(f_fn(a)), float(f_fn(b))
                    xr = (a + b) / 2.0 if root_algo == "Bisection" else (b - fb * (b - a) / (fb - fa))
                    fxr = float(f_fn(xr))
                    err = abs(b - a) / 2.0 if root_algo == "Bisection" else abs(fxr)
                    rows_r.append({"Iter": it, "a": round(a, 6), "b": round(b, 6), "x_r": round(xr, 6), "f(x_r)": round(fxr, 6), "Error": round(err, 7)})
                    if fa * fxr < 0:
                        b = xr
                    else:
                        a = xr
                else:  # Secant
                    fa, fb = float(f_fn(a)), float(f_fn(b))
                    x_new = b - fb * (b - a) / (fb - fa)
                    err = abs(x_new - b)
                    rows_r.append({"Iter": it, "x_{k-1}": round(a, 6), "x_k": round(b, 6), "x_{k+1}": round(x_new, 6), "f(x_{k+1})": round(float(f_fn(x_new)), 6), "Error": round(err, 7)})
                    a, b = b, x_new
                if err < 1e-6:
                    break
            st.dataframe(pd.DataFrame(rows_r), use_container_width=True)

    elif num_cat.startswith("2."):
        aug_txt = st.text_area("✍️ Augmented Matrix [A | B] (each row in a line):", "10 -1 2 0 6\n-1 11 -1 3 25\n2 -1 10 -1 -11\n0 3 -1 8 15", height=105)
        gs_col1, gs_col2, gs_col3 = st.columns(3)
        with gs_col1:
            lin_algo = st.selectbox("Method:", ["Gauss-Seidel", "Jacobi", "LU Decomposition & Exact"])
        with gs_col2:
            sor_w = st.slider("SOR Relaxation Factor (ω):", 0.5, 1.9, 1.0, 0.05)
        with gs_col3:
            piv_en = st.checkbox("Auto-Pivoting (Diagonal Dominance)", value=True)

        if st.button("⚙️ Solve Linear System [A|B]", type="primary"):
            M = np.array([list(map(float, r.replace(",", " ").split())) for r in aug_txt.strip().split("\n") if r.strip()])
            A, B = M[:, :-1].copy(), M[:, -1].copy()
            n = len(B)
            if piv_en:
                for i in range(n):
                    mr = i + int(np.argmax(np.abs(A[i:, i])))
                    A[[i, mr]], B[[i, mr]] = A[[mr, i]], B[[mr, i]]
            if "LU" in lin_algo:
                L, U, _ = sp.Matrix(A).LUdecomposition()
                st.latex(f"L = {sp.latex(L.evalf(4))}, \\quad U = {sp.latex(U.evalf(4))}")
                st.success(f"Exact Solution: `{np.round(np.linalg.solve(A, B), 6)}`")
            else:
                x = np.zeros(n)
                it_rows = []
                for it in range(1, 26):
                    x_old = x.copy()
                    for i in range(n):
                        if lin_algo == "Gauss-Seidel":
                            s1 = np.dot(A[i, :i], x[:i])
                            s2 = np.dot(A[i, i + 1 :], x_old[i + 1 :])
                            x[i] = (1 - sor_w) * x_old[i] + sor_w * ((B[i] - s1 - s2) / A[i, i])
                        else:
                            s = np.dot(A[i, :], x_old) - A[i, i] * x_old[i]
                            x[i] = (B[i] - s) / A[i, i]
                    err = float(np.max(np.abs(x - x_old)))
                    row_d = {"Iter": it, **{f"x{j+1}": round(x[j], 6) for j in range(n)}, "Max Error": round(err, 7)}
                    it_rows.append(row_d)
                    if err < 1e-6:
                        break
                st.dataframe(pd.DataFrame(it_rows), use_container_width=True)
                st.info(f"🎯 Exact Solution for Comparison: `{np.round(np.linalg.solve(A, B), 6)}`")

    elif num_cat.startswith("3."):
        oc1, oc2, oc3, oc4, oc5 = st.columns(5)
        with oc1:
            ode_f_str = st.text_input("dy/dx = f(x, y):", value="x + y")
        with oc2:
            x0_v = st.number_input("x0:", value=0.0)
        with oc3:
            y0_v = st.number_input("y(x0) = y0:", value=1.0)
        with oc4:
            h_v = st.number_input("Step size (h):", value=0.1, format="%.3f")
        with oc5:
            steps_n = st.number_input("Number of steps (N):", value=10, min_value=1, max_value=100)

        if st.button("⚙️ Compare Euler, Heun (RK2), and Runge-Kutta 4th Order (RK4)", type="primary"):
            x_s, y_s = sp.symbols("x y")
            f_xy = sp.lambdify((x_s, y_s), sp.sympify(ode_f_str), "numpy")
            xv = float(x0_v)
            ye, yh, yrk = float(y0_v), float(y0_v), float(y0_v)
            ode_rows = [{"Step": 0, "x": xv, "Euler y": ye, "Heun (RK2) y": yh, "RK4 y": yrk}]
            for s_idx in range(1, int(steps_n) + 1):
                # Euler
                ye_next = ye + h_v * float(f_xy(xv, ye))
                # Heun
                k1_h = float(f_xy(xv, yh))
                k2_h = float(f_xy(xv + h_v, yh + h_v * k1_h))
                yh_next = yh + (h_v / 2.0) * (k1_h + k2_h)
                # RK4
                k1 = h_v * float(f_xy(xv, yrk))
                k2 = h_v * float(f_xy(xv + 0.5 * h_v, yrk + 0.5 * k1))
                k3 = h_v * float(f_xy(xv + 0.5 * h_v, yrk + 0.5 * k2))
                k4 = h_v * float(f_xy(xv + h_v, yrk + k3))
                yrk_next = yrk + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

                xv = round(xv + h_v, 6)
                ye, yh, yrk = ye_next, yh_next, yrk_next
                ode_rows.append({"Step": s_idx, "x": xv, "Euler y": round(ye, 6), "Heun (RK2) y": round(yh, 6), "RK4 y": round(yrk, 6), "k1": round(k1, 5), "k2": round(k2, 5), "k3": round(k3, 5), "k4": round(k4, 5)})
            df_ode_num = pd.DataFrame(ode_rows)
            st.dataframe(df_ode_num, use_container_width=True)

    else:
        ic1, ic2, ic3, ic4 = st.columns(4)
        with ic1:
            int_f_str = st.text_input("Function f(x) to Integrate:", value="1 / (1 + x**2)")
        with ic2:
            ia_v = st.number_input("Lower bound a:", value=0.0)
        with ic3:
            ib_v = st.number_input("Upper bound b:", value=1.0)
        with ic4:
            in_sub = st.number_input("Subintervals n (multiple of 6 for both rules):", value=6, min_value=2)

        pts_interp = st.text_input("Interpolation Points (x0,y0 ; x1,y1 ; ...):", value="0,1 ; 1,2 ; 2,5 ; 3,10")
        if st.button("⚙️ Compute Trapezoidal, Simpson 1/3, Simpson 3/8 & Interpolating Polynomial", type="primary"):
            x_s = sp.Symbol("x")
            f_s = sp.sympify(int_f_str)
            f_fn = sp.lambdify(x_s, f_s, "numpy")
            n_i = int(in_sub)
            xn = np.linspace(ia_v, ib_v, n_i + 1)
            yn = f_fn(xn)
            h_i = (ib_v - ia_v) / n_i
            trap_i = (h_i / 2.0) * (yn[0] + 2.0 * np.sum(yn[1:-1]) + yn[-1])
            simp13 = (h_i / 3.0) * (yn[0] + 4.0 * np.sum(yn[1:-1:2]) + 2.0 * np.sum(yn[2:-2:2]) + yn[-1]) if n_i % 2 == 0 else None
            simp38 = (3.0 * h_i / 8.0) * (yn[0] + 3.0 * np.sum(yn[1:-1][np.arange(1, n_i) % 3 != 0]) + 2.0 * np.sum(yn[3:-1:3]) + yn[-1]) if n_i % 3 == 0 else None
            exact_i = float(sp.integrate(f_s, (x_s, ia_v, ib_v)).evalf())

            st.write(f"**Trapezoidal Rule:** `{trap_i:.7f}` | **Simpson's 1/3:** `{simp13:.7f if simp13 else 'N/A'}` | **Simpson's 3/8:** `{simp38:.7f if simp38 else 'N/A'}` | **Exact:** `{exact_i:.7f}`")

            # Lagrange Polynomial
            pts = [list(map(float, p.split(","))) for p in pts_interp.split(";") if "," in p]
            px, py = [p[0] for p in pts], [p[1] for p in pts]
            poly_l = sum(py[i] * sp.prod([(x_s - px[j]) / (px[i] - px[j]) for j in range(len(px)) if i != j]) for i in range(len(px)))
            st.latex(f"P_{{\\text{{Lagrange}}}}(x) = {sp.latex(sp.expand(poly_l))}")

    render_camera_ai_section(
        "tab3_num",
        "الطرق العددية والتحليل العددي الهندسي (Numerical Methods)",
        "حل مسائل التحليل العددي بالجداول الكاملة: Bisection, False Position, Newton-Raphson, Secant, Fixed-Point, Gauss-Seidel, Jacobi, LU, Power Method, Lagrange & Newton Divided Differences, Splines, Least Squares, Trapezoidal, Simpson 1/3 & 3/8, Romberg, Gaussian Quadrature, Euler, Heun, RK4, Finite Difference PDEs.",
        lambda: num_cat,
    )


# ==============================================================================
# TAB 4: المعادلات التفاضلية والتحويلات الهندسية (ODEs, Laplace, Fourier & Vector Calc)
# ==============================================================================
with TABS[3]:
    st.markdown("""
    <div class='sec-header'>
        <h3>📈 4. المعادلات التفاضلية، تحويلات لابلاس وفورييه، وحساب المتجهات | ODEs, Laplace & Vector Calculus</h3>
        <p>حل المعادلات التفاضلية العادية من الرتب العليا (<b>Higher-Order ODEs</b>)، تحويل لابلاس المباشر والعكسي بالكسور الجزئية (<b>Laplace & Inverse Laplace</b>)، ومصفوفة جاكوبيان وهيسيان (<b>Jacobian & Hessian Matrices</b>).</p>
    </div>
    """, unsafe_allow_html=True)

    adv_math_op = st.selectbox(
        "📐 Select Operation | اختر العملية الرياضية المتقدمة:",
        [
            "1. Laplace Transform F(s) = L{f(t)} | تحويل لابلاس المباشر",
            "2. Inverse Laplace Transform + Partial Fractions L^-1{F(s)} | تحويل لابلاس العكسي بالكسور الجزئية",
            "3. Higher-Order Differential Equation (ODE Solver) | حل المعادلات التفاضلية",
            "4. Multivariable Gradient, Divergence, Curl & Jacobian Matrix | المتجهات ومصفوفة جاكوبيان",
        ],
    )
    math_in_str = st.text_input(
        "✍️ Expression / Equation (e.g. (s + 3) / (s**2 + 5*s + 6) or t**2 * exp(-3*t) or y(x).diff(x,2) + 5*y(x).diff(x) + 6*y(x) - exp(-x)):",
        value="(2*s + 5) / (s**2 + 5*s + 6)",
    )
    if st.button("⚙️ Compute Symbolic Solution | احسب الحل الرياضي", type="primary"):
        try:
            x, y, z, t, s = sp.symbols("x y z t s")
            t_pos = sp.Symbol("t", positive=True)
            if adv_math_op.startswith("1."):
                expr_t = sp.sympify(math_in_str).subs(t, t_pos)
                F_s = sp.laplace_transform(expr_t, t_pos, s, noconds=True)
                st.latex(f"\\mathcal{{L}}\\left\\{{{sp.latex(expr_t)}\\right\\}} = {sp.latex(F_s)}")
            elif adv_math_op.startswith("2."):
                expr_s = sp.sympify(math_in_str)
                pf_s = sp.apart(expr_s, s)
                f_t = sp.inverse_laplace_transform(expr_s, s, t_pos)
                st.latex(f"F(s) = {sp.latex(expr_s)} = {sp.latex(pf_s)}")
                st.latex(f"f(t) = \\mathcal{{L}}^{{-1}}\\{{F(s)\\}} = {sp.latex(f_t)}")
            elif adv_math_op.startswith("3."):
                y_fn = sp.Function("y")
                ode_e = eval(math_in_str, {"y": y_fn, "x": x, "sp": sp, "sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "Eq": sp.Eq})
                sol_o = sp.dsolve(ode_e, y_fn(x))
                st.latex(sp.latex(sol_o))
            else:
                funcs = [sp.sympify(f_item.strip()) for f_item in math_in_str.split(",")]
                M_f = sp.Matrix(funcs)
                J_mat = M_f.jacobian([x, y, z][: len(funcs)])
                st.latex(f"\\mathbf{{J}} = {sp.latex(J_mat)}, \\quad \\det(\\mathbf{{J}}) = {sp.latex(sp.simplify(J_mat.det()))}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    render_camera_ai_section(
        "tab4_math",
        "المعادلات التفاضلية، تحويلات لابلاس وفورييه، والتكامل المتعدد (Advanced Engineering Mathematics)",
        "حل المعادلات التفاضلية العادية والجزئية (ODEs & PDEs)، متسلسلات وتحويلات فورييه، تحويل لابلاس، Z-Transform، التكامل الثنائي والثلاثي، ونظريات Green, Stokes, Gauss.",
        lambda: f"{adv_math_op}: {math_in_str}",
    )


# ==============================================================================
# TAB 5: الميكانيكا التطبيقية ومقاومة المواد (Mohr's Circle & Beam SFD / BMD)
# ==============================================================================
with TABS[4]:
    st.markdown("""
    <div class='sec-header'>
        <h3>⚙️ 5. الميكانيكا التطبيقية ومقاومة المواد | Mechanics of Materials (Mohr's Circle & Beam SFD/BMD)</h3>
        <p>تحليل الإجهادات المستوية ورسم <b>دائرة مور (Mohr's Circle)</b>، وحساب ردود الأفعال ورسم <b>مخططات قوى القص وعزوم الانحناء للكمرات (Shear Force & Bending Moment Diagrams - SFD & BMD)</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    mech_sub = st.radio(
        "⚙️ Select Mechanics Tool | اختر الأداة الميكانيكية:",
        [
            "1. Plane Stress Transformation & Mohr's Circle | تحويل الإجهادات ودائرة مور",
            "2. Simply Supported Beam SFD & BMD Plotter | رسم منحنيات القص والعزوم للكمرات",
        ],
        horizontal=True,
    )

    if mech_sub.startswith("1."):
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            sig_x = st.number_input("σx (MPa):", value=80.0)
        with mc2:
            sig_y = st.number_input("σy (MPa):", value=-40.0)
        with mc3:
            tau_xy = st.number_input("τxy (MPa):", value=35.0)
        with mc4:
            th_deg = st.number_input("Plane Angle θ (deg):", value=30.0)

        if st.button("⚙️ Compute Principal Stresses & Plot Mohr's Circle", type="primary"):
            s_avg = (sig_x + sig_y) / 2.0
            R = math.hypot((sig_x - sig_y) / 2.0, tau_xy)
            s1, s2 = s_avg + R, s_avg - R
            th_p = 0.5 * math.degrees(math.atan2(2.0 * tau_xy, sig_x - sig_y))
            rad_2th = math.radians(2.0 * th_deg)
            sx_p = s_avg + ((sig_x - sig_y) / 2.0) * math.cos(rad_2th) + tau_xy * math.sin(rad_2th)
            txy_p = -((sig_x - sig_y) / 2.0) * math.sin(rad_2th) + tau_xy * math.cos(rad_2th)

            st.latex(f"\\sigma_{{avg}} = {s_avg:.2f}\\,\\text{{MPa}}, \\quad R = \\tau_{{max}} = {R:.2f}\\,\\text{{MPa}}, \\quad \\sigma_1 = {s1:.2f}\\,\\text{{MPa}}, \\quad \\sigma_2 = {s2:.2f}\\,\\text{{MPa}}, \\quad \\theta_p = {th_p:.2f}^\\circ")
            st.info(f"📐 At inclined plane $\\theta = {th_deg}^\\circ$: $\\sigma_{{x'}} = {sx_p:.2f}\\,\\text{{MPa}}, \\quad \\tau_{{x'y'}} = {txy_p:.2f}\\,\\text{{MPa}}$")

            fig_m, ax_m = plt.subplots(figsize=(5.5, 5))
            ang_c = np.linspace(0, 2 * np.pi, 300)
            ax_m.plot(s_avg + R * np.cos(ang_c), R * np.sin(ang_c), color="#1D4ED8", linewidth=2.2)
            ax_m.plot([sig_x, sig_y], [-tau_xy, tau_xy], "ro--", linewidth=1.8, label="X-Y Diameter")
            ax_m.scatter([s1, s2, s_avg], [0, 0, 0], color="#15803D", s=60, zorder=5)
            ax_m.axhline(0, color="black", linewidth=1)
            ax_m.axvline(0, color="black", linewidth=1)
            ax_m.set_aspect("equal")
            ax_m.set_title("Mohr's Circle for 2D Plane Stress")
            ax_m.set_xlabel("Normal Stress σ (MPa)")
            ax_m.set_ylabel("Shear Stress τ (MPa)")
            ax_m.grid(True, linestyle=":")
            st.pyplot(fig_m)

    else:
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            L_beam = st.number_input("Beam Length L (m):", value=6.0, min_value=1.0)
        with bc2:
            P_load = st.number_input("Point Load P (kN):", value=20.0)
        with bc3:
            a_pos = st.number_input("Point Load Position a (m):", value=2.0, min_value=0.0)
        with bc4:
            w_udl = st.number_input("Uniform Load w (kN/m over full span):", value=5.0)

        if st.button("⚙️ Calculate Reactions & Plot SFD / BMD", type="primary"):
            Rb = (P_load * a_pos + w_udl * L_beam * (L_beam / 2.0)) / L_beam
            Ra = P_load + w_udl * L_beam - Rb
            st.success(f"⚖️ **Support Reactions:** $R_A = {Ra:.3f}\\,\\text{{kN}}, \\quad R_B = {Rb:.3f}\\,\\text{{kN}}$")

            x_b = np.linspace(0, L_beam, 500)
            V_x = Ra - w_udl * x_b - np.where(x_b >= a_pos, P_load, 0.0)
            M_x = Ra * x_b - 0.5 * w_udl * (x_b ** 2) - np.where(x_b >= a_pos, P_load * (x_b - a_pos), 0.0)

            fig_b, (ax_v, ax_m) = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True)
            ax_v.plot(x_b, V_x, color="#DC2626", linewidth=2)
            ax_v.fill_between(x_b, V_x, color="#FECACA", alpha=0.5)
            ax_v.axhline(0, color="black", linewidth=1)
            ax_v.set_ylabel("Shear V (kN)")
            ax_v.set_title(f"Shear Force (SFD) & Bending Moment (BMD) | Max M = {np.max(np.abs(M_x)):.2f} kN·m")
            ax_v.grid(True, linestyle=":")

            ax_m.plot(x_b, M_x, color="#1D4ED8", linewidth=2)
            ax_m.fill_between(x_b, M_x, color="#BFDBFE", alpha=0.5)
            ax_m.axhline(0, color="black", linewidth=1)
            ax_m.set_ylabel("Moment M (kN·m)")
            ax_m.set_xlabel("Beam Position x (m)")
            ax_m.grid(True, linestyle=":")
            st.pyplot(fig_b)

    render_camera_ai_section(
        "tab5_mech",
        "الميكانيكا التطبيقية ومقاومة المواد وتصميم الماكينات (Mechanics of Materials & Machine Design)",
        "تحليل صور الكمرات (Beams SFD/BMD)، الجمالونات (Trusses)، دائرة مور، الالتواء (Torsion)، انبعاج الأعمدة (Buckling)، والاهتزازات الميكانيكية (Mechanical Vibrations).",
        lambda: mech_sub,
    )


# ==============================================================================
# TAB 6: الدوائر الكهربية، القوى ثلاثية الأوجه، والمحولات (Circuits & Power)
# ==============================================================================
with TABS[5]:
    st.markdown("""
    <div class='sec-header'>
        <h3>⚡ 6. الدوائر الكهربية، القوى ثلاثية الأوجه، والمحولات | Electric Circuits, 3-Phase & Transformers</h3>
        <p>تحليل اختبارات المحولات الكهربية (<b>Open-Circuit & Short-Circuit Tests</b>)، حسابات القدرة ثلاثية الأوجه وتحسين معامل القدرة (<b>3-Phase Power Factor Correction</b>)، وحل معادلات العقد والمسارات (<b>Nodal & Mesh</b>).</p>
    </div>
    """, unsafe_allow_html=True)

    tc1, tc2, tc3, tc4, tc5, tc6 = st.columns(6)
    with tc1:
        v_oc = st.number_input("Voc (V):", value=220.0)
    with tc2:
        i_oc = st.number_input("Ioc (A):", value=1.2)
    with tc3:
        p_oc = st.number_input("Poc (W):", value=100.0)
    with tc4:
        v_sc = st.number_input("Vsc (V):", value=35.0)
    with tc5:
        i_sc = st.number_input("Isc (A):", value=10.0)
    with tc6:
        p_sc = st.number_input("Psc (W):", value=220.0)

    if st.button("⚙️ Compute Transformer Equivalent Circuit (Rc, Xm, Req, Xeq)", type="primary"):
        pf_0 = p_oc / (v_oc * i_oc)
        iw = i_oc * pf_0
        im = i_oc * math.sqrt(max(0.0, 1.0 - pf_0 ** 2))
        rc = v_oc / iw if iw > 0 else 0.0
        xm = v_oc / im if im > 0 else 0.0
        req = p_sc / (i_sc ** 2)
        zeq = v_sc / i_sc
        xeq = math.sqrt(max(0.0, zeq ** 2 - req ** 2))
        st.latex(f"R_c = {rc:.2f}\\,\\Omega, \\quad X_m = {xm:.2f}\\,\\Omega, \\quad R_{{eq}} = {req:.3f}\\,\\Omega, \\quad X_{{eq}} = {xeq:.3f}\\,\\Omega, \\quad Z_{{eq}} = {zeq:.3f}\\,\\Omega")

    render_camera_ai_section(
        "tab6_elec",
        "الدوائر الكهربية، الآلات، والقوى الكهربية (Electrical Engineering)",
        "حل مسائل دوائر التيار المستمر والمتردد (Nodal, Mesh, Thevenin, Norton, Superposition, Transient RLC)، الأنظمة ثلاثية الأوجه، المحولات، والآلات الكهربية.",
        lambda: f"Transformer OC/SC: Voc={v_oc}, Ioc={i_oc}, Poc={p_oc}, Vsc={v_sc}, Isc={i_sc}, Psc={p_sc}",
    )


# ==============================================================================
# TAB 7: هندسة التحكم الآلي وحالة الفراغ (Control Systems, Routh, Bode & State-Space)
# ==============================================================================
with TABS[6]:
    st.markdown("""
    <div class='sec-header'>
        <h3>🎛️ 7. هندسة التحكم الآلي وحالة الفراغ | Control Systems, Routh-Hurwitz, Bode & State-Space</h3>
        <p>تحليل دالة النقل <b>G(s)</b>، حساب الأقطاب ومعامل التخميد ($\omega_n, \zeta, M_p, T_s$)، ورسم <b>Step Response & Bode Plot</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        num_g = st.text_input("Numerator Coefficients of G(s):", value="25")
    with cc2:
        den_g = st.text_input("Denominator Coefficients of G(s) (e.g. s^2 + 3s + 25 -> 1, 3, 25):", value="1, 3, 25")

    if st.button("⚙️ Analyze Transfer Function, Stability, Step Response & Bode Plot", type="primary"):
        num_arr = [float(x.strip()) for x in num_g.split(",") if x.strip()]
        den_arr = [float(x.strip()) for x in den_g.split(",") if x.strip()]
        tf_sys = signal.TransferFunction(num_arr, den_arr)
        poles = np.roots(den_arr)
        st.write(f"**System Poles:** `{np.round(poles, 4)}` | **Stable:** `{'YES ✅' if np.all(np.real(poles) < 0) else 'NO ⚠️'}`")

        t_s, y_s = signal.step(tf_sys)
        w_b, mag_b, phase_b = signal.bode(tf_sys)

        fc1, fc2 = st.columns(2)
        with fc1:
            fig_st, ax_st = plt.subplots(figsize=(6, 3.2))
            ax_st.plot(t_s, y_s, color="#1D4ED8", linewidth=2)
            ax_st.set_title("Unit Step Response y(t)")
            ax_st.grid(True, linestyle=":")
            st.pyplot(fig_st)
        with fc2:
            fig_bd, ax_bd = plt.subplots(figsize=(6, 3.2))
            ax_bd.semilogx(w_b, mag_b, color="#DC2626", linewidth=2)
            ax_bd.set_title("Bode Magnitude Plot (dB)")
            ax_bd.grid(True, which="both", linestyle=":")
            st.pyplot(fig_bd)

    render_camera_ai_section(
        "tab7_ctrl",
        "هندسة التحكم الآلي والـ PLC والميكاترونكس (Control Systems & Automation)",
        "حل مسائل Block Diagram Reduction, Signal Flow Graph (Mason's Formula), Routh-Hurwitz Stability Criterion, Root Locus, Bode & Nyquist, PID Tuning, State-Space Controllability/Observability, و PLC Ladder Logic.",
        lambda: f"G(s) Num={num_g}, Den={den_g}",
    )


# ==============================================================================
# TAB 8: البرمجة الهندسية وسجل التقارير (Engineering Code Sandbox & Reports)
# ==============================================================================
with TABS[7]:
    st.markdown("""
    <div class='sec-header'>
        <h3>💻 8. البرمجة الهندسية وتصدير التقارير | Engineering Code Sandbox & Reports</h3>
        <p>تنفيذ أكواد بايثون العلمية مباشرة، تصحيح الأكواد الهندسية (<b>Python, C++, MATLAB, Arduino/AVR</b>)، وتصدير سجل جميع العمليات.</p>
    </div>
    """, unsafe_allow_html=True)

    py_code = st.text_area(
        "✍️ Python / NumPy / SymPy Engineering Sandbox:",
        value="import numpy as np\nA = np.array([[4, 1], [2, 3]])\nprint('Eigenvalues:', np.linalg.eigvals(A))",
        height=110,
    )
    if st.button("▶️ Execute Python Code | تشغيل الكود", type="primary"):
        buf_out = io.StringIO()
        old_out = sys.stdout
        sys.stdout = buf_out
        try:
            exec(py_code, {"np": np, "sp": sp, "pd": pd, "math": math})
            st.code(buf_out.getvalue() or "Executed successfully.", language="text")
        except Exception as e:
            st.error(f"Execution Error: {e}")
        finally:
            sys.stdout = old_out

    if st.session_state.history:
        st.markdown("#### 📑 سجل العمليات الهندسية | Session History:")
        df_h = pd.DataFrame(st.session_state.history)
        st.dataframe(df_h, use_container_width=True)
        st.download_button(
            "📥 Download Full Engineering Report (CSV)",
            df_h.to_csv(index=False).encode("utf-8-sig"),
            "Engineering_Solutions_Report.csv",
            "text/csv",
        )

    render_camera_ai_section(
        "tab8_code",
        "البرمجة الهندسية والامتحانات الشاملة (MATLAB / C++ / Python / Embedded C)",
        "كتابة وتصحيح الأكواد البرمجية الهندسية وحل أوراق الامتحانات الهندسية الشاملة.",
        lambda: py_code,
    )
