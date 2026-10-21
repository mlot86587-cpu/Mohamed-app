import os
import io
import re
import sys
import json
import math
import cmath
import base64
import datetime
import urllib.request
import urllib.error
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

st.set_page_config(
    page_title="Universal Engineering Solver | المنصة الهندسية العامة",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 97%; }
    .eng-banner {
        background: linear-gradient(135deg, rgb(15, 23, 42) 0%, rgb(30, 58, 138) 55%, rgb(15, 118, 110) 100%);
        color: white; padding: 18px 24px; border-radius: 12px; margin-bottom: 14px;
        direction: rtl; text-align: right; box-shadow: 0 4px 14px rgba(0,0,0,0.2);
    }
    .eng-banner h1 { margin: 0; font-size: 23px; color: white; }
    .eng-banner p { margin: 5px 0 0 0; font-size: 13.5px; opacity: 0.92; }
    .sec-header {
        background: rgb(248, 250, 252); border: 1px solid rgb(203, 213, 225); border-right: 6px solid rgb(29, 78, 216);
        padding: 10px 16px; border-radius: 8px; margin-bottom: 12px; direction: rtl; text-align: right;
    }
    .ai-cam-box {
        background: rgb(240, 249, 255); border: 2px dashed rgb(2, 132, 199); padding: 14px;
        border-radius: 10px; margin-top: 14px; direction: rtl; text-align: right;
    }
    .priority-badge-1 {
        background: rgb(22, 163, 74); color: white; padding: 4px 12px; border-radius: 6px; font-weight: bold;
    }
    .priority-badge-2 {
        background: rgb(37, 99, 235); color: white; padding: 4px 12px; border-radius: 6px; font-weight: bold;
    }
    .priority-badge-rej {
        background: rgb(220, 38, 38); color: white; padding: 4px 12px; border-radius: 6px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def get_default_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


if "history" not in st.session_state:
    st.session_state.history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = get_default_api_key()
if "custom_tables" not in st.session_state:
    st.session_state.custom_tables = {}


def add_log(module_name, method_used, inputs_str, outputs_str):
    st.session_state.history.append({
        "ID": len(st.session_state.history) + 1,
        "Time | الوقت": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Module | القسم": module_name,
        "Method | الطريقة": method_used,
        "Inputs | المدخلات": str(inputs_str)[:300],
        "Result | النتيجة": str(outputs_str)[:500],
    })


def eval_math_expr(expr_str, **variables):
    safe_env = {
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10,
        "sqrt": np.sqrt, "abs": np.abs, "pi": np.pi, "e": np.e,
        "np": np, "math": math, "cmath": cmath,
    }
    safe_env.update(variables)
    return eval(str(expr_str).replace("^", "**"), {"__builtins__": {}}, safe_env)


def numerical_derivative(expr_str, x_val, h=1e-6):
    return (eval_math_expr(expr_str, x=x_val + h) - eval_math_expr(expr_str, x=x_val - h)) / (2.0 * h)


def norm_cdf(z_val):
    return 0.5 * (1.0 + math.erf(z_val / math.sqrt(2.0)))


with st.sidebar:
    st.markdown("<h4>⚙️ الإعدادات الهندسية | System Config</h4>", unsafe_allow_html=True)
    key_side = st.text_input(
        "🔑 Gemini API Key:",
        value=st.session_state.api_key,
        type="password",
        key="sidebar_api_key_in",
    )
    if key_side.strip() != st.session_state.api_key:
        st.session_state.api_key = key_side.strip()

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


st.markdown("""
<div class='eng-banner'>
    <h1>⚙️ المنصة الهندسية العامة لحل جميع المسائل | Universal Engineering Solver</h1>
    <p>حل أي مسألة هندسية جديدة بالجداول الحرة أو بالكاميرا والذكاء الاصطناعي • AON / AOA / Lags / Crashing • Economy & Multi-Project NPV • Numerical Methods • ODEs & Laplace • Mechanics & Beams • Circuits & Control</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.api_key:
    with st.expander("🔑 تفعيل الكاميرا والذكاء الاصطناعي على الموبايل (أدخل مفتاح Gemini API Key هنا مباشرة)", expanded=True):
        col_k1, col_k2 = st.columns([3, 1.2])
        with col_k1:
            main_key_in = st.text_input(
                "الصق مفتاح Gemini API Key هنا لتفعيل حل الصور والمسائل بالذكاء الاصطناعي:",
                type="password",
                placeholder="AIzaSy...",
                key="main_top_api_input",
            )
        with col_k2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✅ تفعيل المفتاح الآن", use_container_width=True):
                st.session_state.api_key = main_key_in.strip()
                st.rerun()
        st.caption("💡 ملاحظة: جميع الجداول والحسابات البرمجية والرسومات الهندسية في الأسفل تعمل فوراً لأي مسألة جديدة حتى بدون مفتاح API.")


def apply_vision_filter(pil_img, mode_str, rot_angle=0):
    if pil_img is None:
        return None
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass
    img = pil_img.convert("RGB")
    if rot_angle != 0:
        img = img.rotate(-rot_angle, expand=True)
    gray = ImageOps.grayscale(img)
    arr = np.array(gray, dtype=np.float32)

    if "Otsu" in mode_str:
        hist, _ = np.histogram(arr.flatten(), 256, [0, 256])
        total = arr.size
        sum_total = np.dot(np.arange(256), hist)
        sum_b, w_b, max_var, thresh = 0.0, 0, 0.0, 128
        for idx in range(256):
            w_b += hist[idx]
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break
            sum_b += idx * hist[idx]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            var_between = float(w_b) * float(w_f) * ((m_b - m_f) ** 2)
            if var_between > max_var:
                max_var = var_between
                thresh = idx
        return Image.fromarray(np.where(arr > thresh, 255, 0).astype(np.uint8))

    elif "Adaptive" in mode_str:
        blurred = np.array(gray.filter(ImageFilter.BoxBlur(7)), dtype=np.float32)
        return Image.fromarray(np.where(arr > (blurred - 6.0), 255, 0).astype(np.uint8))

    elif "Contrast" in mode_str:
        c_img = ImageEnhance.Contrast(img).enhance(1.9)
        return ImageEnhance.Sharpness(c_img).enhance(2.3)

    elif "Edges" in mode_str:
        return ImageOps.invert(gray.filter(ImageFilter.FIND_EDGES))

    return img


def call_gemini_rest_api(domain_title, domain_guide, text_data, pil_img=None, mode="solve", target_columns=None):
    if not st.session_state.api_key:
        return "❌ **تنبيه:** يرجى وضع مفتاح `Gemini API Key` في الشريط العلوي لتفعيل الذكاء الاصطناعي، أو استخدم زر **(⚙️ الحل البرمجي للجدول)** أعلى الكاميرا لحل المسألة فوراً بدون مفتاح."
    try:
        if mode == "json_table" and target_columns:
            cols_str = json.dumps(target_columns, ensure_ascii=False)
            prompt_text = (
                f"You are a precise Engineering Table Extractor for ({domain_title}). "
                f"Read the image or text and return ONLY a valid JSON array of objects where each object has exact keys: {cols_str}. "
                "If a predecessor or successor is empty or none, use '-'. Output pure JSON array only."
            )
        elif mode == "ocr":
            prompt_text = (
                f"You are an Engineering OCR Scanner for ({domain_title}). "
                "Extract all tables, activities, predecessors, successors, durations, lags, cash flows, and equations accurately from the image."
            )
        else:
            prompt_text = (
                f"أنت بروفيسور في كلية الهندسة متخصص في ({domain_title}).\n"
                f"أسلوب الإجابة المطلوب: {ai_detail_mode}.\n"
                f"توجيهات التخصص: {domain_guide}\n"
                "قم بحل المسألة الهندسية المرفقة (سواء من الصورة أو من الجدول) باللغتين العربية والإنجليزية للمصطلحات، متضمناً:\n"
                "1. **جدول المعطيات والمطلوب (Given Table & Required)**.\n"
                "2. **القوانين الهندسية (Governing Equations)** بصيغة LaTeX واضحة.\n"
                "3. **خطوات الحل والجداول التفصيلية الكاملة لكل عنصر/نشاط (Step-by-Step Calculation & Tables)**.\n"
                "4. **الناتج النهائي والترتيب الهندسي (Final Answer & Critical Path / Priority)**.\n"
                f"\nبيانات المسألة النصية:\n{text_data}"
            )

        parts_list = [{"text": prompt_text}]
        if pil_img is not None:
            buf = io.BytesIO()
            pil_img.convert("RGB").save(buf, format="JPEG", quality=90)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            parts_list.append({"inline_data": {"mime_type": "image/jpeg", "data": b64_str}})

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{ai_model_name}:generateContent?key={st.session_state.api_key}"
        payload = {
            "contents": [{"parts": parts_list}],
            "generationConfig": {"temperature": ai_temp_val, "maxOutputTokens": 4096},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            candidates = resp_data.get("candidates", [])
            if candidates:
                out_parts = candidates[0].get("content", {}).get("parts", [])
                return "\n".join(p.get("text", "") for p in out_parts)
            return "⚠️ لم يتم استلام نص من النموذج."
    except urllib.error.HTTPError as http_err:
        return f"❌ **API HTTP Error ({http_err.code}):** `{http_err.read().decode('utf-8', errors='ignore')}`"
    except Exception as exc:
        return f"❌ **AI Connection Error:** `{exc}`"


def render_camera_ai_section(key_id, domain_title, domain_guide, get_context_fn, table_state_key=None, table_cols=None):
    st.markdown(
        f"<div class='ai-cam-box'><b>📸🤖 وحدة الكاميرا الذكية، الملء التلقائي للجداول، والـ AI | Camera, Auto-Table & AI Solver — ({domain_title})</b></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1.8, 2.2, 1])
    with c1:
        src = st.radio(
            "📷 Image Source | مصدر الصورة:",
            ["Table Only | الجدول فقط", "Camera | كاميرا مباشرة", "Upload | رفع صورة"],
            horizontal=True,
            key=f"{key_id}_src",
        )
    with c2:
        flt = st.selectbox(
            "🔍 Vision Filter | فلتر تحسين الصورة:",
            [
                "Contrast & Sharpness | زيادة التباين والحدة للرسومات",
                "Otsu Thresholding | تصفية المعادلات والجداول",
                "Adaptive Threshold | إزالة الظلال والإضاءة الضعيفة",
                "Edges Detection | إبراز الحواف والشبكات الهندسية",
                "Original | الصورة الأصلية",
            ],
            key=f"{key_id}_flt",
        )
    with c3:
        rot = st.selectbox("🔄 Rotate | تدوير يدوي إضافي:", [0, 90, 180, 270], key=f"{key_id}_rot")

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
        proc_img = apply_vision_filter(raw_img, flt, rot)
        ic1, ic2 = st.columns(2)
        with ic1:
            st.image(ImageOps.exif_transpose(raw_img), caption="Original Image (Auto-Oriented) | الصورة الأصلية", use_container_width=True)
        with ic2:
            st.image(proc_img, caption=f"Processed Image ({flt})", use_container_width=True)

    extra_q = st.text_input(
        "💡 Specific AI Instruction | توجيه خاص للذكاء الاصطناعي (اختياري):",
        placeholder="مثال: حل المسألة الموجودة في الصورة بالتفصيل وارسم الجدول...",
        key=f"{key_id}_note",
    )

    b1, b2, b3 = st.columns(3)
    with b1:
        solve_ai_clicked = st.button(
            "🤖 Solve with AI | حل شامل بالـ AI",
            type="primary",
            use_container_width=True,
            key=f"{key_id}_btn_solve",
        )
    with b2:
        ocr_ai_clicked = st.button(
            "📝 Extract Text/Equations (OCR) | استخراج النص",
            use_container_width=True,
            key=f"{key_id}_btn_ocr",
        )
    with b3:
        autofill_clicked = st.button(
            "📊 Auto-Fill Table from Image | ملء الجدول تلقائياً",
            use_container_width=True,
            key=f"{key_id}_btn_autofill",
        )

    if autofill_clicked:
        if proc_img is None:
            st.warning("⚠️ يرجى تصوير جدول المسألة بالكاميرا أو رفع صورته أولاً ليتم ملء الجدول تلقائياً.")
        elif not table_state_key or not table_cols:
            st.info("ℹ️ استخدم زر استخراج النص (OCR) أو الحل الشامل لهذا القسم.")
        else:
            with st.spinner("📊 جاري قراءة الجدول من الصورة وملء الخلايا تلقائياً ليتم حلها ورسمها برمجياً..."):
                raw_json = call_gemini_rest_api(
                    domain_title, domain_guide, extra_q, proc_img, mode="json_table", target_columns=table_cols
                )
                match = re.search(r"\[.*\]", raw_json, re.DOTALL)
                if match:
                    try:
                        parsed_list = json.loads(match.group(0))
                        new_df = pd.DataFrame(parsed_list)
                        st.session_state.custom_tables[table_state_key] = new_df
                        st.success("✅ تم نقل المسألة من الصورة إلى الجدول التفاعلي بنجاح! اضغط الآن على زر الحل البرمجي أعلى الكاميرا.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"⚠️ تعذر تحويل البيانات إلى جدول: {e}\n{raw_json}")
                else:
                    st.error(f"⚠️ تعذر قراءة الجدول تلقائياً:\n{raw_json}")

    if ocr_ai_clicked:
        if proc_img is None:
            st.warning("⚠️ يرجى التقاط صورة بالكاميرا أو رفعها أولاً.")
        else:
            with st.spinner("🔍 Scanning image with Gemini Vision OCR..."):
                res_ocr = call_gemini_rest_api(domain_title, domain_guide, extra_q, proc_img, mode="ocr")
                st.markdown("<h4>📝 المستخرج البصري | Extracted Problem Data:</h4>", unsafe_allow_html=True)
                st.markdown(res_ocr)
                add_log(domain_title, "Vision OCR", "Image Input", res_ocr)

    if solve_ai_clicked:
        ctx_str = get_context_fn()
        if extra_q.strip():
            ctx_str += f"\nUser Note: {extra_q.strip()}"
        if proc_img is None and not ctx_str.strip():
            st.warning("⚠️ املأ الجدول أو التقط صورة المسألة أولاً.")
        else:
            with st.spinner(f"🤖 Solving ({domain_title}) problem step-by-step..."):
                res_solve = call_gemini_rest_api(domain_title, domain_guide, ctx_str, proc_img, mode="solve")
                st.markdown(f"<h4>🤖 الحل الهندسي التفصيلي | AI Engineering Solution ({domain_title}):</h4>", unsafe_allow_html=True)
                st.markdown(res_solve)
                add_log(domain_title, "Gemini AI Solver", ctx_str, res_solve)


def parse_inline_dep_token(token_str):
    tok = token_str.strip()
    if not tok or tok in ["-", "None", "nan"]:
        return None
    m_colon = re.match(r"^([A-Za-z0-9_]+)\s*[:\(\s]\s*(FS|SS|FF|SF)\s*([+\-]?\d*\.?\d*)\)?$", tok, re.IGNORECASE)
    if m_colon:
        act_n = m_colon.group(1).strip()
        rel_t = m_colon.group(2).upper()
        lag_s = m_colon.group(3).strip()
        lag_v = float(lag_s) if lag_s and lag_s not in ["+", "-"] else 0.0
        return (act_n, rel_t, lag_v)
    return (tok, "FS", 0.0)


def detect_network_cycles_and_dangling(tasks, order_keys):
    visited = {}
    cycle_path = []

    def dfs(node, path_stack):
        visited[node] = 1
        for succ in order_keys:
            for p_id, _, _ in tasks[succ]["preds"]:
                if p_id == node:
                    if visited.get(succ, 0) == 1:
                        cycle_path.extend(path_stack + [succ])
                        return True
                    if visited.get(succ, 0) == 0:
                        if dfs(succ, path_stack + [succ]):
                            return True
        visited[node] = 2
        return False

    has_cycle = False
    for k in order_keys:
        if visited.get(k, 0) == 0:
            if dfs(k, [k]):
                has_cycle = True
                break

    start_nodes = [k for k in order_keys if not tasks[k]["preds"]]
    end_nodes = [k for k in order_keys if not any(any(p == k for p, _, _ in tasks[s]["preds"]) for s in order_keys)]
    return has_cycle, cycle_path, start_nodes, end_nodes


def build_aon_9cell_svg(tasks, order_keys):
    layers = {}
    for _ in range(len(order_keys) + 2):
        for k in order_keys:
            preds = [p for p, _, _ in tasks[k]["preds"] if p in tasks]
            layers[k] = 0 if not preds else max(layers.get(p, 0) for p in preds) + 1
    max_layer = max(layers.values()) if layers else 0
    layer_groups = {}
    for k in order_keys:
        layer_groups.setdefault(layers[k], []).append(k)

    max_in_col = max(len(v) for v in layer_groups.values()) if layer_groups else 1
    svg_w = max(840, (max_layer + 1) * 230 + 80)
    svg_h = max(360, max_in_col * 150 + 90)

    coords = {}
    for ly, nodes in layer_groups.items():
        x_c = 45 + ly * 220
        for idx, k in enumerate(nodes):
            y_c = 45 + idx * 140 + (max_in_col - len(nodes)) * 60
            coords[k] = (x_c, y_c)

    svg_lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {svg_w} {svg_h}' style='width:100%; height:auto; background:white; border:1px solid rgb(203,213,225); border-radius:8px;'>",
        "<defs>",
        "<marker id='arrCrit' markerWidth='10' markerHeight='7' refX='9' refY='3.5' orient='auto'><polygon points='0 0, 10 3.5, 0 7' fill='rgb(219,39,119)'/></marker>",
        "<marker id='arrNorm' markerWidth='10' markerHeight='7' refX='9' refY='3.5' orient='auto'><polygon points='0 0, 10 3.5, 0 7' fill='rgb(71,85,105)'/></marker>",
        "</defs>",
    ]

    for k in order_keys:
        x2, y2 = coords[k]
        for p_id, r_type, lag in tasks[k]["preds"]:
            if p_id in coords:
                x1, y1 = coords[p_id]
                sx, sy = (x1, y1 + 70) if r_type in ["SS", "SF"] else (x1 + 150, y1 + 45)
                ex, ey = (x2 + 150, y2 + 75) if r_type in ["FF", "SF"] else (x2, y2 + 45)
                is_crit_edge = tasks[p_id]["crit_any"] and tasks[k]["crit_any"]
                col = "rgb(219,39,119)" if is_crit_edge else "rgb(71,85,105)"
                dash = "stroke-dasharray='5,3'" if lag != 0 or r_type != "FS" else ""
                mk = "url(#arrCrit)" if is_crit_edge else "url(#arrNorm)"
                svg_lines.append(f"<line x1='{sx}' y1='{sy}' x2='{ex}' y2='{ey}' stroke='{col}' stroke-width='2.2' {dash} marker-end='{mk}'/>")
                mx, my = (sx + ex) / 2.0, (sy + ey) / 2.0 - 7
                lbl_rel = f"Lag {lag:g} ({r_type})" if lag != 0 else (r_type if r_type != "FS" else "")
                if lbl_rel:
                    svg_lines.append(f"<text x='{mx}' y='{my}' font-size='11' font-weight='bold' fill='rgb(190,24,93)' text-anchor='middle'>{lbl_rel}</text>")

    for k in order_keys:
        x, y = coords[k]
        d = tasks[k]
        is_c = d["crit_any"]
        bd = "rgb(219,39,119)" if is_c else "rgb(30,58,138)"
        left_bg = "rgb(254,205,211)" if d["crit_start"] else "rgb(248,250,252)"
        right_bg = "rgb(254,205,211)" if d["crit_finish"] else "rgb(248,250,252)"
        mid_bg = "rgb(253,242,248)" if is_c else "rgb(248,250,252)"
        center_fill = "rgb(236,72,153)" if is_c else "rgb(219,234,254)"
        desc_short = str(d.get("desc", ""))[:12]

        svg_lines.append(f"<rect x='{x}' y='{y}' width='45' height='90' fill='{left_bg}'/>")
        svg_lines.append(f"<rect x='{x+45}' y='{y}' width='60' height='90' fill='{mid_bg}'/>")
        svg_lines.append(f"<rect x='{x+105}' y='{y}' width='45' height='90' fill='{right_bg}'/>")
        svg_lines.append(f"<rect x='{x+45}' y='{y+30}' width='60' height='30' fill='{center_fill}' opacity='0.4'/>")
        svg_lines.append(f"<rect x='{x}' y='{y}' width='150' height='90' rx='4' fill='none' stroke='{bd}' stroke-width='2.2'/>")

        svg_lines.append(f"<line x1='{x}' y1='{y+30}' x2='{x+150}' y2='{y+30}' stroke='{bd}' stroke-width='1.3'/>")
        svg_lines.append(f"<line x1='{x}' y1='{y+60}' x2='{x+150}' y2='{y+60}' stroke='{bd}' stroke-width='1.3'/>")
        svg_lines.append(f"<line x1='{x+45}' y1='{y}' x2='{x+45}' y2='{y+90}' stroke='{bd}' stroke-width='1.3'/>")
        svg_lines.append(f"<line x1='{x+105}' y1='{y}' x2='{x+105}' y2='{y+90}' stroke='{bd}' stroke-width='1.3'/>")

        svg_lines.append(f"<text x='{x+22}' y='{y+20}' font-size='11' font-weight='bold' text-anchor='middle' fill='black'>{d['ES']:g}</text>")
        svg_lines.append(f"<text x='{x+75}' y='{y+20}' font-size='13' font-weight='bold' text-anchor='middle' fill='{bd}'>{k}</text>")
        svg_lines.append(f"<text x='{x+127}' y='{y+20}' font-size='11' font-weight='bold' text-anchor='middle' fill='black'>{d['EF']:g}</text>")

        svg_lines.append(f"<text x='{x+22}' y='{y+49}' font-size='11' font-weight='bold' text-anchor='middle' fill='rgb(190,24,93)'>{d['start_sl']:g}</text>")
        svg_lines.append(f"<text x='{x+75}' y='{y+49}' font-size='9.5' text-anchor='middle' fill='black'>{desc_short}</text>")
        svg_lines.append(f"<text x='{x+127}' y='{y+49}' font-size='11' font-weight='bold' text-anchor='middle' fill='rgb(190,24,93)'>{d['finish_sl']:g}</text>")

        svg_lines.append(f"<text x='{x+22}' y='{y+79}' font-size='11' font-weight='bold' text-anchor='middle' fill='black'>{d['LS']:g}</text>")
        svg_lines.append(f"<text x='{x+75}' y='{y+79}' font-size='11' font-weight='bold' text-anchor='middle' fill='black'>{d['dur']:g}</text>")
        svg_lines.append(f"<text x='{x+127}' y='{y+79}' font-size='11' font-weight='bold' text-anchor='middle' fill='black'>{d['LF']:g}</text>")

    lx, ly_leg = 15, svg_h - 78
    svg_lines.append(f"<rect x='{lx}' y='{ly_leg}' width='135' height='66' fill='white' stroke='black' stroke-width='1.5'/>")
    svg_lines.append(f"<line x1='{lx}' y1='{ly_leg+22}' x2='{lx+135}' y2='{ly_leg+22}' stroke='black' stroke-width='1'/>")
    svg_lines.append(f"<line x1='{lx}' y1='{ly_leg+44}' x2='{lx+135}' y2='{ly_leg+44}' stroke='black' stroke-width='1'/>")
    svg_lines.append(f"<line x1='{lx+42}' y1='{ly_leg}' x2='{lx+42}' y2='{ly_leg+66}' stroke='black' stroke-width='1'/>")
    svg_lines.append(f"<line x1='{lx+93}' y1='{ly_leg}' x2='{lx+93}' y2='{ly_leg+66}' stroke='black' stroke-width='1'/>")
    svg_lines.append(f"<text x='{lx+21}' y='{ly_leg+15}' font-size='9.5' text-anchor='middle'>ES</text>")
    svg_lines.append(f"<text x='{lx+67}' y='{ly_leg+15}' font-size='9.5' font-weight='bold' text-anchor='middle'>ID</text>")
    svg_lines.append(f"<text x='{lx+114}' y='{ly_leg+15}' font-size='9.5' text-anchor='middle'>EF</text>")
    svg_lines.append(f"<text x='{lx+21}' y='{ly_leg+36}' font-size='9.5' text-anchor='middle'>SL(s)</text>")
    svg_lines.append(f"<text x='{lx+67}' y='{ly_leg+36}' font-size='8.5' text-anchor='middle'>Desc</text>")
    svg_lines.append(f"<text x='{lx+114}' y='{ly_leg+36}' font-size='9.5' text-anchor='middle'>SL(f)</text>")
    svg_lines.append(f"<text x='{lx+21}' y='{ly_leg+58}' font-size='9.5' text-anchor='middle'>LS</text>")
    svg_lines.append(f"<text x='{lx+67}' y='{ly_leg+58}' font-size='9.5' text-anchor='middle'>DUR</text>")
    svg_lines.append(f"<text x='{lx+114}' y='{ly_leg+58}' font-size='9.5' text-anchor='middle'>LF</text>")

    svg_lines.append("</svg>")
    return "\n".join(svg_lines)


def build_aoa_slide_svg(tasks, order_keys, proj_dur):
    events = {1: {"E": 0.0, "L": 0.0, "col": 0, "row": 0}}
    ev_counter = 1
    end_event_of = {}
    edges_aoa = []

    for idx, k in enumerate(order_keys):
        preds = [p for p, _, _ in tasks[k]["preds"] if p in tasks]
        if not preds:
            u_ev = 1
        elif len(preds) == 1:
            u_ev = end_event_of[preds[0]]
        else:
            ev_counter += 1
            u_ev = ev_counter
            max_c = max(events[end_event_of[p]]["col"] for p in preds) + 1
            events[u_ev] = {"E": tasks[k]["ES"], "L": tasks[k]["LS"], "col": max_c, "row": idx % 3}
            for p_i, p_name in enumerate(preds):
                edges_aoa.append({
                    "u": end_event_of[p_name], "v": u_ev,
                    "name": f"X{p_i+1}", "dur": 0.0,
                    "ES": tasks[p_name]["EF"], "EF": tasks[p_name]["EF"],
                    "LS": tasks[k]["LS"], "LF": tasks[k]["LS"],
                    "slack": 0.0, "dummy": True, "crit": False,
                })
        ev_counter += 1
        v_ev = ev_counter
        events[v_ev] = {"E": tasks[k]["EF"], "L": tasks[k]["LF"], "col": events[u_ev]["col"] + 1, "row": idx % 3}
        end_event_of[k] = v_ev
        edges_aoa.append({
            "u": u_ev, "v": v_ev,
            "name": k, "dur": tasks[k]["dur"],
            "ES": tasks[k]["ES"], "EF": tasks[k]["EF"],
            "LS": tasks[k]["LS"], "LF": tasks[k]["LF"],
            "slack": tasks[k]["TF"], "dummy": False, "crit": tasks[k]["crit_any"],
        })

    max_col = max(e["col"] for e in events.values()) if events else 1
    svg_w = max(820, (max_col + 1) * 175 + 90)
    svg_h = 380
    ev_coords = {}
    col_counts = {}
    for ev_id, edata in events.items():
        c = edata["col"]
        r_idx = col_counts.get(c, 0)
        col_counts[c] = r_idx + 1
        ev_coords[ev_id] = (65 + c * 165, 80 + (r_idx % 3) * 115)

    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {svg_w} {svg_h}' style='width:100%; height:auto; background:rgb(253,242,248); border:1px solid rgb(219,39,119); border-radius:8px;'>",
        "<defs><marker id='arrAOASlide' markerWidth='10' markerHeight='7' refX='28' refY='3.5' orient='auto'><polygon points='0 0, 10 3.5, 0 7' fill='rgb(190,24,93)'/></marker></defs>",
    ]
    for ed in edges_aoa:
        x1, y1 = ev_coords[ed["u"]]
        x2, y2 = ev_coords[ed["v"]]
        col = "rgb(190,24,93)" if ed["crit"] else ("rgb(100,116,139)" if ed["dummy"] else "rgb(30,58,138)")
        dash = "stroke-dasharray='5,4'" if ed["dummy"] else ""
        svg.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{col}' stroke-width='2.3' {dash} marker-end='url(#arrAOASlide)'/>")
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        svg.append(f"<text x='{mx}' y='{my-8}' font-size='11.5' font-weight='bold' fill='{col}' text-anchor='middle'>{ed['name']}</text>")
        svg.append(f"<text x='{mx}' y='{my+14}' font-size='10.5' fill='black' text-anchor='middle'>{ed['dur']:g}</text>")

    for ev_id, (cx, cy) in ev_coords.items():
        ev_d = events[ev_id]
        sl_v = round(ev_d["L"] - ev_d["E"], 2)
        svg.append(f"<circle cx='{cx}' cy='{cy}' r='22' fill='white' stroke='black' stroke-width='1.8'/>")
        svg.append(f"<text x='{cx}' y='{cy+4}' font-size='12' font-weight='bold' text-anchor='middle' fill='black'>{ev_id}</text>")

        bx, by = cx - 26, cy - 64
        svg.append(f"<rect x='{bx}' y='{by}' width='52' height='36' fill='white' stroke='black' stroke-width='1.2'/>")
        svg.append(f"<line x1='{bx}' y1='{by+18}' x2='{bx+52}' y2='{by+18}' stroke='black' stroke-width='1'/>")
        svg.append(f"<line x1='{bx+26}' y1='{by+18}' x2='{bx+26}' y2='{by+36}' stroke='black' stroke-width='1'/>")
        svg.append(f"<text x='{bx+26}' y='{by+13}' font-size='10' font-weight='bold' text-anchor='middle' fill='rgb(190,24,93)'>{sl_v:g}</text>")
        svg.append(f"<text x='{bx+13}' y='{by+31}' font-size='9.5' text-anchor='middle' fill='black'>{ev_d['E']:g}</text>")
        svg.append(f"<text x='{bx+39}' y='{by+31}' font-size='9.5' text-anchor='middle' fill='black'>{ev_d['L']:g}</text>")

    svg.append("</svg>")
    return "\n".join(svg)


TABS = st.tabs([
    "1️⃣ شبكات الأعمال العامة (AON / AOA / Lags / Crashing)",
    "2️⃣ الفوائد وتقييم المشروعات العام (Economy & NPV)",
    "3️⃣ الطرق العددية العامة (Numerical Methods)",
    "4️⃣ التفاضل ولابلاس العام (ODEs & Laplace)",
    "5️⃣ الميكانيكا والكمرات العامة (Mohr & Beams)",
    "6️⃣ الدوائر والمحولات العامة (Circuits & Power)",
    "7️⃣ التحكم وجدول راوث العام (Control & Routh)",
    "8️⃣ البرمجة والتقارير (Code & History)",
])


with TABS[0]:
    st.markdown("""
    <div class='sec-header'>
        <h3>📊 1. الحل العام لجميع مسائل شبكات الأعمال | Universal Project Management Solver (AON, AOA, Lags, Crashing & PERT)</h3>
        <p>أدخل <b>أي مسألة جديدة</b> مهما كان عدد أنشطتها أو نوع علاقاتها (<b>Predecessors</b> أو <b>Successors</b> أو <b>Lags: FS, SS, FF, SF</b>)، أو اختر من القوالب الجاهزة.</p>
    </div>
    """, unsafe_allow_html=True)

    c_pm1, c_pm2, c_pm3, c_pm4 = st.columns([1.6, 0.8, 1.0, 1.0])
    with c_pm1:
        pm_preset = st.selectbox(
            "🛠️ وضع الإدخال أو القالب | Problem Mode:",
            [
                "🆕 1. Custom New Network | إدخال مسألة شبكات جديدة (جدول حر لأي مسألة)",
                "📉 2. Custom Project Crashing | مسألة ضغط مشروعات بأقل تكلفة (Time-Cost)",
                "🎲 3. Custom PERT Network | مسألة احتمالات بيرت (to, tm, tp)",
                "📚 4. Lecture Example 4 (KOLL Business Center AOA)",
                "📚 5. Lecture Figure 6.21 (Network Using Lags A-to-F)",
                "📚 6. Lecture Example 6.20 (Lag Exercises A-to-K)",
                "📚 7. Lecture Example 3 (Steel Cylinder Successors A-to-T)",
                "📚 8. Lecture Example 6 (AOA Network A-to-R)",
            ],
        )
    with c_pm2:
        custom_n_acts = st.number_input("🔢 عدد الأنشطة:", min_value=2, max_value=40, value=6, step=1)
    with c_pm3:
        table_orient = st.radio("📐 اتجاه الجدول:", ["رأسي (Vertical)", "أفقي (Horizontal)"], horizontal=True)
    with c_pm4:
        dep_mode = st.radio(
            "🔗 قراءة العلاقات:",
            ["تلقائي (Auto-Detect)", "Predecessors", "Successors"],
            horizontal=True,
        )

    if pm_preset.startswith("🆕 1."):
        letters = [chr(65 + i) if i < 26 else f"A{i-25}" for i in range(int(custom_n_acts))]
        init_pm_df = pd.DataFrame([
            {
                "Activity": letters[i],
                "Description": f"Activity {letters[i]}",
                "Predecessors": "-" if i == 0 else letters[i - 1],
                "Successors": "-",
                "Activity Time": float(4 + (i % 5)),
            }
            for i in range(int(custom_n_acts))
        ])
        init_lags_df = pd.DataFrame([
            {"From Activity": "", "To Activity": "", "Relation (FS/SS/FF/SF)": "FS", "Lag Time": 0.0}
        ])
    elif "Example 4" in pm_preset:
        init_pm_df = pd.DataFrame([
            {"Activity": "A", "Description": "Application approval", "Predecessors": "-", "Successors": "B, C, D", "Activity Time": 5.0},
            {"Activity": "B", "Description": "Construction plans", "Predecessors": "A", "Successors": "E, F", "Activity Time": 15.0},
            {"Activity": "C", "Description": "Traffic study", "Predecessors": "A", "Successors": "E, F", "Activity Time": 10.0},
            {"Activity": "D", "Description": "Service availability check", "Predecessors": "A", "Successors": "F", "Activity Time": 5.0},
            {"Activity": "E", "Description": "Staff report", "Predecessors": "B, C", "Successors": "H", "Activity Time": 15.0},
            {"Activity": "F", "Description": "Commission approval", "Predecessors": "B, C, D", "Successors": "G", "Activity Time": 10.0},
            {"Activity": "G", "Description": "Wait for construction", "Predecessors": "F", "Successors": "H", "Activity Time": 170.0},
            {"Activity": "H", "Description": "Occupancy", "Predecessors": "E, G", "Successors": "-", "Activity Time": 35.0},
        ])
        init_lags_df = pd.DataFrame(columns=["From Activity", "To Activity", "Relation (FS/SS/FF/SF)", "Lag Time"])
    elif "Figure 6.21" in pm_preset:
        init_pm_df = pd.DataFrame([
            {"Activity": "A", "Description": "Design system", "Predecessors": "-", "Successors": "-", "Activity Time": 5.0},
            {"Activity": "B", "Description": "Order hardware", "Predecessors": "A:SS+3", "Successors": "-", "Activity Time": 1.0},
            {"Activity": "C", "Description": "Install hardware", "Predecessors": "B:FS+4", "Successors": "-", "Activity Time": 2.0},
            {"Activity": "D", "Description": "Install software", "Predecessors": "A, C", "Successors": "-", "Activity Time": 6.0},
            {"Activity": "E", "Description": "Test system", "Predecessors": "A, D:SS+2", "Successors": "-", "Activity Time": 2.0},
            {"Activity": "F", "Description": "System documentation", "Predecessors": "A, D:FF+2, E:FF+2", "Successors": "-", "Activity Time": 3.0},
        ])
        init_lags_df = pd.DataFrame([
            {"From Activity": "A", "To Activity": "B", "Relation (FS/SS/FF/SF)": "SS", "Lag Time": 3.0},
            {"From Activity": "B", "To Activity": "C", "Relation (FS/SS/FF/SF)": "FS", "Lag Time": 4.0},
            {"From Activity": "D", "To Activity": "E", "Relation (FS/SS/FF/SF)": "SS", "Lag Time": 2.0},
            {"From Activity": "D", "To Activity": "F", "Relation (FS/SS/FF/SF)": "FF", "Lag Time": 2.0},
            {"From Activity": "E", "To Activity": "F", "Relation (FS/SS/FF/SF)": "FF", "Lag Time": 2.0},
        ])
    elif "Example 6.20" in pm_preset:
        init_pm_df = pd.DataFrame([
            {"Activity": "A", "Description": "Task A", "Predecessors": "-", "Successors": "-", "Activity Time": 2.0},
            {"Activity": "B", "Description": "Task B", "Predecessors": "A:SS+5", "Successors": "-", "Activity Time": 4.0},
            {"Activity": "C", "Description": "Task C", "Predecessors": "A", "Successors": "-", "Activity Time": 8.0},
            {"Activity": "D", "Description": "Task D", "Predecessors": "A", "Successors": "-", "Activity Time": 9.0},
            {"Activity": "E", "Description": "Task E", "Predecessors": "B, C:FF+7", "Successors": "-", "Activity Time": 2.0},
            {"Activity": "F", "Description": "Task F", "Predecessors": "C, D:SS+4", "Successors": "-", "Activity Time": 7.0},
            {"Activity": "G", "Description": "Task G", "Predecessors": "D", "Successors": "-", "Activity Time": 4.0},
            {"Activity": "H", "Description": "Task H", "Predecessors": "E:FS+10, F:SS+5", "Successors": "-", "Activity Time": 5.0},
            {"Activity": "I", "Description": "Task I", "Predecessors": "F", "Successors": "-", "Activity Time": 4.0},
            {"Activity": "J", "Description": "Task J", "Predecessors": "G:FS+10", "Successors": "-", "Activity Time": 7.0},
            {"Activity": "K", "Description": "Task K", "Predecessors": "H:FF+8, I, J:FF+8", "Successors": "-", "Activity Time": 3.0},
        ])
        init_lags_df = pd.DataFrame([
            {"From Activity": "A", "To Activity": "B", "Relation (FS/SS/FF/SF)": "SS", "Lag Time": 5.0},
            {"From Activity": "C", "To Activity": "E", "Relation (FS/SS/FF/SF)": "FF", "Lag Time": 7.0},
            {"From Activity": "D", "To Activity": "F", "Relation (FS/SS/FF/SF)": "SS", "Lag Time": 4.0},
            {"From Activity": "E", "To Activity": "H", "Relation (FS/SS/FF/SF)": "FS", "Lag Time": 10.0},
            {"From Activity": "F", "To Activity": "H", "Relation (FS/SS/FF/SF)": "SS", "Lag Time": 5.0},
            {"From Activity": "G", "To Activity": "J", "Relation (FS/SS/FF/SF)": "FS", "Lag Time": 10.0},
            {"From Activity": "H", "To Activity": "K", "Relation (FS/SS/FF/SF)": "FF", "Lag Time": 8.0},
            {"From Activity": "J", "To Activity": "K", "Relation (FS/SS/FF/SF)": "FF", "Lag Time": 8.0},
        ])
    elif "Example 3" in pm_preset:
        acts_3 = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T"]
        succ_3 = ["D","E, F","G","I, J","L","K","H, M","N","P","Q","R","Q","N","O, S","R","R","R","-","T","-"]
        time_3 = [10, 14, 3, 14, 15, 11, 8, 1, 13, 5, 13, 6, 4, 7, 4, 7, 4, 3, 6, 10]
        init_pm_df = pd.DataFrame([
            {"Activity": a, "Description": f"Step {a}", "Predecessors": "-", "Successors": s, "Activity Time": float(t)}
            for a, s, t in zip(acts_3, succ_3, time_3)
        ])
        init_lags_df = pd.DataFrame(columns=["From Activity", "To Activity", "Relation (FS/SS/FF/SF)", "Lag Time"])
    else:
        acts_6 = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R"]
        pred_6 = ["-","-","-","A","B","B","C","D","E, F","F","G","G","K, L","H","I","J","M","Q"]
        time_6 = [10, 14, 3, 14, 15, 11, 8, 13, 6, 13, 1, 4, 7, 7, 4, 3, 6, 10]
        init_pm_df = pd.DataFrame([
            {"Activity": a, "Description": f"Act {a}", "Predecessors": p, "Successors": "-", "Activity Time": float(t)}
            for a, p, t in zip(acts_6, pred_6, time_6)
        ])
        init_lags_df = pd.DataFrame(columns=["From Activity", "To Activity", "Relation (FS/SS/FF/SF)", "Lag Time"])

    if "tab1_pm_df" in st.session_state.custom_tables:
        init_pm_df = st.session_state.custom_tables["tab1_pm_df"]

    if "2. Custom Project Crashing" in pm_preset:
        st.markdown("<h4>📉 جدول ضغط المشروعات العام (أدخل أي أنشطة وتكاليف) | Universal Project Crashing Table:</h4>", unsafe_allow_html=True)
        crash_init = pd.DataFrame([
            {"Activity": "A", "Predecessors": "-", "Normal Time": 6.0, "Crash Time": 4.0, "Normal Cost": 1000.0, "Crash Cost": 1600.0},
            {"Activity": "B", "Predecessors": "A", "Normal Time": 8.0, "Crash Time": 5.0, "Normal Cost": 1500.0, "Crash Cost": 2100.0},
            {"Activity": "C", "Predecessors": "A", "Normal Time": 5.0, "Crash Time": 3.0, "Normal Cost": 800.0, "Crash Cost": 1300.0},
            {"Activity": "D", "Predecessors": "B", "Normal Time": 6.0, "Crash Time": 4.0, "Normal Cost": 1200.0, "Crash Cost": 1500.0},
            {"Activity": "E", "Predecessors": "C", "Normal Time": 7.0, "Crash Time": 5.0, "Normal Cost": 1400.0, "Crash Cost": 1800.0},
            {"Activity": "F", "Predecessors": "D, E", "Normal Time": 4.0, "Crash Time": 2.0, "Normal Cost": 900.0, "Crash Cost": 1700.0},
        ])
        ed_crash = st.data_editor(crash_init, num_rows="dynamic", use_container_width=True, key="crash_editor")
        c_cr1, c_cr2 = st.columns(2)
        with c_cr1:
            target_crash_dur = st.number_input("🎯 Target Crashed Project Duration | المدة المستهدفة بعد الضغط:", value=16.0, step=1.0)
        with c_cr2:
            indirect_cost_day = st.number_input("💰 Indirect Cost per Day | التكلفة غير المباشرة لليوم الواحد:", value=250.0, step=50.0)

        if st.button("⚙️ Solve Crashing Table Step-by-Step", type="primary"):
            c_tasks = {}
            c_keys = []
            base_direct_cost = 0.0
            slope_rows = []
            for _, r in ed_crash.iterrows():
                a_id = str(r["Activity"]).strip()
                if not a_id or a_id == "nan":
                    continue
                nt, ct = float(r["Normal Time"]), float(r["Crash Time"])
                nc, cc = float(r["Normal Cost"]), float(r["Crash Cost"])
                max_c = max(0.0, nt - ct)
                slope = ((cc - nc) / max_c) if max_c > 0 else float("inf")
                preds = [p.strip() for p in str(r["Predecessors"]).split(",") if p.strip() and p.strip() != "-"]
                c_tasks[a_id] = {"dur": nt, "min_dur": ct, "slope": slope, "preds": preds}
                c_keys.append(a_id)
                base_direct_cost += nc
                slope_rows.append({"Activity": a_id, "Normal Time": nt, "Crash Time": ct, "Max Crash Days": max_c, "Crash Cost / Day (Slope)": round(slope, 2) if slope != float("inf") else "N/A"})

            st.dataframe(pd.DataFrame(slope_rows), use_container_width=True)

            def eval_cpm_simple(t_dict):
                for k in c_keys:
                    es = max([t_dict[p]["EF"] for p in t_dict[k]["preds"] if p in t_dict] + [0.0])
                    t_dict[k]["ES"] = es
                    t_dict[k]["EF"] = es + t_dict[k]["dur"]
                p_dur = max(t_dict[k]["EF"] for k in c_keys)
                for k in reversed(c_keys):
                    succs = [s for s in c_keys if k in t_dict[s]["preds"]]
                    lf = min([t_dict[s]["LS"] for s in succs] + [p_dur])
                    t_dict[k]["LF"] = lf
                    t_dict[k]["LS"] = lf - t_dict[k]["dur"]
                    t_dict[k]["crit"] = abs(t_dict[k]["LS"] - t_dict[k]["ES"]) < 1e-5
                return p_dur

            curr_dur = eval_cpm_simple(c_tasks)
            curr_direct = base_direct_cost
            crash_steps = [{
                "Step": 0, "Crashed Activity": "None (Normal)",
                "Project Duration": curr_dur, "Added Crash Cost": 0.0,
                "Total Direct Cost": curr_direct,
                "Indirect Cost": curr_dur * indirect_cost_day,
                "Total Project Cost": curr_direct + curr_dur * indirect_cost_day,
                "Critical Activities": ", ".join([k for k in c_keys if c_tasks[k]["crit"]]),
            }]
            step_i = 1
            while curr_dur > target_crash_dur and step_i <= 40:
                crit_cands = [k for k in c_keys if c_tasks[k]["crit"] and c_tasks[k]["dur"] > c_tasks[k]["min_dur"]]
                if not crit_cands:
                    break
                best_k = min(crit_cands, key=lambda k: c_tasks[k]["slope"])
                c_tasks[best_k]["dur"] -= 1.0
                added_c = c_tasks[best_k]["slope"]
                curr_direct += added_c
                curr_dur = eval_cpm_simple(c_tasks)
                crash_steps.append({
                    "Step": step_i, "Crashed Activity": f"{best_k} (-1)",
                    "Project Duration": curr_dur, "Added Crash Cost": round(added_c, 2),
                    "Total Direct Cost": round(curr_direct, 2),
                    "Indirect Cost": round(curr_dur * indirect_cost_day, 2),
                    "Total Project Cost": round(curr_direct + curr_dur * indirect_cost_day, 2),
                    "Critical Activities": ", ".join([k for k in c_keys if c_tasks[k]["crit"]]),
                })
                step_i += 1
            df_cr_steps = pd.DataFrame(crash_steps)
            st.dataframe(df_cr_steps, use_container_width=True)
            opt_row = df_cr_steps.loc[df_cr_steps["Total Project Cost"].idxmin()]
            st.success(f"🏆 **Optimal Minimum-Cost Duration:** `{opt_row['Project Duration']}` | **Minimum Total Cost:** `${opt_row['Total Project Cost']:,.2f}`")

    elif "3. Custom PERT" in pm_preset:
        pert_init = pd.DataFrame([
            {"Activity": "A", "Predecessors": "-", "Optimistic (to)": 3.0, "Most Likely (tm)": 5.0, "Pessimistic (tp)": 13.0},
            {"Activity": "B", "Predecessors": "A", "Optimistic (to)": 4.0, "Most Likely (tm)": 6.0, "Pessimistic (tp)": 8.0},
            {"Activity": "C", "Predecessors": "A", "Optimistic (to)": 5.0, "Most Likely (tm)": 8.0, "Pessimistic (tp)": 17.0},
            {"Activity": "D", "Predecessors": "B, C", "Optimistic (to)": 2.0, "Most Likely (tm)": 4.0, "Pessimistic (tp)": 6.0},
        ])
        edited_pert = st.data_editor(pert_init, num_rows="dynamic", use_container_width=True, key="pert_table_editor")
        td_val = st.number_input("🎯 Target Project Deadline (Td):", value=22.0)
        if st.button("⚙️ Solve PERT Network, Critical Path & Z-Probability", type="primary"):
            edited_pert["Expected Time (te)"] = (edited_pert["Optimistic (to)"] + 4.0 * edited_pert["Most Likely (tm)"] + edited_pert["Pessimistic (tp)"]) / 6.0
            edited_pert["Std Dev (σ)"] = (edited_pert["Pessimistic (tp)"] - edited_pert["Optimistic (to)"]) / 6.0
            edited_pert["Variance (σ²)"] = edited_pert["Std Dev (σ)"] ** 2
            st.dataframe(edited_pert, use_container_width=True)
            te_sum = float(edited_pert["Expected Time (te)"].sum())
            var_sum = float(edited_pert["Variance (σ²)"].sum())
            sd_tot = math.sqrt(var_sum)
            z_sc = (td_val - te_sum) / sd_tot if sd_tot > 0 else 0.0
            st.success(f"📈 **Expected Sum ($T_e$):** `{te_sum:.2f}` | **Z-Score:** `{z_sc:.4f}` | **Probability $P(T \\le {td_val})$:** `{norm_cdf(z_sc)*100:.2f}%`")

    else:
        st.caption(
            "💡 **إدخال حر لأي مسألة:** يمكنك تعديل أي خلية أو إضافة/حذف صفوف، كما يمكنك كتابة الـ Lags مباشرة داخل خانة Predecessors أو Successors بصيغة `A:SS+3` أو `B:FF+2` أو في جدول الـ Lags بالأسفل."
        )
        col_btn_clr, _ = st.columns([1.3, 4])
        if col_btn_clr.button("🗑️ تفريغ الجدول لمسألة جديدة فارغة"):
            letters_blank = [chr(65 + i) for i in range(int(custom_n_acts))]
            st.session_state.custom_tables["tab1_pm_df"] = pd.DataFrame([
                {"Activity": a, "Description": "", "Predecessors": "-", "Successors": "-", "Activity Time": 1.0}
                for a in letters_blank
            ])
            st.rerun()

        if "رأسي" in table_orient:
            edited_pm_df = st.data_editor(init_pm_df, num_rows="dynamic", use_container_width=True, key="pm_vert_editor")
        else:
            col_to_use = "Successors" if "Successors" in dep_mode else "Predecessors"
            horiz_init = init_pm_df.set_index("Activity")[[col_to_use, "Activity Time"]].T
            edited_horiz = st.data_editor(horiz_init, use_container_width=True, key="pm_horiz_editor")
            reconstructed = []
            for col_act in edited_horiz.columns:
                dep_val = str(edited_horiz.loc[edited_horiz.index[0], col_act])
                t_val = float(edited_horiz.loc["Activity Time", col_act])
                reconstructed.append({
                    "Activity": str(col_act),
                    "Description": str(col_act),
                    "Predecessors": dep_val if col_to_use == "Predecessors" else "-",
                    "Successors": dep_val if col_to_use == "Successors" else "-",
                    "Activity Time": t_val,
                })
            edited_pm_df = pd.DataFrame(reconstructed)

        with st.expander("🔗 جدول علاقات الـ Lags الإضافي (FS, SS, FF, SF) — اختياري", expanded=("Lag" in pm_preset or "6.21" in pm_preset)):
            edited_lags_df = st.data_editor(init_lags_df, num_rows="dynamic", use_container_width=True, key="pm_lags_editor")

        if st.button("⚙️ حل المسألة ورسم الشبكات (Solve Any Network & Draw AON / AOA)", type="primary"):
            try:
                tasks = {}
                order_keys = []
                for _, row in edited_pm_df.iterrows():
                    act_id = str(row["Activity"]).strip()
                    if not act_id or act_id == "nan":
                        continue
                    dur_v = float(row["Activity Time"]) if pd.notna(row["Activity Time"]) else 0.0
                    desc_v = str(row.get("Description", act_id)) if pd.notna(row.get("Description", "")) else act_id
                    tasks[act_id] = {
                        "dur": dur_v, "desc": desc_v, "preds": [],
                        "ES": 0.0, "EF": dur_v, "LS": 0.0, "LF": dur_v,
                        "start_sl": 0.0, "finish_sl": 0.0, "TF": 0.0, "FF": 0.0,
                        "crit_start": False, "crit_finish": False, "crit_any": False,
                    }
                    order_keys.append(act_id)

                has_any_preds = any(
                    str(r.get("Predecessors", "-")).strip() not in ["-", "None", "nan", ""]
                    for _, r in edited_pm_df.iterrows()
                )
                has_any_succs = any(
                    str(r.get("Successors", "-")).strip() not in ["-", "None", "nan", ""]
                    for _, r in edited_pm_df.iterrows()
                )

                use_preds = True
                use_succs = False
                if "Predecessors" in dep_mode:
                    use_preds, use_succs = True, False
                elif "Successors" in dep_mode:
                    use_preds, use_succs = False, True
                else:
                    use_preds = has_any_preds
                    use_succs = has_any_succs and not has_any_preds

                if use_preds:
                    for _, row in edited_pm_df.iterrows():
                        act_id = str(row["Activity"]).strip()
                        if act_id not in tasks:
                            continue
                        raw_p = str(row.get("Predecessors", "-")).strip()
                        if raw_p and raw_p not in ["-", "None", "nan", ""]:
                            for p_tok in raw_p.split(","):
                                parsed = parse_inline_dep_token(p_tok)
                                if parsed and parsed[0] in tasks:
                                    tasks[act_id]["preds"].append([parsed[0], parsed[1], parsed[2]])

                if use_succs:
                    for _, row in edited_pm_df.iterrows():
                        act_id = str(row["Activity"]).strip()
                        if act_id not in tasks:
                            continue
                        raw_s = str(row.get("Successors", "-")).strip()
                        if raw_s and raw_s not in ["-", "None", "nan", ""]:
                            for s_tok in raw_s.split(","):
                                parsed = parse_inline_dep_token(s_tok)
                                if parsed and parsed[0] in tasks:
                                    tasks[parsed[0]]["preds"].append([act_id, parsed[1], parsed[2]])

                for _, lrow in edited_lags_df.iterrows():
                    u_a = str(lrow.get("From Activity", "")).strip()
                    v_a = str(lrow.get("To Activity", "")).strip()
                    rel_t = str(lrow.get("Relation (FS/SS/FF/SF)", "FS")).strip().upper()[:2]
                    lag_num = float(lrow.get("Lag Time", 0.0)) if pd.notna(lrow.get("Lag Time", 0.0)) else 0.0
                    if u_a in tasks and v_a in tasks:
                        replaced = False
                        for p_item in tasks[v_a]["preds"]:
                            if p_item[0] == u_a:
                                p_item[1] = rel_t
                                p_item[2] = lag_num
                                replaced = True
                        if not replaced:
                            tasks[v_a]["preds"].append([u_a, rel_t, lag_num])

                has_cyc, cyc_nodes, st_nodes, end_nodes = detect_network_cycles_and_dangling(tasks, order_keys)
                if has_cyc:
                    st.error(f"🚨 **تم اكتشاف مسار دائري مغلق (Loop):** `{' ➔ '.join(cyc_nodes)}`")
                else:
                    st.caption(f"✅ **فحص الشبكة:** أنشطة البداية = `{st_nodes}` | أنشطة النهاية = `{end_nodes}`")

                for _ in range(len(order_keys) + 4):
                    for t_id in order_keys:
                        d_curr = tasks[t_id]["dur"]
                        es_cands = [0.0]
                        ef_cands = [d_curr]
                        for p_id, r_type, lag in tasks[t_id]["preds"]:
                            p_es, p_ef = tasks[p_id]["ES"], tasks[p_id]["EF"]
                            if r_type == "FS":
                                es_cands.append(p_ef + lag)
                            elif r_type == "SS":
                                es_cands.append(p_es + lag)
                            elif r_type == "FF":
                                ef_cands.append(p_ef + lag)
                            elif r_type == "SF":
                                ef_cands.append(p_es + lag)
                        es_best = max(es_cands)
                        ef_best = max(max(ef_cands), es_best + d_curr)
                        es_final = max(es_best, ef_best - d_curr)
                        tasks[t_id]["ES"] = es_final
                        tasks[t_id]["EF"] = es_final + d_curr

                proj_dur = max(tasks[k]["EF"] for k in order_keys) if order_keys else 0.0

                for k in order_keys:
                    tasks[k]["LF"] = proj_dur
                    tasks[k]["LS"] = proj_dur - tasks[k]["dur"]

                for _ in range(len(order_keys) + 4):
                    for t_id in reversed(order_keys):
                        d_curr = tasks[t_id]["dur"]
                        lf_cands = [proj_dur]
                        ls_cands = [proj_dur - d_curr]
                        for succ_id in order_keys:
                            for p_id, r_type, lag in tasks[succ_id]["preds"]:
                                if p_id == t_id:
                                    s_ls, s_lf = tasks[succ_id]["LS"], tasks[succ_id]["LF"]
                                    if r_type == "FS":
                                        lf_cands.append(s_ls - lag)
                                    elif r_type == "FF":
                                        lf_cands.append(s_lf - lag)
                                    elif r_type == "SS":
                                        ls_cands.append(s_ls - lag)
                                    elif r_type == "SF":
                                        ls_cands.append(s_lf - lag)
                        lf_best = min(lf_cands)
                        ls_best = min(min(ls_cands), lf_best - d_curr)
                        lf_final = min(lf_best, ls_best + d_curr)
                        tasks[t_id]["LF"] = lf_final
                        tasks[t_id]["LS"] = lf_final - d_curr

                crit_full, crit_partial = [], []
                out_rows = []
                for k in order_keys:
                    d = tasks[k]
                    s_sl = round(d["LS"] - d["ES"], 4)
                    f_sl = round(d["LF"] - d["EF"], 4)
                    d["start_sl"] = s_sl
                    d["finish_sl"] = f_sl
                    d["TF"] = min(s_sl, f_sl)
                    d["crit_start"] = abs(s_sl) < 1e-5
                    d["crit_finish"] = abs(f_sl) < 1e-5
                    d["crit_any"] = d["crit_start"] or d["crit_finish"]

                    if d["crit_start"] and d["crit_finish"]:
                        crit_full.append(k)
                        c_status = "🔴 Full Critical (Start & Finish)"
                    elif d["crit_start"]:
                        crit_partial.append(f"{k} (Start Only)")
                        c_status = "🟠 Start Only Critical"
                    elif d["crit_finish"]:
                        crit_partial.append(f"{k} (Finish Only)")
                        c_status = "🟠 Finish Only Critical"
                    else:
                        c_status = "🟢 Non-Critical"

                    ff_cands = []
                    for succ_id in order_keys:
                        for p_id, r_type, lag in tasks[succ_id]["preds"]:
                            if p_id == k:
                                s_es, s_ef = tasks[succ_id]["ES"], tasks[succ_id]["EF"]
                                if r_type == "FS":
                                    ff_cands.append(s_es - lag - d["EF"])
                                elif r_type == "SS":
                                    ff_cands.append(s_es - lag - d["ES"])
                                elif r_type == "FF":
                                    ff_cands.append(s_ef - lag - d["EF"])
                                elif r_type == "SF":
                                    ff_cands.append(s_ef - lag - d["ES"])
                    d["FF"] = max(0.0, round(min(ff_cands), 4)) if ff_cands else round(proj_dur - d["EF"], 4)

                    out_rows.append({
                        "Activity": k,
                        "Duration (DUR)": d["dur"],
                        "Predecessors Used": ", ".join([f"{p}({rt}{lg:+g})" for p, rt, lg in d["preds"]]) or "-",
                        "ES": d["ES"],
                        "EF": d["EF"],
                        "LS": d["LS"],
                        "LF": d["LF"],
                        "Start Slack (LS-ES)": d["start_sl"],
                        "Finish Slack (LF-EF)": d["finish_sl"],
                        "Free Float (FF)": d["FF"],
                        "Critical Status": c_status,
                    })

                m1, m2, m3 = st.columns(3)
                m1.success(f"⏱️ **Total Project Time:** `{proj_dur:g}`")
                m2.error(f"🔥 **Critical Path:** `{' ➔ '.join(crit_full)}`")
                if crit_partial:
                    m3.warning(f"⚡ **Partial Critical (Start/Finish Only):** `{', '.join(crit_partial)}`")
                else:
                    m3.info("✅ All critical nodes have both Start & Finish critical.")

                st.dataframe(pd.DataFrame(out_rows), use_container_width=True)

                aon_svg_str = build_aon_9cell_svg(tasks, order_keys)
                st.markdown("<h4>📐 1. رسم شبكة AON (صندوق الـ 9 خانات القياسي):</h4>", unsafe_allow_html=True)
                st.markdown(aon_svg_str, unsafe_allow_html=True)
                st.download_button("📥 Download AON (.svg)", aon_svg_str.encode("utf-8"), "AON_Network.svg", "image/svg+xml")

                aoa_svg_str = build_aoa_slide_svg(tasks, order_keys, proj_dur)
                st.markdown("<h4>🏹 2. رسم شبكة الأسهم (AOA Network):</h4>", unsafe_allow_html=True)
                st.markdown(aoa_svg_str, unsafe_allow_html=True)
                st.download_button("📥 Download AOA (.svg)", aoa_svg_str.encode("utf-8"), "AOA_Network.svg", "image/svg+xml")

                add_log("Project Management", pm_preset, edited_pm_df.to_dict(), f"Total Time={proj_dur}, CP={crit_full}")
            except Exception as err:
                st.error(f"❌ Error solving network: {err}")

    render_camera_ai_section(
        "tab1_pm",
        "إدارة المشروعات والشبكات (Project Management: AON / AOA / Lags / Crashing / PERT)",
        "حل أي مسألة شبكات أعمال AON و AOA، العلاقات FS, SS, FF, SF مع الـ Lags، ضغط المشروعات Crashing، وترتيب المسار الحرج.",
        lambda: edited_pm_df.to_string() if "edited_pm_df" in locals() else "Project Network",
        table_state_key="tab1_pm_df",
        table_cols=["Activity", "Description", "Predecessors", "Successors", "Activity Time"],
    )


with TABS[1]:
    st.markdown("""
    <div class='sec-header'>
        <h3>💰 2. الحل العام لجميع مسائل الفوائد والاقتصاد الهندسي | Universal Engineering Economy Solver</h3>
        <p>حدد <b>أي عدد من المشروعات</b> و<b>أي عدد من السنوات</b>، أو ابنِ <b>أي مسألة تدفقات نقدية عامة (P, F, A, G, g)</b> في فترات زمنية مختلفة لحساب القيمة المكافئة عند أي سنة.</p>
    </div>
    """, unsafe_allow_html=True)

    eco_mode_sel = st.radio(
        "💰 Select Economy Solver Mode | اختر نوع مسألة الفوائد والاقتصاد الهندسي:",
        [
            "1. Universal Multi-Project NPV, IRR & Payback Table (حدد أي عدد مشروعات وسنوات + التضخم)",
            "2. Universal Cash Flow & Equivalence Builder (حل أي مسألة فوائد عامة تجمع بين دفعات مفردة ومنتظمة ومتدرجة)",
            "3. Compound Interest Factors & Depreciation Schedule (معاملات الفوائد وجداول الإهلاك SL, SYD, DDB)",
        ],
    )

    if eco_mode_sel.startswith("1."):
        c_cfg1, c_cfg2, c_cfg3, c_cfg4, c_cfg5 = st.columns(5)
        with c_cfg1:
            num_projs = st.number_input("🔢 عدد المشروعات:", min_value=1, max_value=8, value=3, step=1)
        with c_cfg2:
            num_years = st.number_input("📅 عدد السنوات (N):", min_value=1, max_value=25, value=5, step=1)
        with c_cfg3:
            req_ror = st.number_input("📈 معدل العائد r (%):", value=20.0, step=1.0)
        with c_cfg4:
            inf_rate = st.number_input("🎈 معدل التضخم f (%):", value=2.0, step=0.5)
        with c_cfg5:
            load_slide_ex = st.checkbox("📚 تحميل أرقام مثال السلايد (Page 52)", value=False)

        if load_slide_ex and int(num_projs) == 3 and int(num_years) == 5:
            p_titles = ["Time Fades Away", "On the Beach", "Tonight's the Night"]
            init_npv_table = pd.DataFrame([
                {"Year": 0, "Time Fades Away - Investment": 600000.0, "Time Fades Away - Revenue": 0.0, "On the Beach - Investment": 400000.0, "On the Beach - Revenue": 0.0, "Tonight's the Night - Investment": 200000.0, "Tonight's the Night - Revenue": 0.0},
                {"Year": 1, "Time Fades Away - Investment": 0.0, "Time Fades Away - Revenue": 600000.0, "On the Beach - Investment": 0.0, "On the Beach - Revenue": 400000.0, "Tonight's the Night - Investment": 0.0, "Tonight's the Night - Revenue": 200000.0},
                {"Year": 2, "Time Fades Away - Investment": 0.0, "Time Fades Away - Revenue": 75000.0, "On the Beach - Investment": 0.0, "On the Beach - Revenue": 100000.0, "Tonight's the Night - Investment": 0.0, "Tonight's the Night - Revenue": 125000.0},
                {"Year": 3, "Time Fades Away - Investment": 0.0, "Time Fades Away - Revenue": 20000.0, "On the Beach - Investment": 0.0, "On the Beach - Revenue": 25000.0, "Tonight's the Night - Investment": 0.0, "Tonight's the Night - Revenue": 75000.0},
                {"Year": 4, "Time Fades Away - Investment": 0.0, "Time Fades Away - Revenue": 15000.0, "On the Beach - Investment": 0.0, "On the Beach - Revenue": 20000.0, "Tonight's the Night - Investment": 0.0, "Tonight's the Night - Revenue": 20000.0},
                {"Year": 5, "Time Fades Away - Investment": 0.0, "Time Fades Away - Revenue": 10000.0, "On the Beach - Investment": 0.0, "On the Beach - Revenue": 10000.0, "Tonight's the Night - Investment": 0.0, "Tonight's the Night - Revenue": 10000.0},
            ])
        else:
            p_titles = [f"Project {j+1}" for j in range(int(num_projs))]
            rows_dyn = []
            for yr in range(int(num_years) + 1):
                r_d = {"Year": yr}
                for p_t in p_titles:
                    r_d[f"{p_t} - Investment"] = 100000.0 if yr == 0 else 0.0
                    r_d[f"{p_t} - Revenue"] = 0.0 if yr == 0 else 35000.0
                rows_dyn.append(r_d)
            init_npv_table = pd.DataFrame(rows_dyn)

        if "tab2_npv_df" in st.session_state.custom_tables:
            init_npv_table = st.session_state.custom_tables["tab2_npv_df"]

        edited_npv_df = st.data_editor(init_npv_table, num_rows="dynamic", use_container_width=True, key="npv_universal_editor")

        if st.button("⚙️ Solve All Projects (NPV Equations, IRR, Payback & Priority)", type="primary"):
            k_disc = (req_ror + inf_rate) / 100.0
            st.info(f"📌 **Combined Discount Rate ($i = r + f$):** `{k_disc*100:.2f}%`")
            results_summary = []
            proj_cfs_map = {}

            for p_title in p_titles:
                inv_col = f"{p_title} - Investment"
                rev_col = f"{p_title} - Revenue"
                npv_total = 0.0
                terms_latex = []
                cfs_list = []
                cum_s, cum_d = 0.0, 0.0
                pb_s, pb_d = None, None

                for _, r_row in edited_npv_df.iterrows():
                    yr = int(r_row["Year"])
                    inv_v = float(r_row.get(inv_col, 0.0)) if pd.notna(r_row.get(inv_col, 0.0)) else 0.0
                    rev_v = float(r_row.get(rev_col, 0.0)) if pd.notna(r_row.get(rev_col, 0.0)) else 0.0
                    net_cf = rev_v - inv_v
                    cfs_list.append(net_cf)
                    pv_term = net_cf / ((1.0 + k_disc) ** yr)
                    npv_total += pv_term

                    ps, pd_prev = cum_s, cum_d
                    cum_s += net_cf
                    cum_d += pv_term
                    if yr > 0 and ps < 0 and cum_s >= 0 and pb_s is None:
                        pb_s = (yr - 1) + abs(ps) / net_cf
                    if yr > 0 and pd_prev < 0 and cum_d >= 0 and pb_d is None:
                        pb_d = (yr - 1) + abs(pd_prev) / pv_term

                    if yr == 0 and net_cf != 0:
                        terms_latex.append(f"{net_cf:,.0f}")
                    elif yr > 0 and net_cf != 0:
                        sign_str = "+" if net_cf > 0 else "-"
                        pwr_str = f"^{{{yr}}}" if yr > 1 else ""
                        terms_latex.append(f"{sign_str} {abs(net_cf):,.0f}\\frac{{1}}{{(1+{k_disc:g}){pwr_str}}}")

                proj_cfs_map[p_title] = cfs_list
                irr_est = 0.15
                for _ in range(45):
                    f_v = sum(cf / ((1.0 + irr_est) ** t) for t, cf in enumerate(cfs_list))
                    df_v = sum(-t * cf / ((1.0 + irr_est) ** (t + 1)) for t, cf in enumerate(cfs_list) if t > 0)
                    if abs(df_v) < 1e-10:
                        break
                    irr_est -= f_v / df_v

                results_summary.append({
                    "Project": p_title, "NPV": npv_total, "IRR_pct": irr_est * 100.0,
                    "Payback_Simple": f"{pb_s:.2f} yrs" if pb_s is not None else "Not Reached",
                    "Payback_Disc": f"{pb_d:.2f} yrs" if pb_d is not None else "Not Reached",
                    "LaTeX": " ".join(terms_latex) + f" = \\${npv_total:,.2f}",
                })

            funded_sorted = sorted([x for x in results_summary if x["NPV"] > 0], key=lambda item: item["NPV"], reverse=True)
            rank_map = {item["Project"]: idx + 1 for idx, item in enumerate(funded_sorted)}

            summary_table = []
            for item in results_summary:
                p_name = item["Project"]
                npv_v = item["NPV"]
                rk = rank_map.get(p_name, None)
                badge_html = (
                    f"<span class='priority-badge-1'>Priority #{rk} (Fund ✅)</span>" if rk == 1
                    else (f"<span class='priority-badge-2'>Priority #{rk} (Fund ✅)</span>" if rk is not None
                          else "<span class='priority-badge-rej'>Reject ❌ (Negative NPV)</span>")
                )
                st.markdown(f"**🔹 {p_name}:** {badge_html}", unsafe_allow_html=True)
                st.latex(f"NPV = {item['LaTeX']}")
                summary_table.append({
                    "Project": p_name,
                    "NPV ($)": round(npv_v, 2),
                    "IRR (%)": round(item["IRR_pct"], 2),
                    "Simple Payback": item["Payback_Simple"],
                    "Discounted Payback": item["Payback_Disc"],
                    "Priority Rank": f"Priority {rk}" if rk else "Rejected",
                })
            st.dataframe(pd.DataFrame(summary_table), use_container_width=True)

            rates_grid = np.linspace(0.0, 0.40, 21)
            sens_rows = [{"Discount Rate (%)": round(rg * 100, 1), **{pt: sum(cf / ((1 + rg) ** t) for t, cf in enumerate(proj_cfs_map[pt])) for pt in p_titles}} for rg in rates_grid]
            st.line_chart(pd.DataFrame(sens_rows).set_index("Discount Rate (%)"))

    elif eco_mode_sel.startswith("2."):
        st.markdown("<h4>🧮 باني مسائل التدفقات النقدية العام (أدخل أي دفعات مفردة أو منتظمة أو متدرجة بأي توقيت):</h4>", unsafe_allow_html=True)
        c_eq1, c_eq2, c_eq3 = st.columns(3)
        with c_eq1:
            gen_i = st.number_input("Interest Rate i (%):", value=10.0, step=0.5)
        with c_eq2:
            gen_horizon = st.number_input("Total Horizon N (Years):", value=10, min_value=1, max_value=60)
        with c_eq3:
            target_t = st.number_input("Target Equivalence Year (t_eq):", value=0, min_value=0, max_value=60)

        init_gen_cf = pd.DataFrame([
            {"ItemType (Single/Uniform/ArithGrad/GeomGrad)": "Single", "Amount ($)": -20000.0, "Start Year": 0, "End Year": 0, "Gradient G or g(%)": 0.0},
            {"ItemType (Single/Uniform/ArithGrad/GeomGrad)": "Uniform", "Amount ($)": 5000.0, "Start Year": 1, "End Year": 6, "Gradient G or g(%)": 0.0},
            {"ItemType (Single/Uniform/ArithGrad/GeomGrad)": "ArithGrad", "Amount ($)": 0.0, "Start Year": 1, "End Year": 6, "Gradient G or g(%)": 500.0},
            {"ItemType (Single/Uniform/ArithGrad/GeomGrad)": "Single", "Amount ($)": 8000.0, "Start Year": 10, "End Year": 10, "Gradient G or g(%)": 0.0},
        ])
        ed_gen_cf = st.data_editor(init_gen_cf, num_rows="dynamic", use_container_width=True, key="gen_cf_builder")

        if st.button("⚙️ Solve Universal Cash Flow Diagram & Find P0, FN, AW, and Value at t_eq", type="primary"):
            i_r = gen_i / 100.0
            N_tot = int(gen_horizon)
            cf_timeline = np.zeros(N_tot + 1, dtype=float)
            for _, row in ed_gen_cf.iterrows():
                itype = str(row["ItemType (Single/Uniform/ArithGrad/GeomGrad)"]).strip().lower()
                amt = float(row["Amount ($)"])
                t1 = max(0, min(N_tot, int(row["Start Year"])))
                t2 = max(t1, min(N_tot, int(row["End Year"])))
                grad = float(row["Gradient G or g(%)"])
                if "single" in itype:
                    cf_timeline[t1] += amt
                elif "uniform" in itype:
                    for t in range(t1, t2 + 1):
                        cf_timeline[t] += amt
                elif "arith" in itype:
                    for t in range(t1, t2 + 1):
                        cf_timeline[t] += amt + grad * (t - t1)
                elif "geom" in itype:
                    g_r = grad / 100.0
                    for t in range(t1, t2 + 1):
                        cf_timeline[t] += amt * ((1.0 + g_r) ** (t - t1))

            pw_0 = sum(cf_timeline[t] / ((1.0 + i_r) ** t) for t in range(N_tot + 1))
            fw_N = pw_0 * ((1.0 + i_r) ** N_tot)
            val_at_teq = pw_0 * ((1.0 + i_r) ** int(target_t))
            ap_f = (i_r * ((1.0 + i_r) ** N_tot)) / (((1.0 + i_r) ** N_tot) - 1.0) if i_r > 0 else (1.0 / N_tot)
            aw_all = pw_0 * ap_f

            st.latex(f"P_0 = {pw_0:,.2f}, \\quad F_{{{N_tot}}} = {fw_N:,.2f}, \\quad A_{{1..{N_tot}}} = {aw_all:,.2f}, \\quad V_{{t={int(target_t)}}} = {val_at_teq:,.2f}")
            df_tl = pd.DataFrame({"Year (t)": range(N_tot + 1), "Net Cash Flow ($)": np.round(cf_timeline, 2), "Present Worth at t=0": [round(cf_timeline[t] / ((1 + i_r) ** t), 2) for t in range(N_tot + 1)]})
            st.dataframe(df_tl, use_container_width=True)
            st.bar_chart(df_tl.set_index("Year (t)")["Net Cash Flow ($)"])

    else:
        init_factors_df = pd.DataFrame([
            {"Case": "Custom 1", "Interest i (%)": 10.0, "Periods n": 5, "Present (P)": 10000.0, "Future (F)": 0.0, "Uniform (A)": 0.0, "Gradient (G)": 0.0},
        ])
        edited_fac_in = st.data_editor(init_factors_df, num_rows="dynamic", use_container_width=True, key="eco_fac_editor")
        if st.button("⚙️ Compute Interest Factors & Equivalent Values", type="primary"):
            out_fac_rows = []
            for _, r in edited_fac_in.iterrows():
                i = float(r["Interest i (%)"]) / 100.0
                n = int(r["Periods n"])
                fp = (1.0 + i) ** n
                pf = 1.0 / fp
                pa = (fp - 1.0) / (i * fp) if i > 0 else float(n)
                ap = 1.0 / pa if pa > 0 else 0.0
                fa = (fp - 1.0) / i if i > 0 else float(n)
                af = 1.0 / fa if fa > 0 else 0.0
                pg = (1.0 / i) * (pa - n / fp) if i > 0 else 0.0
                ag = (1.0 / i) - (n / (fp - 1.0)) if i > 0 else 0.0
                p_eq = float(r["Present (P)"]) + float(r["Future (F)"]) * pf + float(r["Uniform (A)"]) * pa + float(r["Gradient (G)"]) * pg
                out_fac_rows.append({
                    "Case": r["Case"], "i (%)": r["Interest i (%)"], "n": n,
                    "(F/P)": round(fp, 5), "(P/F)": round(pf, 5),
                    "(P/A)": round(pa, 5), "(A/P)": round(ap, 5),
                    "(F/A)": round(fa, 5), "(A/F)": round(af, 5),
                    "(P/G)": round(pg, 5), "(A/G)": round(ag, 5),
                    "Equivalent PW": round(p_eq, 2), "Equivalent FW": round(p_eq * fp, 2), "Equivalent AW": round(p_eq * ap, 2),
                })
            st.dataframe(pd.DataFrame(out_fac_rows), use_container_width=True)

    render_camera_ai_section(
        "tab2_eco",
        "الفوائد وتقييم المشروعات والاقتصاد الهندسي (Universal Engineering Economy)",
        "حل أي مسألة اقتصاد هندسي وفوائد مركبة، تقييم المشروعات NPV, IRR, Payback, B/C، التضخم، والإهلاك.",
        lambda: "Economy Problem",
        table_state_key="tab2_npv_df",
        table_cols=list(init_npv_table.columns) if eco_mode_sel.startswith("1.") else None,
    )


with TABS[2]:
    st.markdown("""
    <div class='sec-header'>
        <h3>🧮 3. الحل العام لجميع مسائل الطرق العددية | Universal Numerical Methods Solver</h3>
        <p>حدد <b>حجم النظام $N \times N$</b> لأي عدد من المعادلات الخطية، أو أدخل <b>أي دالة غير خطية $f(x)$</b>، أو <b>أي جدول نقاط $(x, y)$</b> للاستكمال والتكامل العددي وتوفيق المنحنيات، أو <b>أي معادلة تفاضلية</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    num_mode_tab = st.radio(
        "🧮 Select Numerical Solver | اختر نوع مسألة الطرق العددية:",
        [
            "1. Universal Linear System N×N [A | B] (Gauss-Seidel, Jacobi, SOR, Gaussian Elimination, LU & Cramer)",
            "2. Universal Root Finder (Bisection, False Position, Newton-Raphson, Secant, Fixed-Point g(x))",
            "3. Universal Tabular Data (Newton Divided Differences, Lagrange, Least-Squares Regression & Integration)",
            "4. Universal ODE Initial Value Problem (Euler, Heun RK2, Midpoint & Runge-Kutta RK4)",
        ],
    )

    if num_mode_tab.startswith("1."):
        c_n1, c_n2, c_n3, c_n4 = st.columns(4)
        with c_n1:
            sys_dim_n = st.number_input("🔢 عدد المعادلات (N):", min_value=2, max_value=10, value=3, step=1)
        with c_n2:
            iter_algo = st.selectbox("الخوارزمية:", ["Gauss-Seidel", "Jacobi", "Direct LU & Cramer's Rule"])
        with c_n3:
            sor_omega = st.number_input("معامل الاسترخاء (ω):", value=1.0, min_value=0.1, max_value=1.95, step=0.05)
        with c_n4:
            tol_eps = st.number_input("نسبة الخطأ المسموحة (%):", value=0.001, format="%.5f")

        n_int_dim = int(sys_dim_n)
        default_mat_rows = []
        for i in range(n_int_dim):
            r_dict = {f"x{j+1}": (10.0 if i == j else -1.0) for j in range(n_int_dim)}
            r_dict["Constant B"] = float(i + 5)
            default_mat_rows.append(r_dict)
        init_mat_df = pd.DataFrame(default_mat_rows)

        if "tab3_mat_df" in st.session_state.custom_tables and len(st.session_state.custom_tables["tab3_mat_df"].columns) == n_int_dim + 1:
            init_mat_df = st.session_state.custom_tables["tab3_mat_df"]

        ed_mat_df = st.data_editor(init_mat_df, num_rows="dynamic", use_container_width=True, key=f"num_mat_ed_{n_int_dim}")
        auto_piv = st.checkbox("✅ تفعيل الترتيب التلقائي للصفوف لتحقيق القطر المهيمن (Auto-Pivoting)", value=True)

        if st.button("⚙️ Solve N×N Linear System & Show Full Steps", type="primary"):
            M_arr = ed_mat_df.to_numpy(dtype=float)
            n_r = M_arr.shape[0]
            A_m = M_arr[:, :n_r].copy()
            B_v = M_arr[:, -1].copy()
            if auto_piv:
                for i in range(n_r):
                    mr = i + int(np.argmax(np.abs(A_m[i:, i])))
                    A_m[[i, mr]], B_v[[i, mr]] = A_m[[mr, i]], B_v[[mr, i]]

            exact_x = np.linalg.solve(A_m, B_v)
            det_a = float(np.linalg.det(A_m))
            st.info(f"📌 **Determinant $\\det(A) =$** `{det_a:.6f}` | **Exact Solution:** `{np.round(exact_x, 6)}`")

            if "Direct" not in iter_algo:
                x_v = np.zeros(n_r)
                iter_log = []
                for it in range(1, 50):
                    x_old = x_v.copy()
                    for i in range(n_r):
                        if iter_algo == "Gauss-Seidel":
                            s1 = np.dot(A_m[i, :i], x_v[:i])
                            s2 = np.dot(A_m[i, i + 1 :], x_old[i + 1 :])
                            x_gs = (B_v[i] - s1 - s2) / A_m[i, i]
                            x_v[i] = (1.0 - sor_omega) * x_old[i] + sor_omega * x_gs
                        else:
                            s = np.dot(A_m[i, :], x_old) - A_m[i, i] * x_old[i]
                            x_v[i] = (B_v[i] - s) / A_m[i, i]
                    ea_pct = float(np.max(np.abs((x_v - x_old) / np.where(np.abs(x_v) > 1e-12, x_v, 1.0)))) * 100.0
                    et_pct = float(np.max(np.abs((exact_x - x_v) / np.where(np.abs(exact_x) > 1e-12, exact_x, 1.0)))) * 100.0
                    iter_log.append({
                        "Iter": it,
                        **{f"x{j+1}": round(x_v[j], 6) for j in range(n_r)},
                        "εa (%)": round(ea_pct, 5),
                        "εt (%)": round(et_pct, 5),
                    })
                    if ea_pct < tol_eps:
                        break
                st.dataframe(pd.DataFrame(iter_log), use_container_width=True)

    elif num_mode_tab.startswith("2."):
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        with rc1:
            r_method = st.selectbox("Method:", ["Newton-Raphson", "Bisection", "False Position", "Secant", "Fixed-Point x = g(x)"])
        with rc2:
            f_in_str = st.text_input("f(x) or g(x):", value="x**3 - 2*x - 5")
        with rc3:
            x0_in = st.number_input("a / x0:", value=2.0, format="%.4f")
        with rc4:
            x1_in = st.number_input("b / x1:", value=3.0, format="%.4f")
        with rc5:
            max_it_r = st.number_input("Max Iterations:", value=15, min_value=1, max_value=100)

        if st.button("⚙️ Solve Root & Generate Iteration Table", type="primary"):
            a, b = float(x0_in), float(x1_in)
            r_rows = []
            for it in range(1, int(max_it_r) + 1):
                if r_method == "Newton-Raphson":
                    fx = float(eval_math_expr(f_in_str, x=a))
                    dfx = float(numerical_derivative(f_in_str, a))
                    xn = a - fx / dfx
                    ea = abs((xn - a) / xn) * 100.0 if xn != 0 else 0.0
                    r_rows.append({"Iter": it, "x_i": round(a, 6), "f(x_i)": round(fx, 6), "f'(x_i)": round(dfx, 6), "x_{i+1}": round(xn, 6), "εa (%)": round(ea, 5)})
                    a = xn
                elif r_method == "Fixed-Point x = g(x)":
                    xn = float(eval_math_expr(f_in_str, x=a))
                    ea = abs((xn - a) / xn) * 100.0 if xn != 0 else 0.0
                    r_rows.append({"Iter": it, "x_i": round(a, 6), "x_{i+1} = g(x_i)": round(xn, 6), "εa (%)": round(ea, 5)})
                    a = xn
                elif r_method in ["Bisection", "False Position"]:
                    fa, fb = float(eval_math_expr(f_in_str, x=a)), float(eval_math_expr(f_in_str, x=b))
                    xr = 0.5 * (a + b) if r_method == "Bisection" else (b - fb * (b - a) / (fb - fa))
                    fxr = float(eval_math_expr(f_in_str, x=xr))
                    ea = abs(b - a) / abs(xr) * 50.0 if xr != 0 else 0.0
                    r_rows.append({"Iter": it, "a": round(a, 6), "b": round(b, 6), "x_r": round(xr, 6), "f(x_r)": round(fxr, 6), "εa (%)": round(ea, 5)})
                    if fa * fxr < 0:
                        b = xr
                    else:
                        a = xr
                else:
                    fa, fb = float(eval_math_expr(f_in_str, x=a)), float(eval_math_expr(f_in_str, x=b))
                    xn = b - fb * (b - a) / (fb - fa)
                    ea = abs((xn - b) / xn) * 100.0 if xn != 0 else 0.0
                    r_rows.append({"Iter": it, "x_{i-1}": round(a, 6), "x_i": round(b, 6), "x_{i+1}": round(xn, 6), "f(x_{i+1})": round(float(eval_math_expr(f_in_str, x=xn)), 6), "εa (%)": round(ea, 5)})
                    a, b = b, xn
                if ea < 1e-5:
                    break
            st.dataframe(pd.DataFrame(r_rows), use_container_width=True)

    elif num_mode_tab.startswith("3."):
        init_pts_df = pd.DataFrame([
            {"x": 0.0, "y = f(x)": 1.0},
            {"x": 1.0, "y = f(x)": 2.5},
            {"x": 2.0, "y = f(x)": 5.0},
            {"x": 3.0, "y = f(x)": 10.5},
        ])
        ed_pts_df = st.data_editor(init_pts_df, num_rows="dynamic", use_container_width=True, key="num_pts_editor")
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            x_eval_pt = st.number_input("🎯 قيمة x المطلوب حساب f(x) عندها:", value=1.5)
        with c_p2:
            reg_deg = st.number_input("📈 درجة كثيرة حدود Regression (Least-Squares):", value=2, min_value=1, max_value=6)

        if st.button("⚙️ Solve Divided Differences, Lagrange, Regression & Integration", type="primary"):
            xv = ed_pts_df["x"].to_numpy(dtype=float)
            yv = ed_pts_df["y = f(x)"].to_numpy(dtype=float)
            n_p = len(xv)
            dd = np.zeros((n_p, n_p + 1))
            dd[:, 0], dd[:, 1] = xv, yv
            for j in range(2, n_p + 1):
                for i in range(n_p - j + 1):
                    dd[i, j] = (dd[i + 1, j - 1] - dd[i, j - 1]) / (dd[i + j - 1, 0] - dd[i, 0])
            cols_dd = ["x", "f(x)"] + [f"Δ^{k}f" for k in range(1, n_p)]
            st.dataframe(pd.DataFrame(np.round(dd, 6), columns=cols_dd), use_container_width=True)

            val_interp = dd[0, 1]
            prod_term = 1.0
            for k in range(1, n_p):
                prod_term *= (x_eval_pt - xv[k - 1])
                val_interp += dd[0, k + 1] * prod_term
            trap_tab = np.trapz(yv, xv)
            poly_reg = np.polyfit(xv, yv, min(int(reg_deg), n_p - 1))
            st.success(f"📌 **Interpolated $P({x_eval_pt}) =$** `{val_interp:.6f}` | **Trapezoidal Integral $\\int y\\,dx =$** `{trap_tab:.6f}` | **Least-Squares Poly Coeffs:** `{np.round(poly_reg, 5)}`")

    else:
        rk_in_df = pd.DataFrame([
            {"Function f(x, y)": "x + y", "x0": 0.0, "y0": 1.0, "Step h": 0.1, "Steps N": 10},
        ])
        ed_rk_df = st.data_editor(rk_in_df, use_container_width=True, key="rk_table_editor")
        if st.button("⚙️ Generate Euler, Heun (RK2), Midpoint & RK4 Table", type="primary"):
            r0 = ed_rk_df.iloc[0]
            f_str = str(r0["Function f(x, y)"])
            xv = float(r0["x0"])
            ye = yh = ym = yrk = float(r0["y0"])
            h = float(r0["Step h"])
            steps = int(r0["Steps N"])
            rows_ode = [{"Step": 0, "x": xv, "Euler y": ye, "Heun (RK2) y": yh, "Midpoint y": ym, "RK4 y": yrk}]
            for s_i in range(1, steps + 1):
                ye_n = ye + h * float(eval_math_expr(f_str, x=xv, y=ye))
                k1h = float(eval_math_expr(f_str, x=xv, y=yh))
                k2h = float(eval_math_expr(f_str, x=xv + h, y=yh + h * k1h))
                yh_n = yh + 0.5 * h * (k1h + k2h)

                k1m = float(eval_math_expr(f_str, x=xv, y=ym))
                ym_n = ym + h * float(eval_math_expr(f_str, x=xv + 0.5 * h, y=ym + 0.5 * h * k1m))

                k1 = h * float(eval_math_expr(f_str, x=xv, y=yrk))
                k2 = h * float(eval_math_expr(f_str, x=xv + 0.5 * h, y=yrk + 0.5 * k1))
                k3 = h * float(eval_math_expr(f_str, x=xv + 0.5 * h, y=yrk + 0.5 * k2))
                k4 = h * float(eval_math_expr(f_str, x=xv + h, y=yrk + k3))
                yrk_n = yrk + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                xv = round(xv + h, 5)
                ye, yh, ym, yrk = ye_n, yh_n, ym_n, yrk_n
                rows_ode.append({"Step": s_i, "x": xv, "Euler y": round(ye, 6), "Heun (RK2) y": round(yh, 6), "Midpoint y": round(ym, 6), "RK4 y": round(yrk, 6)})
            st.dataframe(pd.DataFrame(rows_ode), use_container_width=True)

    render_camera_ai_section(
        "tab3_num",
        "الطرق العددية والجداول التكرارية (Universal Numerical Methods)",
        "حل أي مسألة طرق عددية بالجداول الكاملة وحساب نسب الخطأ التقريبية والحقيقية.",
        lambda: num_mode_tab,
    )


with TABS[3]:
    st.markdown("""
    <div class='sec-header'>
        <h3>📈 4. الحل العام لتحويلات لابلاس والكسور الجزئية والمعادلات التفاضلية | Universal ODEs & Laplace Solver</h3>
        <p>أدخل معاملات <b>أي دالة كسرية $F(s) = N(s)/D(s)$</b> لتفكيكها إلى كسور جزئية وإيجاد تحويل لابلاس العكسي $\mathcal{L}^{-1}\{F(s)\}$، أو حل <b>أي معادلة تفاضلية من أي رتبة</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    c_lap1, c_lap2 = st.columns(2)
    with c_lap1:
        num_lap_str = st.text_input("🔢 معاملات البسط Numerator N(s) (من أعلى أس إلى الحد المطلق):", value="2, 5")
    with c_lap2:
        den_lap_str = st.text_input("🔢 معاملات المقام Denominator D(s) (مثال s^2 + 5s + 6 ➔ 1, 5, 6):", value="1, 5, 6")

    if st.button("⚙️ Decompose Partial Fractions & Find Inverse Laplace f(t)", type="primary"):
        num_p = np.array([float(x.strip()) for x in num_lap_str.split(",") if x.strip()])
        den_p = np.array([float(x.strip()) for x in den_lap_str.split(",") if x.strip()])
        poles_s = np.roots(den_p)
        d_den_p = np.polyder(den_p)
        terms_ft = []
        pf_rows = []
        for idx, p_val in enumerate(poles_s):
            res_val = np.polyval(num_p, p_val) / np.polyval(d_den_p, p_val)
            pf_rows.append({"Pole (s_i)": str(np.round(p_val, 4)), "Residue (K_i)": str(np.round(res_val, 4))})
            if abs(p_val.imag) < 1e-6:
                terms_ft.append(f"({res_val.real:.4f}) e^{{{p_val.real:.4f}t}}")
            else:
                terms_ft.append(f"({np.round(res_val, 3)}) e^{{({np.round(p_val, 3)})t}}")
        st.dataframe(pd.DataFrame(pf_rows), use_container_width=True)
        st.latex("f(t) = \\mathcal{L}^{-1}\\{F(s)\\} = " + " + ".join(terms_ft))

    ode_tab_df = pd.DataFrame([
        {"Case": "Custom ODE", "a (y'')": 1.0, "b (y')": 4.0, "c (y)": 13.0, "y(0)": 1.0, "y'(0)": 0.0},
    ])
    ed_ode_tab = st.data_editor(ode_tab_df, num_rows="dynamic", use_container_width=True, key="ode_tab_editor")
    if st.button("⚙️ Solve All ODE Rows in Table", type="primary"):
        ode_res = []
        for _, r in ed_ode_tab.iterrows():
            rts = np.roots([float(r["a (y'')"]), float(r["b (y')"]), float(r["c (y)"])])
            ode_res.append({
                "Case": r["Case"],
                "Root r1": str(np.round(rts[0], 4)),
                "Root r2": str(np.round(rts[1], 4)),
                "Damping Type": "Complex (Underdamped)" if not np.isreal(rts[0]) else ("Repeated (Critical)" if abs(rts[0] - rts[1]) < 1e-6 else "Real Distinct (Overdamped)"),
            })
        st.dataframe(pd.DataFrame(ode_res), use_container_width=True)

    render_camera_ai_section(
        "tab4_math",
        "المعادلات التفاضلية وتحويلات لابلاس وفورييه (Universal ODEs, Laplace & Calculus)",
        "حل أي معادلة تفاضلية أو تكامل أو تحويل لابلاس وفورييه خطوة بخطوة.",
        lambda: f"Laplace N={num_lap_str}, D={den_lap_str}",
    )


with TABS[4]:
    st.markdown("""
    <div class='sec-header'>
        <h3>⚙️ 5. الحل العام للكمرات متعددة الأحمال ودائرة مور | Universal Multi-Load Beam SFD/BMD & Mohr's Circle</h3>
        <p>أضف <b>أي عدد من الأحمال المركزية أو الموزعة أو العزوم</b> في أي موضع على الكمرة، وسيقوم البرنامج بحساب ردود الأفعال ورسم منحنيي القص والعزوم <b>SFD & BMD</b> لأي مسألة.</p>
    </div>
    """, unsafe_allow_html=True)

    mech_choice = st.radio("⚙️ اختر الأداة:", ["1. Universal Multi-Load Beam Solver (SFD & BMD)", "2. Universal Stress Transformation & Mohr's Circle"], horizontal=True)

    if mech_choice.startswith("1."):
        b_c1, b_c2, b_c3 = st.columns(3)
        with b_c1:
            beam_L = st.number_input("📏 طول الكمرة الكلي L (m):", value=8.0, min_value=0.5)
        with b_c2:
            supp_A_x = st.number_input("📍 موضع الركيزة الأولى x_A (m):", value=0.0, min_value=0.0)
        with b_c3:
            supp_B_x = st.number_input("📍 موضع الركيزة الثانية x_B (m):", value=8.0, min_value=0.1)

        init_loads_df = pd.DataFrame([
            {"Load Type (Point / UDL / Moment)": "Point", "Magnitude (kN or kN/m or kN.m)": 20.0, "Start Position x1 (m)": 2.0, "End Position x2 (m for UDL)": 2.0},
            {"Load Type (Point / UDL / Moment)": "UDL", "Magnitude (kN or kN/m or kN.m)": 10.0, "Start Position x1 (m)": 2.0, "End Position x2 (m for UDL)": 6.0},
            {"Load Type (Point / UDL / Moment)": "Point", "Magnitude (kN or kN/m or kN.m)": 15.0, "Start Position x1 (m)": 6.0, "End Position x2 (m for UDL)": 6.0},
        ])
        ed_loads = st.data_editor(init_loads_df, num_rows="dynamic", use_container_width=True, key="beam_loads_editor")

        if st.button("⚙️ Solve Beam Reactions & Plot SFD / BMD for All Loads", type="primary"):
            tot_Fy = 0.0
            sum_M_about_A = 0.0
            for _, r in ed_loads.iterrows():
                ltype = str(r["Load Type (Point / UDL / Moment)"]).strip().lower()
                mag = float(r["Magnitude (kN or kN/m or kN.m)"])
                x1 = float(r["Start Position x1 (m)"])
                x2 = float(r["End Position x2 (m for UDL)"])
                if "point" in ltype:
                    tot_Fy += mag
                    sum_M_about_A += mag * (x1 - supp_A_x)
                elif "udl" in ltype:
                    span_w = max(0.0, x2 - x1)
                    f_eq = mag * span_w
                    x_cent = x1 + 0.5 * span_w
                    tot_Fy += f_eq
                    sum_M_about_A += f_eq * (x_cent - supp_A_x)
                elif "moment" in ltype:
                    sum_M_about_A += mag

            span_AB = supp_B_x - supp_A_x if abs(supp_B_x - supp_A_x) > 1e-6 else 1.0
            R_B = sum_M_about_A / span_AB
            R_A = tot_Fy - R_B
            st.success(f"⚖️ **Support Reactions:** $R_A = {R_A:.3f}\\,\\text{{kN}}$ (at $x={supp_A_x}\\,\\text{{m}}$) | $R_B = {R_B:.3f}\\,\\text{{kN}}$ (at $x={supp_B_x}\\,\\text{{m}}$)")

            x_grid = np.linspace(0.0, beam_L, 400)
            V_arr = np.zeros_like(x_grid)
            M_arr = np.zeros_like(x_grid)
            for idx, xv in enumerate(x_grid):
                v_val, m_val = 0.0, 0.0
                if xv >= supp_A_x:
                    v_val += R_A
                    m_val += R_A * (xv - supp_A_x)
                if xv >= supp_B_x:
                    v_val += R_B
                    m_val += R_B * (xv - supp_B_x)
                for _, r in ed_loads.iterrows():
                    ltype = str(r["Load Type (Point / UDL / Moment)"]).strip().lower()
                    mag = float(r["Magnitude (kN or kN/m or kN.m)"])
                    x1 = float(r["Start Position x1 (m)"])
                    x2 = float(r["End Position x2 (m for UDL)"])
                    if "point" in ltype and xv >= x1:
                        v_val -= mag
                        m_val -= mag * (xv - x1)
                    elif "udl" in ltype and xv > x1:
                        eff_end = min(xv, x2)
                        length_active = max(0.0, eff_end - x1)
                        f_act = mag * length_active
                        v_val -= f_act
                        m_val -= f_act * (xv - (x1 + 0.5 * length_active))
                    elif "moment" in ltype and xv >= x1:
                        m_val -= mag
                V_arr[idx] = v_val
                M_arr[idx] = m_val

            st.info(f"📊 **Max Shear $|V|_{{max}} =$** `{np.max(np.abs(V_arr)):.3f} kN` | **Max Bending Moment $|M|_{{max}} =$** `{np.max(np.abs(M_arr)):.3f} kN·m`")
            df_sfd_bmd = pd.DataFrame({"x (m)": np.round(x_grid, 3), "Shear V (kN)": V_arr, "Moment M (kN.m)": M_arr}).set_index("x (m)")
            st.line_chart(df_sfd_bmd)

    else:
        stress_df = pd.DataFrame([
            {"Element": "Custom Point", "σx (MPa)": 80.0, "σy (MPa)": -40.0, "τxy (MPa)": 35.0, "Angle θ (deg)": 30.0},
        ])
        ed_stress = st.data_editor(stress_df, num_rows="dynamic", use_container_width=True, key="stress_editor")
        if st.button("⚙️ Compute Principal Stresses & Draw Mohr's Circle", type="primary"):
            s_out = []
            for _, r in ed_stress.iterrows():
                sx, sy, txy, th = float(r["σx (MPa)"]), float(r["σy (MPa)"]), float(r["τxy (MPa)"]), float(r["Angle θ (deg)"])
                s_avg = 0.5 * (sx + sy)
                rad_m = math.hypot(0.5 * (sx - sy), txy)
                rad_2th = math.radians(2.0 * th)
                sx_p = s_avg + 0.5 * (sx - sy) * math.cos(rad_2th) + txy * math.sin(rad_2th)
                txy_p = -0.5 * (sx - sy) * math.sin(rad_2th) + txy * math.cos(rad_2th)
                s_out.append({
                    "Element": r["Element"], "σ_avg": round(s_avg, 2),
                    "σ1 Max": round(s_avg + rad_m, 2), "σ2 Min": round(s_avg - rad_m, 2),
                    "τ_max": round(rad_m, 2), "θp (deg)": round(0.5 * math.degrees(math.atan2(2 * txy, sx - sy)), 2),
                    "σx' at θ": round(sx_p, 2), "τx'y' at θ": round(txy_p, 2),
                })
            st.dataframe(pd.DataFrame(s_out), use_container_width=True)

    render_camera_ai_section(
        "tab5_mech",
        "الميكانيكا التطبيقية ومقاومة المواد (Universal Beams, Trusses & Mohr's Circle)",
        "حل أي مسألة كمرات (SFD & BMD) أو جمالونات (Trusses) أو دائرة مور للإجهادات.",
        lambda: mech_choice,
    )


with TABS[5]:
    st.markdown("""
    <div class='sec-header'>
        <h3>⚡ 6. الحل العام للدوائر الكهربية (AC/DC Mesh & Nodal) والمحولات | Universal Circuits & Transformers</h3>
        <p>حل <b>أي نظام معادلات عقد أو مسارات ($N \times N$ AC/DC Phasor Circuit)</b> بالممانعات المركبة $R+jX$، أو حسابات المحولات والكفاءة وتنظيم الجهد عند أي حمل.</p>
    </div>
    """, unsafe_allow_html=True)

    el_mode = st.radio("⚡ اختر القسم الكهربي:", ["1. Universal AC/DC Mesh & Nodal Phasor Solver [Z][I] = [V]", "2. Transformer Equivalent Circuit, Efficiency & Voltage Regulation"], horizontal=True)

    if el_mode.startswith("1."):
        st.caption("📌 أدخل الممانعات المركبة بصيغة بايثون مثل `10+5j` أو `8-4j` أو أرقام حقيقية لدوائر التيار المستمر:")
        ac_init = pd.DataFrame([
            {"Z_col1": "10+5j", "Z_col2": "-5j", "Source V": "100+0j"},
            {"Z_col1": "-5j", "Z_col2": "20+10j", "Source V": "0+0j"},
        ])
        ed_ac = st.data_editor(ac_init, num_rows="dynamic", use_container_width=True, key="ac_mesh_editor")
        if st.button("⚙️ Solve AC/DC Currents / Node Voltages (Magnitude & Phase Angle)", type="primary"):
            n_eq = len(ed_ac)
            Z_mat = np.zeros((n_eq, n_eq), dtype=complex)
            V_vec = np.zeros(n_eq, dtype=complex)
            for i in range(n_eq):
                for j in range(n_eq):
                    Z_mat[i, j] = complex(str(ed_ac.iloc[i, j]).replace(" ", ""))
                V_vec[i] = complex(str(ed_ac.iloc[i, -1]).replace(" ", ""))
            I_sol = np.linalg.solve(Z_mat, V_vec)
            phasor_rows = []
            for idx, val_c in enumerate(I_sol, 1):
                mag_v = abs(val_c)
                ang_v = math.degrees(cmath.phase(val_c))
                phasor_rows.append({"Variable": f"I_{idx} / V_{idx}", "Rectangular (Real + j Imag)": f"{val_c.real:.4f} + {val_c.imag:.4f}j", "Magnitude": round(mag_v, 4), "Phase Angle (deg)": round(ang_v, 2)})
            st.dataframe(pd.DataFrame(phasor_rows), use_container_width=True)

    else:
        tr_df = pd.DataFrame([
            {"Transformer": "Custom Unit", "Voc (V)": 220.0, "Ioc (A)": 1.5, "Poc (W)": 120.0, "Vsc (V)": 45.0, "Isc (A)": 10.0, "Psc (W)": 250.0, "Load PF": 0.85},
        ])
        ed_tr = st.data_editor(tr_df, num_rows="dynamic", use_container_width=True, key="tr_table_editor")
        if st.button("⚙️ Compute Equivalent Circuit, Efficiency & Voltage Regulation", type="primary"):
            tr_out = []
            for _, r in ed_tr.iterrows():
                voc, ioc, poc = float(r["Voc (V)"]), float(r["Ioc (A)"]), float(r["Poc (W)"])
                vsc, isc, psc = float(r["Vsc (V)"]), float(r["Isc (A)"]), float(r["Psc (W)"])
                pf_load = float(r.get("Load PF", 0.85))
                pf0 = poc / (voc * ioc)
                rc = voc / (ioc * pf0)
                xm = voc / (ioc * math.sqrt(max(0.0, 1.0 - pf0 ** 2)))
                req = psc / (isc ** 2)
                zeq = vsc / isc
                xeq = math.sqrt(max(0.0, zeq ** 2 - req ** 2))
                sin_load = math.sqrt(max(0.0, 1.0 - pf_load ** 2))
                v_reg_pct = ((isc * (req * pf_load + xeq * sin_load)) / voc) * 100.0
                p_out = voc * isc * pf_load
                eff_pct = (p_out / (p_out + poc + psc)) * 100.0
                tr_out.append({"Unit": r["Transformer"], "Rc (Ω)": round(rc, 2), "Xm (Ω)": round(xm, 2), "Req (Ω)": round(req, 3), "Xeq (Ω)": round(xeq, 3), "V.Reg (%)": round(v_reg_pct, 2), "Efficiency (%)": round(eff_pct, 2)})
            st.dataframe(pd.DataFrame(tr_out), use_container_width=True)

    render_camera_ai_section(
        "tab6_elec",
        "الدوائر الكهربية والمحولات والقوى ثلاثية الأوجه (Universal Electrical Solver)",
        "حل أي مسألة دوائر كهربية أو محولات أو قوى ثلاثية الأوجه.",
        lambda: el_mode,
    )


with TABS[6]:
    st.markdown("""
    <div class='sec-header'>
        <h3>🎛️ 7. الحل العام لهندسة التحكم وجدول راوث | Universal Control Systems & Routh-Hurwitz</h3>
    </div>
    """, unsafe_allow_html=True)

    routh_coeffs_str = st.text_input(
        "🔢 أدخل معاملات معادلة المقام لأي نظام تحكم (a_n s^n + ... + a_0 = 0):",
        value="1, 2, 3, 6, 5, 3",
    )
    if st.button("⚙️ Generate Full Routh-Hurwitz Stability Table & Roots", type="primary"):
        coeffs = [float(x.strip()) for x in routh_coeffs_str.split(",") if x.strip()]
        deg_n = len(coeffs) - 1
        cols_cnt = (len(coeffs) + 1) // 2
        routh = np.zeros((deg_n + 1, cols_cnt), dtype=float)
        routh[0, : len(coeffs[0::2])] = coeffs[0::2]
        routh[1, : len(coeffs[1::2])] = coeffs[1::2]

        for i in range(2, deg_n + 1):
            if np.all(np.abs(routh[i - 1, :]) < 1e-9):
                p_deg = deg_n - i + 2
                for j in range(cols_cnt):
                    pwr = p_deg - 2 * j
                    if pwr >= 0:
                        routh[i - 1, j] = routh[i - 2, j] * pwr
            if abs(routh[i - 1, 0]) < 1e-9:
                routh[i - 1, 0] = 1e-5
            for j in range(cols_cnt - 1):
                a1, b1 = routh[i - 2, 0], routh[i - 1, 0]
                a2, b2 = routh[i - 2, j + 1], routh[i - 1, j + 1]
                routh[i, j] = (b1 * a2 - a1 * b2) / b1

        row_labels = [f"s^{deg_n - i}" for i in range(deg_n + 1)]
        col_labels = [f"Col {j+1}" for j in range(cols_cnt)]
        st.dataframe(pd.DataFrame(np.round(routh, 5), index=row_labels, columns=col_labels), use_container_width=True)

        first_col = routh[:, 0]
        sign_changes = sum(1 for k in range(len(first_col) - 1) if first_col[k] * first_col[k + 1] < 0)
        exact_roots = np.roots(coeffs)
        if sign_changes == 0:
            st.success(f"✅ **System is Stable!** (0 sign changes) | Roots: `{np.round(exact_roots, 4)}`")
        else:
            st.error(f"⚠️ **System is Unstable!** (`{sign_changes}` Right-Half Plane poles) | Roots: `{np.round(exact_roots, 4)}`")

    ctrl_df = pd.DataFrame([
        {"System": "Custom Sys", "wn (rad/s)": 6.0, "Damping Ratio (ζ)": 0.5, "DC Gain (K)": 1.0},
    ])
    ed_ctrl = st.data_editor(ctrl_df, num_rows="dynamic", use_container_width=True, key="ctrl_table_editor")
    if st.button("⚙️ Compute Transient Specifications (Mp%, Tp, Ts, Tr)", type="primary"):
        c_out = []
        for _, r in ed_ctrl.iterrows():
            wn, z = float(r["wn (rad/s)"]), float(r["Damping Ratio (ζ)"])
            wd = wn * math.sqrt(max(0.0, 1.0 - z ** 2))
            mp = 100.0 * math.exp(-z * math.pi / math.sqrt(1.0 - z ** 2)) if 0 < z < 1 else 0.0
            tp = math.pi / wd if wd > 0 else 0.0
            ts = 4.0 / (z * wn) if z > 0 else 0.0
            c_out.append({"System": r["System"], "wd (rad/s)": round(wd, 3), "Overshoot Mp (%)": round(mp, 2), "Peak Time Tp (s)": round(tp, 3), "Settling Time Ts 2% (s)": round(ts, 3)})
        st.dataframe(pd.DataFrame(c_out), use_container_width=True)

    render_camera_ai_section(
        "tab7_ctrl",
        "هندسة التحكم الآلي والـ PLC (Universal Control Systems)",
        "حل أي مسألة تحكم آلي Routh-Hurwitz, Root Locus, Bode, Block Diagrams, PID, و PLC.",
        lambda: f"Routh Polynomial: {routh_coeffs_str}",
    )


with TABS[7]:
    st.markdown("""
    <div class='sec-header'>
        <h3>💻 8. البرمجة الهندسية وسجل التقارير التفاعلي | Code Sandbox & History Table</h3>
    </div>
    """, unsafe_allow_html=True)

    py_code = st.text_area(
        "✍️ Python / NumPy Engineering Sandbox:",
        value="import numpy as np\nA = np.array([[4, 1], [2, 3]])\nprint('Eigenvalues:', np.linalg.eigvals(A))",
        height=100,
    )
    if st.button("▶️ Execute Python Code", type="primary"):
        buf_out = io.StringIO()
        old_out = sys.stdout
        sys.stdout = buf_out
        try:
            exec(py_code, {"np": np, "pd": pd, "math": math, "cmath": cmath})
            st.code(buf_out.getvalue() or "Executed successfully.", language="text")
        except Exception as e:
            st.error(f"Execution Error: {e}")
        finally:
            sys.stdout = old_out

    if st.session_state.history:
        df_h = pd.DataFrame(st.session_state.history)
        st.dataframe(df_h, use_container_width=True)
        st.download_button(
            "📥 Download Full Report (CSV)",
            df_h.to_csv(index=False).encode("utf-8-sig"),
            "Engineering_Solutions_Report.csv",
            "text/csv",
        )

    render_camera_ai_section(
        "tab8_code",
        "البرمجة الهندسية والامتحانات الشاملة (MATLAB / C++ / Python)",
        "كتابة وتصحيح الأكواد البرمجية الهندسية وحل أوراق الامتحانات الهندسية الشاملة.",
        lambda: py_code,
    )
