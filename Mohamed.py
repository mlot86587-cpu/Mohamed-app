import os
import io
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
    page_title="Engineering Solver Hub | المنصة الهندسية المتقدمة",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 97%; }
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
    st.markdown("### ⚙️ الإعدادات الهندسية | System Config")
    key_in = st.text_input(
        "🔑 Gemini API Key:",
        value=st.session_state.api_key,
        type="password",
        help="أدخل مفتاح Google Gemini API لتفعيل الحل الذكي وقراءة صور الكاميرا.",
    )
    st.session_state.api_key = key_in.strip()
    if st.session_state.api_key:
        st.success("✅ AI & Vision Active | الذكاء الاصطناعي مفعل")
    else:
        st.warning("⚠️ Enter API Key for AI | الجداول والحل البرمجي يعملان فوراً")

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


def apply_vision_filter(pil_img, mode_str, rot_angle=0):
    if pil_img is None:
        return None
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


def call_gemini_rest_api(domain_title, domain_guide, text_data, pil_img=None, mode="solve"):
    if not st.session_state.api_key:
        return "❌ **Error:** يرجى إدخال مفتاح `Gemini API Key` في القائمة الجانبية أولاً."
    try:
        if mode == "ocr":
            prompt_text = (
                f"You are an Engineering OCR Scanner for ({domain_title}). "
                "Extract all tables, activities, predecessors, successors, durations, lags, cash flows, and equations accurately from the image."
            )
        else:
            prompt_text = (
                f"أنت بروفيسور في كلية الهندسة متخصص في ({domain_title}).\n"
                f"أسلوب الإجابة المطلوب: {ai_detail_mode}.\n"
                f"توجيهات التخصص: {domain_guide}\n"
                "قم بحل المسألة الهندسية باللغتين العربية والإنجليزية للمصطلحات، متضمناً:\n"
                "1. **جدول المعطيات والمطلوب (Given Table & Required)**.\n"
                "2. **القوانين الهندسية (Governing Equations)** بصيغة LaTeX واضحة.\n"
                "3. **خطوات الحل والجداول التفصيلية (Step-by-Step Calculation & Tables)**.\n"
                "4. **الناتج النهائي والترتيب الهندسي (Final Answer & Priority/Critical Path)**.\n"
                f"\nبيانات الجدول المدخلة:\n{text_data}"
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


def render_camera_ai_section(key_id, domain_title, domain_guide, get_context_fn):
    st.markdown(
        f"<div class='ai-cam-box'><b>📸🤖 وحدة الكاميرا الذكية ومعالجة الصور والـ AI | Camera, Vision & AI Solver — ({domain_title})</b></div>",
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
        proc_img = apply_vision_filter(raw_img, flt, rot)
        ic1, ic2 = st.columns(2)
        with ic1:
            st.image(raw_img, caption="Original Image | الصورة الأصلية", use_container_width=True)
        with ic2:
            st.image(proc_img, caption=f"Processed Image ({flt})", use_container_width=True)

    extra_q = st.text_input(
        "💡 Specific AI Instruction | توجيه خاص للذكاء الاصطناعي (اختياري):",
        placeholder="مثال: اشرح خطوات المسار الأمامي والخلفي، أو رتب المشروعات حسب الـ NPV...",
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
            "📝 Extract Table/Data from Image (OCR) | استخراج الجدول من الصورة",
            use_container_width=True,
            key=f"{key_id}_btn_ocr",
        )

    if ocr_ai_clicked:
        if proc_img is None:
            st.warning("⚠️ يرجى التقاط صورة بالكاميرا أو رفعها أولاً.")
        else:
            with st.spinner("🔍 Scanning image with Gemini Vision OCR..."):
                res_ocr = call_gemini_rest_api(domain_title, domain_guide, extra_q, proc_img, mode="ocr")
                st.markdown("#### 📝 المستخرج البصري | Extracted Problem Data:")
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
                st.markdown(f"#### 🤖 الحل الهندسي التفصيلي | AI Engineering Solution ({domain_title}):")
                st.markdown(res_solve)
                add_log(domain_title, "Gemini AI Solver", ctx_str, res_solve)


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
        f"<svg viewBox='0 0 {svg_w} {svg_h}' style='width:100%; height:auto; background:white; border:1px solid rgb(203,213,225); border-radius:8px;'>",
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
                if r_type in ["SS", "SF"]:
                    sx, sy = x1, y1 + 70
                else:
                    sx, sy = x1 + 150, y1 + 45
                if r_type in ["FF", "SF"]:
                    ex, ey = x2 + 150, y2 + 75
                else:
                    ex, ey = x2, y2 + 45

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
        bg = "rgb(253,242,248)" if is_c else "rgb(248,250,252)"
        bd = "rgb(219,39,119)" if is_c else "rgb(30,58,138)"
        center_fill = "rgb(236,72,153)" if is_c else "rgb(219,234,254)"
        desc_short = str(d.get("desc", ""))[:12]

        svg_lines.append(f"<rect x='{x}' y='{y}' width='150' height='90' rx='4' fill='{bg}' stroke='{bd}' stroke-width='2.2'/>")
        svg_lines.append(f"<rect x='{x+45}' y='{y+30}' width='60' height='30' fill='{center_fill}' opacity='0.35'/>")
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
        f"<svg viewBox='0 0 {svg_w} {svg_h}' style='width:100%; height:auto; background:rgb(253,242,248); border:1px solid rgb(219,39,119); border-radius:8px;'>",
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


st.markdown("""
<div class='eng-banner'>
    <h1>⚙️ المنصة الهندسية الجامعية المتقدمة | Advanced Engineering Solver Platform</h1>
    <p>Interactive Editable Tables (st.data_editor) Across All Engineering Courses • AON (9-Cell Lag Box) & AOA Networks • Multi-Project NPV & Inflation Priority Table • Numerical Methods • Mechanics & Control • Live Camera & AI</p>
</div>
""", unsafe_allow_html=True)

TABS = st.tabs([
    "1️⃣ شبكات الأعمال (AON / AOA / Lags)",
    "2️⃣ الفوائد وتقييم المشروعات (NPV & Economy)",
    "3️⃣ الطرق العددية (Numerical Tables)",
    "4️⃣ التفاضل ولابلاس (ODEs & Laplace)",
    "5️⃣ الميكانيكا والكمرات (Mohr & Beams)",
    "6️⃣ الدوائر والمحولات (Circuits & Power)",
    "7️⃣ التحكم الآلي (Control Systems)",
    "8️⃣ البرمجة والتقارير (Code & History)",
])


with TABS[0]:
    st.markdown("""
    <div class='sec-header'>
        <h3>📊 1. شبكات الأعمال والمسار الحرج | Project Management Networks (AON, AOA, Lags & PERT)</h3>
        <p>املأ الجدول التفاعلي مباشرة كما في المحاضرة (سواء بالاعتماديات السابقة <b>Predecessors</b> أو اللاحقة <b>Successors</b>، وبالوضع الرأسي أو الأفقي، مع جدول <b>Lags</b> وصندوق الـ 9 خانات القياسي).</p>
    </div>
    """, unsafe_allow_html=True)

    c_pm1, c_pm2, c_pm3 = st.columns([1.6, 1.2, 1.2])
    with c_pm1:
        pm_preset = st.selectbox(
            "📚 Load Lecture Slide Example | اختر مثالاً جاهزاً من المحاضرة:",
            [
                "1. Example 4 (Slide 21 - KOLL Business Center AOA Network)",
                "2. Figure 6.21 (Slide 47 - Network Using Lags A-to-F)",
                "3. Example 6.20 (Slide 48 - Lag Exercises A-to-K)",
                "4. Example 3 (Slide 31 - Steel Cylinder AON with Successors A-to-T)",
                "5. Example 6 (Slide 35 - AOA Network A-to-R)",
                "6. PERT Probabilistic Table (to, tm, tp)",
            ],
        )
    with c_pm2:
        table_orient = st.radio(
            "📐 شكل الجدول | Table Layout:",
            ["Vertical Table | جدول رأسي", "Horizontal Table | جدول أفقي"],
            horizontal=True,
        )
    with c_pm3:
        dep_mode = st.radio(
            "🔗 نوع الرابط | Dependency Column:",
            ["Predecessors (السابق)", "Successors (اللاحق)"],
            horizontal=True,
            index=1 if "Example 3" in pm_preset else 0,
        )

    if "Example 4" in pm_preset:
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
            {"Activity": "A", "Description": "Design system", "Predecessors": "-", "Successors": "B, D, E, F", "Activity Time": 5.0},
            {"Activity": "B", "Description": "Order hardware", "Predecessors": "A", "Successors": "C", "Activity Time": 1.0},
            {"Activity": "C", "Description": "Install hardware", "Predecessors": "B", "Successors": "D", "Activity Time": 2.0},
            {"Activity": "D", "Description": "Install software", "Predecessors": "A, C", "Successors": "E, F", "Activity Time": 6.0},
            {"Activity": "E", "Description": "Test system", "Predecessors": "A, D", "Successors": "F", "Activity Time": 2.0},
            {"Activity": "F", "Description": "System documentation", "Predecessors": "A, D, E", "Successors": "-", "Activity Time": 3.0},
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
            {"Activity": "A", "Description": "Task A", "Predecessors": "-", "Successors": "B, C, D", "Activity Time": 2.0},
            {"Activity": "B", "Description": "Task B", "Predecessors": "A", "Successors": "E", "Activity Time": 4.0},
            {"Activity": "C", "Description": "Task C", "Predecessors": "A", "Successors": "E, F", "Activity Time": 8.0},
            {"Activity": "D", "Description": "Task D", "Predecessors": "A", "Successors": "F, G", "Activity Time": 9.0},
            {"Activity": "E", "Description": "Task E", "Predecessors": "B, C", "Successors": "H", "Activity Time": 2.0},
            {"Activity": "F", "Description": "Task F", "Predecessors": "C, D", "Successors": "H, I", "Activity Time": 7.0},
            {"Activity": "G", "Description": "Task G", "Predecessors": "D", "Successors": "J", "Activity Time": 4.0},
            {"Activity": "H", "Description": "Task H", "Predecessors": "E, F", "Successors": "K", "Activity Time": 5.0},
            {"Activity": "I", "Description": "Task I", "Predecessors": "F", "Successors": "K", "Activity Time": 4.0},
            {"Activity": "J", "Description": "Task J", "Predecessors": "G", "Successors": "K", "Activity Time": 7.0},
            {"Activity": "K", "Description": "Task K", "Predecessors": "H, I, J", "Successors": "-", "Activity Time": 3.0},
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

    if "6. PERT" not in pm_preset:
        st.markdown("#### 📋 1. الجدول التفاعلي لأنشطة المشروع | Interactive Project Activities Table:")
        if "Vertical" in table_orient:
            edited_pm_df = st.data_editor(init_pm_df, num_rows="dynamic", use_container_width=True, key="pm_vert_editor")
        else:
            horiz_init = init_pm_df.set_index("Activity")[["Predecessors" if "Predecessors" in dep_mode else "Successors", "Activity Time"]].T
            edited_horiz = st.data_editor(horiz_init, use_container_width=True, key="pm_horiz_editor")
            reconstructed = []
            for col_act in edited_horiz.columns:
                dep_val = str(edited_horiz.loc[edited_horiz.index[0], col_act])
                t_val = float(edited_horiz.loc["Activity Time", col_act])
                reconstructed.append({
                    "Activity": str(col_act),
                    "Description": str(col_act),
                    "Predecessors": dep_val if "Predecessors" in dep_mode else "-",
                    "Successors": dep_val if "Successors" in dep_mode else "-",
                    "Activity Time": t_val,
                })
            edited_pm_df = pd.DataFrame(reconstructed)

        with st.expander("🔗 جدول علاقات التأخير والتداخل (Lags Table: FS, SS, FF, SF) — اضغط للفتح أو التعديل", expanded=("Lag" in pm_preset or "6.21" in pm_preset)):
            edited_lags_df = st.data_editor(init_lags_df, num_rows="dynamic", use_container_width=True, key="pm_lags_editor")

        if st.button("⚙️ Solve Network Table & Generate AON (9-Cell) / AOA / Schedule", type="primary"):
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

                if "Predecessors" in dep_mode:
                    for _, row in edited_pm_df.iterrows():
                        act_id = str(row["Activity"]).strip()
                        if act_id not in tasks:
                            continue
                        raw_p = str(row.get("Predecessors", "-")).strip()
                        if raw_p and raw_p not in ["-", "None", "nan", ""]:
                            for p_tok in raw_p.split(","):
                                p_clean = p_tok.strip()
                                if p_clean in tasks:
                                    tasks[act_id]["preds"].append([p_clean, "FS", 0.0])
                else:
                    for _, row in edited_pm_df.iterrows():
                        act_id = str(row["Activity"]).strip()
                        if act_id not in tasks:
                            continue
                        raw_s = str(row.get("Successors", "-")).strip()
                        if raw_s and raw_s not in ["-", "None", "nan", ""]:
                            for s_tok in raw_s.split(","):
                                s_clean = s_tok.strip()
                                if s_clean in tasks:
                                    tasks[s_clean]["preds"].append([act_id, "FS", 0.0])

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
                        c_status = "🟠 Start Only on Critical Path"
                    elif d["crit_finish"]:
                        crit_partial.append(f"{k} (Finish Only)")
                        c_status = "🟠 Finish Only on Critical Path"
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
                        "Description": d["desc"],
                        "Duration (DUR)": d["dur"],
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
                m1.success(f"⏱️ **Total Time:** `{proj_dur:g}`")
                m2.error(f"🔥 **Critical Path:** `{' ➔ '.join(crit_full)}`")
                if crit_partial:
                    m3.warning(f"⚡ **Start/Finish Only Critical:** `{', '.join(crit_partial)}`")
                else:
                    m3.info("✅ All critical activities have both Start & Finish critical.")

                st.markdown("#### 📊 جدول المخرجات الكامل (Detailed Schedule & Floats Table):")
                st.dataframe(pd.DataFrame(out_rows), use_container_width=True)

                st.markdown("#### 📐 1. رسم شبكة AON بصندوق الـ 9 خانات القياسي (3×3 Lag Node Legend: ES|ID|EF / SL|Desc|SL / LS|DUR|LF):")
                st.markdown(build_aon_9cell_svg(tasks, order_keys), unsafe_allow_html=True)

                st.markdown("#### 🏹 2. رسم شبكة الأسهم (AOA Network - Forward Pass, Backward Pass & Slack Boxes):")
                st.markdown(build_aoa_slide_svg(tasks, order_keys, proj_dur), unsafe_allow_html=True)

                add_log("Project Management", pm_preset, edited_pm_df.to_dict(), f"Total Time={proj_dur}, CP={crit_full}")
            except Exception as err:
                st.error(f"❌ Error solving network: {err}")

    else:
        pert_init = pd.DataFrame([
            {"Activity": "A", "Optimistic (to)": 3.0, "Most Likely (tm)": 5.0, "Pessimistic (tp)": 13.0},
            {"Activity": "B", "Optimistic (to)": 4.0, "Most Likely (tm)": 6.0, "Pessimistic (tp)": 8.0},
            {"Activity": "C", "Optimistic (to)": 5.0, "Most Likely (tm)": 8.0, "Pessimistic (tp)": 17.0},
            {"Activity": "D", "Optimistic (to)": 2.0, "Most Likely (tm)": 4.0, "Pessimistic (tp)": 6.0},
        ])
        edited_pert = st.data_editor(pert_init, num_rows="dynamic", use_container_width=True, key="pert_table_editor")
        td_val = st.number_input("🎯 Target Project Deadline (Td):", value=25.0)
        if st.button("⚙️ Compute PERT Table & Probability", type="primary"):
            edited_pert["Expected Time (te)"] = (edited_pert["Optimistic (to)"] + 4.0 * edited_pert["Most Likely (tm)"] + edited_pert["Pessimistic (tp)"]) / 6.0
            edited_pert["Std Dev (σ)"] = (edited_pert["Pessimistic (tp)"] - edited_pert["Optimistic (to)"]) / 6.0
            edited_pert["Variance (σ²)"] = edited_pert["Std Dev (σ)"] ** 2
            st.dataframe(edited_pert, use_container_width=True)
            te_sum = float(edited_pert["Expected Time (te)"].sum())
            var_sum = float(edited_pert["Variance (σ²)"].sum())
            sd_tot = math.sqrt(var_sum)
            z_sc = (td_val - te_sum) / sd_tot if sd_tot > 0 else 0.0
            st.success(f"📈 **Expected Total Time ($T_e$):** `{te_sum:.2f}` | **Z-Score:** `{z_sc:.4f}` | **Probability:** `{norm_cdf(z_sc)*100:.2f}%`")

    render_camera_ai_section(
        "tab1_pm",
        "إدارة المشروعات والشبكات (Project Management: AON / AOA / Lags / PERT)",
        "حل مسائل شبكات الأعمال AON و AOA، العلاقات FS, SS, FF, SF مع الـ Lags، حسابات ES, EF, LS, LF, Start Slack, Finish Slack, Free Slack، وترتيب المسار الحرج.",
        lambda: edited_pm_df.to_string() if "6. PERT" not in pm_preset else edited_pert.to_string(),
    )


with TABS[1]:
    st.markdown("""
    <div class='sec-header'>
        <h3>💰 2. الفوائد وتقييم المشروعات والتدفقات النقدية | Engineering Economy & Multi-Project NPV</h3>
        <p>جدول المقارنة بين المشروعات مع معدل العائد والتضخم (مطابق لـ <b>Example, Problem 6, page 52 — Broken Arrow Records</b>) مع كتابة معادلات الـ <b>NPV</b> التفصيلية وترتيب الأولويات (Priority Order)، بالإضافة إلى جداول معاملات الفوائد والإهلاك.</p>
    </div>
    """, unsafe_allow_html=True)

    eco_mode_sel = st.radio(
        "💰 Select Economy Table Mode | اختر نوع جدول الفوائد والاقتصاد الهندسي:",
        [
            "1. Multi-Project NPV Selection Table with Inflation (Broken Arrow Slide Page 52) | جدول مقارنة المشروعات بالـ NPV والتضخم",
            "2. Compound Interest Factors Interactive Table (F/P, P/F, P/A, A/P, P/G, A/G) | جدول معاملات الفوائد المركبة",
            "3. Depreciation Interactive Table (SL, SYD, DDB) | جدول حسابات الإهلاك",
        ],
        horizontal=True,
    )

    if eco_mode_sel.startswith("1."):
        c_inf1, c_inf2, c_inf3 = st.columns(3)
        with c_inf1:
            req_ror = st.number_input("📈 Required Rate of Return (%) | معدل العائد المطلوب:", value=20.0, step=1.0)
        with c_inf2:
            inf_rate = st.number_input("🎈 Predicted Inflation Rate (%) | معدل التضخم المتوقع:", value=2.0, step=0.5)
        with c_inf3:
            comb_mode = st.selectbox(
                "🧮 Combined Rate Formula | طريقة دمج الفائدة والتضخم:",
                [
                    "Additive Slide Formula (i = r + f = 20% + 2% = 22%)",
                    "Exact Fisher Formula (i = r + f + r*f)",
                ],
            )

        p1_name, p2_name, p3_name = st.columns(3)
        with p1_name:
            proj1_title = st.text_input("Project 1 Name:", value="Time Fades Away")
        with p2_name:
            proj2_title = st.text_input("Project 2 Name:", value="On the Beach")
        with p3_name:
            proj3_title = st.text_input("Project 3 Name:", value="Tonight's the Night")

        init_npv_table = pd.DataFrame([
            {"Year": 0, f"{proj1_title} - Investment": 600000.0, f"{proj1_title} - Revenue Stream": 0.0, f"{proj2_title} - Investment": 400000.0, f"{proj2_title} - Revenue Stream": 0.0, f"{proj3_title} - Investment": 200000.0, f"{proj3_title} - Revenue Stream": 0.0},
            {"Year": 1, f"{proj1_title} - Investment": 0.0, f"{proj1_title} - Revenue Stream": 600000.0, f"{proj2_title} - Investment": 0.0, f"{proj2_title} - Revenue Stream": 400000.0, f"{proj3_title} - Investment": 0.0, f"{proj3_title} - Revenue Stream": 200000.0},
            {"Year": 2, f"{proj1_title} - Investment": 0.0, f"{proj1_title} - Revenue Stream": 75000.0, f"{proj2_title} - Investment": 0.0, f"{proj2_title} - Revenue Stream": 100000.0, f"{proj3_title} - Investment": 0.0, f"{proj3_title} - Revenue Stream": 125000.0},
            {"Year": 3, f"{proj1_title} - Investment": 0.0, f"{proj1_title} - Revenue Stream": 20000.0, f"{proj2_title} - Investment": 0.0, f"{proj2_title} - Revenue Stream": 25000.0, f"{proj3_title} - Investment": 0.0, f"{proj3_title} - Revenue Stream": 75000.0},
            {"Year": 4, f"{proj1_title} - Investment": 0.0, f"{proj1_title} - Revenue Stream": 15000.0, f"{proj2_title} - Investment": 0.0, f"{proj2_title} - Revenue Stream": 20000.0, f"{proj3_title} - Investment": 0.0, f"{proj3_title} - Revenue Stream": 20000.0},
            {"Year": 5, f"{proj1_title} - Investment": 0.0, f"{proj1_title} - Revenue Stream": 10000.0, f"{proj2_title} - Investment": 0.0, f"{proj2_title} - Revenue Stream": 10000.0, f"{proj3_title} - Investment": 0.0, f"{proj3_title} - Revenue Stream": 10000.0},
        ])

        st.markdown("#### 📋 جدول الاستثمارات والتدفقات النقدية للمشروعات (Interactive Multi-Project Cash Flow Table):")
        edited_npv_df = st.data_editor(init_npv_table, num_rows="dynamic", use_container_width=True, key="npv_multi_editor")

        if st.button("⚙️ Compute Step-by-Step NPV Equations & Order of Priority", type="primary"):
            r_dec = req_ror / 100.0
            f_dec = inf_rate / 100.0
            k_disc = (r_dec + f_dec) if "Additive" in comb_mode else (r_dec + f_dec + r_dec * f_dec)
            st.info(f"📌 **Combined Discount Rate ($i = r + f$):** `{k_disc*100:.2f}%` (`{k_disc:.4f}`)")

            projects_list = [proj1_title, proj2_title, proj3_title]
            results_summary = []

            for p_title in projects_list:
                inv_col = f"{p_title} - Investment"
                rev_col = f"{p_title} - Revenue Stream"
                npv_total = 0.0
                terms_latex = []
                for _, r_row in edited_npv_df.iterrows():
                    yr = int(r_row["Year"])
                    inv_v = float(r_row.get(inv_col, 0.0)) if pd.notna(r_row.get(inv_col, 0.0)) else 0.0
                    rev_v = float(r_row.get(rev_col, 0.0)) if pd.notna(r_row.get(rev_col, 0.0)) else 0.0
                    net_cf = rev_v - inv_v
                    pv_term = net_cf / ((1.0 + k_disc) ** yr)
                    npv_total += pv_term
                    if yr == 0:
                        if net_cf != 0:
                            terms_latex.append(f"{net_cf:,.0f}")
                    else:
                        if net_cf != 0:
                            sign_str = "+" if net_cf > 0 else "-"
                            pwr_str = f"^{{{yr}}}" if yr > 1 else ""
                            terms_latex.append(f"{sign_str} {abs(net_cf):,.0f}\\frac{{1}}{{(1+{k_disc:g}){pwr_str}}}")

                eq_full = " ".join(terms_latex) + f" = \\${npv_total:,.2f}"
                results_summary.append({"Project": p_title, "NPV": npv_total, "LaTeX": eq_full})

            funded_sorted = sorted([x for x in results_summary if x["NPV"] > 0], key=lambda item: item["NPV"], reverse=True)
            rank_map = {item["Project"]: idx + 1 for idx, item in enumerate(funded_sorted)}

            st.markdown("#### 🧮 معادلات صافي القيمة الحالية وترتيب الأولويات (Step-by-Step NPV Equations & Priority):")
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
                    "Project Name | اسم المشروع": p_name,
                    "Net Present Value (NPV)": round(npv_v, 2),
                    "Decision | قرار التمويل": "Fund (تمويل)" if npv_v > 0 else "Do Not Fund (رفض)",
                    "Priority Rank | ترتيب الأولوية": f"Priority {rk}" if rk else "Rejected",
                })
            st.dataframe(pd.DataFrame(summary_table), use_container_width=True)
            add_log("Engineering Economy", "Multi-Project NPV Table", edited_npv_df.to_dict(), summary_table)

    elif eco_mode_sel.startswith("2."):
        init_factors_df = pd.DataFrame([
            {"Case": "Case 1", "Interest i (%)": 10.0, "Periods n": 5, "Present (P)": 10000.0, "Future (F)": 0.0, "Uniform (A)": 0.0, "Gradient (G)": 0.0},
            {"Case": "Case 2", "Interest i (%)": 12.0, "Periods n": 8, "Present (P)": 0.0, "Future (F)": 0.0, "Uniform (A)": 2500.0, "Gradient (G)": 300.0},
            {"Case": "Case 3", "Interest i (%)": 8.0, "Periods n": 10, "Present (P)": 0.0, "Future (F)": 50000.0, "Uniform (A)": 0.0, "Gradient (G)": 0.0},
        ])
        edited_fac_in = st.data_editor(init_factors_df, num_rows="dynamic", use_container_width=True, key="eco_fac_editor")
        if st.button("⚙️ Compute Interest Factors & Equivalent P, F, A for All Rows", type="primary"):
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
                    "(F/P)": round(fp, 4), "(P/F)": round(pf, 4),
                    "(P/A)": round(pa, 4), "(A/P)": round(ap, 4),
                    "(F/A)": round(fa, 4), "(A/F)": round(af, 4),
                    "(P/G)": round(pg, 4), "(A/G)": round(ag, 4),
                    "Total Equivalent PW (P)": round(p_eq, 2),
                    "Total Equivalent FW (F)": round(p_eq * fp, 2),
                    "Total Equivalent AW (A)": round(p_eq * ap, 2),
                })
            st.dataframe(pd.DataFrame(out_fac_rows), use_container_width=True)

    else:
        dep_in_df = pd.DataFrame([
            {"Asset": "Machine A", "Initial Cost (B)": 100000.0, "Salvage Value (S)": 10000.0, "Useful Life (N)": 5},
        ])
        ed_dep_in = st.data_editor(dep_in_df, num_rows="dynamic", use_container_width=True, key="dep_in_editor")
        if st.button("⚙️ Generate Full Depreciation Schedule Table", type="primary"):
            r0 = ed_dep_in.iloc[0]
            B_v, S_v, N_v = float(r0["Initial Cost (B)"]), float(r0["Salvage Value (S)"]), int(r0["Useful Life (N)"])
            syd_s = N_v * (N_v + 1) / 2.0
            bv_sl, bv_syd, bv_ddb = B_v, B_v, B_v
            d_rows = []
            for yr in range(1, N_v + 1):
                dsl = (B_v - S_v) / N_v
                bv_sl -= dsl
                dsyd = ((N_v - yr + 1) / syd_s) * (B_v - S_v)
                bv_syd -= dsyd
                dddb = min((2.0 / N_v) * bv_ddb, max(0.0, bv_ddb - S_v))
                bv_ddb -= dddb
                d_rows.append({
                    "Year": yr,
                    "SL Depr.": round(dsl, 2), "SL Book Value": round(bv_sl, 2),
                    "SYD Depr.": round(dsyd, 2), "SYD Book Value": round(bv_syd, 2),
                    "DDB Depr.": round(dddb, 2), "DDB Book Value": round(bv_ddb, 2),
                })
            st.dataframe(pd.DataFrame(d_rows), use_container_width=True)

    render_camera_ai_section(
        "tab2_eco",
        "الفوائد وتقييم المشروعات والاقتصاد الهندسي (NPV, IRR, Inflation & Interest Tables)",
        "حل مسائل تقييم المشروعات وحسابات NPV مع التضخم وترتيب الأولويات، جداول التدفقات النقدية، معاملات الفوائد المركبة، والإهلاك.",
        lambda: edited_npv_df.to_string() if eco_mode_sel.startswith("1.") else "Economy Table",
    )


with TABS[2]:
    st.markdown("""
    <div class='sec-header'>
        <h3>🧮 3. الجداول التفاعلية للطرق العددية | Interactive Numerical Methods Tables</h3>
        <p>املأ الجداول التفاعلية مباشرة لحل الأنظمة الخطية (<b>Gauss-Seidel & Jacobi Augmented Matrix Table</b>)، وجدول نقاط الاستكمال والتكامل العددي (<b>Interpolation & Simpson/Trapezoidal Table</b>)، وجدول المعادلات غير الخطية والـ <b>RK4</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    num_mode_tab = st.radio(
        "🧮 Select Numerical Table | اختر جدول الطرق العددية:",
        [
            "1. Augmented Matrix Table [A | B] (Gauss-Seidel, Jacobi & Exact LU)",
            "2. Tabular Data Points (x, y) for Interpolation (Newton Divided Differences & Lagrange) & Numerical Integration",
            "3. Root Finding & RK4 ODE Parameters Table",
        ],
        horizontal=True,
    )

    if num_mode_tab.startswith("1."):
        init_mat_df = pd.DataFrame([
            {"x1": 10.0, "x2": -1.0, "x3": 2.0, "x4": 0.0, "Constant B": 6.0},
            {"x1": -1.0, "x2": 11.0, "x3": -1.0, "x4": 3.0, "Constant B": 25.0},
            {"x1": 2.0, "x2": -1.0, "x3": 10.0, "x4": -1.0, "Constant B": -11.0},
            {"x1": 0.0, "x2": 3.0, "x3": -1.0, "x4": 8.0, "Constant B": 15.0},
        ])
        st.markdown("#### 📋 املأ جدول المصفوفة الموسعة $[A \\mid B]$ مباشرة:")
        ed_mat_df = st.data_editor(init_mat_df, num_rows="dynamic", use_container_width=True, key="num_mat_editor")
        c_gs1, c_gs2 = st.columns(2)
        with c_gs1:
            iter_algo = st.selectbox("Iterative Method:", ["Gauss-Seidel", "Jacobi"])
        with c_gs2:
            auto_piv = st.checkbox("Auto-Pivoting for Diagonal Dominance", value=True)

        if st.button("⚙️ Solve Matrix Table & Show Iteration Table", type="primary"):
            M_arr = ed_mat_df.to_numpy(dtype=float)
            n_r = M_arr.shape[0]
            A_m = M_arr[:, :n_r].copy()
            B_v = M_arr[:, -1].copy()
            if auto_piv:
                for i in range(n_r):
                    mr = i + int(np.argmax(np.abs(A_m[i:, i])))
                    A_m[[i, mr]], B_v[[i, mr]] = A_m[[mr, i]], B_v[[mr, i]]
            x_v = np.zeros(n_r)
            iter_log = []
            for it in range(1, 25):
                x_old = x_v.copy()
                for i in range(n_r):
                    if iter_algo == "Gauss-Seidel":
                        s1 = np.dot(A_m[i, :i], x_v[:i])
                        s2 = np.dot(A_m[i, i + 1 :], x_old[i + 1 :])
                        x_v[i] = (B_v[i] - s1 - s2) / A_m[i, i]
                    else:
                        s = np.dot(A_m[i, :], x_old) - A_m[i, i] * x_old[i]
                        x_v[i] = (B_v[i] - s) / A_m[i, i]
                err = float(np.max(np.abs(x_v - x_old)))
                iter_log.append({"Iteration": it, **{f"x{j+1}": round(x_v[j], 6) for j in range(n_r)}, "Max Error": round(err, 7)})
                if err < 1e-6:
                    break
            st.dataframe(pd.DataFrame(iter_log), use_container_width=True)
            st.success(f"🎯 **Exact Solution:** `{np.round(np.linalg.solve(A_m, B_v), 6)}`")

    elif num_mode_tab.startswith("2."):
        init_pts_df = pd.DataFrame([
            {"x": 0.0, "y = f(x)": 1.0},
            {"x": 1.0, "y = f(x)": 2.0},
            {"x": 2.0, "y = f(x)": 5.0},
            {"x": 3.0, "y = f(x)": 10.0},
            {"x": 4.0, "y = f(x)": 17.0},
        ])
        ed_pts_df = st.data_editor(init_pts_df, num_rows="dynamic", use_container_width=True, key="num_pts_editor")
        x_eval_pt = st.number_input("🎯 Value of x to Interpolate f(x):", value=2.5)
        if st.button("⚙️ Build Divided Differences Table & Integrate Tabular Data", type="primary"):
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
            st.success(f"📌 **Interpolated Value $P({x_eval_pt}) =$** `{val_interp:.6f}` | **Trapezoidal Integral $\\int y\\,dx =$** `{trap_tab:.6f}`")

    else:
        rk_in_df = pd.DataFrame([
            {"Function f(x, y)": "x + y", "x0": 0.0, "y0": 1.0, "Step h": 0.1, "Steps N": 10},
        ])
        ed_rk_df = st.data_editor(rk_in_df, use_container_width=True, key="rk_table_editor")
        if st.button("⚙️ Generate Euler, Heun & RK4 Table", type="primary"):
            r0 = ed_rk_df.iloc[0]
            f_str = str(r0["Function f(x, y)"])
            xv, ye, yh, yrk = float(r0["x0"]), float(r0["y0"]), float(r0["y0"]), float(r0["y0"])
            h = float(r0["Step h"])
            steps = int(r0["Steps N"])
            rows_ode = [{"Step": 0, "x": xv, "Euler y": ye, "Heun y": yh, "RK4 y": yrk}]
            for s_i in range(1, steps + 1):
                ye_n = ye + h * float(eval_math_expr(f_str, x=xv, y=ye))
                k1h = float(eval_math_expr(f_str, x=xv, y=yh))
                k2h = float(eval_math_expr(f_str, x=xv + h, y=yh + h * k1h))
                yh_n = yh + 0.5 * h * (k1h + k2h)
                k1 = h * float(eval_math_expr(f_str, x=xv, y=yrk))
                k2 = h * float(eval_math_expr(f_str, x=xv + 0.5 * h, y=yrk + 0.5 * k1))
                k3 = h * float(eval_math_expr(f_str, x=xv + 0.5 * h, y=yrk + 0.5 * k2))
                k4 = h * float(eval_math_expr(f_str, x=xv + h, y=yrk + k3))
                yrk_n = yrk + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                xv = round(xv + h, 5)
                ye, yh, yrk = ye_n, yh_n, yrk_n
                rows_ode.append({"Step": s_i, "x": xv, "Euler y": round(ye, 6), "Heun y": round(yh, 6), "RK4 y": round(yrk, 6)})
            st.dataframe(pd.DataFrame(rows_ode), use_container_width=True)

    render_camera_ai_section(
        "tab3_num",
        "الطرق العددية والجداول التكرارية (Numerical Methods Tables)",
        "حل مسائل الطرق العددية بالجداول الكاملة: Gauss-Seidel, Jacobi, Newton-Raphson, Bisection, Secant, Divided Differences, Simpson, RK4.",
        lambda: num_mode_tab,
    )


with TABS[3]:
    st.markdown("""
    <div class='sec-header'>
        <h3>📈 4. جدول المعادلات التفاضلية وتحويلات لابلاس | ODEs & Laplace Interactive Table</h3>
    </div>
    """, unsafe_allow_html=True)

    ode_tab_df = pd.DataFrame([
        {"Case": "ODE 1 (Underdamped)", "a (y'')": 1.0, "b (y')": 2.0, "c (y)": 10.0, "y(0)": 1.0, "y'(0)": 0.0},
        {"Case": "ODE 2 (Overdamped)", "a (y'')": 1.0, "b (y')": 5.0, "c (y)": 6.0, "y(0)": 2.0, "y'(0)": -1.0},
        {"Case": "ODE 3 (Critically Damped)", "a (y'')": 1.0, "b (y')": 4.0, "c (y)": 4.0, "y(0)": 1.0, "y'(0)": 2.0},
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
        "المعادلات التفاضلية وتحويلات لابلاس وفورييه (ODEs, Laplace & Fourier)",
        "حل المعادلات التفاضلية وتحويلات لابلاس وفورييه والتكاملات المتعددة خطوة بخطوة.",
        lambda: ed_ode_tab.to_string(),
    )


with TABS[4]:
    st.markdown("""
    <div class='sec-header'>
        <h3>⚙️ 5. جداول الميكانيكا، دائرة مور، وأحمال الكمرات | Mechanics, Mohr's Circle & Beam Loads Tables</h3>
    </div>
    """, unsafe_allow_html=True)

    stress_df = pd.DataFrame([
        {"Element": "Point A", "σx (MPa)": 80.0, "σy (MPa)": -40.0, "τxy (MPa)": 35.0, "Angle θ (deg)": 30.0},
        {"Element": "Point B", "σx (MPa)": 120.0, "σy (MPa)": 40.0, "τxy (MPa)": -30.0, "Angle θ (deg)": 45.0},
    ])
    ed_stress = st.data_editor(stress_df, num_rows="dynamic", use_container_width=True, key="stress_editor")
    if st.button("⚙️ Compute Principal Stresses Table (σ1, σ2, τmax)", type="primary"):
        s_out = []
        for _, r in ed_stress.iterrows():
            sx, sy, txy, th = float(r["σx (MPa)"]), float(r["σy (MPa)"]), float(r["τxy (MPa)"]), float(r["Angle θ (deg)"])
            s_avg = 0.5 * (sx + sy)
            rad_m = math.hypot(0.5 * (sx - sy), txy)
            s_out.append({
                "Element": r["Element"],
                "σ_avg (MPa)": round(s_avg, 2),
                "σ1 Max (MPa)": round(s_avg + rad_m, 2),
                "σ2 Min (MPa)": round(s_avg - rad_m, 2),
                "τ_max (MPa)": round(rad_m, 2),
                "θp (deg)": round(0.5 * math.degrees(math.atan2(2 * txy, sx - sy)), 2),
            })
        st.dataframe(pd.DataFrame(s_out), use_container_width=True)

    render_camera_ai_section(
        "tab5_mech",
        "الميكانيكا التطبيقية ومقاومة المواد (Mohr's Circle, Beams SFD/BMD & Trusses)",
        "حل مسائل الكمرات (SFD & BMD)، الجمالونات (Trusses)، ودائرة مور للإجهادات.",
        lambda: ed_stress.to_string(),
    )


with TABS[5]:
    st.markdown("""
    <div class='sec-header'>
        <h3>⚡ 6. جدول اختبارات المحولات والدوائر الكهربية | Transformers & AC Circuits Table</h3>
    </div>
    """, unsafe_allow_html=True)

    tr_df = pd.DataFrame([
        {"Transformer": "Unit 1", "Voc (V)": 220.0, "Ioc (A)": 1.5, "Poc (W)": 120.0, "Vsc (V)": 45.0, "Isc (A)": 10.0, "Psc (W)": 250.0},
    ])
    ed_tr = st.data_editor(tr_df, num_rows="dynamic", use_container_width=True, key="tr_table_editor")
    if st.button("⚙️ Compute Equivalent Circuit Parameters Table", type="primary"):
        tr_out = []
        for _, r in ed_tr.iterrows():
            voc, ioc, poc = float(r["Voc (V)"]), float(r["Ioc (A)"]), float(r["Poc (W)"])
            vsc, isc, psc = float(r["Vsc (V)"]), float(r["Isc (A)"]), float(r["Psc (W)"])
            pf0 = poc / (voc * ioc)
            rc = voc / (ioc * pf0)
            xm = voc / (ioc * math.sqrt(max(0.0, 1.0 - pf0 ** 2)))
            req = psc / (isc ** 2)
            zeq = vsc / isc
            xeq = math.sqrt(max(0.0, zeq ** 2 - req ** 2))
            tr_out.append({"Transformer": r["Transformer"], "PF0": round(pf0, 4), "Rc (Ω)": round(rc, 2), "Xm (Ω)": round(xm, 2), "Req (Ω)": round(req, 3), "Xeq (Ω)": round(xeq, 3), "Zeq (Ω)": round(zeq, 3)})
        st.dataframe(pd.DataFrame(tr_out), use_container_width=True)

    render_camera_ai_section(
        "tab6_elec",
        "الدوائر الكهربية والمحولات والقوى ثلاثية الأوجه (Electrical Engineering)",
        "حل مسائل الدوائر الكهربية والمحولات والقدرة ثلاثية الأوجه.",
        lambda: ed_tr.to_string(),
    )


with TABS[6]:
    st.markdown("""
    <div class='sec-header'>
        <h3>🎛️ 7. جدول أنظمة التحكم الآلي | Control Systems Transfer Functions Table</h3>
    </div>
    """, unsafe_allow_html=True)

    ctrl_df = pd.DataFrame([
        {"System": "Sys 1", "wn (rad/s) or sqrt(c)": 5.0, "Damping Ratio (ζ)": 0.4, "DC Gain (K)": 1.0},
        {"System": "Sys 2", "wn (rad/s) or sqrt(c)": 8.0, "Damping Ratio (ζ)": 0.707, "DC Gain (K)": 2.0},
    ])
    ed_ctrl = st.data_editor(ctrl_df, num_rows="dynamic", use_container_width=True, key="ctrl_table_editor")
    if st.button("⚙️ Compute Transient Specifications Table (Mp%, Tp, Ts)", type="primary"):
        c_out = []
        for _, r in ed_ctrl.iterrows():
            wn, z = float(r["wn (rad/s) or sqrt(c)"]), float(r["Damping Ratio (ζ)"])
            wd = wn * math.sqrt(max(0.0, 1.0 - z ** 2))
            mp = 100.0 * math.exp(-z * math.pi / math.sqrt(1.0 - z ** 2)) if 0 < z < 1 else 0.0
            tp = math.pi / wd if wd > 0 else 0.0
            ts = 4.0 / (z * wn) if z > 0 else 0.0
            c_out.append({"System": r["System"], "wd (rad/s)": round(wd, 3), "Overshoot Mp (%)": round(mp, 2), "Peak Time Tp (s)": round(tp, 3), "Settling Time Ts 2% (s)": round(ts, 3)})
        st.dataframe(pd.DataFrame(c_out), use_container_width=True)

    render_camera_ai_section(
        "tab7_ctrl",
        "هندسة التحكم الآلي والـ PLC (Control Systems & Automation)",
        "حل مسائل Routh-Hurwitz, Root Locus, Bode, Block Diagrams, PID, و PLC Ladder Logic.",
        lambda: ed_ctrl.to_string(),
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
