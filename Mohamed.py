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
    .main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 97%; }
    .eng-banner {
        background: linear-gradient(135deg, rgb(15, 23, 42) 0%, rgb(30, 58, 138) 55%, rgb(15, 118, 110) 100%);
        color: white; padding: 20px 24px; border-radius: 12px; margin-bottom: 16px;
        direction: rtl; text-align: right; box-shadow: 0 4px 14px rgba(0,0,0,0.2);
    }
    .eng-banner h1 { margin: 0; font-size: 24px; color: white; }
    .eng-banner p { margin: 6px 0 0 0; font-size: 14px; opacity: 0.92; }
    .sec-header {
        background: rgb(248, 250, 252); border: 1px solid rgb(203, 213, 225); border-right: 6px solid rgb(29, 78, 216);
        padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; direction: rtl; text-align: right;
    }
    .ai-cam-box {
        background: rgb(240, 249, 255); border: 2px dashed rgb(2, 132, 199); padding: 14px;
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
    clean_expr = expr_str.replace("^", "**")
    return eval(clean_expr, {"__builtins__": {}}, safe_env)


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
        bin_arr = np.where(arr > thresh, 255, 0).astype(np.uint8)
        return Image.fromarray(bin_arr)

    elif "Adaptive" in mode_str:
        blurred = np.array(gray.filter(ImageFilter.BoxBlur(7)), dtype=np.float32)
        ad_arr = np.where(arr > (blurred - 6.0), 255, 0).astype(np.uint8)
        return Image.fromarray(ad_arr)

    elif "Contrast" in mode_str:
        c_img = ImageEnhance.Contrast(img).enhance(1.9)
        return ImageEnhance.Sharpness(c_img).enhance(2.3)

    elif "Edges" in mode_str:
        ed = gray.filter(ImageFilter.FIND_EDGES)
        return ImageOps.invert(ed)

    return img


def call_gemini_rest_api(domain_title, domain_guide, text_data, pil_img=None, mode="solve"):
    if not st.session_state.api_key:
        return "❌ **Error:** يرجى إدخال مفتاح `Gemini API Key` في القائمة الجانبية أولاً."
    try:
        if mode == "ocr":
            prompt_text = (
                f"You are an Engineering OCR Scanner for ({domain_title}). "
                "Extract all equations, tables, network nodes, durations, lags, and numbers accurately from the image. "
                "Provide both LaTeX notation and ready-to-copy text format."
            )
        else:
            prompt_text = (
                f"أنت بروفيسور في كلية الهندسة متخصص في ({domain_title}).\n"
                f"أسلوب الإجابة المطلوب: {ai_detail_mode}.\n"
                f"توجيهات التخصص: {domain_guide}\n"
                "قم بحل المسألة الهندسية باللغتين العربية والإنجليزية للمصطلحات، متضمناً:\n"
                "1. **المعطيات والمطلوب (Given & Required)** مع الوحدات القياسية.\n"
                "2. **القوانين الهندسية (Governing Equations)** بصيغة LaTeX واضحة.\n"
                "3. **خطوات الحل والجداول التفصيلية (Step-by-Step Calculation & Tables)**.\n"
                "4. **الناتج النهائي والتعليق الهندسي (Final Answer & Engineering Interpretation)**.\n"
                f"\nبيانات المسألة النصية:\n{text_data}"
            )

        parts_list = [{"text": prompt_text}]
        if pil_img is not None:
            buf = io.BytesIO()
            pil_img.convert("RGB").save(buf, format="JPEG", quality=90)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            parts_list.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": b64_str,
                }
            })

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
        err_body = http_err.read().decode("utf-8", errors="ignore")
        return f"❌ **API HTTP Error ({http_err.code}):** `{err_body}`"
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
            ["Text Only | نص فقط", "Camera | كاميرا مباشرة", "Upload | رفع صورة"],
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
                res_ocr = call_gemini_rest_api(domain_title, domain_guide, extra_q, proc_img, mode="ocr")
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
                res_solve = call_gemini_rest_api(domain_title, domain_guide, ctx_str, proc_img, mode="solve")
                st.markdown(f"#### 🤖 الحل الهندسي التفصيلي | AI Engineering Solution ({domain_title}):")
                st.markdown(res_solve)
                add_log(domain_title, "Gemini AI Solver", ctx_str, res_solve)


def build_aon_svg(tasks, order_keys):
    layers = {}
    for k in order_keys:
        preds = [p for p, _, _ in tasks[k]["preds"] if p in tasks]
        layers[k] = 0 if not preds else max(layers.get(p, 0) for p in preds) + 1
    max_layer = max(layers.values()) if layers else 0
    layer_groups = {}
    for k in order_keys:
        ly = layers[k]
        layer_groups.setdefault(ly, []).append(k)

    max_in_col = max(len(v) for v in layer_groups.values()) if layer_groups else 1
    svg_w = max(780, (max_layer + 1) * 210 + 60)
    svg_h = max(320, max_in_col * 135 + 60)

    coords = {}
    for ly, nodes in layer_groups.items():
        x_c = 40 + ly * 205
        for idx, k in enumerate(nodes):
            y_c = 40 + idx * 125 + (max_in_col - len(nodes)) * 55
            coords[k] = (x_c, y_c)

    svg_lines = [
        f"<svg viewBox='0 0 {svg_w} {svg_h}' style='width:100%; height:auto; background:white; border:1px solid rgb(203,213,225); border-radius:8px;'>",
        "<defs>",
        "<marker id='arrowCrit' markerWidth='10' markerHeight='7' refX='9' refY='3.5' orient='auto'><polygon points='0 0, 10 3.5, 0 7' fill='rgb(220,38,38)'/></marker>",
        "<marker id='arrowNorm' markerWidth='10' markerHeight='7' refX='9' refY='3.5' orient='auto'><polygon points='0 0, 10 3.5, 0 7' fill='rgb(71,85,105)'/></marker>",
        "</defs>",
    ]

    for k in order_keys:
        x2, y2 = coords[k]
        for p_id, r_type, lag in tasks[k]["preds"]:
            if p_id in coords:
                x1, y1 = coords[p_id]
                sx, sy = x1 + 135, y1 + 42
                ex, ey = x2, y2 + 42
                is_crit_edge = (abs(tasks[p_id]["TF"]) < 1e-5) and (abs(tasks[k]["TF"]) < 1e-5)
                col = "rgb(220,38,38)" if is_crit_edge else "rgb(71,85,105)"
                mk = "url(#arrowCrit)" if is_crit_edge else "url(#arrowNorm)"
                svg_lines.append(f"<line x1='{sx}' y1='{sy}' x2='{ex}' y2='{ey}' stroke='{col}' stroke-width='2.3' marker-end='{mk}'/>")
                mx, my = (sx + ex) / 2.0, (sy + ey) / 2.0 - 6
                lbl_rel = f"{r_type}{lag:+g}" if (r_type != "FS" or lag != 0) else "FS"
                svg_lines.append(f"<text x='{mx}' y='{my}' font-size='11' font-weight='bold' fill='rgb(185,28,28)' text-anchor='middle'>{lbl_rel}</text>")

    for k in order_keys:
        x, y = coords[k]
        d = tasks[k]
        is_c = abs(d["TF"]) < 1e-5
        bg = "rgb(254,226,226)" if is_c else "rgb(219,234,254)"
        bd = "rgb(220,38,38)" if is_c else "rgb(29,78,216)"
        svg_lines.append(f"<rect x='{x}' y='{y}' width='135' height='84' rx='6' fill='{bg}' stroke='{bd}' stroke-width='2'/>")
        svg_lines.append(f"<line x1='{x}' y1='{y+28}' x2='{x+135}' y2='{y+28}' stroke='{bd}' stroke-width='1.2'/>")
        svg_lines.append(f"<line x1='{x}' y1='{y+56}' x2='{x+135}' y2='{y+56}' stroke='{bd}' stroke-width='1.2'/>")
        svg_lines.append(f"<line x1='{x+45}' y1='{y}' x2='{x+45}' y2='{y+84}' stroke='{bd}' stroke-width='1.2'/>")
        svg_lines.append(f"<line x1='{x+90}' y1='{y}' x2='{x+90}' y2='{y+84}' stroke='{bd}' stroke-width='1.2'/>")
        svg_lines.append(f"<text x='{x+22}' y='{y+19}' font-size='11' text-anchor='middle' fill='black'>ES:{d['ES']:g}</text>")
        svg_lines.append(f"<text x='{x+67}' y='{y+19}' font-size='11' font-weight='bold' text-anchor='middle' fill='black'>D:{d['dur']:g}</text>")
        svg_lines.append(f"<text x='{x+112}' y='{y+19}' font-size='11' text-anchor='middle' fill='black'>EF:{d['EF']:g}</text>")
        svg_lines.append(f"<text x='{x+67}' y='{y+47}' font-size='13' font-weight='bold' text-anchor='middle' fill='{bd}'>{k}</text>")
        svg_lines.append(f"<text x='{x+22}' y='{y+74}' font-size='11' text-anchor='middle' fill='black'>LS:{d['LS']:g}</text>")
        svg_lines.append(f"<text x='{x+67}' y='{y+74}' font-size='11' font-weight='bold' text-anchor='middle' fill='rgb(185,28,28)'>TF:{d['TF']:g}</text>")
        svg_lines.append(f"<text x='{x+112}' y='{y+74}' font-size='11' text-anchor='middle' fill='black'>LF:{d['LF']:g}</text>")

    svg_lines.append("</svg>")
    return "\n".join(svg_lines)


def build_aoa_svg(tasks, order_keys):
    events = {10: {"t": 0.0, "col": 0, "row": 0}}
    ev_counter = 10
    end_event_of = {}
    edges_aoa = []

    for idx, k in enumerate(order_keys):
        preds = [p for p, _, _ in tasks[k]["preds"] if p in tasks]
        if not preds:
            u_ev = 10
        elif len(preds) == 1:
            u_ev = end_event_of[preds[0]]
        else:
            ev_counter += 10
            u_ev = ev_counter
            max_c = max(events[end_event_of[p]]["col"] for p in preds) + 1
            events[u_ev] = {"t": tasks[k]["ES"], "col": max_c, "row": idx % 3}
            for p_i, p_name in enumerate(preds):
                edges_aoa.append((end_event_of[p_name], u_ev, f"Dummy d{p_i+1}(0)", True, False))
        ev_counter += 10
        v_ev = ev_counter
        events[v_ev] = {"t": tasks[k]["EF"], "col": events[u_ev]["col"] + 1, "row": idx % 3}
        end_event_of[k] = v_ev
        edges_aoa.append((u_ev, v_ev, f"{k} (D={tasks[k]['dur']:g})", False, abs(tasks[k]["TF"]) < 1e-5))

    max_col = max(e["col"] for e in events.values()) if events else 1
    svg_w = max(760, (max_col + 1) * 165 + 80)
    svg_h = 340
    ev_coords = {}
    col_counts = {}
    for ev_id, edata in events.items():
        c = edata["col"]
        r_idx = col_counts.get(c, 0)
        col_counts[c] = r_idx + 1
        ev_coords[ev_id] = (60 + c * 155, 65 + (r_idx % 3) * 105)

    svg = [
        f"<svg viewBox='0 0 {svg_w} {svg_h}' style='width:100%; height:auto; background:white; border:1px solid rgb(203,213,225); border-radius:8px;'>",
        "<defs><marker id='arrAOA' markerWidth='10' markerHeight='7' refX='26' refY='3.5' orient='auto'><polygon points='0 0, 10 3.5, 0 7' fill='rgb(15,118,110)'/></marker></defs>",
    ]
    for u, v, lbl, is_dummy, is_crit in edges_aoa:
        x1, y1 = ev_coords[u]
        x2, y2 = ev_coords[v]
        col = "rgb(220,38,38)" if is_crit else ("rgb(100,116,139)" if is_dummy else "rgb(15,118,110)")
        dash = "stroke-dasharray='6,4'" if is_dummy else ""
        svg.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{col}' stroke-width='2.2' {dash} marker-end='url(#arrAOA)'/>")
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0 - 8
        svg.append(f"<text x='{mx}' y='{my}' font-size='11' font-weight='bold' fill='{col}' text-anchor='middle'>{lbl}</text>")

    for ev_id, (cx, cy) in ev_coords.items():
        t_v = events[ev_id]["t"]
        svg.append(f"<circle cx='{cx}' cy='{cy}' r='24' fill='rgb(254,240,138)' stroke='rgb(202,138,4)' stroke-width='2'/>")
        svg.append(f"<line x1='{cx-24}' y1='{cy}' x2='{cx+24}' y2='{cy}' stroke='rgb(202,138,4)' stroke-width='1.2'/>")
        svg.append(f"<text x='{cx}' y='{cy-5}' font-size='11' font-weight='bold' text-anchor='middle' fill='black'>E{ev_id}</text>")
        svg.append(f"<text x='{cx}' y='{cy+14}' font-size='10' text-anchor='middle' fill='rgb(180,83,9)'>t={t_v:g}</text>")

    svg.append("</svg>")
    return "\n".join(svg)


def build_gantt_svg(tasks, order_keys, proj_dur):
    svg_w = 800
    row_h = 36
    svg_h = max(180, len(order_keys) * row_h + 60)
    scale_x = 640.0 / max(1.0, proj_dur)
    svg = [f"<svg viewBox='0 0 {svg_w} {svg_h}' style='width:100%; height:auto; background:white; border:1px solid rgb(203,213,225); border-radius:8px;'>"]
    for idx, k in enumerate(order_keys):
        d = tasks[k]
        y = 30 + idx * row_h
        bx = 90 + d["ES"] * scale_x
        bw = max(4.0, d["dur"] * scale_x)
        col = "rgb(220,38,38)" if abs(d["TF"]) < 1e-5 else "rgb(37,99,235)"
        svg.append(f"<text x='75' y='{y+18}' font-size='12' font-weight='bold' text-anchor='end' fill='black'>{k} ({d['ES']:g}-{d['EF']:g})</text>")
        svg.append(f"<rect x='{bx}' y='{y+4}' width='{bw}' height='20' rx='4' fill='{col}'/>")
        if d["TF"] > 0:
            fw = d["TF"] * scale_x
            svg.append(f"<rect x='{bx+bw}' y='{y+8}' width='{fw}' height='12' rx='2' fill='rgb(203,213,225)' stroke='rgb(100,116,139)' stroke-dasharray='3,2'/>")
    svg.append("</svg>")
    return "\n".join(svg)


st.markdown("""
<div class='eng-banner'>
    <h1>⚙️ المنصة الهندسية الجامعية المتقدمة | Advanced Engineering Solver Platform</h1>
    <p>Project Management (AON / AOA / Lags / PERT / Gantt) • Engineering Economy (Interest Factors / CFD / IRR / Depreciation) • Numerical Methods • ODEs & Laplace • Mechanics & Mohr's Circle • 3-Phase & Transformers • Control Systems • Live Camera & AI</p>
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


with TABS[0]:
    st.markdown("""
    <div class='sec-header'>
        <h3>📊 1. إدارة المشروعات والشبكات الهندسية | Project Management (CPM, AON, AOA, Lags & PERT)</h3>
        <p>حساب المسار الحرج (Critical Path Method) مع دعم كامل لجميع العلاقات (<b>FS, SS, FF, SF</b>) وفترات التأخير والتداخل (<b>Lags / Leads</b>)، وحساب (<b>ES, EF, LS, LF, TF, FF</b>)، ورسم مخطط <b>AON</b> القياسي، ومخطط الأسهم <b>AOA</b>، ومخطط <b>Gantt Chart</b>، وتقييم <b>PERT</b> الاحتمالي.</p>
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
            "📌 **Input Format | طريقة الإدخال:** في كل سطر اكتب: `Activity, Duration, Predecessors` "
            "| مثال للعلاقات والـ Lags: `A:FS+2` أو `B:SS+3` أو `C:FF+4` أو `D:SF+1` (أو اكتب اسم النشاط فقط مثل `A` لعلاقة `FS+0`)."
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
            "✍️ Activities, Durations & Predecessor Relations (FS/SS/FF/SF ± Lag) | جدول الأنشطة والعلاقات:",
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

                for k in order_keys:
                    tasks[k]["LF"] = proj_dur
                    tasks[k]["LS"] = proj_dur - tasks[k]["dur"]

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

                st.markdown("#### 📐 1. مخطط الشبكة المستطيلة (AON - Activity on Node Diagram with Lags):")
                st.markdown(build_aon_svg(tasks, order_keys), unsafe_allow_html=True)

                st.markdown("#### 🏹 2. مخطط شبكة الأسهم والأحداث (AOA - Activity on Arrow & Dummy Activities):")
                st.markdown(build_aoa_svg(tasks, order_keys), unsafe_allow_html=True)

                st.markdown("#### 📅 3. مخطط جانت الزمني وفائض الوقت (Project Gantt Chart & Float Bars):")
                st.markdown(build_gantt_svg(tasks, order_keys, proj_dur), unsafe_allow_html=True)

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
            prob_pct = norm_cdf(z_val) * 100.0
            st.latex(f"T_e = \\sum t_e = {tot_te:.3f}, \\quad \\sigma_T = \\sqrt{{\\sum \\sigma^2}} = {proj_sd:.4f}, \\quad Z = \\frac{{T_d - T_e}}{{\\sigma_T}} = {z_val:.4f}")
            st.success(f"📈 **احتمالية إنهاء المشروع خلال ({target_deadline}) وحدة زمنية | Probability P(T ≤ Td):** `{prob_pct:.2f}%`")

    render_camera_ai_section(
        "tab1_pm",
        "إدارة المشروعات والشبكات (Project Management: AON / AOA / Lags / PERT / Crashing)",
        "حل مسائل شبكات الأعمال AON و AOA، العلاقات المتقدمة FS, SS, FF, SF مع الـ Lags، ضغط المشروعات بأقل تكلفة (Project Crashing)، تسوية الموارد (Resource Leveling)، وتحليل PERT.",
        lambda: pdm_text if pm_sub.startswith("1.") else pert_in,
    )


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
        if st.button("⚙️ Evaluate Project (PW/NPV, FW, AW, IRR, B/C) & Plot CFD", type="primary"):
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

            st.markdown("#### 📊 مخطط التدفقات النقدية | Cash Flow Diagram (CFD):")
            max_abs = max(abs(x) for x in cfs) if cfs else 1.0
            svg_cfd = ["<svg viewBox='0 0 760 240' style='width:100%; height:auto; background:white; border:1px solid rgb(203,213,225); border-radius:8px;'>"]
            svg_cfd.append("<line x1='50' y1='120' x2='710' y2='120' stroke='black' stroke-width='2'/>")
            step_x = 620.0 / max(1, len(cfs) - 1)
            for idx, cf_v in enumerate(cfs):
                cx = 60 + idx * step_x
                bar_h = (abs(cf_v) / max_abs) * 85.0
                if cf_v >= 0:
                    svg_cfd.append(f"<line x1='{cx}' y1='120' x2='{cx}' y2='{120-bar_h}' stroke='rgb(22,163,74)' stroke-width='4'/>")
                    svg_cfd.append(f"<polygon points='{cx-5},{125-bar_h} {cx+5},{125-bar_h} {cx},{115-bar_h}' fill='rgb(22,163,74)'/>")
                    svg_cfd.append(f"<text x='{cx}' y='{110-bar_h}' font-size='10' font-weight='bold' fill='rgb(22,163,74)' text-anchor='middle'>+{cf_v:g}</text>")
                else:
                    svg_cfd.append(f"<line x1='{cx}' y1='120' x2='{cx}' y2='{120+bar_h}' stroke='rgb(220,38,38)' stroke-width='4'/>")
                    svg_cfd.append(f"<polygon points='{cx-5},{115+bar_h} {cx+5},{115+bar_h} {cx},{125+bar_h}' fill='rgb(220,38,38)'/>")
                    svg_cfd.append(f"<text x='{cx}' y='{138+bar_h}' font-size='10' font-weight='bold' fill='rgb(220,38,38)' text-anchor='middle'>{cf_v:g}</text>")
                svg_cfd.append(f"<text x='{cx}' y='135' font-size='11' fill='black' text-anchor='middle'>n={idx}</text>")
            svg_cfd.append("</svg>")
            st.markdown("\n".join(svg_cfd), unsafe_allow_html=True)

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
            st.dataframe(pd.DataFrame(dep_rows), use_container_width=True)

    render_camera_ai_section(
        "tab2_eco",
        "الاقتصاد الهندسي وحسابات الفوائد (Engineering Economy)",
        "حل مسائل الفوائد المركبة، مخططات التدفق النقدي (Cash Flow Diagrams)، المقارنة بين البدائل الهندسية بأعمار مختلفة (LCM, PW, AW, Incremental IRR, B/C)، تحليل الإحلال (Replacement Analysis)، نقطة التعادل، والتضخم والإهلاك.",
        lambda: eco_tab_mode,
    )


with TABS[2]:
    st.markdown("""
    <div class='sec-header'>
        <h3>🧮 3. الطرق العددية والتحليل العددي الهندسي | Numerical Methods</h3>
        <p>جذور المعادلات غير الخطية (<b>Bisection, False Position, Newton-Raphson, Secant</b>)، الأنظمة الخطية (<b>Gauss-Seidel + SOR + Auto-Pivoting, Jacobi, Cramers Rule</b>)، الاستكمال (<b>Lagrange & Newton Divided Differences</b>)، والتكامل والمعادلات التفاضلية عددياً (<b>Trapezoidal, Simpson 1/3 & 3/8, Euler, Heun, Runge-Kutta RK4</b>).</p>
    </div>
    """, unsafe_allow_html=True)

    num_cat = st.selectbox(
        "🧮 Select Numerical Category | اختر فئة الطرق العددية:",
        [
            "1. Non-Linear Equations Root Finding (Bisection, False Position, Newton-Raphson, Secant)",
            "2. Linear Systems Iterative & Direct (Gauss-Seidel with Auto-Pivoting & SOR, Jacobi, Exact)",
            "3. Numerical ODEs Initial Value Problems (Euler, Heun's Modified Euler, Runge-Kutta 4th Order RK4)",
            "4. Numerical Integration & Interpolation (Trapezoidal, Simpson's 1/3 & 3/8, Newton Divided Differences)",
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
            rows_r = []
            a, b = float(r_x0), float(r_x1)
            for it in range(1, 20):
                if root_algo == "Newton-Raphson":
                    fx = float(eval_math_expr(f_root_str, x=a))
                    dfx = float(numerical_derivative(f_root_str, a))
                    x_new = a - fx / dfx
                    err = abs(x_new - a)
                    rows_r.append({"Iter": it, "x_k": round(a, 6), "f(x_k)": round(fx, 6), "f'(x_k)": round(dfx, 6), "x_{k+1}": round(x_new, 6), "Error": round(err, 7)})
                    a = x_new
                elif root_algo in ["Bisection", "False Position (Regula Falsi)"]:
                    fa, fb = float(eval_math_expr(f_root_str, x=a)), float(eval_math_expr(f_root_str, x=b))
                    xr = (a + b) / 2.0 if root_algo == "Bisection" else (b - fb * (b - a) / (fb - fa))
                    fxr = float(eval_math_expr(f_root_str, x=xr))
                    err = abs(b - a) / 2.0 if root_algo == "Bisection" else abs(fxr)
                    rows_r.append({"Iter": it, "a": round(a, 6), "b": round(b, 6), "x_r": round(xr, 6), "f(x_r)": round(fxr, 6), "Error": round(err, 7)})
                    if fa * fxr < 0:
                        b = xr
                    else:
                        a = xr
                else:
                    fa, fb = float(eval_math_expr(f_root_str, x=a)), float(eval_math_expr(f_root_str, x=b))
                    x_new = b - fb * (b - a) / (fb - fa)
                    err = abs(x_new - b)
                    rows_r.append({"Iter": it, "x_{k-1}": round(a, 6), "x_k": round(b, 6), "x_{k+1}": round(x_new, 6), "f(x_{k+1})": round(float(eval_math_expr(f_root_str, x=x_new)), 6), "Error": round(err, 7)})
                    a, b = b, x_new
                if err < 1e-6:
                    break
            st.dataframe(pd.DataFrame(rows_r), use_container_width=True)

    elif num_cat.startswith("2."):
        aug_txt = st.text_area("✍️ Augmented Matrix [A | B] (each row in a line):", "10 -1 2 0 6\n-1 11 -1 3 25\n2 -1 10 -1 -11\n0 3 -1 8 15", height=105)
        gs_col1, gs_col2, gs_col3 = st.columns(3)
        with gs_col1:
            lin_algo = st.selectbox("Method:", ["Gauss-Seidel", "Jacobi"])
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
            st.info(f"🎯 Exact Solution (NumPy LU): `{np.round(np.linalg.solve(A, B), 6)}`")

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
            xv = float(x0_v)
            ye, yh, yrk = float(y0_v), float(y0_v), float(y0_v)
            ode_rows = [{"Step": 0, "x": xv, "Euler y": ye, "Heun (RK2) y": yh, "RK4 y": yrk, "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0}]
            for s_idx in range(1, int(steps_n) + 1):
                ye_next = ye + h_v * float(eval_math_expr(ode_f_str, x=xv, y=ye))
                k1_h = float(eval_math_expr(ode_f_str, x=xv, y=yh))
                k2_h = float(eval_math_expr(ode_f_str, x=xv + h_v, y=yh + h_v * k1_h))
                yh_next = yh + (h_v / 2.0) * (k1_h + k2_h)

                k1 = h_v * float(eval_math_expr(ode_f_str, x=xv, y=yrk))
                k2 = h_v * float(eval_math_expr(ode_f_str, x=xv + 0.5 * h_v, y=yrk + 0.5 * k1))
                k3 = h_v * float(eval_math_expr(ode_f_str, x=xv + 0.5 * h_v, y=yrk + 0.5 * k2))
                k4 = h_v * float(eval_math_expr(ode_f_str, x=xv + h_v, y=yrk + k3))
                yrk_next = yrk + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

                xv = round(xv + h_v, 6)
                ye, yh, yrk = ye_next, yh_next, yrk_next
                ode_rows.append({"Step": s_idx, "x": xv, "Euler y": round(ye, 6), "Heun (RK2) y": round(yh, 6), "RK4 y": round(yrk, 6), "k1": round(k1, 5), "k2": round(k2, 5), "k3": round(k3, 5), "k4": round(k4, 5)})
            st.dataframe(pd.DataFrame(ode_rows), use_container_width=True)

    else:
        ic1, ic2, ic3, ic4 = st.columns(4)
        with ic1:
            int_f_str = st.text_input("Function f(x) to Integrate:", value="1 / (1 + x**2)")
        with ic2:
            ia_v = st.number_input("Lower bound a:", value=0.0)
        with ic3:
            ib_v = st.number_input("Upper bound b:", value=1.0)
        with ic4:
            in_sub = st.number_input("Subintervals n (multiple of 6):", value=6, min_value=2)

        pts_interp = st.text_input("Interpolation Points (x0,y0 ; x1,y1 ; ...):", value="0,1 ; 1,2 ; 2,5 ; 3,10")
        if st.button("⚙️ Compute Trapezoidal, Simpson 1/3, Simpson 3/8 & Divided Differences", type="primary"):
            n_i = int(in_sub)
            xn = np.linspace(ia_v, ib_v, n_i + 1)
            yn = np.array([float(eval_math_expr(int_f_str, x=xv)) for xv in xn])
            h_i = (ib_v - ia_v) / n_i
            trap_i = (h_i / 2.0) * (yn[0] + 2.0 * np.sum(yn[1:-1]) + yn[-1])
            simp13 = (h_i / 3.0) * (yn[0] + 4.0 * np.sum(yn[1:-1:2]) + 2.0 * np.sum(yn[2:-2:2]) + yn[-1]) if n_i % 2 == 0 else 0.0
            simp38 = (3.0 * h_i / 8.0) * (yn[0] + 3.0 * np.sum(yn[1:-1][np.arange(1, n_i) % 3 != 0]) + 2.0 * np.sum(yn[3:-1:3]) + yn[-1]) if n_i % 3 == 0 else 0.0
            st.success(f"📐 **Trapezoidal:** `{trap_i:.7f}` | **Simpson's 1/3:** `{simp13:.7f}` | **Simpson's 3/8:** `{simp38:.7f}`")

            pts = [list(map(float, p.split(","))) for p in pts_interp.split(";") if "," in p]
            n_p = len(pts)
            dd_table = np.zeros((n_p, n_p + 1))
            for i in range(n_p):
                dd_table[i, 0] = pts[i][0]
                dd_table[i, 1] = pts[i][1]
            for j in range(2, n_p + 1):
                for i in range(n_p - j + 1):
                    dd_table[i, j] = (dd_table[i + 1, j - 1] - dd_table[i, j - 1]) / (dd_table[i + j - 1, 0] - dd_table[i, 0])
            cols_dd = ["x", "f(x)"] + [f"Order {k}" for k in range(1, n_p)]
            st.markdown("#### 📋 جدول الفروق المقسومة لنيوتن | Newton's Divided Differences Table:")
            st.dataframe(pd.DataFrame(np.round(dd_table, 5), columns=cols_dd), use_container_width=True)

    render_camera_ai_section(
        "tab3_num",
        "الطرق العددية والتحليل العددي الهندسي (Numerical Methods)",
        "حل مسائل التحليل العددي بالجداول الكاملة: Bisection, False Position, Newton-Raphson, Secant, Fixed-Point, Gauss-Seidel, Jacobi, LU, Lagrange & Newton Divided Differences, Splines, Least Squares, Trapezoidal, Simpson 1/3 & 3/8, Euler, Heun, RK4.",
        lambda: num_cat,
    )


with TABS[3]:
    st.markdown("""
    <div class='sec-header'>
        <h3>📈 4. المعادلات التفاضلية، تحويلات لابلاس وفورييه، والمصفوفات المتقدمة | ODEs, Laplace & Eigenvalues</h3>
        <p>حل الأنظمة التفاضلية من الرتبة الثانية، تفكيك الكسور الجزئية لتحويلات لابلاس العكسية (<b>Residue & Partial Fractions</b>)، وحساب القيم والمتجهات الذاتية (<b>Eigenvalues & Eigenvectors</b>).</p>
    </div>
    """, unsafe_allow_html=True)

    c_ode1, c_ode2, c_ode3 = st.columns(3)
    with c_ode1:
        a_ode = st.number_input("Coefficient a (a·y'' + b·y' + c·y = 0):", value=1.0)
    with c_ode2:
        b_ode = st.number_input("Coefficient b:", value=5.0)
    with c_ode3:
        c_ode = st.number_input("Coefficient c:", value=6.0)

    if st.button("⚙️ Solve Characteristic Equation & Homogeneous 2nd-Order ODE", type="primary"):
        roots_c = np.roots([a_ode, b_ode, c_ode])
        r1, r2 = roots_c[0], roots_c[1]
        st.write(f"**Characteristic Roots ($a r^2 + b r + c = 0$):** `r1 = {r1:.4f}`, `r2 = {r2:.4f}`")
        if np.isreal(r1) and abs(r1 - r2) > 1e-7:
            st.latex(f"y(t) = C_1 e^{{{r1.real:.3f}t}} + C_2 e^{{{r2.real:.3f}t}}")
        elif abs(r1 - r2) <= 1e-7:
            st.latex(f"y(t) = (C_1 + C_2 t) e^{{{r1.real:.3f}t}}")
        else:
            alpha, beta = r1.real, abs(r1.imag)
            st.latex(f"y(t) = e^{{{alpha:.3f}t}} \\left(C_1 \\cos({beta:.3f}t) + C_2 \\sin({beta:.3f}t)\\right)")

    render_camera_ai_section(
        "tab4_math",
        "المعادلات التفاضلية، تحويلات لابلاس وفورييه، والتكامل المتعدد (Advanced Engineering Mathematics)",
        "حل المعادلات التفاضلية العادية والجزئية (ODEs & PDEs)، متسلسلات وتحويلات فورييه، تحويل لابلاس المباشر والعكسي، Z-Transform، التكامل الثنائي والثلاثي، ومصفوفة جاكوبيان.",
        lambda: f"ODE: {a_ode}y'' + {b_ode}y' + {c_ode}y = 0",
    )


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

        if st.button("⚙️ Compute Principal Stresses & Draw Mohr's Circle", type="primary"):
            s_avg = (sig_x + sig_y) / 2.0
            R = math.hypot((sig_x - sig_y) / 2.0, tau_xy)
            s1, s2 = s_avg + R, s_avg - R
            th_p = 0.5 * math.degrees(math.atan2(2.0 * tau_xy, sig_x - sig_y))
            rad_2th = math.radians(2.0 * th_deg)
            sx_p = s_avg + ((sig_x - sig_y) / 2.0) * math.cos(rad_2th) + tau_xy * math.sin(rad_2th)
            txy_p = -((sig_x - sig_y) / 2.0) * math.sin(rad_2th) + tau_xy * math.cos(rad_2th)

            st.latex(f"\\sigma_{{avg}} = {s_avg:.2f}\\,\\text{{MPa}}, \\quad R = \\tau_{{max}} = {R:.2f}\\,\\text{{MPa}}, \\quad \\sigma_1 = {s1:.2f}\\,\\text{{MPa}}, \\quad \\sigma_2 = {s2:.2f}\\,\\text{{MPa}}, \\quad \\theta_p = {th_p:.2f}^\\circ")
            st.info(f"📐 At inclined plane $\\theta = {th_deg}^\\circ$: $\\sigma_{{x'}} = {sx_p:.2f}\\,\\text{{MPa}}, \\quad \\tau_{{x'y'}} = {txy_p:.2f}\\,\\text{{MPa}}$")

            scale_m = 110.0 / max(1.0, R + abs(s_avg))
            cx_m = 200 + s_avg * scale_m
            cy_m = 160
            rad_px = R * scale_m
            x1_p, y1_p = 200 + sig_x * scale_m, 160 + tau_xy * scale_m
            x2_p, y2_p = 200 + sig_y * scale_m, 160 - tau_xy * scale_m
            svg_mohr = f"""
            <svg viewBox='0 0 420 320' style='width:420px; height:auto; background:white; border:1px solid rgb(203,213,225); border-radius:8px;'>
                <line x1='20' y1='160' x2='400' y2='160' stroke='black' stroke-width='1.5'/>
                <line x1='200' y1='20' x2='200' y2='300' stroke='black' stroke-width='1.5'/>
                <circle cx='{cx_m}' cy='{cy_m}' r='{rad_px}' fill='rgba(59,130,246,0.12)' stroke='rgb(29,78,216)' stroke-width='2.5'/>
                <line x1='{x1_p}' y1='{y1_p}' x2='{x2_p}' y2='{y2_p}' stroke='rgb(220,38,38)' stroke-width='2' stroke-dasharray='5,3'/>
                <circle cx='{cx_m}' cy='{cy_m}' r='4' fill='black'/>
                <text x='210' y='15' font-size='12' font-weight='bold'>Mohr's Circle (σ1={s1:.1f}, σ2={s2:.1f}, τmax={R:.1f})</text>
            </svg>
            """
            st.markdown(svg_mohr, unsafe_allow_html=True)

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

            x_b = np.linspace(0, L_beam, 200)
            V_x = Ra - w_udl * x_b - np.where(x_b >= a_pos, P_load, 0.0)
            M_x = Ra * x_b - 0.5 * w_udl * (x_b ** 2) - np.where(x_b >= a_pos, P_load * (x_b - a_pos), 0.0)
            df_beam = pd.DataFrame({"x (m)": np.round(x_b, 3), "Shear Force V (kN)": V_x, "Bending Moment M (kN·m)": M_x}).set_index("x (m)")
            st.markdown("#### 📉 Shear Force Diagram (SFD) & Bending Moment Diagram (BMD):")
            st.line_chart(df_beam)

    render_camera_ai_section(
        "tab5_mech",
        "الميكانيكا التطبيقية ومقاومة المواد وتصميم الماكينات (Mechanics of Materials & Machine Design)",
        "تحليل صور الكمرات (Beams SFD/BMD)، الجمالونات (Trusses)، دائرة مور، الالتواء (Torsion)، انبعاج الأعمدة (Buckling)، والاهتزازات الميكانيكية.",
        lambda: mech_sub,
    )


with TABS[5]:
    st.markdown("""
    <div class='sec-header'>
        <h3>⚡ 6. الدوائر الكهربية، القوى ثلاثية الأوجه، والمحولات | Electric Circuits, 3-Phase & Transformers</h3>
        <p>تحليل اختبارات المحولات الكهربية (<b>Open-Circuit & Short-Circuit Tests</b>)، حسابات القدرة ثلاثية الأوجه وتحسين معامل القدرة (<b>3-Phase Power Factor Correction</b>).</p>
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
        "حل مسائل دوائر التيار المستمر والمتردد (Nodal, Mesh, Thevenin, Norton, Transient RLC)، الأنظمة ثلاثية الأوجه، المحولات، والآلات الكهربية.",
        lambda: f"Transformer OC/SC: Voc={v_oc}, Ioc={i_oc}, Poc={p_oc}, Vsc={v_sc}, Isc={i_sc}, Psc={p_sc}",
    )


with TABS[6]:
    st.markdown("""
    <div class='sec-header'>
        <h3>🎛️ 7. هندسة التحكم الآلي وحالة الفراغ | Control Systems, Routh-Hurwitz, Bode & State-Space</h3>
        <p>تحليل دالة النقل <b>G(s)</b>، حساب الأقطاب ومعامل التخميد ($\omega_n, \zeta, M_p, T_s$)، ورسم <b>Step Response & Bode Magnitude</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        num_g = st.text_input("Numerator Coefficients of G(s):", value="25")
    with cc2:
        den_g = st.text_input("Denominator Coefficients of G(s) (e.g. s^2 + 3s + 25 -> 1, 3, 25):", value="1, 3, 25")

    if st.button("⚙️ Analyze Transfer Function, Stability, Step Response & Bode", type="primary"):
        num_arr = [float(x.strip()) for x in num_g.split(",") if x.strip()]
        den_arr = [float(x.strip()) for x in den_g.split(",") if x.strip()]
        poles = np.roots(den_arr)
        st.write(f"**System Poles:** `{np.round(poles, 4)}` | **Stable:** `{'YES ✅' if np.all(np.real(poles) < 0) else 'NO ⚠️'}`")

        if len(den_arr) == 3 and den_arr[0] > 0:
            wn = math.sqrt(den_arr[2] / den_arr[0])
            zeta = (den_arr[1] / den_arr[0]) / (2.0 * wn)
            dc_gain = num_arr[-1] / den_arr[-1]
            mp = 100.0 * math.exp(-zeta * math.pi / math.sqrt(1.0 - zeta ** 2)) if 0 < zeta < 1 else 0.0
            ts = 4.0 / (zeta * wn) if zeta > 0 else 0.0
            st.latex(f"\\omega_n = {wn:.3f}\\,\\text{{rad/s}}, \\quad \\zeta = {zeta:.3f}, \\quad M_p = {mp:.2f}\\%, \\quad T_s(2\\%) = {ts:.3f}\\,\\text{{s}}")

            t_vec = np.linspace(0, max(5.0, ts * 1.5), 250)
            if 0 < zeta < 1:
                wd = wn * math.sqrt(1.0 - zeta ** 2)
                phi = math.acos(zeta)
                y_vec = dc_gain * (1.0 - (np.exp(-zeta * wn * t_vec) / math.sqrt(1.0 - zeta ** 2)) * np.sin(wd * t_vec + phi))
            else:
                y_vec = dc_gain * (1.0 - np.exp(-wn * t_vec))
            st.markdown("#### 📈 Unit Step Response y(t):")
            st.line_chart(pd.DataFrame({"t (s)": np.round(t_vec, 3), "y(t)": y_vec}).set_index("t (s)"))

        w_freqs = np.logspace(-1, 2, 150)
        s_jw = 1j * w_freqs
        H_jw = np.polyval(num_arr, s_jw) / np.polyval(den_arr, s_jw)
        mag_db = 20.0 * np.log10(np.maximum(1e-12, np.abs(H_jw)))
        st.markdown("#### 📊 Bode Magnitude Plot (dB vs rad/s):")
        st.line_chart(pd.DataFrame({"w (rad/s)": np.round(w_freqs, 2), "Magnitude (dB)": mag_db}).set_index("w (rad/s)"))

    render_camera_ai_section(
        "tab7_ctrl",
        "هندسة التحكم الآلي والـ PLC والميكاترونكس (Control Systems & Automation)",
        "حل مسائل Block Diagram Reduction, Signal Flow Graph (Mason's Formula), Routh-Hurwitz Stability Criterion, Root Locus, Bode & Nyquist, PID Tuning, State-Space, و PLC Ladder Logic.",
        lambda: f"G(s) Num={num_g}, Den={den_g}",
    )


with TABS[7]:
    st.markdown("""
    <div class='sec-header'>
        <h3>💻 8. البرمجة الهندسية وتصدير التقارير | Engineering Code Sandbox & Reports</h3>
        <p>تنفيذ أكواد بايثون العلمية مباشرة، تصحيح الأكواد الهندسية (<b>Python, C++, MATLAB, Arduino/AVR</b>)، وتصدير سجل جميع العمليات.</p>
    </div>
    """, unsafe_allow_html=True)

    py_code = st.text_area(
        "✍️ Python / NumPy Engineering Sandbox:",
        value="import numpy as np\nA = np.array([[4, 1], [2, 3]])\nprint('Eigenvalues:', np.linalg.eigvals(A))",
        height=110,
    )
    if st.button("▶️ Execute Python Code | تشغيل الكود", type="primary"):
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
