import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad

# === إعدادات الصفحة ===
st.set_page_config(page_title="Numerical Integration", page_icon="✨", layout="wide")

# === تصميم CSS ===
st.markdown("""
<style>
.control-panel { background-color: #f8f9fa; padding: 20px; border-radius: 15px; border: 1px solid #e9ecef; }
.result-card { background-color: #ffffff; padding: 20px; border-radius: 15px; border-left: 5px solid #007bff; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; }
.exact-card { border-left: 5px solid #28a745; }
.card-title { color: #6c757d; font-size: 16px; font-weight: bold; margin-bottom: 5px; }
.card-value { color: #212529; font-size: 28px; font-weight: bold; }
.card-subtext { color: #dc3545; font-size: 14px; margin-top: 5px; }
div[data-testid="column"] { padding: 0 2px !important; }
.stButton > button { width: 100% !important; padding: 2px 5px !important; font-size: 14px !important; font-weight: bold !important; height: 35px !important; margin-bottom: 5px !important; }
</style>
""", unsafe_allow_html=True)

if 'func_text' not in st.session_state:
    st.session_state.func_text = ""

def append_to_func(text):
    st.session_state.func_text += text

def clear_func():
    st.session_state.func_text = ""

# === تقسيم الشاشة ===
col_control, col_display = st.columns([1, 2.2])

# ==========================================
# ⚙️ العمود الأيسر: لوحة التحكم والآلة الحاسبة
# ==========================================
with col_control:
    st.markdown("### 🎛️ إعدادات المسألة")
    st.markdown("---")
    
    func_input = st.text_input("الدالة الرياضية f(x):", key="func_text", placeholder="مثال: x**2 + sin(x)")
    
    st.markdown("<small style='color:gray;'>لوحة الرموز الرياضية:</small>", unsafe_allow_html=True)
    r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
    r1c1.button("sin", on_click=append_to_func, args=("sin(",))
    r1c2.button("cos", on_click=append_to_func, args=("cos(",))
    r1c3.button("tan", on_click=append_to_func, args=("tan(",))
    r1c4.button("π", on_click=append_to_func, args=("pi",))
    r1c5.button("🧹", on_click=clear_func, type="secondary", help="مسح الكل")

    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
    r2c1.button("sin⁻¹", on_click=append_to_func, args=("asin(",))
    r2c2.button("cos⁻¹", on_click=append_to_func, args=("acos(",))
    r2c3.button("tan⁻¹", on_click=append_to_func, args=("atan(",))
    r2c4.button("e", on_click=append_to_func, args=("e",))
    r2c5.button("√x", on_click=append_to_func, args=("sqrt(",))

    r3c1, r3c2, r3c3, r3c4, r3c5 = st.columns(5)
    r3c1.button("ln", on_click=append_to_func, args=("ln(",))
    r3c2.button("x²", on_click=append_to_func, args=("**2",))
    r3c3.button("xʸ", on_click=append_to_func, args=("**",))
    r3c4.button("÷", on_click=append_to_func, args=("/",))
    r3c5.button("( )", on_click=append_to_func, args=("()",))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    angle_mode = st.radio("نظام الزوايا (للدوال المثلثية):", ["راديان (Rad)", "درجات (Deg)"], horizontal=True)
    
    # --- التعديل الجديد: خانات الحدود أصبحت نصية لتقبل الرموز ---
    c1, c2 = st.columns(2)
    with c1:
        a_str = st.text_input("الحد الأدنى (a):", value="", placeholder="مثال: 0, pi, e")
    with c2:
        b_str = st.text_input("الحد الأقصى (b):", value="", placeholder="مثال: pi/2, exp(1)")
        
    method = st.selectbox("اختر طريقة الحل:", [
        "شبه المنحرف (Trapezoidal)", 
        "سمبسون 1/3 (Simpson 1/3)", 
        "سمبسون 3/8 (Simpson 3/8)"
    ])
    
    input_type = st.radio("طريقة الإدخال:", ["لدي عدد القطاعات (n)", "لدي حجم الخطوة (h)"])
    val = st.number_input("أدخل القيمة (n أو h):", value=None, min_value=0.0001, placeholder="اكتب رقم...")
    
    st.markdown("<br>", unsafe_allow_html=True)
    calc_btn = st.button("🚀 احسب وارسم", type="primary", use_container_width=True)

# ==========================================
# 📊 العمود الأيمن: شاشة النتائج والرسم
# ==========================================
with col_display:
    if calc_btn:
        if not func_input.strip():
            st.warning("⚠️ يرجى إدخال الدالة الرياضية أولاً.")
            st.stop()
        if not a_str.strip() or not b_str.strip() or val is None:
            st.warning("⚠️ يرجى تعبئة خانات الحد الأدنى والأقصى والقيمة قبل الحساب.")
            st.stop()
            
        func_str = func_input.replace('^', '**').replace('ln', 'log')
        
        try:
            # --- ترجمة الرموز في حدود التكامل إلى أرقام ---
            a_val = float(sp.sympify(a_str, locals={'e': sp.E}))
            b_val = float(sp.sympify(b_str, locals={'e': sp.E}))
        except Exception:
            st.error("❌ خطأ: يرجى كتابة حدود التكامل بشكل صحيح (أرقام أو رموز مثل pi و e).")
            st.stop()
        
        if a_val >= b_val:
            st.error("❌ خطأ: الحد الأقصى يجب أن يكون أكبر من الحد الأدنى.")
        else:
            x = sp.Symbol('x')
            try:
                custom_dict = {'e': sp.E}
                if "Deg" in angle_mode:
                    custom_dict.update({
                        'sin': lambda arg: sp.sin(arg * sp.pi / 180),
                        'cos': lambda arg: sp.cos(arg * sp.pi / 180),
                        'tan': lambda arg: sp.tan(arg * sp.pi / 180),
                        'asin': lambda arg: sp.asin(arg) * 180 / sp.pi,
                        'acos': lambda arg: sp.acos(arg) * 180 / sp.pi,
                        'atan': lambda arg: sp.atan(arg) * 180 / sp.pi,
                    })

                f_expr = sp.sympify(func_str, locals=custom_dict)
                f = sp.lambdify(x, f_expr, "numpy")
                f(a_val) 
                
                exact_val, exact_error = quad(f, a_val, b_val)
                
                if input_type == "لدي حجم الخطوة (h)":
                    h = val
                    n_calc = (b_val - a_val) / h
                    n = int(round(n_calc))
                    if abs(n_calc - n) > 1e-6:
                        st.error(f"❌ خطأ: الخطوة h={h} لا تقسم المسافة بشكل صحيح.")
                        st.stop()
                else:
                    n = int(val)
                    h = (b_val - a_val) / n

                if "1/3" in method and n % 2 != 0:
                    st.error(f"❌ خطأ: طريقة سمبسون 1/3 تتطلب (n) زوجياً. إدخالك يعطي n = {n}")
                    st.stop()
                elif "3/8" in method and n % 3 != 0:
                    st.error(f"❌ خطأ: طريقة سمبسون 3/8 تتطلب (n) من مضاعفات 3. إدخالك يعطي n = {n}")
                    st.stop()

                x_vals = np.linspace(a_val, b_val, n + 1)
                y_vals = f(x_vals)
                if isinstance(y_vals, (int, float)):
                    y_vals = np.full_like(x_vals, y_vals)

                if "شبه المنحرف" in method:
                    approx_val = (h / 2) * (y_vals[0] + 2 * np.sum(y_vals[1:-1]) + y_vals[-1])
                elif "1/3" in method:
                    approx_val = (h / 3) * (y_vals[0] + 4 * np.sum(y_vals[1:-1:2]) + 2 * np.sum(y_vals[2:-2:2]) + y_vals[-1])
                elif "3/8" in method:
                    integral = y_vals[0] + y_vals[-1]
                    for i in range(1, n):
                        integral += 2 * y_vals[i] if i % 3 == 0 else 3 * y_vals[i]
                    approx_val = (3 * h / 8) * integral

                abs_error = abs(exact_val - approx_val)

                st.markdown("### 📈 ملخص النتائج")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown(f"""
                    <div class='result-card exact-card'>
                        <div class='card-title'>🎯 القيمة الدقيقة (Exact)</div>
                        <div class='card-value'>{exact_val:.6f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with res_col2:
                    st.markdown(f"""
                    <div class='result-card'>
                        <div class='card-title'>🛠️ نتيجتك ({method.split(' ')[0]})</div>
                        <div class='card-value'>{approx_val:.6f}</div>
                        <div class='card-subtext'>الخطأ الفعلي: {abs_error:.2e} | n = {n} | h = {h:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("### 🎨 الرسم البياني للمساحة")
                x_smooth = np.linspace(a_val, b_val, 500)
                y_smooth = f(x_smooth)
                if isinstance(y_smooth, (int, float)):
                    y_smooth = np.full_like(x_smooth, y_smooth)

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(x_smooth, y_smooth, color='#007bff', linewidth=2, label='f(x)')
                
                if n <= 100:
                    for i in range(n):
                        xs = [x_vals[i], x_vals[i], x_vals[i+1], x_vals[i+1]]
                        ys = [0, y_vals[i], y_vals[i+1], 0]
                        ax.fill(xs, ys, color='#007bff', alpha=0.15, edgecolor='#0056b3', linewidth=1)
                else:
                    ax.fill_between(x_smooth, 0, y_smooth, color='#007bff', alpha=0.2)

                ax.axhline(0, color='black', linewidth=1)
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_ylabel("f(x)")
                ax.set_xlabel("x")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].
 
