import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad

# === إعدادات الصفحة ===
st.set_page_config(page_title="Numerical Analysis Pro", page_icon="🔢", layout="wide")

# === تصميم CSS (لجعل الأزرار تبدو كآلة حاسبة حقيقية متلاصقة) ===
st.markdown("""
<style>
.control-panel { background-color: #f8f9fa; padding: 20px; border-radius: 15px; border: 1px solid #e9ecef; }
.result-card { background-color: #ffffff; padding: 20px; border-radius: 15px; border-left: 5px solid #007bff; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; }
.exact-card { border-left: 5px solid #28a745; }
.card-title { color: #6c757d; font-size: 16px; font-weight: bold; margin-bottom: 5px; }
.card-value { color: #212529; font-size: 28px; font-weight: bold; }
.card-subtext { color: #dc3545; font-size: 14px; margin-top: 5px; }

/* تقليل المسافات بين أزرار الآلة الحاسبة وتكبير الخط */
div[data-testid="column"] { padding: 0 3px !important; }
.stButton > button { 
    width: 100% !important; 
    padding: 0px !important; 
    font-size: 18px !important; 
    font-weight: bold !important; 
    height: 42px !important; 
    margin-bottom: 4px !important; 
}
</style>
""", unsafe_allow_html=True)

# === إدارة حالة النص ===
if 'func_text' not in st.session_state:
    st.session_state.func_text = ""

def append_to_func(text):
    st.session_state.func_text += text

def clear_func():
    st.session_state.func_text = ""

def backspace_func():
    st.session_state.func_text = st.session_state.func_text[:-1]

# ==========================================
# 📌 القائمة الجانبية (لاختيار القسم)
# ==========================================
st.sidebar.title("🧮 الأقسام الرياضية")
app_mode = st.sidebar.radio("اختر العملية المطلوبة:", 
    ["📈 التكامل العددي (Integration)", "🎯 حل المعادلات (Root Finding)"]
)
st.sidebar.markdown("---")
st.sidebar.info("💡 اكتب الدالة باستخدام الآلة الحاسبة المدمجة لتجنب أخطاء لوحة مفاتيح الهاتف.")

# === تقسيم الشاشة الأساسي ===
col_control, col_display = st.columns([1, 2.5])

# ==========================================
# ⚙️ العمود الأيسر: لوحة التحكم والآلة الحاسبة
# ==========================================
with col_control:
    st.markdown(f"### 🎛️ إعدادات {app_mode.split(' ')[1]}")
    st.markdown("---")
    
    func_input = st.text_input("الدالة الرياضية f(x):", key="func_text", placeholder="استخدم الأزرار بالأسفل للكتابة...")
    
    # --- 🧮 لوحة الآلة الحاسبة المتكاملة ---
    st.markdown("<small style='color:gray;'>الآلة الحاسبة:</small>", unsafe_allow_html=True)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.button("sin", on_click=append_to_func, args=("sin(",))
    c2.button("cos", on_click=append_to_func, args=("cos(",))
    c3.button("tan", on_click=append_to_func, args=("tan(",))
    c4.button("π", on_click=append_to_func, args=("pi",))
    c5.button("🧹", on_click=clear_func, type="secondary", help="مسح الكل")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.button("sin⁻¹", on_click=append_to_func, args=("asin(",))
    c2.button("cos⁻¹", on_click=append_to_func, args=("acos(",))
    c3.button("tan⁻¹", on_click=append_to_func, args=("atan(",))
    c4.button("e^( )", on_click=append_to_func, args=("exp(",)) # الزرار بالشكل المطلوب
    c5.button("√", on_click=append_to_func, args=("sqrt(",))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.button("7", on_click=append_to_func, args=("7",))
    c2.button("8", on_click=append_to_func, args=("8",))
    c3.button("9", on_click=append_to_func, args=("9",))
    c4.button("DEL", on_click=backspace_func) # زرار مسح حرف بحرف
    c5.button("x", on_click=append_to_func, args=("x",))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.button("4", on_click=append_to_func, args=("4",))
    c2.button("5", on_click=append_to_func, args=("5",))
    c3.button("6", on_click=append_to_func, args=("6",))
    c4.button("×", on_click=append_to_func, args=("*",))
    c5.button("÷", on_click=append_to_func, args=("/",))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.button("1", on_click=append_to_func, args=("1",))
    c2.button("2", on_click=append_to_func, args=("2",))
    c3.button("3", on_click=append_to_func, args=("3",))
    c4.button("+", on_click=append_to_func, args=("+",))
    c5.button("-", on_click=append_to_func, args=("-",))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.button("0", on_click=append_to_func, args=("0",))
    c2.button(".", on_click=append_to_func, args=(".",))
    c3.button("(", on_click=append_to_func, args=("(",))
    c4.button(")", on_click=append_to_func, args=(")",))
    c5.button("xʸ", on_click=append_to_func, args=("**",))

    st.markdown("<br>", unsafe_allow_html=True)
    
    angle_mode = st.radio("نظام الزوايا:", ["راديان (Rad)", "درجات (Deg)"], horizontal=True)

    if "Root" in app_mode:
        method = st.selectbox("اختر طريقة الحل:", ["طريقة التنصيف (Bisection)", "نيوتن-رافسون (Newton-Raphson)"])
        if "Bisection" in method:
            col1, col2 = st.columns(2)
            with col1:
                a_str = st.text_input("من (a):", value="", placeholder="مثال: 0")
            with col2:
                b_str = st.text_input("إلى (b):", value="", placeholder="مثال: 3")
        else:
            a_str = st.text_input("نقطة التخمين المبدئية (x0):", value="", placeholder="مثال: 1")
            b_str = "0"
            
        tol_str = st.text_input("نسبة الخطأ المقبولة (Tolerance):", value="1e-6")
        
    else:
        col1, col2 = st.columns(2)
        with col1:
            a_str = st.text_input("الحد الأدنى (a):", value="", placeholder="مثال: 0")
        with col2:
            b_str = st.text_input("الحد الأقصى (b):", value="", placeholder="مثال: pi")
            
        method = st.selectbox("اختر طريقة الحل:", ["شبه المنحرف (Trapezoidal)", "سمبسون 1/3 (Simpson 1/3)", "سمبسون 3/8 (Simpson 3/8)"])
        input_type = st.radio("طريقة الإدخال:", ["لدي عدد القطاعات (n)", "لدي حجم الخطوة (h)"])
        val = st.number_input("أدخل القيمة (n أو h):", value=None, min_value=0.0001, placeholder="اكتب رقم...")

    st.markdown("<br>", unsafe_allow_html=True)
    calc_btn = st.button("🚀 احسب وارسم", type="primary", use_container_width=True)

# ==========================================
# 📊 العمود الأيمن: شاشة النتائج والرسم
# ==========================================
with col_display:
    if calc_btn:
        if not func_input.strip() or not a_str.strip():
            st.warning("⚠️ يرجى تعبئة الدالة والحدود أولاً.")
            st.stop()
            
        # 💡 السر هنا: مسح أي مسافات حطها كيبورد الموبايل بالغلط
        func_clean = func_input.replace('^', '**').replace('ln', 'log').replace(' ', '')
        a_clean = a_str.replace(' ', '')
        b_clean = b_str.replace(' ', '')
        
        x = sp.Symbol('x')
        custom_dict = {'e': sp.E}
        
        if "Deg" in angle_mode:
            custom_dict.update({
                'sin': lambda arg: sp.sin(arg * sp.pi / 180),
                'cos': lambda arg: sp.cos(arg * sp.pi / 180),
                'tan': lambda arg: sp.tan(arg * sp.pi / 180),
            })

        try:
            f_expr = sp.sympify(func_clean, locals=custom_dict)
            f = sp.lambdify(x, f_expr, "numpy")
            a_val = float(sp.sympify(a_clean, locals={'e': sp.E}))
            b_val = float(sp.sympify(b_clean, locals={'e': sp.E}))
            
            if "Root" in app_mode:
                tol_clean = tol_str.replace(' ', '') # مسح مسافات الـ Tolerance
                tol_val = float(sp.sympify(tol_clean, locals={'e': sp.E}))
            
        except Exception as e:
            st.error(f"❌ خطأ: يرجى كتابة الدالة والأرقام بشكل صحيح. تأكد من إغلاق الأقواس. (السبب البرمجي: {e})")
            st.stop()

        # ========================================================
        # 🟢 تنفيذ كود حل المعادلات (Root Finding)
        # ========================================================
        if "Root" in app_mode:
            root = None
            iterations = 0
            error = 0
            
            if "Bisection" in method:
                if a_val >= b_val:
                    st.error("❌ خطأ: النقطة (b) يجب أن تكون أكبر من (a).")
                    st.stop()
                
                fa = f(a_val)
                fb = f(b_val)
                
                if fa * fb > 0:
                    st.error(f"❌ خطأ: الدالة لا تقطع محور السينات بين {a_val} و {b_val}.")
                    st.stop()
                
                a_temp, b_temp = a_val, b_val
                for i in range(100):
                    c = (a_temp + b_temp) / 2.0
                    fc = f(c)
                    if abs(fc) < tol_val or (b_temp - a_temp) / 2.0 < tol_val:
                        root, iterations, error = c, i+1, abs(fc)
                        break
                    if fa * fc < 0:
                        b_temp = c
                    else:
                        a_temp = c
                        fa = fc
                if root is None: root, iterations, error = c, 100, abs(fc)

            else:
                try:
                    df_expr = sp.diff(f_expr, x)
                    df = sp.lambdify(x, df_expr, "numpy")
                except Exception:
                    st.error("❌ فشل في حساب اشتقاق الدالة.")
                    st.stop()
                
                x_curr = a_val
                for i in range(100):
                    fx = f(x_curr)
                    dfx = df(x_curr)
                    if dfx == 0:
                        st.error("❌ المشتقة تساوي صفر، طريقة نيوتن تفشل هنا.")
                        st.stop()
                        
                    x_next = x_curr - fx / dfx
                    if abs(x_next - x_curr) < tol_val:
                        root, iterations, error = x_next, i+1, abs(f(x_next))
                        break
                    x_curr = x_next
                if root is None: root, iterations, error = x_curr, 100, abs(f(x_curr))

            st.markdown("### 🎯 ملخص إيجاد الجذر")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown(f"<div class='result-card exact-card'><div class='card-title'>📍 قيمة الجذر (Root)</div><div class='card-value'>{root:.6f}</div></div>", unsafe_allow_html=True)
            with res_col2:
                st.markdown(f"<div class='result-card'><div class='card-title'>⚙️ الأداء (Performance)</div><div class='card-value' style='font-size:22px;'>الخطأ: {error:.2e}</div><div class='card-subtext'>تم الوصول للحل بعد {iterations} خطوة</div></div>", unsafe_allow_html=True)
                
            st.markdown("### 🎨 الرسم البياني لتقاطع الدالة")
            x_min = root - 2 if root != 0 else -2
            x_max = root + 2 if root != 0 else 2
            if "Bisection" in method: x_min, x_max = min(a_val, root-1), max(b_val, root+1)

            x_smooth = np.linspace(x_min, x_max, 500)
            y_smooth = f(x_smooth)
            if isinstance(y_smooth, (int, float)): y_smooth = np.full_like(x_smooth, y_smooth)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x_smooth, y_smooth, color='#2d82ff', linewidth=2.5, label='f(x)')
            ax.axhline(0, color='#ff4b4b', linewidth=1.5, linestyle='--')
            ax.plot(root, 0, marker='o', color='#28a745', markersize=10, label=f'Root: {root:.4f}')
            
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#dddddd')
            ax.spines['bottom'].set_color('#dddddd')
            st.pyplot(fig)


        # ========================================================
        # 🔵 تنفيذ كود التكامل (Integration)
        # ========================================================
        else:
            if val is None:
                st.warning("⚠️ يرجى إدخال قيمة n أو h للتكامل.")
                st.stop()
            if a_val >= b_val:
                st.error("❌ خطأ: الحد الأقصى يجب أن يكون أكبر من الحد الأدنى.")
                st.stop()
                
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
                st.error(f"❌ خطأ: طريقة سمبسون 1/3 تتطلب (n) زوجياً.")
                st.stop()
            elif "3/8" in method and n % 3 != 0:
                st.error(f"❌ خطأ: طريقة سمبسون 3/8 تتطلب (n) من مضاعفات 3.")
                st.stop()

            x_vals = np.linspace(a_val, b_val, n + 1)
            y_vals = f(x_vals)
            if isinstance(y_vals, (int, float)): y_vals = np.full_like(x_vals, y_vals)

            if "شبه المنحرف" in method: approx_val = (h / 2) * (y_vals[0] + 2 * np.sum(y_vals[1:-1]) + y_vals[-1])
            elif "1/3" in method: approx_val = (h / 3) * (y_vals[0] + 4 * np.sum(y_vals[1:-1:2]) + 2 * np.sum(y_vals[2:-2:2]) + y_vals[-1])
            elif "3/8" in method:
                integral = y_vals[0] + y_vals[-1]
                for i in range(1, n): integral += 2 * y_vals[i] if i % 3 == 0 else 3 * y_vals[i]
                approx_val = (3 * h / 8) * integral

            abs_error = abs(exact_val - approx_val)

            st.markdown("### 📈 ملخص النتائج")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown(f"<div class='result-card exact-card'><div class='card-title'>🎯 القيمة الدقيقة (Exact)</div><div class='card-value'>{exact_val:.6f}</div></div>", unsafe_allow_html=True)
            with res_col2:
                st.markdown(f"<div class='result-card'><div class='card-title'>🛠️ نتيجتك ({method.split(' ')[0]})</div><div class='card-value'>{approx_val:.6f}</div><div class='card-subtext'>الخطأ الفعلي: {abs_error:.2e} | n = {n} | h = {h:.4f}</div></div>", unsafe_allow_html=True)

            st.markdown("### 🎨 الرسم البياني للمساحة")
            x_smooth = np.linspace(a_val, b_val, 500)
            y_smooth = f(x_smooth)
            if isinstance(y_smooth, (int, float)): y_smooth = np.full_like(x_smooth, y_smooth)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x_smooth, y_smooth, color='#2d82ff', linewidth=2.5, label='f(x)')
            
            if n <= 100:
                for i in range(n):
                    xs = [x_vals[i], x_vals[i], x_vals[i+1], x_vals[i+1]]
                    ys = [0, y_vals[i], y_vals[i+1], 0]
                    ax.fill(xs, ys, color='#2d82ff', alpha=0.15, edgecolor='#0056b3', linewidth=1)
            else:
                ax.fill_between(x_smooth, 0, y_smooth, color='#2d82ff', alpha=0.15)
            
            ax.axhline(0, color='black', linewidth=1, alpha=0.7)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#dddddd')
            ax.spines['bottom'].set_color('#dddddd')
            st.pyplot(fig)
            
    else:
        st.info(f"👋 أهلاً بك! أنت الآن في وضع {app_mode.split(' ')[1]}. قم بإدخال البيانات واضغط على احسب.")
                 
