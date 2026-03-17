import streamlit as st
import numpy as np
import sympy as sp
from scipy.integrate import quad
import plotly.graph_objects as go
import pandas as pd

# === إعدادات الصفحة ===
st.set_page_config(page_title="Numerical Analysis Pro", page_icon="🔢", layout="wide")

# === تصميم CSS ===
st.markdown("""
<style>
.result-card { background-color: var(--secondary-background-color); padding: 20px; border-radius: 15px; border-left: 5px solid #007bff; margin-bottom: 20px; }
.exact-card { border-left: 5px solid #28a745; }
.card-title { font-size: 16px; font-weight: bold; margin-bottom: 5px; opacity: 0.8; }
.card-value { font-size: 28px; font-weight: bold; color: #2d82ff; }
.exact-value { color: #28a745; }
.card-subtext { font-size: 14px; margin-top: 5px; opacity: 0.7; }
div[data-testid="column"] { padding: 0 3px !important; }
.stButton > button { width: 100% !important; padding: 0px !important; font-size: 18px !important; font-weight: bold !important; height: 42px !important; margin-bottom: 4px !important; }
</style>
""", unsafe_allow_html=True)

# === إدارة حالة النص والسجل ===
if 'func_text' not in st.session_state: st.session_state.func_text = ""
if 'preset' not in st.session_state: st.session_state.preset = "اختر مثالاً..."
if 'history' not in st.session_state: st.session_state.history = []

def append_to_func(text): st.session_state.func_text += text
def clear_func(): st.session_state.func_text = ""
def backspace_func(): st.session_state.func_text = st.session_state.func_text[:-1]

def apply_preset():
    p = st.session_state.preset
    if p == "تكامل: دالة أسية (جرسية)": st.session_state.func_text = "exp(-x**2)"
    elif p == "تكامل: دالة مثلثية": st.session_state.func_text = "x * sin(x)"
    elif p == "جذور: دالة تكعيبية": st.session_state.func_text = "x**3 - x - 2"
    elif p == "جذور: دالة أسية": st.session_state.func_text = "exp(-x) - x"

# ==========================================
# 📌 القائمة الجانبية
# ==========================================
st.sidebar.title("🧮 الأقسام الرياضية")
app_mode = st.sidebar.radio("اختر العملية المطلوبة:", [
    "📈 التكامل العددي (Integration)", 
    "🎯 حل المعادلات (Root Finding)",
    "🧮 أنظمة المعادلات الخطية (Linear Systems)" # القسم الجديد
])
st.sidebar.markdown("---")

if "Linear" not in app_mode:
    st.sidebar.selectbox("💡 أمثلة سريعة للتدريب:", 
        ["اختر مثالاً...", "تكامل: دالة أسية (جرسية)", "تكامل: دالة مثلثية", "جذور: دالة تكعيبية", "جذور: دالة أسية"], 
        key='preset', on_change=apply_preset)

# --- 🎯 إضافة شريط للتحكم في دقة الأرقام ---
st.sidebar.markdown("---")
dp = st.sidebar.slider("🎯 دقة الأرقام (Decimal Places):", min_value=2, max_value=10, value=6)

# --- 🕒 تصميم سجل العمليات ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🕒 سجل العمليات (History)")
if st.sidebar.button("🗑️ مسح السجل", use_container_width=True):
    st.session_state.history = []
    st.rerun()

if not st.session_state.history:
    st.sidebar.info("السجل فارغ. ابدأ الحساب الآن!")
else:
    for item in reversed(st.session_state.history):
        st.sidebar.success(item)

# === تقسيم الشاشة الأساسي ===
col_control, col_display = st.columns([1, 2.5])

# ==========================================
# ⚙️ العمود الأيسر: لوحة التحكم
# ==========================================
with col_control:
    st.markdown(f"### 🎛️ إعدادات {app_mode.split(' ')[1]}")
    
    # -----------------------------------------------------
    # 1. إعدادات قسم أنظمة المعادلات الخطية (الجديد)
    # -----------------------------------------------------
    if "Linear" in app_mode:
        st.info("قم بإدخال معاملات المصفوفة والثوابت في الجدول أدناه.")
        method = st.selectbox("اختر طريقة الحل:", ["جاكوبي (Jacobi)", "جاوس-سيدل (Gauss-Seidel)"])
        
        n_vars = st.number_input("عدد المتغيرات (المعادلات):", min_value=2, max_value=10, value=3)
        
        # إنشاء مصفوفة افتراضية مسيطرة قطرياً (Diagonally Dominant) عشان متعملش Divergence
        default_matrix = np.zeros((n_vars, n_vars + 1))
        if n_vars == 3:
            default_matrix = np.array([[5, -1, 1, 10], [2, 8, -1, 11], [-1, 1, 4, 3]], dtype=float)
            
        cols = [f"x{i+1}" for i in range(n_vars)] + ["= b"]
        df_matrix = pd.DataFrame(default_matrix, columns=cols)
        
        st.markdown("**المصفوفة الموسعة $[A | b]$:**")
        edited_df = st.data_editor(df_matrix, use_container_width=True, hide_index=True)
        
        tol_str = st.text_input("نسبة الخطأ المقبولة (Tolerance):", value="1e-6")
        
    # -----------------------------------------------------
    # 2. إعدادات الأقسام القديمة (التكامل والجذور)
    # -----------------------------------------------------
    else:
        func_input = st.text_input("الدالة الرياضية f(x):", key="func_text", placeholder="استخدم الآلة الحاسبة للكتابة...")
        
        with st.expander("🧮 إظهار / إخفاء الآلة الحاسبة", expanded=False):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.button("sin", on_click=append_to_func, args=("sin(",))
            c2.button("cos", on_click=append_to_func, args=("cos(",))
            c3.button("tan", on_click=append_to_func, args=("tan(",))
            c4.button("π", on_click=append_to_func, args=("pi",))
            c5.button("🧹", on_click=clear_func, type="secondary")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.button("sin⁻¹", on_click=append_to_func, args=("asin(",))
            c2.button("cos⁻¹", on_click=append_to_func, args=("acos(",))
            c3.button("tan⁻¹", on_click=append_to_func, args=("atan(",))
            c4.button("e^( )", on_click=append_to_func, args=("exp(",))
            c5.button("√", on_click=append_to_func, args=("sqrt(",))

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.button("7", on_click=append_to_func, args=("7",))
            c2.button("8", on_click=append_to_func, args=("8",))
            c3.button("9", on_click=append_to_func, args=("9",))
            c4.button("DEL", on_click=backspace_func)
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

        angle_mode = st.radio("نظام الزوايا:", ["راديان (Rad)", "درجات (Deg)"], horizontal=True)

        if "Root" in app_mode:
            method = st.selectbox("اختر طريقة الحل:", ["طريقة التنصيف (Bisection)", "الوضع الخاطئ (False Position)", "نيوتن-رافسون (Newton)", "طريقة القاطع (Secant)"])
            if method in ["طريقة التنصيف (Bisection)", "الوضع الخاطئ (False Position)"]:
                col1, col2 = st.columns(2)
                with col1: a_str = st.text_input("من (a):", value="0")
                with col2: b_str = st.text_input("إلى (b):", value="2")
            elif "Newton" in method:
                a_str = st.text_input("نقطة التخمين المبدئية (x0):", value="1")
                b_str = "0"
            elif "Secant" in method:
                col1, col2 = st.columns(2)
                with col1: a_str = st.text_input("التخمين الأول (x0):", value="0")
                with col2: b_str = st.text_input("التخمين الثاني (x1):", value="1")
            tol_str = st.text_input("نسبة الخطأ المقبولة (Tolerance):", value="1e-6")
            
        else:
            col1, col2 = st.columns(2)
            with col1: a_str = st.text_input("الحد الأدنى (a):", value="0")
            with col2: b_str = st.text_input("الحد الأقصى (b):", value="2")
            input_type = st.radio("طريقة الإدخال:", ["لدي عدد القطاعات (n)", "لدي حجم الخطوة (h)"])
            val = st.number_input("أدخل القيمة (n أو h):", value=10.0, min_value=0.0001)

    st.markdown("<br>", unsafe_allow_html=True)
    calc_btn = st.button("🚀 احسب", type="primary", use_container_width=True)

# ==========================================
# 📊 العمود الأيمن: شاشة النتائج والرسم
# ==========================================
with col_display:
    if calc_btn:
        
        # ========================================================
        # 🟢 1. تنفيذ كود أنظمة المعادلات (الجديد)
        # ========================================================
        if "Linear" in app_mode:
            try:
                tol_clean = tol_str.replace(' ', '')
                tol_val = float(sp.sympify(tol_clean, locals={'e': sp.E}))
            except Exception:
                st.error("❌ تأكد من كتابة نسبة الخطأ بشكل صحيح.")
                st.stop()
                
            # سحب المصفوفات من الجدول
            A = edited_df.iloc[:, :-1].values.astype(float)
            b = edited_df.iloc[:, -1].values.astype(float)
            n = len(b)
            
            # التحقق من أصفار القطر الرئيسي
            if np.any(np.diag(A) == 0):
                st.error("❌ يوجد أصفار على القطر الرئيسي (Main Diagonal). يرجى إعادة ترتيب المعادلات لتجنب القسمة على صفر.")
                st.stop()

            st.markdown("### 📚 القانون الرياضي المستخدم:")
            if "Jacobi" in method:
                st.latex(r"x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j \neq i} a_{ij} x_j^{(k)} \right)")
            else:
                st.latex(r"x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)} \right)")

            x = np.zeros(n) # التخمين المبدئي بأصفار
            steps_data = []
            iterations = 0
            final_error = 0
            max_iter = 100
            
            for it in range(max_iter):
                x_new = np.zeros(n)
                for i in range(n):
                    s = 0
                    for j in range(n):
                        if i != j:
                            if "Jacobi" in method:
                                s += A[i, j] * x[j]
                            else: # Gauss-Seidel
                                s += A[i, j] * (x_new[j] if j < i else x[j])
                    x_new[i] = (b[i] - s) / A[i, i]
                
                # حساب نسبة الخطأ (أقصى فرق بين القديم والجديد)
                error = np.max(np.abs(x_new - x))
                
                # تخزين الخطوة في الجدول
                step_dict = {"Iteration": it + 1}
                for i in range(n):
                    step_dict[f"x{i+1}"] = f"{x_new[i]:.{dp}f}"
                step_dict["Error"] = f"{error:.2e}"
                steps_data.append(step_dict)
                
                iterations = it + 1
                final_error = error
                
                if error < tol_val:
                    break
                x = x_new.copy()

            if iterations == max_iter:
                st.warning("⚠️ وصل البرنامج للحد الأقصى من المحاولات (100) وقد لا يكون الحل تقارب (Diverged). تأكد أن المصفوفة مسيطرة قطرياً (Diagonally Dominant).")

            # عرض النتيجة النهائية
            st.markdown("### 📍 قيم المتغيرات النهائية")
            cols_res = st.columns(n)
            for i in range(n):
                with cols_res[i]:
                    st.markdown(f"<div class='result-card exact-card'><div class='card-title'>المتغير $x_{i+1}$</div><div class='card-value exact-value'>{x_new[i]:.{dp}f}</div></div>", unsafe_allow_html=True)
            
            st.info(f"تم الوصول للحل بعد **{iterations} خطوة** | أقصى خطأ: **{final_error:.2e}**")
            
            # تسجيل في السجل
            st.session_state.history.append(f"🧮 **نظام خطي ({method.split(' ')[0]}):**\nالمتغيرات: `{n}`\nالخطوات: `{iterations}`")

            # عرض الجدول وزر التحميل
            st.markdown("### 📝 جدول خطوات الحل (Iterations Table)")
            df_steps = pd.DataFrame(steps_data)
            st.dataframe(df_steps, use_container_width=True, hide_index=True)
            
            csv = df_steps.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 تحميل الجدول كملف إكسيل (CSV)", data=csv, file_name='linear_iterations.csv', mime='text/csv', use_container_width=True)


        # ========================================================
        # 🔵 2. تنفيذ الأقسام القديمة (الجذور والتكامل)
        # ========================================================
        else:
            if not func_input.strip() or not a_str.strip(): st.warning("⚠️ يرجى تعبئة الدالة والحدود أولاً."); st.stop()
                
            func_clean = func_input.replace('^', '**').replace('ln', 'log').replace(' ', '')
            a_clean = a_str.replace(' ', '')
            b_clean = b_str.replace(' ', '')
            
            x = sp.Symbol('x')
            custom_dict = {'e': sp.E}
            if "Deg" in angle_mode: custom_dict.update({'sin': lambda arg: sp.sin(arg * sp.pi / 180), 'cos': lambda arg: sp.cos(arg * sp.pi / 180), 'tan': lambda arg: sp.tan(arg * sp.pi / 180)})

            try:
                f_expr = sp.sympify(func_clean, locals=custom_dict)
                f = sp.lambdify(x, f_expr, "numpy")
                a_val = float(sp.sympify(a_clean, locals={'e': sp.E}))
                b_val = float(sp.sympify(b_clean, locals={'e': sp.E}))
                if "Root" in app_mode:
                    tol_clean = tol_str.replace(' ', '')
                    tol_val = float(sp.sympify(tol_clean, locals={'e': sp.E}))
            except Exception as e: st.error(f"❌ خطأ: يرجى كتابة الدالة والأرقام بشكل صحيح. (التفاصيل: {e})"); st.stop()

            # --------- جذور المعادلات ---------
            if "Root" in app_mode:
                root, iterations, error = None, 0, 0
                steps_data = [] 
                
                st.markdown("### 📚 القانون الرياضي المستخدم:")
                if "Bisection" in method: st.latex(r"x_{root} = \frac{a + b}{2}")
                elif "False Position" in method: st.latex(r"x_{root} = \frac{a \cdot f(b) - b \cdot f(a)}{f(b) - f(a)}")
                elif "Newton" in method: st.latex(r"x_{i+1} = x_i - \frac{f(x_i)}{f'(x_i)}")
                elif "Secant" in method: st.latex(r"x_{i+1} = x_i - f(x_i) \frac{x_i - x_{i-1}}{f(x_i) - f(x_{i-1})}")
                
                if method in ["طريقة التنصيف (Bisection)", "الوضع الخاطئ (False Position)"]:
                    if a_val >= b_val: st.error("❌ النقطة (b) يجب أن تكون أكبر من (a)."); st.stop()
                    fa, fb = f(a_val), f(b_val)
                    if fa * fb > 0: st.error(f"❌ الدالة لا تقطع محور السينات بين {a_val} و {b_val}. (f(a) و f(b) نفس الإشارة)"); st.stop()
                    
                    a_temp, b_temp = a_val, b_val
                    for i in range(100):
                        if "Bisection" in method: c = (a_temp + b_temp) / 2.0
                        else: c = (a_temp * f(b_temp) - b_temp * f(a_temp)) / (f(b_temp) - f(a_temp))
                        fc = f(c)
                        current_error = abs(b_temp - a_temp) / 2.0 if "Bisection" in method else abs(fc)
                        steps_data.append({"Iteration": i + 1, "a": f"{a_temp:.{dp}f}", "b": f"{b_temp:.{dp}f}", "Root (c)": f"{c:.{dp}f}", "f(c)": f"{fc:.{dp}f}", "Error": f"{current_error:.2e}"})
                        if abs(fc) < tol_val or current_error < tol_val: root, iterations, error = c, i+1, abs(fc); break
                        if fa * fc < 0: b_temp = c
                        else: a_temp, fa = c, fc
                    if root is None: root, iterations, error = c, 100, abs(fc)

                elif "Newton" in method:
                    try: df_expr = sp.diff(f_expr, x); df = sp.lambdify(x, df_expr, "numpy")
                    except Exception: st.error("❌ فشل في حساب اشتقاق الدالة."); st.stop()
                    
                    x_curr = a_val 
                    for i in range(100):
                        fx, dfx = f(x_curr), df(x_curr)
                        if dfx == 0: st.error("❌ المشتقة تساوي صفر، طريقة نيوتن تفشل هنا."); st.stop()
                        x_next = x_curr - fx / dfx
                        current_error = abs(x_next - x_curr)
                        steps_data.append({"Iteration (i)": i + 1, "X_i": f"{x_curr:.{dp}f}", "f(X_i)": f"{fx:.{dp}f}", "f'(X_i)": f"{dfx:.{dp}f}", "X_i+1": f"{x_next:.{dp}f}", "Error": f"{current_error:.2e}"})
                        if current_error < tol_val: root, iterations, error = x_next, i+1, abs(f(x_next)); break
                        x_curr = x_next
                    if root is None: root, iterations, error = x_curr, 100, abs(f(x_curr))

                elif "Secant" in method:
                    x0, x1 = a_val, b_val
                    for i in range(100):
                        f0, f1 = f(x0), f(x1)
                        if f1 - f0 == 0: st.error("❌ القسمة على صفر، طريقة القاطع تفشل هنا."); st.stop()
                        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
                        current_error = abs(x2 - x1)
                        steps_data.append({"Iteration (i)": i + 1, "X_{i-1}": f"{x0:.{dp}f}", "X_i": f"{x1:.{dp}f}", "f(X_i)": f"{f1:.{dp}f}", "X_{i+1}": f"{x2:.{dp}f}", "Error": f"{current_error:.2e}"})
                        if current_error < tol_val: root, iterations, error = x2, i+1, abs(f(x2)); break
                        x0, x1 = x1, x2
                    if root is None: root, iterations, error = x2, 100, abs(f(x2))

                res_col1, res_col2 = st.columns(2)
                with res_col1: st.markdown(f"<div class='result-card exact-card'><div class='card-title'>📍 قيمة الجذر النهائي (Root)</div><div class='card-value exact-value'>{root:.{dp}f}</div></div>", unsafe_allow_html=True)
                with res_col2: st.markdown(f"<div class='result-card'><div class='card-title'>⚙️ الأداء (Performance)</div><div class='card-value' style='font-size:22px;'>الخطأ: {error:.2e}</div><div class='card-subtext'>تم الوصول للحل بعد {iterations} خطوة</div></div>", unsafe_allow_html=True)
                
                st.session_state.history.append(f"🎯 **جذر:** `{func_input}`\n\n**النتيجة:** `{root:.{dp}f}`")

                st.markdown("### 📝 جدول خطوات الحل (Iterations Table)")
                if len(steps_data) > 0:
                    df_steps = pd.DataFrame(steps_data)
                    st.dataframe(df_steps, use_container_width=True, hide_index=True)
                    csv = df_steps.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 تحميل الجدول كملف إكسيل (CSV)", data=csv, file_name='iterations.csv', mime='text/csv', use_container_width=True)

                st.markdown("### 🎨 الرسم البياني التفاعلي")
                x_min, x_max = root - 2, root + 2
                if method in ["طريقة التنصيف (Bisection)", "الوضع الخاطئ (False Position)"]: x_min, x_max = min(a_val, root-1), max(b_val, root+1)
                x_smooth = np.linspace(x_min, x_max, 500)
                y_smooth = f(x_smooth)
                if isinstance(y_smooth, (int, float)): y_smooth = np.full_like(x_smooth, y_smooth)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name='f(x)', line=dict(color='#2d82ff', width=3)))
                fig.add_trace(go.Scatter(x=[x_min, x_max], y=[0, 0], mode='lines', name='y=0', line=dict(color='#ff4b4b', width=2, dash='dash')))
                fig.add_trace(go.Scatter(x=[root], y=[0], mode='markers', name='Root', marker=dict(color='#28a745', size=12, symbol='circle')))
                fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

            # --------- التكامل ---------
            else:
                st.markdown("### 📚 القوانين الرياضية (Numerical Methods):")
                st.latex(r"\text{Trapezoidal: } \int_{a}^{b} f(x) dx \approx \frac{h}{2} \left[ f(x_0) + 2 \sum_{i=1}^{n-1} f(x_i) + f(x_n) \right]")
                st.latex(r"\text{Simpson 1/3: } \int_{a}^{b} f(x) dx \approx \frac{h}{3} \left[ f(x_0) + 4 \sum_{i \text{ odd}} f(x_i) + 2 \sum_{i \text{ even}} f(x_i) + f(x_n) \right]")
                st.latex(r"\text{Simpson 3/8: } \int_{a}^{b} f(x) dx \approx \frac{3h}{8} \left[ f(x_0) + 3 \sum_{i \neq 3k} f(x_i) + 2 \sum_{i = 3k} f(x_i) + f(x_n) \right]")

                if a_val >= b_val: st.error("❌ الحد الأقصى يجب أن يكون أكبر من الحد الأدنى."); st.stop()
                exact_val, _ = quad(f, a_val, b_val)
                
                if input_type == "لدي حجم الخطوة (h)": h = val; n = int(round((b_val - a_val) / h))
                else: n = int(val); h = (b_val - a_val) / n

                x_vals = np.linspace(a_val, b_val, n + 1)
                y_vals = f(x_vals)
                if isinstance(y_vals, (int, float)): y_vals = np.full_like(x_vals, y_vals)

                trap_val = (h / 2) * (y_vals[0] + 2 * np.sum(y_vals[1:-1]) + y_vals[-1])
                simp13_val = (h / 3) * (y_vals[0] + 4 * np.sum(y_vals[1:-1:2]) + 2 * np.sum(y_vals[2:-2:2]) + y_vals[-1]) if n % 2 == 0 else None
                simp38_val = None
                if n % 3 == 0:
                    integral = y_vals[0] + y_vals[-1]
                    for i in range(1, n): integral += 2 * y_vals[i] if i % 3 == 0 else 3 * y_vals[i]
                    simp38_val = (3 * h / 8) * integral

                st.markdown(f"<div class='result-card exact-card'><div class='card-title'>🎯 القيمة الدقيقة التحليلية (Exact Integration)</div><div class='card-value exact-value'>{exact_val:.{dp}f}</div></div>", unsafe_allow_html=True)
                st.session_state.history.append(f"📈 **تكامل:** `{func_input}`\n\n**من:** `{a_val}` **إلى:** `{b_val}`\n\n**النتيجة:** `{exact_val:.{dp}f}`")

                st.markdown("### ⚖️ مقارنة طرق التكامل")
                df_data = []
                for m_name, m_val in [("شبه المنحرف (Trapezoidal)", trap_val), ("سمبسون 1/3 (Simpson 1/3)", simp13_val), ("سمبسون 3/8 (Simpson 3/8)", simp38_val)]:
                    if m_val is not None: df_data.append({"الطريقة": m_name, "النتيجة التقريبية": f"{m_val:.{dp}f}", "الخطأ المطلق": f"{abs(exact_val - m_val):.2e}"})
                    else: df_data.append({"الطريقة": m_name, "النتيجة التقريبية": f"❌ يتطلب شروط معينة لـ n", "الخطأ المطلق": "-"})
                
                st.dataframe(pd.DataFrame(df_data), use_container_width=True, hide_index=True)

                st.markdown("### 🎨 الرسم البياني التفاعلي للمساحة")
                x_smooth = np.linspace(a_val, b_val, 500)
                y_smooth = f(x_smooth)
                if isinstance(y_smooth, (int, float)): y_smooth = np.full_like(x_smooth, y_smooth)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name='f(x)', line=dict(color='#2d82ff', width=3)))
                fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, fill='tozeroy', fillcolor='rgba(45, 130, 255, 0.2)', mode='none', name='Area'))
                fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
    else:
        st.info(f"👋 أهلاً بك! أنت الآن في وضع {app_mode.split(' ')[1]}. أدخل بياناتك واضغط على احسب.")
