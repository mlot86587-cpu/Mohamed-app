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
.result-card { background-color: #f8f9fa; padding: 20px; border-radius: 15px; border-left: 5px solid #007bff; margin-bottom: 20px; }
.exact-card { border-left: 5px solid #28a745; }
.card-title { font-size: 16px; font-weight: bold; margin-bottom: 5px; color: #6c757d; }
.card-value { font-size: 28px; font-weight: bold; color: #2d82ff; }
.exact-value { color: #28a745; }
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
    elif p == "تفاضلية: دالة خطية": st.session_state.func_text = "x + y"

# ==========================================
# 📌 القائمة الجانبية
# ==========================================
st.sidebar.title("🧮 الأقسام الرياضية")
app_mode = st.sidebar.radio("اختر العملية المطلوبة:", [
    "📈 التكامل العددي (Integration)", 
    "🎯 حل المعادلات (Root Finding)",
    "🧮 أنظمة المعادلات الخطية (Linear Systems)",
    "📉 المعادلات التفاضلية (ODE Solvers)"
])
st.sidebar.markdown("---")

if "Linear" not in app_mode:
    st.sidebar.selectbox("💡 أمثلة سريعة للتدريب:", 
        ["اختر مثالاً...", "تكامل: دالة أسية (جرسية)", "تكامل: دالة مثلثية", "جذور: دالة تكعيبية", "جذور: دالة أسية", "تفاضلية: دالة خطية"], 
        key='preset', on_change=apply_preset)

dp = st.sidebar.slider("🎯 دقة الأرقام (Decimal Places):", min_value=2, max_value=10, value=6)

# === تقسيم الشاشة الأساسي ===
col_control, col_display = st.columns([1, 2.5])

# ==========================================
# ⚙️ العمود الأيسر: لوحة التحكم
# ==========================================
with col_control:
    st.markdown(f"### 🎛️ إعدادات {app_mode.split(' ')[1]}")
    
    # --- 1. أنظمة المعادلات الخطية ---
    if "Linear" in app_mode:
        st.info("💡 يمكنك نسخ المصفوفة من Excel ولصقها مباشرة في الجدول (Ctrl+V).")
        method = st.selectbox("اختر طريقة الحل:", ["جاكوبي (Jacobi)", "جاوس-سيدل (Gauss-Seidel)"])
        n_vars = st.number_input("عدد المتغيرات (المعادلات):", min_value=2, max_value=50, value=3)
        default_matrix = np.array([[5, -1, 1, 10], [2, 8, -1, 11], [-1, 1, 4, 3]], dtype=float) if n_vars == 3 else np.zeros((n_vars, n_vars + 1))
        cols = [f"x{i+1}" for i in range(n_vars)] + ["= b"]
        df_matrix = pd.DataFrame(default_matrix, columns=cols)
        st.markdown(f"**المصفوفة الموسعة:**")
        edited_df = st.data_editor(df_matrix, use_container_width=True, hide_index=True)
        tol_str = st.text_input("نسبة الخطأ المقبولة (Tolerance):", value="1e-6")
        
    # --- 2. الأقسام المعتمدة على الدوال ---
    else:
        func_label = "المعادلة التفاضلية y' = f(x,y):" if "ODE" in app_mode else "الدالة الرياضية f(x):"
        func_input = st.text_input(func_label, key="func_text", placeholder="اكتب الدالة هنا...")
        
        with st.expander("🧮 إظهار / إخفاء الآلة الحاسبة", expanded=False):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.button("sin", on_click=append_to_func, args=("sin(",))
            c2.button("cos", on_click=append_to_func, args=("cos(",))
            c3.button("tan", on_click=append_to_func, args=("tan(",))
            c4.button("π", on_click=append_to_func, args=("pi",))
            c5.button("🧹", on_click=clear_func)
            
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
            c5.button("e^", on_click=append_to_func, args=("exp(",))

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
            c5.button("y", on_click=append_to_func, args=("y",))

        angle_mode = st.radio("نظام الزوايا:", ["راديان (Rad)", "درجات (Deg)"], horizontal=True)

        if "Root" in app_mode:
            method = st.selectbox("اختر طريقة الحل:", ["نيوتن-رافسون (Newton)", "طريقة التنصيف (Bisection)", "الوضع الخاطئ (False Position)", "طريقة القاطع (Secant)"])
            if method in ["طريقة التنصيف (Bisection)", "الوضع الخاطئ (False Position)"]:
                col1, col2 = st.columns(2)
                with col1: a_str = st.text_input("من (a):", value="0")
                with col2: b_str = st.text_input("إلى (b):", value="2")
            elif "Newton" in method:
                a_str = st.text_input("قيمة البداية (x0):", value="1") # X0 as requested
                b_str = "0"
            elif "Secant" in method:
                col1, col2 = st.columns(2)
                with col1: a_str = st.text_input("التخمين الأول (x0):", value="0")
                with col2: b_str = st.text_input("التخمين الثاني (x1):", value="1")
            
            tol_str = st.text_input("نسبة الخطأ المقبولة (Tolerance):", value="1e-6")
            
        elif "ODE" in app_mode:
            method = st.selectbox("اختر طريقة الحل:", ["طريقة أويلر (Euler)", "رانجي-كوتا (Runge-Kutta 4th)"])
            col1, col2 = st.columns(2)
            with col1: x0_str = st.text_input("قيمة البداية (x0):", value="0")
            with col2: y0_str = st.text_input("قيمة البداية (y0):", value="1")
            x_end_str = st.text_input("حساب y عند النقطة (x_end):", value="2")
            input_type = st.radio("طريقة الإدخال:", ["عدد الخطوات (n)", "حجم الخطوة (h)"])
            val = st.number_input("القيمة (n أو h):", value=10.0, min_value=0.0001)

        else: # Integration
            col1, col2 = st.columns(2)
            with col1: a_str = st.text_input("الحد الأدنى (a):", value="0")
            with col2: b_str = st.text_input("الحد الأقصى (b):", value="2")
            input_type = st.radio("طريقة الإدخال:", ["عدد القطاعات (n)", "حجم الخطوة (h)"])
            val = st.number_input("القيمة (n أو h):", value=10.0, min_value=0.0001)

    st.markdown("<br>", unsafe_allow_html=True)
    calc_btn = st.button("🚀 احسب وارسم", type="primary", use_container_width=True)

# ==========================================
# 📊 العمود الأيمن: شاشة النتائج والرسم
# ==========================================
with col_display:
    if calc_btn:
        # 🟢 1. أنظمة المعادلات الخطية
        if "Linear" in app_mode:
            try: tol_val = float(sp.sympify(tol_str.replace(' ', ''), locals={'e': sp.E}))
            except: st.error("❌ خطأ في Tolerance"); st.stop()
            A_input, b_input = edited_df.iloc[:, :-1].values.astype(float), edited_df.iloc[:, -1].values.astype(float)
            n = len(b_input)
            indices, used_rows = [], set()
            for i in range(n):
                max_row, max_val = -1, -1
                for r in range(n):
                    if r not in used_rows and abs(A_input[r, i]) > max_val: max_val, max_row = abs(A_input[r, i]), r
                indices.append(max_row); used_rows.add(max_row)
            A, b = A_input[indices], b_input[indices]
            if indices != list(range(n)): st.success("✨ تم ترتيب المصفوفة لضمان الحل (Diagonally Dominant)")
            if np.any(np.diag(A) == 0): st.error("❌ يوجد صفر على القطر الرئيسي"); st.stop()
            
            x_vals = np.zeros(n)
            steps_data, iterations, final_error, max_iter = [], 0, 0, 500
            for it in range(max_iter):
                x_new = np.zeros(n)
                for i in range(n):
                    s = sum(A[i, j] * (x_vals[j] if "Jacobi" in method else (x_new[j] if j < i else x_vals[j])) for j in range(n) if i != j)
                    x_new[i] = (b[i] - s) / A[i, i]
                error = np.max(np.abs(x_new - x_vals))
                
                # --- تم حل مشكلة الـ SyntaxError هنا ---
                step_dict = {"Iteration": it + 1}
                for i in range(n): step_dict[f"x{i+1}"] = f"{x_new[i]:.{dp}f}"
                step_dict["Error"] = f"{error:.2e}"
                steps_data.append(step_dict)
                # ----------------------------------------
                
                iterations, final_error = it + 1, error
                if error < tol_val or error > 1e10: break
                x_vals = x_new.copy()
                
            st.markdown("### 📍 قيم المتغيرات النهائية")
            st.dataframe(pd.DataFrame({"المتغير": [f"x{i+1}" for i in range(n)], "القيمة النهائية": [f"{x_new[i]:.{dp}f}" for i in range(n)]}).T, use_container_width=True)
            st.markdown("### 📝 جدول خطوات الحل")
            st.dataframe(pd.DataFrame(steps_data), use_container_width=True, hide_index=True)

        # 🔵 2. الأقسام المعتمدة على الدوال (الجذور، التكامل، ODE)
        else:
            if not func_input.strip(): st.warning("⚠️ يرجى تعبئة الدالة أولاً."); st.stop()
            func_clean = func_input.replace('^', '**').replace('ln', 'log').replace(' ', '')
            x, y = sp.Symbol('x'), sp.Symbol('y')
            custom_dict = {'e': sp.E}
            if "Deg" in angle_mode: custom_dict.update({'sin': lambda arg: sp.sin(arg * sp.pi / 180), 'cos': lambda arg: sp.cos(arg * sp.pi / 180), 'tan': lambda arg: sp.tan(arg * sp.pi / 180)})

            try:
                f_expr = sp.sympify(func_clean, locals=custom_dict)
                f = sp.lambdify((x, y) if "ODE" in app_mode else x, f_expr, "numpy")
            except Exception as e: st.error(f"❌ خطأ في الدالة. (التفاصيل: {e})"); st.stop()

            # --------- جذور المعادلات (Newton Raphson) ---------
            if "Root" in app_mode:
                try:
                    a_val = float(sp.sympify(a_str.replace(' ', ''), locals={'e': sp.E}))
                    b_val = float(sp.sympify(b_str.replace(' ', ''), locals={'e': sp.E}))
                    tol_val = float(sp.sympify(tol_str.replace(' ', ''), locals={'e': sp.E}))
                except: st.error("❌ تأكد من إدخال الأرقام بشكل صحيح."); st.stop()

                root, iterations, error, steps_data = None, 0, 0, []

                if "Newton" in method:
                    try: 
                        df_expr = sp.diff(f_expr, x)
                        df = sp.lambdify(x, df_expr, "numpy")
                    except: st.error("❌ فشل في حساب اشتقاق الدالة."); st.stop()
                    
                    x_curr = a_val # Initial guess X0
                    for i in range(100): # Max Iterations
                        fx, dfx = f(x_curr), df(x_curr)
                        if dfx == 0: st.error("❌ المشتقة تساوي صفر. فشل الحل."); st.stop()
                        x_next = x_curr - fx / dfx
                        current_err = abs(x_next - x_curr)
                        
                        # --- تم حل مشكلة التنسيق الطويل هنا ---
                        steps_data.append({
                            "Iteration": i+1, 
                            "X_i": f"{x_curr:.{dp}f}", 
                            "f(X_i)": f"{fx:.{dp}f}", 
                            "f'(X_i)": f"{dfx:.{dp}f}", 
                            "X_i+1": f"{x_next:.{dp}f}", 
                            "Error": f"{current_err:.2e}"
                        })
                        
                        if current_err < tol_val: 
                            root, iterations, error = x_next, i+1, current_err
                            break
                        x_curr = x_next
                    if root is None: root, iterations, error = x_curr, 100, current_err

                elif "Bisection" in method or "False Position" in method:
                    fa, fb = f(a_val), f(b_val)
                    if fa * fb > 0: st.error("❌ الدالة لا تقطع محور السينات بين النقطتين."); st.stop()
                    a_temp, b_temp = a_val, b_val
                    for i in range(100):
                        c = (a_temp + b_temp) / 2.0 if "Bisection" in method else (a_temp * f(b_temp) - b_temp * f(a_temp)) / (f(b_temp) - f(a_temp))
                        fc, current_err = f(c), (abs(b_temp - a_temp) / 2.0 if "Bisection" in method else abs(f(c)))
                        steps_data.append({"Iteration": i+1, "a": f"{a_temp:.{dp}f}", "b": f"{b_temp:.{dp}f}", "Root (c)": f"{c:.{dp}f}", "f(c)": f"{fc:.{dp}f}", "Error": f"{current_err:.2e}"})
                        if abs(fc) < tol_val or current_err < tol_val: root, iterations, error = c, i+1, abs(fc); break
                        if fa * fc < 0: b_temp = c
                        else: a_temp, fa = c, fc

                elif "Secant" in method:
                    x0, x1 = a_val, b_val
                    for i in range(100):
                        f0, f1 = f(x0), f(x1)
                        if f1 - f0 == 0: st.error("❌ القسمة على صفر."); st.stop()
                        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
                        current_err = abs(x2 - x1)
                        steps_data.append({"Iteration": i+1, "X_i-1": f"{x0:.{dp}f}", "X_i": f"{x1:.{dp}f}", "f(X_i)": f"{f1:.{dp}f}", "X_i+1": f"{x2:.{dp}f}", "Error": f"{current_err:.2e}"})
                        if current_err < tol_val: root, iterations, error = x2, i+1, current_err; break
                        x0, x1 = x1, x2

                st.markdown(f"<div class='result-card exact-card'><div class='card-title'>📍 قيمة الجذر النهائي (Root)</div><div class='card-value exact-value'>{root:.{dp}f}</div><div class='card-subtext'>المحاولات: {iterations} | الخطأ: {error:.2e}</div></div>", unsafe_allow_html=True)
                
                st.markdown("### 📝 جدول المحاولات (Iterations Table)")
                st.dataframe(pd.DataFrame(steps_data), use_container_width=True, hide_index=True)

                x_min, x_max = root - 2, root + 2
                x_smooth = np.linspace(x_min, x_max, 500)
                y_smooth = [f(val) for val in x_smooth]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name='f(x)', line=dict(color='#2d82ff')))
                fig.add_trace(go.Scatter(x=[x_min, x_max], y=[0, 0], mode='lines', name='y=0', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=[root], y=[0], mode='markers', name='Root', marker=dict(color='green', size=12)))
                st.plotly_chart(fig, use_container_width=True)

            # --------- المعادلات التفاضلية (ODE Solvers) ---------
            elif "ODE" in app_mode:
                try:
                    x0, y0, x_end = float(sp.sympify(x0_str)), float(sp.sympify(y0_str)), float(sp.sympify(x_end_str))
                except: st.error("❌ خطأ في الأرقام المدخلة."); st.stop()

                n = int(round((x_end - x0) / val)) if "h" in input_type else int(val)
                h = val if "h" in input_type else (x_end - x0) / n

                x_vals, y_vals, steps_data = [x0], [y0], []
                for i in range(n):
                    xi, yi = x_vals[-1], y_vals[-1]
                    if "Euler" in method:
                        slope = f(xi, yi)
                        y_next = yi + h * slope
                        steps_data.append({"Step": i, "x_i": f"{xi:.{dp}f}", "y_i": f"{yi:.{dp}f}", "f(x,y)": f"{slope:.{dp}f}", "y_next": f"{y_next:.{dp}f}"})
                    else:
                        k1 = h * f(xi, yi)
                        k2 = h * f(xi + h/2, yi + k1/2)
                        k3 = h * f(xi + h/2, yi + k2/2)
                        k4 = h * f(xi + h, yi + k3)
                        y_next = yi + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
                        steps_data.append({"Step": i, "x_i": f"{xi:.{dp}f}", "y_i": f"{yi:.{dp}f}", "k1": f"{k1:.{dp}f}", "k2": f"{k2:.{dp}f}", "k3": f"{k3:.{dp}f}", "k4": f"{k4:.{dp}f}", "y_next": f"{y_next:.{dp}f}"})
                    x_vals.append(xi + h)
                    y_vals.append(y_next)

                st.markdown(f"<div class='result-card exact-card'><div class='card-title'>📍 قيمة y التقريبية عند x = {x_end}</div><div class='card-value exact-value'>y = {y_vals[-1]:.{dp}f}</div></div>", unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(steps_data), use_container_width=True, hide_index=True)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', name='مسار التقريب', line=dict(color='#2d82ff')))
                st.plotly_chart(fig, use_container_width=True)

            # --------- التكامل العددي (Integration) ---------
            else:
                try:
                    a_val = float(sp.sympify(a_str.replace(' ', ''), locals={'e': sp.E}))
                    b_val = float(sp.sympify(b_str.replace(' ', ''), locals={'e': sp.E}))
                except: st.error("❌ تأكد من إدخال حدود التكامل بشكل صحيح."); st.stop()

                exact_val, _ = quad(f, a_val, b_val)
                n = int(round((b_val - a_val) / val)) if "h" in input_type else int(val)
                h = val if "h" in input_type else (b_val - a_val) / n

                x_vals = np.linspace(a_val, b_val, n + 1)
                y_vals = [f(v) for v in x_vals]

                trap_val = (h / 2) * (y_vals[0] + 2 * sum(y_vals[1:-1]) + y_vals[-1])
                simp13_val = (h / 3) * (y_vals[0] + 4 * sum(y_vals[1:-1:2]) + 2 * sum(y_vals[2:-2:2]) + y_vals[-1]) if n % 2 == 0 else None
                
                simp38_val = None
                if n % 3 == 0:
                    integral = y_vals[0] + y_vals[-1]
                    for i in range(1, n): integral += 2 * y_vals[i] if i % 3 == 0 else 3 * y_vals[i]
                    simp38_val = (3 * h / 8) * integral

                st.markdown(f"<div class='result-card exact-card'><div class='card-title'>🎯 القيمة الدقيقة (Exact)</div><div class='card-value exact-value'>{exact_val:.{dp}f}</div></div>", unsafe_allow_html=True)
                
                df_data = []
                for m_name, m_val in [("Trapezoidal", trap_val), ("Simpson 1/3", simp13_val), ("Simpson 3/8", simp38_val)]:
                    df_data.append({"الطريقة": m_name, "النتيجة": f"{m_val:.{dp}f}" if m_val else "❌ n غير متوافقة", "الخطأ": f"{abs(exact_val - m_val):.2e}" if m_val else "-"})
                st.dataframe(pd.DataFrame(df_data), use_container_width=True, hide_index=True)

                x_smooth = np.linspace(a_val, b_val, 500)
                y_smooth = [f(v) for v in x_smooth]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name='f(x)', line=dict(color='#2d82ff')))
                fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, fill='tozeroy', fillcolor='rgba(45,130,255,0.2)', mode='none', name='Area'))
                st.plotly_chart(fig, use_container_width=True)
