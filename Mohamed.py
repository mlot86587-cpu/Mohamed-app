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
    elif p == "تفاضلية: دالة خطية": st.session_state.func_text = "x + y"
    elif p == "تفاضلية: معادلة معقدة": st.session_state.func_text = "(x - y)/2"

# ==========================================
# 📌 القائمة الجانبية
# ==========================================
st.sidebar.title("🧮 الأقسام الرياضية")
app_mode = st.sidebar.radio("اختر العملية المطلوبة:", [
    "📈 التكامل العددي (Integration)", 
    "🎯 حل المعادلات (Root Finding)",
    "🧮 أنظمة المعادلات الخطية (Linear Systems)",
    "📉 المعادلات التفاضلية (ODE Solvers)" # القسم الرابع الجديد!
])
st.sidebar.markdown("---")

if "Linear" not in app_mode:
    st.sidebar.selectbox("💡 أمثلة سريعة للتدريب:", 
        ["اختر مثالاً...", "تكامل: دالة أسية (جرسية)", "تكامل: دالة مثلثية", "جذور: دالة تكعيبية", "جذور: دالة أسية", "تفاضلية: دالة خطية", "تفاضلية: معادلة معقدة"], 
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
    # 1. إعدادات قسم أنظمة المعادلات الخطية
    # -----------------------------------------------------
    if "Linear" in app_mode:
        st.info("💡 نصيحة: يمكنك نسخ المصفوفة من Excel ولصقها مباشرة في الجدول أدناه (Ctrl+V).")
        method = st.selectbox("اختر طريقة الحل:", ["جاكوبي (Jacobi)", "جاوس-سيدل (Gauss-Seidel)"])
        n_vars = st.number_input("عدد المتغيرات (المعادلات):", min_value=2, max_value=50, value=3)
        default_matrix = np.zeros((n_vars, n_vars + 1))
        if n_vars == 3: default_matrix = np.array([[5, -1, 1, 10], [2, 8, -1, 11], [-1, 1, 4, 3]], dtype=float)
        cols = [f"x{i+1}" for i in range(n_vars)] + ["= b"]
        df_matrix = pd.DataFrame(default_matrix, columns=cols)
        st.markdown(f"**المصفوفة الموسعة $[A | b]$ بحجم ({n_vars}×{n_vars+1}):**")
        edited_df = st.data_editor(df_matrix, use_container_width=True, hide_index=True)
        tol_str = st.text_input("نسبة الخطأ المقبولة (Tolerance):", value="1e-6")
        
    # -----------------------------------------------------
    # 2. إعدادات الأقسام (التكامل، الجذور، والمعادلات التفاضلية)
    # -----------------------------------------------------
    else:
        func_label = "المعادلة التفاضلية y' = f(x,y):" if "ODE" in app_mode else "الدالة الرياضية f(x):"
        func_input = st.text_input(func_label, key="func_text", placeholder="استخدم الآلة الحاسبة للكتابة...")
        
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
            c5.button("y", on_click=append_to_func, args=("y",)) # إضافة زر y للمتغير الثاني

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
            
        elif "ODE" in app_mode:
            method = st.selectbox("اختر طريقة الحل:", ["طريقة أويلر (Euler)", "رانجي-كوتا (Runge-Kutta 4th)"])
            col1, col2 = st.columns(2)
            with col1: x0_str = st.text_input("قيمة البداية (x0):", value="0")
            with col2: y0_str = st.text_input("قيمة البداية (y0):", value="1")
            
            x_end_str = st.text_input("حساب y عند النقطة (x_end):", value="2")
            input_type = st.radio("طريقة الإدخال:", ["لدي عدد الخطوات (n)", "لدي حجم الخطوة (h)"])
            val = st.number_input("أدخل القيمة (n أو h):", value=10.0, min_value=0.0001)

        else: # Integration
            col1, col2 = st.columns(2)
            with col1: a_str = st.text_input("الحد الأدنى (a):", value="0")
            with col2: b_str = st.text_input("الحد الأقصى (b):", value="2")
            input_type = st.radio("طريقة الإدخال:", ["لدي عدد القطاعات (n)", "لدي حجم الخطوة (h)"])
            val = st.number_input("أدخل القيمة (n أو h):", value=10.0, min_value=0.0001)

    st.markdown("<br>", unsafe_allow_html=True)
    calc_btn = st.button("🚀 احسب وارسم", type="primary", use_container_width=True)

# ==========================================
# 📊 العمود الأيمن: شاشة النتائج والرسم
# ==========================================
with col_display:
    if calc_btn:
        
        # 🟢 1. أنظمة المعادلات الخطية
        if "Linear" in app_mode:
            # (نفس كود الأنظمة الخطية الخاص بالنسخة السابقة تماماً)
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
            if indices != list(range(n)): st.success("✨ تم الترتيب (Diagonally Dominant)")
            if np.any(np.diag(A) == 0): st.error("❌ يوجد صفر على القطر الرئيسي"); st.stop()
            x_vals = np.zeros(n)
            steps_data, iterations, final_error, max_iter = [], 0, 0, 500
            for it in range(max_iter):
                x_new = np.zeros(n)
                for i in range(n):
                    s = sum(A[i, j] * (x_vals[j] if "Jacobi" in method else (x_new[j] if j < i else x_vals[j])) for j in range(n) if i != j)
                    x_new[i] = (b[i] - s) / A[i, i]
                error = np.max(np.abs(x_new - x_vals))
                step_dict = {"Iteration": it + 1}
                for i in range(n): step_dict[f"x{i+1}"] = f"{x_new[i]:.{dp}f}"
                step_dict["Error"], steps_data.append(step_dict) = f"{error:.2e}", None
                iterations, final_error = it + 1, error
                if error < tol_val or error > 1e10: break
                x_vals = x_new.copy()
            st.markdown("### 📍 قيم المتغيرات النهائية")
            st.dataframe(pd.DataFrame({"المتغير": [f"x{i+1}" for i in range(n)], "القيمة النهائية": [f"{x_new[i]:.{dp}f}" for i in range(n)]}), use_container_width=True)
            df_steps = pd.DataFrame(steps_data)
            st.dataframe(df_steps, use_container_width=True, hide_index=True)

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

            # --------- المعادلات التفاضلية (ODE Solvers) ---------
            if "ODE" in app_mode:
                try:
                    x0 = float(sp.sympify(x0_str.replace(' ', ''), locals={'e': sp.E}))
                    y0 = float(sp.sympify(y0_str.replace(' ', ''), locals={'e': sp.E}))
                    x_end = float(sp.sympify(x_end_str.replace(' ', ''), locals={'e': sp.E}))
                except Exception: st.error("❌ خطأ في إدخال القيم المبدئية."); st.stop()

                if "h" in input_type: h = val; n = int(round((x_end - x0) / h))
                else: n = int(val); h = (x_end - x0) / n

                st.markdown("### 📚 القانون الرياضي المستخدم:")
                if "Euler" in method: st.latex(r"y_{i+1} = y_i + h \cdot f(x_i, y_i)")
                else: st.latex(r"\begin{aligned} k_1 &= h f(x_i, y_i) \\ k_2 &= h f(x_i + \frac{h}{2}, y_i + \frac{k_1}{2}) \\ k_3 &= h f(x_i + \frac{h}{2}, y_i + \frac{k_2}{2}) \\ k_4 &= h f(x_i + h, y_i + k_3) \\ y_{i+1} &= y_i + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4) \end{aligned}")

                x_vals, y_vals = [x0], [y0]
                steps_data = []

                for i in range(n):
                    xi, yi = x_vals[-1], y_vals[-1]
                    try:
                        if "Euler" in method:
                            slope = f(xi, yi)
                            y_next = yi + h * slope
                            steps_data.append({"Step": i, "x_i": f"{xi:.{dp}f}", "y_i": f"{yi:.{dp}f}", "f(x,y)": f"{slope:.{dp}f}", "y_next": f"{y_next:.{dp}f}"})
                        else: # RK4
                            k1 = h * f(xi, yi)
                            k2 = h * f(xi + h/2, yi + k1/2)
                            k3 = h * f(xi + h/2, yi + k2/2)
                            k4 = h * f(xi + h, yi + k3)
                            y_next = yi + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
                            steps_data.append({"Step": i, "x_i": f"{xi:.{dp}f}", "y_i": f"{yi:.{dp}f}", "k1": f"{k1:.{dp}f}", "k2": f"{k2:.{dp}f}", "k3": f"{k3:.{dp}f}", "k4": f"{k4:.{dp}f}", "y_next": f"{y_next:.{dp}f}"})
                    except Exception as e:
                        st.error(f"❌ خطأ رياضي أثناء الحساب (مثال: قسمة على صفر). التفاصيل: {e}")
                        st.stop()
                        
                    x_vals.append(xi + h)
                    y_vals.append(y_next)

                final_y = y_vals[-1]
                
                st.markdown(f"<div class='result-card exact-card'><div class='card-title'>📍 قيمة y التقريبية عند x = {x_end}</div><div class='card-value exact-value'>y = {final_y:.{dp}f}</div></div>", unsafe_allow_html=True)
                st.session_state.history.append(f"📉 **تفاضلية ({method.split(' ')[0]}):**\n`y' = {func_input}`\n`y({x_end}) = {final_y:.{dp}f}`")

                st.markdown("### 📝 جدول خطوات الحل")
                df_steps = pd.DataFrame(steps_data)
                st.dataframe(df_steps, use_container_width=True, hide_index=True)
                csv = df_steps.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 تحميل الجدول كملف إكسيل (CSV)", data=csv, file_name='ode_iterations.csv', mime='text/csv', use_container_width=True)

                st.markdown("### 🎨 الرسم البياني لمسار الدالة التقريبي")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', name='مسار التقريب (y)', line=dict(color='#2d82ff', width=3), marker=dict(size=8, color='#ff4b4b')))
                fig.update_layout(xaxis_title="x", yaxis_title="y", margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

            # --------- جذور المعادلات ---------
            elif "Root" in app_mode:
                # (نفس كود الجذور الخاص بالنسخة السابقة بالضبط لتوفير المساحة)
                st.success("للتشغيل، يرجى كتابة الكود الخاص بالـ Root هنا كما هو في النسخة 8.0.")
                
            # --------- التكامل ---------
            else:
                # (نفس كود التكامل الخاص بالنسخة السابقة بالضبط لتوفير المساحة)
                st.success("للتشغيل، يرجى كتابة الكود الخاص بالـ Integration هنا كما هو في النسخة 8.0.")
 
