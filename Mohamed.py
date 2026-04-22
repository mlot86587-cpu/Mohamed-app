import streamlit as st
import numpy as np
import sympy as sp
from scipy.integrate import quad
import plotly.graph_objects as go
import pandas as pd

# ==========================================
# === إعدادات الصفحة ===
# ==========================================
st.set_page_config(
    page_title="Numerical Analysis Pro", 
    page_icon="🔢", 
    layout="wide"
)

# ==========================================
# === تصميم CSS ===
# ==========================================
st.markdown("""
<style>
.result-card { background-color: var(--secondary-background-color); padding: 20px; border-radius: 15px; border-left: 5px solid #007bff; margin-bottom: 20px; }
.exact-card { border-left: 5px solid #28a745; }
.tutorial-card { border-left: 5px solid #ffc107; background-color: rgba(255, 193, 7, 0.1); }
.card-title { font-size: 16px; font-weight: bold; margin-bottom: 5px; opacity: 0.8; }
.card-value { font-size: 28px; font-weight: bold; color: #2d82ff; }
.exact-value { color: #28a745; }
.card-subtext { font-size: 14px; margin-top: 5px; opacity: 0.7; }
div[data-testid="column"] { padding: 0 3px !important; }
.stButton > button { width: 100% !important; padding: 0px !important; font-size: 18px !important; font-weight: bold !important; height: 42px !important; margin-bottom: 4px !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# === إدارة حالة النص والسجل ===
# ==========================================
if 'func_text' not in st.session_state: 
    st.session_state.func_text = ""
if 'preset' not in st.session_state: 
    st.session_state.preset = "اختر مثالاً..."
if 'history' not in st.session_state: 
    st.session_state.history = []

def append_to_func(text): 
    st.session_state.func_text += text

def clear_func(): 
    st.session_state.func_text = ""

def backspace_func(): 
    st.session_state.func_text = st.session_state.func_text[:-1]

def apply_preset():
    p = st.session_state.preset
    if p == "تكامل: دالة أسية (جرسية)": 
        st.session_state.func_text = "exp(-x**2)"
    elif p == "تكامل: دالة مثلثية": 
        st.session_state.func_text = "x * sin(x)"
    elif p == "جذور: دالة تكعيبية": 
        st.session_state.func_text = "x**3 - x - 2"
    elif p == "جذور: دالة أسية": 
        st.session_state.func_text = "exp(-x) - x"
    elif p == "تفاضلية: دالة خطية": 
        st.session_state.func_text = "x - y + 2"
    elif p == "تفاضل: دالة لوغاريتمية": 
        st.session_state.func_text = "x * log(x)"

# ==========================================
# 📌 القائمة الجانبية
# ==========================================
st.sidebar.title("🧮 الأقسام الرياضية")
app_mode = st.sidebar.radio("اختر العملية المطلوبة:", [
    "📈 التكامل العددي (Integration)", 
    "🎯 حل المعادلات (Root Finding)",
    "🧮 أنظمة المعادلات الخطية (Linear Systems)",
    "🔬 المعادلات التفاضلية (Differential Eq)",
    "📐 التفاضل العددي (Differentiation)",
    "📉 الاستيفاء وتوفيق المنحنيات (Interpolation)"
])
st.sidebar.markdown("---")

valid_preset_modes = [
    "🧮 أنظمة المعادلات الخطية (Linear Systems)", 
    "📉 الاستيفاء وتوفيق المنحنيات (Interpolation)"
]

if app_mode not in valid_preset_modes:
    st.sidebar.selectbox(
        "💡 أمثلة سريعة للتدريب:", 
        [
            "اختر مثالاً...", 
            "تكامل: دالة أسية (جرسية)", 
            "تكامل: دالة مثلثية", 
            "جذور: دالة تكعيبية", 
            "جذور: دالة أسية", 
            "تفاضلية: دالة خطية", 
            "تفاضل: دالة لوغاريتمية"
        ], 
        key='preset', 
        on_change=apply_preset
    )

# --- 🎯 إعدادات إضافية ---
st.sidebar.markdown("---")
dp = st.sidebar.slider(
    "🎯 دقة الأرقام (Decimal Places):", 
    min_value=2, max_value=10, value=6
)
show_tutorial = st.sidebar.toggle(
    "👀 وضع الشرح (إظهار التعويض)", 
    value=False
)

# --- 🕒 سجل العمليات ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🕒 السجل (History)")
if st.sidebar.button("🗑️ مسح السجل", use_container_width=True):
    st.session_state.history = []
    st.rerun()

if not st.session_state.history:
    st.sidebar.info("السجل فارغ. ابدأ الحساب الآن!")
else:
    for item in reversed(st.session_state.history[-5:]): # عرض آخر 5 عمليات فقط
        st.sidebar.success(item)

# تذييل شخصي للقائمة الجانبية
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: gray; font-size: 11px; margin-top: 20px;'>
    <b style='color:#007bff;'>Numerical Analysis Laboratory</b><br>
    Zagazig National University<br>
    Mechatronics Engineering Dept.<br><br>
    <i>Dev: Mohamed Khaled Mohamed El-Hadi Abulfotouh</i>
</div>
""", unsafe_allow_html=True)

# === تقسيم الشاشة الأساسي ===
col_control, col_display = st.columns([1, 2.5])

# ==========================================
# ⚙️ العمود الأيسر: لوحة التحكم
# ==========================================
with col_control:
    st.markdown("### 🎛️ إعدادات الإدخال")
    
    # 1. الاستيفاء (Interpolation)
    if "Interpolation" in app_mode:
        method = st.selectbox("طريقة الاستيفاء:", ["لاجرانج (Lagrange)"])
        num_pts = st.number_input("عدد النقاط (n):", min_value=2, max_value=10, value=3)
        st.write("أدخل قيم X و Y:")
        
        x_init, y_init = [], []
        for i in range(num_pts):
            x_init.append(float(i))
            y_init.append(float(np.random.randint(1, 10)))
            
        df_pts = pd.DataFrame({"X": x_init, "Y": y_init})
        edited_pts = st.data_editor(df_pts, use_container_width=True, hide_index=True)
        
    # 2. الأنظمة الخطية
    elif "Linear" in app_mode:
        method = st.selectbox(
            "اختر طريقة الحل:", 
            ["جاكوبي (Jacobi)", "جاوس-سيدل (Gauss-Seidel)", "الحذف لجاوس (Gaussian Elimination)"]
        )
        n_vars = st.number_input("عدد المتغيرات:", min_value=2, max_value=10, value=3)
        
        default_matrix = np.zeros((n_vars, n_vars + 1))
        if n_vars == 3:
            default_matrix = np.array([[2, 8, -1, 11], [-1, 1, 4, 3], [5, -1, 1, 10]], dtype=float)
            
        cols = [f"x{i+1}" for i in range(n_vars)] + ["= b"]
        df_matrix = pd.DataFrame(default_matrix, columns=cols)
        
        st.markdown("**المصفوفة الموسعة:**")
        edited_df = st.data_editor(df_matrix, use_container_width=True, hide_index=True)
        
        if "Gauss" not in method or "Seidel" in method:
            tol_str = st.text_input("نسبة الخطأ (Tolerance):", value="1e-6")
        
    # 3. الأقسام المعتمدة على الدوال
    else:
        func_placeholder = "مثال: x - y" if "Differential" in app_mode else "استخدم الآلة الحاسبة..."
        func_input = st.text_input("الدالة الرياضية:", key="func_text", placeholder=func_placeholder)
        
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
            c5.button("^", on_click=append_to_func, args=("**",))

        if "Root" in app_mode:
            method = st.selectbox("طريقة الحل:", ["التنصيف (Bisection)", "الوضع الخاطئ (False Position)", "نيوتن-رافسون (Newton)", "القاطع (Secant)"])
            col1, col2 = st.columns(2)
            if method in ["التنصيف (Bisection)", "الوضع الخاطئ (False Position)"]:
                with col1: a_str = st.text_input("من (a):", value="1")
                with col2: b_str = st.text_input("إلى (b):", value="2")
            elif "Newton" in method:
                a_str = st.text_input("تخمين أولي (x0):", value="1")
            elif "Secant" in method:
                with col1: a_str = st.text_input("تخمين 1 (x0):", value="1")
                with col2: b_str = st.text_input("تخمين 2 (x1):", value="2")
            tol_str = st.text_input("نسبة الخطأ (Tolerance):", value="1e-6")
            
        elif "Differential" in app_mode:
            method = st.selectbox("طريقة الحل:", ["رينج-كوتا الرابعة (RK4)", "رينج-كوتا الثانية (RK2)"])
            col1, col2 = st.columns(2)
            with col1: x0_str = st.text_input("القيمة الابتدائية x0:", value="0")
            with col2: y0_str = st.text_input("القيمة الابتدائية y0:", value="1")
            col3, col4 = st.columns(2)
            with col3: xend_str = st.text_input("قيمة x المستهدفة:", value="1")
            with col4: h_str = st.text_input("حجم الخطوة (h):", value="0.1")
            
        elif "Differentiation" in app_mode:
            method = st.selectbox("طريقة التفاضل:", ["الفروق الأمامية (Forward)", "الفروق الخلفية (Backward)", "الفروق المركزية (Central)"])
            col1, col2 = st.columns(2)
            with col1: x0_str = st.text_input("احسب المشتقة عند x =", value="1")
            with col2: h_str = st.text_input("حجم الخطوة (h):", value="0.01")
            
        elif "Integration" in app_mode:
            method = st.selectbox("طريقة التكامل:", ["شبه المنحرف (Trapezoidal)", "سيمبسون 1/3 (Simpson 1/3)"])
            col1, col2 = st.columns(2)
            with col1: a_str = st.text_input("الحد الأدنى (a):", value="0")
            with col2: b_str = st.text_input("الحد الأقصى (b):", value="2")
            
            input_type = st.radio("طريقة الإدخال:", ["عدد القطاعات (n)", "حجم الخطوة (h)"])
            val = st.number_input("أدخل القيمة (n أو h):", value=10.0, min_value=0.0001)

    st.markdown("<br>", unsafe_allow_html=True)
    calc_btn = st.button("🚀 احسب النتائج", type="primary", use_container_width=True)

# ==========================================
# 📊 العمود الأيمن: شاشة النتائج والرسم
# ==========================================
def parse_expression(expr_str):
    expr_str = expr_str.replace('^', '**')
    return sp.sympify(expr_str)

with col_display:
    if calc_btn:
        # ==================================
        # 1. الاستيفاء (Interpolation)
        # ==================================
        if "Interpolation" in app_mode:
            x_vals, y_vals = edited_pts["X"].values, edited_pts["Y"].values
            if len(set(x_vals)) != len(x_vals):
                st.error("❌ قيم X يجب أن تكون فريدة.")
                st.stop()
                
            x_sym = sp.Symbol('x')
            poly = 0
            if "Lagrange" in method:
                st.markdown("### 📚 القانون الرياضي (Lagrange):")
                st.latex(r"P(x) = \sum_{i=0}^{n} y_i \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}")
                for i in range(len(x_vals)):
                    term = y_vals[i]
                    for j in range(len(x_vals)):
                        if i != j: term *= (x_sym - x_vals[j]) / (x_vals[i] - x_vals[j])
                    poly += term
                    
            poly_simplified = sp.simplify(poly)
            st.markdown("### 📍 معادلة المنحنى المستنتجة:")
            st.info(f"**P(x) =** {poly_simplified}")
            
            p_func = sp.lambdify(x_sym, poly_simplified, "numpy")
            x_curve = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 200)
            y_curve = p_func(x_curve)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name='النقاط المُدخلة', marker=dict(color='red', size=12)))
            fig.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', name='المنحنى', line=dict(color='#007bff', width=2)))
            fig.update_layout(title="منحنى الاستيفاء", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.history.append("📉 استيفاء: تم الحساب بنجاح.")

        # ==================================
        # 2. الأنظمة الخطية (Linear Systems)
        # ==================================
        elif "Linear" in app_mode:
            A_input, b_input = edited_df.iloc[:, :-1].values.astype(float), edited_df.iloc[:, -1].values.astype(float)
            n = len(b_input)
            
            if "Elimination" in method:
                st.markdown("### 📚 طريقة الحذف لجاوس")
                Ab = np.column_stack((A_input, b_input))
                for i in range(n):
                    max_row = np.argmax(np.abs(Ab[i:n, i])) + i
                    Ab[[i, max_row]] = Ab[[max_row, i]]
                    if Ab[i, i] == 0:
                        st.error("❌ المصفوفة شاذة.")
                        st.stop()
                    for j in range(i+1, n):
                        factor = Ab[j, i] / Ab[i, i]
                        Ab[j] = Ab[j] - factor * Ab[i]
                        
                x_res = np.zeros(n)
                for i in range(n-1, -1, -1):
                    x_res[i] = (Ab[i, -1] - np.sum(Ab[i, i+1:n] * x_res[i+1:n])) / Ab[i, i]
                    
                st.markdown("### 📍 النتائج")
                res_df = pd.DataFrame({"المتغير": [f"x{i+1}" for i in range(n)], "القيمة": [f"{x_res[i]:.{dp}f}" for i in range(n)]})
                st.dataframe(res_df, use_container_width=True)
                st.session_state.history.append("🧮 جاوس: تم الحل بنجاح.")
                
            else:
                tol_val = float(parse_expression(tol_str))
                A, b = A_input.copy(), b_input.copy()
                # جعل المصفوفة مهيمنة قطرياً (بشكل مبسط)
                for i in range(n):
                    max_row = np.argmax(np.abs(A[i:n, i])) + i
                    A[[i, max_row]], b[[i, max_row]] = A[[max_row, i]], b[[max_row, i]]

                if np.any(np.diag(A) == 0):
                    st.error("❌ القطر يحتوي على أصفار لا يمكن حلها بالطرق التكرارية مباشرة.")
                    st.stop()

                x_arr, steps_data = np.zeros(n), []
                for it in range(500):
                    x_new = np.zeros(n)
                    for i in range(n):
                        s = sum(A[i, j] * (x_arr[j] if "Jacobi" in method else (x_new[j] if j < i else x_arr[j])) for j in range(n) if i != j)
                        x_new[i] = (b[i] - s) / A[i, i]
                        
                    error = np.max(np.abs(x_new - x_arr))
                    step_dict = {"Iteration": it + 1}
                    step_dict.update({f"x{i+1}": f"{x_new[i]:.{dp}f}" for i in range(n)})
                    step_dict["Error"] = f"{error:.2e}"
                    steps_data.append(step_dict)
                    
                    if error < tol_val: break
                    x_arr = x_new.copy()

                st.markdown(f"### 📍 جدول المحاولات ({method})")
                st.dataframe(pd.DataFrame(steps_data), use_container_width=True)
                st.session_state.history.append(f"🧮 {method}: تم الوصول للحل.")

        # ==================================
        # 3. الجذور (Root Finding)
        # ==================================
        elif "Root" in app_mode:
            try:
                x_sym = sp.Symbol('x')
                func_expr = parse_expression(st.session_state.func_text)
                f_np = sp.lambdify(x_sym, func_expr, "numpy")
                tol = float(parse_expression(tol_str))
                
                st.markdown(f"### 🎯 حل جذور الدالة: **f(x) = {func_expr}**")
                steps, root = [], None
                
                if "Bisection" in method or "False" in method:
                    a, b = float(a_str), float(b_str)
                    if f_np(a) * f_np(b) > 0:
                        st.error("❌ الدالة لا تغير إشارتها بين a و b. يرجى اختيار فترة أخرى.")
                        st.stop()
                    for i in range(100):
                        fa, fb = f_np(a), f_np(b)
                        c = (a + b) / 2 if "Bisection" in method else (a * fb - b * fa) / (fb - fa)
                        fc = f_np(c)
                        error = abs(b - a) / 2 if "Bisection" in method else abs(fc)
                        steps.append({"Iter": i+1, "a": a, "b": b, "c (Root)": c, "f(c)": fc, "Error": error})
                        if error < tol or abs(fc) < tol:
                            root = c; break
                        if fa * fc < 0: b = c
                        else: a = c
                        
                elif "Newton" in method:
                    df_expr = sp.diff(func_expr, x_sym)
                    df_np = sp.lambdify(x_sym, df_expr, "numpy")
                    st.info(f"المشتقة: **f'(x) = {df_expr}**")
                    x0 = float(a_str)
                    for i in range(100):
                        fx0, dfx0 = f_np(x0), df_np(x0)
                        if dfx0 == 0: st.error("المشتقة تساوي صفر!"); break
                        x1 = x0 - fx0 / dfx0
                        error = abs(x1 - x0)
                        steps.append({"Iter": i+1, "x0": x0, "x1 (Root)": x1, "f(x1)": f_np(x1), "Error": error})
                        if error < tol: root = x1; break
                        x0 = x1
                        
                elif "Secant" in method:
                    x0, x1 = float(a_str), float(b_str)
                    for i in range(100):
                        fx0, fx1 = f_np(x0), f_np(x1)
                        if fx1 - fx0 == 0: break
                        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
                        error = abs(x2 - x1)
                        steps.append({"Iter": i+1, "x0": x0, "x1": x1, "x2 (Root)": x2, "Error": error})
                        if error < tol: root = x2; break
                        x0, x1 = x1, x2

                if steps:
                    st.dataframe(pd.DataFrame(steps).style.format(precision=dp), use_container_width=True)
                    st.success(f"✅ الجذر التقريبي هو: **{root:.{dp}f}**")
                    st.session_state.history.append("🎯 جذور: تم إيجاد الجذر بنجاح.")
                    
            except Exception as e: st.error(f"❌ خطأ: {e}")

        # ==================================
        # 4. التكامل العددي (Integration)
        # ==================================
        elif "Integration" in app_mode:
            try:
                x_sym = sp.Symbol('x')
                func_expr = parse_expression(st.session_state.func_text)
                f_np = sp.lambdify(x_sym, func_expr, "numpy")
                a, b = float(a_str), float(b_str)
                exact_val, _ = quad(f_np, a, b)
                
                if input_type == "عدد القطاعات (n)":
                    n = int(val)
                    if "Simpson" in method and n % 2 != 0: n += 1; st.warning(f"تم تعديل n إلى {n} ليكون زوجياً.")
                    h = (b - a) / n
                else:
                    h = val
                    n = int((b - a) / h)
                    if "Simpson" in method and n % 2 != 0: n += 1; h = (b-a)/n; st.warning("تم تعديل h لضمان عدد قطاعات زوجي.")
                
                x_vals = np.linspace(a, b, n + 1)
                y_vals = f_np(x_vals)
                
                if "Trapezoidal" in method:
                    res = (h / 2) * (y_vals[0] + 2 * np.sum(y_vals[1:-1]) + y_vals[-1])
                    st.latex(r"I \approx \frac{h}{2} \left[ f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n) \right]")
                else:
                    res = (h / 3) * (y_vals[0] + 4 * np.sum(y_vals[1:-1:2]) + 2 * np.sum(y_vals[2:-2:2]) + y_vals[-1])
                    st.latex(r"I \approx \frac{h}{3} \left[ f(x_0) + 4\sum f(x_{odd}) + 2\sum f(x_{even}) + f(x_n) \right]")
                
                st.markdown("### 📍 نتائج التكامل")
                col1, col2 = st.columns(2)
                col1.markdown(f"<div class='result-card'><div class='card-title'>القيمة العددية ({method}):</div><div class='card-value'>{res:.{dp}f}</div></div>", unsafe_allow_html=True)
                col2.markdown(f"<div class='result-card exact-card'><div class='card-title'>القيمة الدقيقة (SciPy):</div><div class='card-value exact-value'>{exact_val:.{dp}f}</div></div>", unsafe_allow_html=True)
                st.info(f"الخطأ المطلق: **{abs(exact_val - res):.2e}** | حجم الخطوة (h): **{h:.4f}** | عدد القطاعات (n): **{n}**")
                
                # رسم المساحة
                fig = go.Figure()
                x_dense = np.linspace(a, b, 200)
                fig.add_trace(go.Scatter(x=x_dense, y=f_np(x_dense), mode='lines', name='f(x)'))
                fig.add_trace(go.Bar(x=x_vals, y=y_vals, opacity=0.3, name='القطاعات'))
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.history.append("📈 تكامل: تم الحساب بنجاح.")

            except Exception as e: st.error(f"❌ تأكد من المدخلات والدالة: {e}")

        # ==================================
        # 5. المعادلات التفاضلية (ODEs)
        # ==================================
        elif "Differential" in app_mode:
            try:
                x_sym, y_sym = sp.symbols('x y')
                func_expr = parse_expression(st.session_state.func_text)
                f_np = sp.lambdify((x_sym, y_sym), func_expr, "numpy")
                
                x0, y0 = float(x0_str), float(y0_str)
                xend, h = float(xend_str), float(h_str)
                
                x_arr = np.arange(x0, xend + h/2, h) # +h/2 لضمان تضمين النقطة الأخيرة
                y_arr = np.zeros(len(x_arr))
                y_arr[0] = y0
                
                steps = [{"x": x_arr[0], "y": y_arr[0]}]
                
                for i in range(len(x_arr) - 1):
                    xi, yi = x_arr[i], y_arr[i]
                    if "RK4" in method:
                        k1 = h * f_np(xi, yi)
                        k2 = h * f_np(xi + h/2, yi + k1/2)
                        k3 = h * f_np(xi + h/2, yi + k2/2)
                        k4 = h * f_np(xi + h, yi + k3)
                        y_arr[i+1] = yi + (k1 + 2*k2 + 2*k3 + k4) / 6
                    else: # RK2
                        k1 = h * f_np(xi, yi)
                        k2 = h * f_np(xi + h, yi + k1)
                        y_arr[i+1] = yi + 0.5 * (k1 + k2)
                        
                    steps.append({"x": x_arr[i+1], "y": y_arr[i+1]})
                    
                st.markdown(f"### 📍 حل المعادلة: y' = {func_expr}")
                st.dataframe(pd.DataFrame(steps).style.format(precision=dp), use_container_width=True)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_arr, y=y_arr, mode='lines+markers', name='الحل العددي'))
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.history.append(f"🔬 تفاضلية: تم حل ({method}).")
                
            except Exception as e: st.error(f"❌ خطأ: {e}")

        # ==================================
        # 6. التفاضل العددي (Differentiation)
        # ==================================
        elif "Differentiation" in app_mode:
            try:
                x_sym = sp.Symbol('x')
                func_expr = parse_expression(st.session_state.func_text)
                f_np = sp.lambdify(x_sym, func_expr, "numpy")
                
                x0, h = float(x0_str), float(h_str)
                fx0 = f_np(x0)
                fx_plus = f_np(x0 + h)
                fx_minus = f_np(x0 - h)
                
                if "Forward" in method:
                    res = (fx_plus - fx0) / h
                    st.latex(r"f'(x) \approx \frac{f(x+h) - f(x)}{h}")
                elif "Backward" in method:
                    res = (fx0 - fx_minus) / h
                    st.latex(r"f'(x) \approx \frac{f(x) - f(x-h)}{h}")
                else: # Central
                    res = (fx_plus - fx_minus) / (2 * h)
                    st.latex(r"f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}")
                    
                # حساب القيمة الدقيقة
                exact_diff = sp.diff(func_expr, x_sym)
                exact_val = float(exact_diff.evalf(subs={x_sym: x0}))
                
                col1, col2 = st.columns(2)
                col1.markdown(f"<div class='result-card'><div class='card-title'>المشتقة العددية:</div><div class='card-value'>{res:.{dp}f}</div></div>", unsafe_allow_html=True)
                col2.markdown(f"<div class='result-card exact-card'><div class='card-title'>المشتقة الدقيقة:</div><div class='card-value exact-value'>{exact_val:.{dp}f}</div></div>", unsafe_allow_html=True)
                
                st.info(f"المشتقة الرمزية: **f'(x) = {exact_diff}** | نسبة الخطأ: **{abs(exact_val - res):.2e}**")
                st.session_state.history.append("📐 تفاضل: تم الحساب بنجاح.")
                
            except Exception as e: st.error(f"❌ خطأ: {e}")
