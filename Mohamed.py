import streamlit as st
import numpy as np
import sympy as sp
from scipy.integrate import quad
import plotly.graph_objects as go
import pandas as pd

# === إعدادات الصفحة ===
st.set_page_config(
    page_title="Numerical Analysis Pro", 
    page_icon="🔢", 
    layout="wide"
)

# === تصميم CSS ===
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

# === إدارة حالة النص والسجل ===
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
    "👀 وضع الشرح (إظهار تعويض الخطوة الأولى)", 
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
    for item in reversed(st.session_state.history):
        st.sidebar.success(item)

# === تقسيم الشاشة الأساسي ===
col_control, col_display = st.columns([1, 2.5])

# ==========================================
# ⚙️ العمود الأيسر: لوحة التحكم
# ==========================================
with col_control:
    st.markdown("### 🎛️ إعدادات الإدخال")
    
    # -----------------------------------------------------
    # 1. الاستيفاء (Interpolation)
    # -----------------------------------------------------
    if "Interpolation" in app_mode:
        method = st.selectbox("طريقة الاستيفاء:", ["لاجرانج (Lagrange)"])
        num_pts = st.number_input("عدد النقاط (n):", min_value=2, max_value=10, value=3)
        st.write("أدخل قيم X و Y:")
        df_pts = pd.DataFrame({
            "X": np.arange(num_pts, dtype=float), 
            "Y": np.random.randint(1, 10, num_pts).astype(float)
        })
        edited_pts = st.data_editor(df_pts, use_container_width=True, hide_index=True)
        
    # -----------------------------------------------------
    # 2. الأنظمة الخطية
    # -----------------------------------------------------
    elif "Linear" in app_mode:
        method = st.selectbox(
            "اختر طريقة الحل:", 
            ["جاكوبي (Jacobi)", "جاوس-سيدل (Gauss-Seidel)", "الحذف لجاوس (Gaussian Elimination)"]
        )
        n_vars = st.number_input(
            "عدد المتغيرات (المعادلات):", 
            min_value=2, max_value=10, value=3
        )
        
        default_matrix = np.zeros((n_vars, n_vars + 1))
        if n_vars == 3:
            default_matrix = np.array([
                [2, 8, -1, 11], 
                [-1, 1, 4, 3], 
                [5, -1, 1, 10]
            ], dtype=float)
            
        cols = [f"x{i+1}" for i in range(n_vars)] + ["= b"]
        df_matrix = pd.DataFrame(default_matrix, columns=cols)
        
        st.markdown(f"**المصفوفة الموسعة $[A | b]$:**")
        edited_df = st.data_editor(df_matrix, use_container_width=True, hide_index=True)
        
        if "Gauss" not in method or "Seidel" in method:
            tol_str = st.text_input("نسبة الخطأ (Tolerance):", value="1e-6")
        
    # -----------------------------------------------------
    # 3. الأقسام المعتمدة على الدوال
    # -----------------------------------------------------
    else:
        if "Differential" in app_mode:
            func_input = st.text_input(
                "الدالة التفاضلية y' = f(x,y):", 
                key="func_text", 
                placeholder="مثال: x - y"
            )
        else:
            func_input = st.text_input(
                "الدالة الرياضية f(x):", 
                key="func_text", 
                placeholder="استخدم الآلة الحاسبة..."
            )
        
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
            c5.button("y", on_click=append_to_func, args=("y",))

        angle_mode = st.radio(
            "نظام الزوايا:", 
            ["راديان (Rad)", "درجات (Deg)"], 
            horizontal=True
        )

        if "Root" in app_mode:
            method = st.selectbox(
                "طريقة الحل:", 
                ["التنصيف (Bisection)", "الوضع الخاطئ (False Position)", "نيوتن-رافسون (Newton)", "القاطع (Secant)"]
            )
            if method in ["التنصيف (Bisection)", "الوضع الخاطئ (False Position)"]:
                col1, col2 = st.columns(2)
                with col1: 
                    a_str = st.text_input("من (a):", value="0")
                with col2: 
                    b_str = st.text_input("إلى (b):", value="2")
            elif "Newton" in method:
                a_str = st.text_input("تخمين أولي (x0):", value="1")
                b_str = "0"
            elif "Secant" in method:
                col1, col2 = st.columns(2)
                with col1: 
                    a_str = st.text_input("تخمين 1 (x0):", value="0")
                with col2: 
                    b_str = st.text_input("تخمين 2 (x1):", value="1")
            
            tol_str = st.text_input("نسبة الخطأ (Tolerance):", value="1e-6")
            
        elif "Differential" in app_mode:
            method = st.selectbox(
                "طريقة الحل:", 
                ["رينج-كوتا الرابعة (RK4)", "رينج-كوتا الثانية (RK2)"]
            )
            col1, col2 = st.columns(2)
            with col1: 
                x0_str = st.text_input("القيمة الابتدائية x0:", value="0")
            with col2: 
                y0_str = st.text_input("القيمة الابتدائية y0:", value="1")
            col3, col4 = st.columns(2)
            with col3: 
                xend_str = st.text_input("قيمة x المستهدفة:", value="1")
            with col4: 
                h_str = st.text_input("حجم الخطوة (h):", value="0.1")
            
        elif "Differentiation" in app_mode:
            method = st.selectbox(
                "طريقة التفاضل:", 
                ["الفروق الأمامية (Forward)", "الفروق الخلفية (Backward)", "الفروق المركزية (Central)"]
            )
            col1, col2 = st.columns(2)
            with col1: 
                x0_str = st.text_input("احسب المشتقة عند x =", value="1")
            with col2: 
                h_str = st.text_input("حجم الخطوة (h):", value="0.01")
            
        elif "Integration" in app_mode:
            method = st.selectbox(
                "طريقة التكامل:", 
                ["شبه المنحرف (Trapezoidal)", "سيمبسون 1/3 (Simpson 1/3)"]
            )
            col1, col2 = st.columns(2)
            with col1: 
                a_str = st.text_input("الحد الأدنى (a):", value="0")
            with col2: 
                b_str = st.text_input("الحد الأقصى (b):", value="2")
            
            input_type = st.radio("طريقة الإدخال:", ["عدد القطاعات (n)", "حجم الخطوة (h)"])
            val = st.number_input("أدخل القيمة (n أو h):", value=10.0, min_value=0.0001)

    st.markdown("<br>", unsafe_allow_html=True)
    calc_btn = st.button("🚀 احسب النتائج", type="primary", use_container_width=True)

# ==========================================
# 📊 العمود الأيمن: شاشة النتائج والرسم
# ==========================================
with col_display:
    if calc_btn:
        
        # ==================================
        # 1. الاستيفاء (Interpolation)
        # ==================================
        if "Interpolation" in app_mode:
            x_vals = edited_pts["X"].values
            y_vals = edited_pts["Y"].values
            
            if len(set(x_vals)) != len(x_vals):
                st.error("❌ قيم X يجب أن تكون فريدة.")
                st.stop()
                
            x_sym = sp.Symbol('x')
            poly = 0
            
            if "Lagrange" in method:
                st.markdown("### 📚 القانون الرياضي (Lagrange Polynomial):")
                st.latex(r"P(x) = \sum_{i=0}^{n} y_i \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}")
                
                for i in range(len(x_vals)):
                    term = y_vals[i]
                    for j in range(len(x_vals)):
                        if i != j:
                            term *= (x_sym - x_vals[j]) / (x_vals[i] - x_vals[j])
                    poly += term
                    
            try:
                poly_simplified = sp.simplify(poly)
            except Exception as e:
                st.error(f"حدث خطأ أثناء تبسيط المعادلة: {e}")
                st.stop()
                
            st.markdown("### 📍 معادلة المنحنى المستنتجة:")
            st.info(f"**P(x) =** {poly_simplified}")
            
            if show_tutorial:
                msg = "<b>💡 توضيح:</b> تم حساب حدود لاجرانج وجمعها للوصول للمعادلة."
                html = f"<div class='result-card tutorial-card'>{msg}</div>"
                st.markdown(html, unsafe_allow_html=True)

            p_func = sp.lambdify(x_sym, poly_simplified, "numpy")
            x_curve = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 200)
            y_curve = p_func(x_curve)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode='markers', 
                name='النقاط المُدخلة', marker=dict(color='red', size=12)
            ))
            fig.add_trace(go.Scatter(
                x=x_curve, y=y_curve, mode='lines', 
                name='المنحنى المستنتج', line=dict(color='#007bff', width=2)
            ))
            fig.update_layout(title="منحنى الاستيفاء", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            hist_msg = f"📉 **استيفاء:** تم استنتاج المعادلة لـ {len(x_vals)} نقاط."
            st.session_state.history.append(hist_msg)

        # ==================================
        # 2. الأنظمة الخطية (Linear Systems)
        # ==================================
        elif "Linear" in app_mode:
            try:
                A_input = edited_df.iloc[:, :-1].values.astype(float)
                b_input = edited_df.iloc[:, -1].values.astype(float)
                n = len(b_input)
                
                if "Gauss" not in method or "Seidel" in method:
                    tol_clean = tol_str.replace(' ', '')
                    tol_val = float(sp.sympify(tol_clean, locals={'e': sp.E}))
            except Exception:
                st.error("❌ تأكد من كتابة الأرقام بشكل صحيح.")
                st.stop()
                
            if "Elimination" in method:
                st.markdown("### 📚 طريقة الحذف لجاوس (Gaussian Elimination)")
                Ab = np.column_stack((A_input, b_input))
                
                for i in range(n):
                    max_row = np.argmax(np.abs(Ab[i:n, i])) + i
                    Ab[[i, max_row]] = Ab[[max_row, i]]
                    
                    if Ab[i, i] == 0:
                        st.error("❌ المصفوفة شاذة (Singular). لا يوجد حل وحيد.")
                        st.stop()
                        
                    for j in range(i+1, n):
                        factor = Ab[j, i] / Ab[i, i]
                        Ab[j] = Ab[j] - factor * Ab[i]
                        
                x_res = np.zeros(n)
                for i in range(n-1, -1, -1):
                    x_res[i] = (Ab[i, -1] - np.sum(Ab[i, i+1:n] * x_res[i+1:n])) / Ab[i, i]
                    
                st.markdown("### 📍 قيم المتغيرات النهائية (الحل المباشر)")
                
                res_df = pd.DataFrame({
                    "المتغير": [f"x{i+1}" for i in range(n)], 
                    "القيمة الدقيقة": [f"{x_res[i]:.{dp}f}" for i in range(n)]
                })
                st.dataframe(res_df, use_container_width=True)
                st.session_state.history.append("🧮 **نظام خطي (جاوس):** تم الحل بنجاح.")
                
            else:
                indices = []
                used_rows = set()
                for i in range(n):
                    max_row, max_val = -1, -1
                    for r in range(n):
                        if r not in used_rows:
                            if abs(A_input[r, i]) > max_val:
                                max_val = abs(A_input[r, i])
                                max_row = r
                    indices.append(max_row)
                    used_rows.add(max_row)

                A, b = A_input[indices], b_input[indices]
                if indices != list(range(n)): 
                    st.success("✨ تم إعادة ترتيب المعادلات لضمان الحل (Pivoting)!")
                if np.any(np.diag(A) == 0): 
                    st.error("❌ يوجد صفر على القطر الرئيسي."); st.stop()

                st.markdown("### 📚 القانون الرياضي المستخدم:")
                if "Jacobi" in method: 
                    st.latex(r"x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j \neq i} a_{ij} x_j^{(k)} \right)")
                else: 
                    st.latex(r"x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)} \right)")

                x_arr = np.zeros(n)
                steps_data, iterations, final_error, max_iter = [], 0, 0, 500
                
                for it in range(max_iter):
                    x_new = np.zeros(n)
                    for i in range(n):
                        s = 0
                        for j in range(n):
                            if i != j:
                                if "Jacobi" in method: 
                                    s += A[i, j] * x_arr[j]
                                else: 
                                    s += A[i, j] * (x_new[j] if j < i else x_arr[j])
                        x_new[i] = (b[i] - s) / A[i, i]
                        
                        if show_tutorial and it == 0 and i == 0:
                            msg = f"<b>💡 تعويض الخطوة 1 للمتغير x1:</b><br> x1 = ( {b[i]} - المجموع ) / {A[i,i]} = {x_new[i]:.{dp}f}"
                            html = f"<div class='result-card tutorial-card'>{msg}</div>"
                            st.markdown(html, unsafe_allow_html=True)
                    
                    error = np.max(np.abs(x_new - x_arr))
                    step_dict = {"Iteration": it + 1}
                    for i in range(n): 
                        step_dict[f"x{i+1}"] = f"{x_new[i]:.{dp}f}"
                    step_dict["Error"] = f"{error:.2e}"
                    steps_data.append(step_dict)
                    
                    iterations, final_error = it + 1, error
                    if error < tol_val: break
                    if error > 1e10: break
                    x_arr = x_new.copy()

                if iterations == max_iter or final_error > 1e10: 
                    st.error("❌ الطريقة فشلت في الوصول لحل (Diverged).")
                    
                st.markdown("### 📍 قيم المتغيرات النهائية")
                res_df = pd.DataFrame({
                    "المتغير": [f"x{i+1}"
