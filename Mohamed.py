import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# إعدادات الصفحة
st.set_page_config(page_title="حاسبة التكامل", page_icon="🧮", layout="centered")

# === كود CSS لإجبار الأزرار تكون جنب بعض على الموبايل ===
st.markdown("""
<style>
/* منع نزول العناصر لسطر جديد وجعلها قابلة للسحب الجانبي (Scroll) */
div[data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    padding-bottom: 5px;
}
/* تصغير المسافات بين العواميد (الأزرار) لتوفير المساحة */
div[data-testid="column"] {
    min-width: fit-content !important;
    padding: 0 3px !important;
}
</style>
""", unsafe_allow_html=True)
# =======================================================

st.title("🧮 حاسبة التكامل العددي التفاعلية")
st.write("أدخل الدالة وحدد المعطيات لرؤية النتيجة والرسم البياني للمساحة تحت المنحنى.")

# --- إدارة حالة النص (لربط الأزرار بخانة الدالة) ---
if 'func_text' not in st.session_state:
    st.session_state.func_text = "e^(-x^2)"

def append_to_func(text):
    st.session_state.func_text += text

def clear_func():
    st.session_state.func_text = ""

# --- واجهة المستخدم ---
func_input = st.text_input("الدالة f(x):", key="func_text")

# --- أزرار المساعدة العلمية ---
st.markdown("<small><b>أزرار مساعدة (اضغط لإضافتها للدالة):</b></small>", unsafe_allow_html=True)

# الصف الأول من الأزرار
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.button("sin()", on_click=append_to_func, args=("sin(",))
c2.button("cos()", on_click=append_to_func, args=("cos(",))
c3.button("tan()", on_click=append_to_func, args=("tan(",))
c4.button("asin()", on_click=append_to_func, args=("asin(",))
c5.button("acos()", on_click=append_to_func, args=("acos(",))
c6.button("atan()", on_click=append_to_func, args=("atan(",))
c7.button("🧹 مسح", on_click=clear_func, type="secondary")

# الصف الثاني من الأزرار
c8, c9, c10, c11, c12, c13, c14 = st.columns(7)
c8.button("ln()", on_click=append_to_func, args=("ln(",))
c9.button("e", on_click=append_to_func, args=("e",))
c10.button("sqrt()", on_click=append_to_func, args=("sqrt(",))
c11.button("pi", on_click=append_to_func, args=("pi",))
c12.button("x²", on_click=append_to_func, args=("**2",))
c13.button("^", on_click=append_to_func, args=("**",))
c14.button("exp()", on_click=append_to_func, args=("exp(",))

st.markdown("---") 

# --- إدخال باقي المعطيات ---
col1, col2 = st.columns(2)
with col1:
    a = st.number_input("الحد الأدنى (a):", value=0.0)
with col2:
    b_val = st.number_input("الحد الأقصى (b):", value=2.0)

method = st.selectbox("طريقة الحل:", [
    "شبه المنحرف (Trapezoidal)", 
    "سمبسون 1/3 (Simpson 1/3)", 
    "سمبسون 3/8 (Simpson 3/8)"
])

input_type = st.radio("المدخلات المتاحة:", ["عدد القطاعات (n)", "حجم الخطوة (h)"])
val = st.number_input("أدخل القيمة (n أو h):", value=4.0, min_value=0.0001)

# --- زر الحساب ---
if st.button("🚀 احسب وارسم", type="primary", use_container_width=True):
    func_str = func_input.replace('^', '**').replace('ln', 'log') 
    
    if a >= b_val:
        st.error("❌ خطأ: الحد الأقصى يجب أن يكون أكبر من الحد الأدنى.")
    else:
        x = sp.Symbol('x')
        try:
            f_expr = sp.sympify(func_str, locals={'e': sp.E})
            f = sp.lambdify(x, f_expr, "numpy")
            f(a) 
            
            if input_type == "حجم الخطوة (h)":
                h = val
                n_calc = (b_val - a) / h
                n = int(round(n_calc))
                if abs(n_calc - n) > 1e-6:
                    st.error(f"❌ خطأ: الخطوة h={h} لا تقسم المسافة بشكل صحيح.")
                    st.stop()
            else:
                n = int(val)
                h = (b_val - a) / n

            if "1/3" in method and n % 2 != 0:
                st.error(f"❌ خطأ: طريقة سمبسون 1/3 تتطلب (n) زوجياً، لكن المدخلات تعطي n = {n}")
                st.stop()
            elif "3/8" in method and n % 3 != 0:
                st.error(f"❌ خطأ: طريقة سمبسون 3/8 تتطلب (n) من مضاعفات 3، لكن المدخلات تعطي n = {n}")
                st.stop()

            x_vals = np.linspace(a, b_val, n + 1)
            y_vals = f(x_vals)
            if isinstance(y_vals, (int, float)):
                y_vals = np.full_like(x_vals, y_vals)

            if "شبه المنحرف" in method:
                result = (h / 2) * (y_vals[0] + 2 * np.sum(y_vals[1:-1]) + y_vals[-1])
            elif "1/3" in method:
                result = (h / 3) * (y_vals[0] + 4 * np.sum(y_vals[1:-1:2]) + 2 * np.sum(y_vals[2:-2:2]) + y_vals[-1])
            elif "3/8" in method:
                integral = y_vals[0] + y_vals[-1]
                for i in range(1, n):
                    if i % 3 == 0:
                        integral += 2 * y_vals[i]
                    else:
                        integral += 3 * y_vals[i]
                result = (3 * h / 8) * integral

            st.success("✅ تم الحساب بنجاح!")
            st.info(f"**النتيجة التقريبية = {float(result):.6f}** \n(عدد القطاعات: {n} | الخطوة: {h})")

            x_smooth = np.linspace(a, b_val, 500)
            y_smooth = f(x_smooth)
            if isinstance(y_smooth, (int, float)):
                y_smooth = np.full_like(x_smooth, y_smooth)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x_smooth, y_smooth, 'b-', linewidth=2, label=f'f(x) = {func_input}')
            
            if n <= 100:
                for i in range(n):
                    xs = [x_vals[i], x_vals[i], x_vals[i+1], x_vals[i+1]]
                    ys = [0, y_vals[i], y_vals[i+1], 0]
                    ax.fill(xs, ys, 'skyblue', edgecolor='black', alpha=0.5, linewidth=1)
            else:
                ax.fill_between(x_smooth, 0, y_smooth, color='skyblue', alpha=0.5)

            ax.axhline(0, color='black', linewidth=1)
            ax.set_title(f"Numerical Integration ({method})", fontsize=12)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            st.pyplot(fig)

        except Exception as e:
            st.error("❌ خطأ: الدالة مكتوبة بشكل خاطئ أو تحتوي على رموز مجهولة. تأكد من استخدام المتغير x.")
