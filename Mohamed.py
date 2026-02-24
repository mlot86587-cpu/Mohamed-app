import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# إعدادات الصفحة
st.set_page_config(page_title="حاسبة التكامل", page_icon="🧮", layout="centered")

# === الكود المعدل لتقليل المسافات وجعل الأزرار متقاربة جداً ===
st.markdown("""
<style>
/* إلغاء المسافات الكبيرة بين العواميد (الأزرار) */
[data-testid="stHorizontalBlock"] {
    gap: 0.5rem !important; /* ده السر اللي هيقربهم من بعض */
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    padding-bottom: 5px;
}
/* إجبار العواميد إنها تاخد مساحة الزرار بس */
[data-testid="column"] {
    width: fit-content !important;
    min-width: fit-content !important;
    flex: 0 0 auto !important;
    padding: 0 !important;
}
/* تصغير الأزرار نفسها وتظبيط الهوامش */
.stButton > button {
    width: auto !important;
    padding: 4px 12px !important;
    min-height: 35px !important;
    margin: 0 !important;
}
/* شكل شريط التمرير */
[data-testid="stHorizontalBlock"]::-webkit-scrollbar {
    height: 4px;
}
[data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb {
    background-color: #555;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)
# =======================================================

st.title("🧮 حاسبة التكامل العددي التفاعلية")
st.write("أدخل الدالة وحدد المعطيات لرؤية النتيجة والرسم البياني للمساحة تحت المنحنى.")

# --- مسح النص الافتراضي (الخانة تبدأ فارغة) ---
if 'func_text' not in st.session_state:
    st.session_state.func_text = ""

def append_to_func(text):
    st.session_state.func_text += text

def clear_func():
    st.session_state.func_text = ""

# --- واجهة المستخدم ---
func_input = st.text_input("الدالة f(x):", key="func_text")

# --- أزرار المساعدة العلمية ---
st.markdown("<small><b>أزرار مساعدة (اسحب الشريط يميناً ويساراً):</b></small>", unsafe_allow_html=True)

cols = st.columns(14)
cols[0].button("sin()", on_click=append_to_func, args=("sin(",))
cols[1].button("cos()", on_click=append_to_func, args=("cos(",))
cols[2].button("tan()", on_click=append_to_func, args=("tan(",))
cols[3].button("ln()", on_click=append_to_func, args=("ln(",))
cols[4].button("e", on_click=append_to_func, args=("e",))
cols[5].button("sqrt()", on_click=append_to_func, args=("sqrt(",))
cols[6].button("pi", on_click=append_to_func, args=("pi",))
cols[7].button("x²", on_click=append_to_func, args=("**2",))
cols[8].button("^", on_click=append_to_func, args=("**",))
cols[9].button("exp()", on_click=append_to_func, args=("exp(",))
cols[10].button("asin()", on_click=append_to_func, args=("asin(",))
cols[11].button("acos()", on_click=append_to_func, args=("acos(",))
cols[12].button("atan()", on_click=append_to_func, args=("atan(",))
cols[13].button("🧹 مسح", on_click=clear_func, type="secondary")

st.markdown("------") 

# --- إدخال باقي المعطيات ---
col1, col2 = st.columns(2)
with col1:
    a = st.number_input("الحد الأدنى (a):", value=0.0)
with col2:
    b_val = st.number_input("الحد الأقصى (b):", value=0.0)

method = st.selectbox("طريقة الحل:", [
    "شبه المنحرف (Trapezoidal)", 
    "سمبسون 1/3 (Simpson 1/3)", 
    "سمبسون 3/8 (Simpson 3/8)"
])

input_type = st.radio("المدخلات المتاحة:", ["عدد القطاعات (n)", "حجم الخطوة (h)"])
val = st.number_input("أدخل القيمة (n أو h):", value=1.0, min_value=0.0001)

# --- زر الحساب ---
if st.button("🚀 احسب وارسم", type="primary", use_container_width=True):
    # التأكد أن المستخدم أدخل دالة ولم يترك الخانة فارغة
    if not func_input.strip():
        st.warning("⚠️ يرجى إدخال الدالة أولاً في الخانة المخصصة.")
        st.stop()
        
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
