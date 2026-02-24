import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad

# === إعدادات الصفحة ===
st.set_page_config(page_title="Numerical Integration", page_icon="✨", layout="wide")

# === تصميم CSS (لوحة التحكم، المربعات، وأزرار الآلة الحاسبة) ===
st.markdown("""
<style>
/* تصميم عمود التحكم */
.control-panel {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e9ecef;
}
/* تصميم مربعات النتائج */
.result-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    border-left: 5px solid #007bff;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
.exact-card {
    border-left: 5px solid #28a745;
}
.card-title { color: #6c757d; font-size: 16px; font-weight: bold; margin-bottom: 5px; }
.card-value { color: #212529; font-size: 28px; font-weight: bold; }
.card-subtext { color: #dc3545; font-size: 14px; margin-top: 5px; }

/* تصميم أزرار الآلة الحاسبة لتكون متقاربة وشيك */
div[data-testid="column"] {
    padding: 0 2px !important;
}
.stButton > button {
    width: 100% !important;
    padding: 2px 5px !important;
    font-size: 14px !important;
    font-weight: bold !important;
    height: 35px !important;
    margin-bottom: 5px !important;
}
</style>
""", unsafe_allow_html=True)

# === إدارة حالة النص (لربط أزرار الآلة الحاسبة بالخانة) ===
if 'func_text' not in st.session_state:
    st.session_state.func_text = "x**2 + sin(x)"

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
    
    # خانة الدالة (مرتبطة بالذاكرة)
    func_input = st.text_input("الدالة الرياضية f(x):", key="func_text")
    
    # --- 🧮 لوحة المفاتيح الرياضية ---
    st.markdown("<small style='color:gray;'>لوحة الرموز الرياضية:</small>", unsafe_allow_html=True)
    
    # الصف الأول
    r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
    r1c1.button("sin", on_click=append_to_func, args=("sin(",))
    r1c2.button("cos", on_click=append_to_func, args=("cos(",))
    r1c3.button("tan", on_click=append_to_func, args=("tan(",))
    r1c4.button("π", on_click=append_to_func, args=("pi",))
    r1c5.button("🧹", on_click=clear_func, type="secondary", help="مسح الكل")

    # الصف الثاني (الدوال العكسية والجذر)
    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
    r2c1.button("sin⁻¹", on_click=append_to_func, args=("asin(",))
    r2c2.button("cos⁻¹", on_click=append_to_func, args=("acos(",))
    r2c3.button("tan⁻¹", on_click=append_to_func, args=("atan(",))
    r2c4.button("e", on_click=append_to_func, args=("e",))
    r2c5.button("√x", on_click=append_to_func, args=("sqrt(",))

    # الصف الثالث (اللوغاريتمات والأسس والقسمة)
    r3c1, r3c2, r3c3, r3c4, r3c5 = st.columns(5)
    r3c1.button("ln", on_click=append_to_func, args=("ln(",))
    r3c2.button("x²", on_click=append_to_func, args=("**2",))
    r3c3.button("xʸ", on_click=append_to_func, args=("**",))
    r3c4.button("÷", on_click=append_to_func, args=("/",))
    r3c5.button("( )", on_click=append_to_func, args=("()",))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # باقي إعدادات المسألة
    c1, c2 = st.columns(2)
    with c1:
        a = st.number_input("الحد الأدنى (a):", value=0.0)
    with c2:
        b_val = st.number_input("الحد الأقصى (b):", value=3.0)
        
    method = st.selectbox("اختر طريقة الحل:", [
        "شبه المنحرف (Trapezoidal)", 
        "سمبسون 1/3 (Simpson 1/3)", 
        "سمبسون 3/8 (Simpson 3/8)"
    ])
    
    input_type = st.radio("طريقة الإدخال:", ["لدي عدد القطاعات (n)", "لدي حجم الخطوة (h)"])
    val = st.number_input("أدخل القيمة (n أو h):", value=10.0, min_value=0.0001)
    
    st.markdown("<br>", unsafe_allow_html=True)
    calc_btn = st.button("🚀 احسب وارسم", type="primary", use_container_width=True)

# ==========================================
# 📊 العمود الأيمن: شاشة النتائج والرسم
# ==========================================
with col_display:
    if calc_btn:
        if not func_input.strip():
            st.warning("⚠️ يرجى إدخال الدالة أولاً في الخانة المخصصة.")
            st.stop()
            
        # تحويل الرموز لصيغة بايثون
        func_str = func_input.replace('^', '**').replace('ln', 'log')
        
        if a >= b_val:
            st.error("❌ خطأ: الحد الأقصى يجب أن يكون أكبر من الحد الأدنى.")
        else:
            x = sp.Symbol('x')
            try:
                # 1. تعريف الدالة
                f_expr = sp.sympify(func_str, locals={'e': sp.E})
                f = sp.lambdify(x, f_expr, "numpy")
                f(a) # اختبار سريع
                
                # 2. حساب القيمة الدقيقة (Exact)
                exact_val, exact_error = quad(f, a, b_val)
                
                # 3. معالجة (n) و (h)
                if input_type == "لدي حجم الخطوة (h)":
                    h = val
                    n_calc = (b_val - a) / h
                    n = int(round(n_calc))
                    if abs(n_calc - n) > 1e-6:
                        st.error(f"❌ خطأ: الخطوة h={h} لا تقسم المسافة بشكل صحيح.")
                        st.stop()
                else:
                    n = int(val)
                    h = (b_val - a) / n

                # 4. التأكد من شروط سمبسون
                if "1/3" in method and n % 2 != 0:
                    st.error(f"❌ خطأ: طريقة سمبسون 1/3 تتطلب (n) زوجياً. إدخالك يعطي n = {n}")
                    st.stop()
                elif "3/8" in method and n % 3 != 0:
                    st.error(f"❌ خطأ: طريقة سمبسون 3/8 تتطلب (n) من مضاعفات 3. إدخالك يعطي n = {n}")
                    st.stop()

                # 5. الحساب العددي بالطريقة المختارة
                x_vals = np.linspace(a, b_val, n + 1)
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

                # --- عرض النتائج في مربعات شيك ---
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

                # --- الرسم البياني ---
                st.markdown("### 🎨 الرسم البياني للمساحة")
                x_smooth = np.linspace(a, b_val, 500)
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
                ax.spines['right'].set_visible(False)
                
                st.pyplot(fig)

            except Exception as e:
                st.error("❌ تأكد من كتابة الدالة بشكل صحيح. استخدم الأزرار لتجنب الأخطاء الإملائية.")
    else:
        st.info("👋 أهلاً بك! قم بإدخال الدالة باستخدام الأزرار الرياضية، حدد المعطيات، ثم اضغط على 'احسب وارسم'.")
