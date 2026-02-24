import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad

# === 1. إعدادات الصفحة (تخطيط عريض) ===
st.set_page_config(page_title="Numerical Integration Dashboard", page_icon="📈", layout="wide")

# === 2. كود CSS لتصميم المربعات (Cards) زي الصورة ===
st.markdown("""
<style>
/* المربع الرئيسي للقيمة الدقيقة */
.exact-card {
    background-color: #f4f9ff;
    border: 1px solid #d2eef9;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    margin-bottom: 20px;
}
.exact-title { color: #888; font-size: 14px; font-weight: bold; }
.exact-value { color: #2d82ff; font-size: 32px; font-weight: bold; margin-top: 5px; }

/* مربعات الطرق الثلاثة */
.method-card {
    background-color: #ffffff;
    border: 1px solid #eaeaea;
    border-radius: 12px;
    padding: 15px;
    text-align: right;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.02);
}
.method-title { color: #444; font-size: 14px; font-weight: bold; margin-bottom: 10px;}
.method-val { color: #222; font-size: 18px; font-weight: bold;}
.method-err { color: #888; font-size: 12px; margin-top: 5px;}

/* تظبيط القائمة الجانبية */
.sidebar-box {
    background-color: #ffffff;
    border: 1px solid #eaeaea;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# === 3. تقسيم الشاشة (عمود للتحكم وعمود للنتائج) ===
col_input, col_output = st.columns([1, 2.5]) # نسبة العرض 1 إلى 2.5

# ==========================================
# ⚙️ العمود الأيسر: مركز العمليات (المدخلات)
# ==========================================
with col_input:
    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("### 🚀 مركز العمليات")
    st.markdown("---")
    
    func_input = st.text_input("🧪 المعادلة الرياضية:", value="sin(2**x)")
    a = st.number_input("🟢 (a) نقطة البداية:", value=2.0)
    b_val = st.number_input("🔴 (b) نقطة النهاية:", value=5.0)
    n = st.number_input("⭐ (n) عدد الشرائح:", value=30, min_value=1, step=1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    calc_btn = st.button("▶️ تشغيل المحرك", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 📊 العمود الأيمن: النتائج والرسم البياني
# ==========================================
with col_output:
    if calc_btn:
        func_str = func_input.replace('^', '**').replace('ln', 'log')
        
        if a >= b_val:
            st.error("❌ خطأ: نقطة النهاية يجب أن تكون أكبر من البداية.")
        else:
            x = sp.Symbol('x')
            try:
                # تحضير الدالة
                f_expr = sp.sympify(func_str, locals={'e': sp.E})
                f = sp.lambdify(x, f_expr, "numpy")
                f(a) # اختبار الدالة
                
                # 1. حساب القيمة الدقيقة باستخدام Scipy
                exact_val, _ = quad(f, a, b_val)
                
                # طباعة القيمة الدقيقة في مربع كبير
                st.markdown(f"""
                <div class='exact-card'>
                    <div class='exact-title'>(Exact) القيمة الدقيقة</div>
                    <div class='exact-value'>{exact_val:.6f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # 2. التجهيز للحساب العددي
                h = (b_val - a) / n
                x_vals = np.linspace(a, b_val, n + 1)
                y_vals = f(x_vals)
                if isinstance(y_vals, (int, float)):
                    y_vals = np.full_like(x_vals, y_vals)

                # دوال فرعية للحساب
                def calc_trapz():
                    return (h / 2) * (y_vals[0] + 2 * np.sum(y_vals[1:-1]) + y_vals[-1])
                
                def calc_simp13():
                    if n % 2 != 0: return None
                    return (h / 3) * (y_vals[0] + 4 * np.sum(y_vals[1:-1:2]) + 2 * np.sum(y_vals[2:-2:2]) + y_vals[-1])
                
                def calc_simp38():
                    if n % 3 != 0: return None
                    integral = y_vals[0] + y_vals[-1]
                    for i in range(1, n):
                        integral += 2 * y_vals[i] if i % 3 == 0 else 3 * y_vals[i]
                    return (3 * h / 8) * integral

                # 3. عرض المربعات الثلاثة
                c1, c2, c3 = st.columns(3)
                
                # --- شبه المنحرف ---
                with c1:
                    trap_val = calc_trapz()
                    err = abs(exact_val - trap_val)
                    st.markdown(f"""
                    <div class='method-card'>
                        <div class='method-title'>📐 (Trapezoidal) شبه المنحرف</div>
                        <div class='method-val'>{trap_val:.6f}</div>
                        <div class='method-err'>الخطأ الفعلي: {err:.2e}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # --- سمبسون 1/3 ---
                with c2:
                    simp13_val = calc_simp13()
                    if simp13_val is not None:
                        err = abs(exact_val - simp13_val)
                        html = f"<div class='method-val'>{simp13_val:.6f}</div><div class='method-err'>الخطأ الفعلي: {err:.2e}</div>"
                    else:
                        html = "<div class='method-err'>❌ تتطلب n عدد زوجي</div>"
                    
                    st.markdown(f"<div class='method-card'><div class='method-title'>🔴 سمبسون 1/3</div>{html}</div>", unsafe_allow_html=True)

                # --- سمبسون 3/8 ---
                with c3:
                    simp38_val = calc_simp38()
                    if simp38_val is not None:
                        err = abs(exact_val - simp38_val)
                        html = f"<div class='method-val'>{simp38_val:.6f}</div><div class='method-err'>الخطأ الفعلي: {err:.2e}</div>"
                    else:
                        html = "<div class='method-err'>❌ تتطلب n مضاعف لـ 3</div>"
                    
                    st.markdown(f"<div class='method-card'><div class='method-title'>🟠 سمبسون 3/8</div>{html}</div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # 4. الرسم البياني (بتصميم احترافي ناعم)
                x_smooth = np.linspace(a, b_val, 1000)
                y_smooth = f(x_smooth)
                if isinstance(y_smooth, (int, float)):
                    y_smooth = np.full_like(x_smooth, y_smooth)

                fig, ax = plt.subplots(figsize=(10, 4))
                
                # رسم المنحنى والتظليل
                ax.plot(x_smooth, y_smooth, color='#2d82ff', linewidth=2)
                ax.fill_between(x_smooth, 0, y_smooth, color='#2d82ff', alpha=0.15)
                
                # تجميل المحاور والخلفية
                ax.axhline(0, color='black', linewidth=0.8, alpha=0.7)
                ax.grid(True, linestyle='--', alpha=0.4)
                
                # إخفاء الإطار الخارجي ليكون نظيفاً
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#dddddd')
                ax.spines['bottom'].set_color('#dddddd')
                
                st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ خطأ في كتابة المعادلة. تأكد من الصيغة الرياضية. ({e})")
    
    # رسالة ترحيب قبل الضغط على الزر
    elif not calc_btn:
        st.info("👈 قم بإدخال المعادلة وتحديد النقاط من 'مركز العمليات' ثم اضغط على 'تشغيل المحرك'.")
