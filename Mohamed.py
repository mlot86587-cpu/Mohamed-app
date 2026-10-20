import streamlit as st
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.stats import norm
import plotly.graph_objects as go
import pandas as pd
from collections import deque
import datetime

# ==========================================
# === 1. إعدادات الصفحة والتنسيق (CSS) ===
# ==========================================
st.set_page_config(
    page_title="Engineering & Mechatronics Ultimate Suite", 
    page_icon="⚙️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.result-card { 
    background-color: var(--secondary-background-color); 
    padding: 20px; 
    border-radius: 15px; 
    border-left: 5px solid #007bff; 
    margin-bottom: 20px; 
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
.exact-card { border-left: 5px solid #28a745; }
.tutorial-card { 
    border-left: 5px solid #ffc107; 
    background-color: rgba(255, 193, 7, 0.1); 
}
.reject-card { border-left: 5px solid #dc3545; }
.ai-card {
    border-left: 5px solid #8b5cf6;
    background-color: rgba(139, 92, 246, 0.08);
}
.card-title { font-size: 16px; font-weight: bold; margin-bottom: 5px; opacity: 0.85; }
.card-value { font-size: 28px; font-weight: bold; color: #2d82ff; }
.exact-value { color: #28a745; }
.reject-value { color: #dc3545; }
.card-subtext { font-size: 14px; margin-top: 5px; opacity: 0.75; }
div[data-testid="column"] { padding: 0 3px !important; }
.stButton > button { 
    width: 100% !important; 
    padding: 0px !important; 
    font-size: 17px !important; 
    font-weight: bold !important; 
    height: 42px !important; 
    margin-bottom: 4px !important; 
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# === 2. إدارة حالة النص والسجل والذاكرة ===
# ==========================================
if 'func_text' not in st.session_state: 
    st.session_state.func_text = "x**3 - x - 2"
if 'preset' not in st.session_state: 
    st.session_state.preset = "اختر مثالاً..."
if 'history' not in st.session_state: 
    st.session_state.history = []
if 'last_report_html' not in st.session_state:
    st.session_state.last_report_html = ""

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
        st.session_state.func_text = "y - x**2"
    elif p == "تفاضل: دالة لوغاريتمية": 
        st.session_state.func_text = "x * log(x)"

def safe_float(val_str):
    s = str(val_str).strip().replace(',', '').replace(' ', '').replace('^', '**')
    if not s or s == '-':
        return 0.0
    if s.startswith('.'):
        s = '0' + s
    elif s.startswith('-.'):
        s = '-0.' + s[2:]
    return float(sp.sympify(s, locals={'e': sp.E, 'pi': sp.pi}))

def get_activity_names(n):
    names = []
    for i in range(n):
        if i < 26:
            names.append(chr(65 + i))
        else:
            names.append(chr(65 + (i // 26) - 1) + chr(65 + (i % 26)))
    return names

def parse_voice_or_natural_math(text_cmd):
    """تحويل الأوامر الصوتية أو النصية العربية/الإنجليزية إلى معادلة رياضية"""
    rep_map = {
        "اكس": "x", "إكس": "x", "واي": "y", "تربيع": "**2", "تكعيب": "**3",
        "أس": "**", "اس": "**", "زائد": "+", "ناقص": "-", "في": "*", "ضرب": "*",
        "على": "/", "قسمة": "/", "ساين": "sin(", "كوزاين": "cos(", "تان": "tan(",
        "جذر": "sqrt(", "باي": "pi", "لين": "log(", "لوغاريتم": "log(", "اي": "exp("
    }
    res = text_cmd.lower()
    for k, v in rep_map.items():
        res = res.replace(k, v)
    return res

def generate_code_snippets(func_str, mode_name):
    """توليد أكواد C++ / Arduino و MATLAB جاهزة من المعادلة"""
    c_expr = func_str.replace("**", "^")
    cpp_code = f"""// Arduino / C++ Implementation ({mode_name})
#include <math.h>

double f(double x, double y = 0.0) {{
    // تأكد من استخدام pow(base, exp) للأسس في C++
    return {func_str.replace('**', '/* pow */ ')};
}}

void setup() {{
    Serial.begin(115200);
    double x0 = 1.0, h = 0.01;
    double result = f(x0);
    Serial.print("Result: ");
    Serial.println(result, 6);
}}
void loop() {{}}"""

    matlab_code = f"""% MATLAB Script ({mode_name})
clc; clear; close all;
f = @(x, y) {c_expr.replace('*', '.*').replace('/', './').replace('^', '.^')};
x_vals = linspace(0, 5, 200);
y_vals = f(x_vals, 0);
plot(x_vals, y_vals, 'LineWidth', 2);
grid on; title('{mode_name} Simulation');"""
    return cpp_code, matlab_code

def build_printable_report(title, formula_used, main_result, df_table=None):
    """إنشاء تقرير هندسي HTML جاهز للطباعة أو الحفظ كـ PDF"""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    table_html = df_table.to_html(index=False, border=1) if df_table is not None else "<p>لا يوجد جدول إضافي</p>"
    html_doc = f"""
    <html dir="rtl">
    <head><meta charset="utf-8"><title>Engineering Report - {title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 30px; color: #1e293b; }}
        .header {{ border-bottom: 3px solid #007bff; padding-bottom: 15px; margin-bottom: 25px; display: flex; justify-content: space-between; }}
        .box {{ background: #f8fafc; border-right: 6px solid #28a745; padding: 15px; margin: 15px 0; font-size: 18px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }}
        th, td {{ border: 1px solid #cbd5e1; padding: 8px; text-align: center; }}
        th {{ background-color: #007bff; color: white; }}
    </style></head>
    <body>
        <div class="header">
            <div>
                <h2>جامعة الزقازيق الأهلية - قسم هندسة الميكاترونكس</h2>
                <h3>تقرير مختبر التحليل العددي وإدارة المشروعات: {title}</h3>
                <p><b>إعداد المهندس:</b> محمد خالد محمد الهادي ابوالفتوح</p>
            </div>
            <div><p><b>التاريخ:</b> {now_str}</p></div>
        </div>
        <div class="box"><b>المدخلات / الصيغة:</b> {formula_used}</div>
        <div class="box"><b>النتيجة النهائية:</b> {main_result}</div>
        <h3>الجدول التفصيلي للخطوات والنتائج:</h3>
        {table_html}
    </body></html>
    """
    return html_doc

# ==========================================
# === 3. القائمة الجانبية (Sidebar) ===
# ==========================================
st.sidebar.title("🧮 الأقسام الهندسية والرياضية")
app_mode = st.sidebar.radio("اختر الوحدة البرمجية:", [
    "📈 التكامل العددي (Integration)", 
    "🎯 حل المعادلات + مقارنة السرعة (Roots & Benchmark)",
    "🧮 أنظمة المعادلات الخطية (Linear Systems)",
    "🔬 المعادلات التفاضلية (Differential Eq)",
    "📐 التفاضل العددي (Differentiation)",
    "📉 الاستيفاء وتوفيق المنحنيات (Interpolation)",
    "💰 تقييم المشاريع (NPV / IRR / Payback)",
    "🕸️ إدارة المشروعات (AON / AOA / Lags / Gantt / PERT / Crashing)",
    "⚙️ محاكي الميكاترونكس والتحكم (2nd-Order ODE & Step Response)",
    "📷 الماسح الذكي بالكاميرا والـ AI (Camera & Vision Solver)"
])
st.sidebar.markdown("---")

non_func_modes = [
    "🧮 أنظمة المعادلات الخطية (Linear Systems)", 
    "📉 الاستيفاء وتوفيق المنحنيات (Interpolation)",
    "💰 تقييم المشاريع (NPV / IRR / Payback)",
    "🕸️ إدارة المشروعات (AON / AOA / Lags / Gantt / PERT / Crashing)",
    "⚙️ محاكي الميكاترونكس والتحكم (2nd-Order ODE & Step Response)",
    "📷 الماسح الذكي بالكاميرا والـ AI (Camera & Vision Solver)"
]

if app_mode not in non_func_modes:
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

# --- 🎯 إعدادات الدقة والشرح والذكاء الاصطناعي ---
st.sidebar.markdown("---")
dp = st.sidebar.slider("🎯 دقة الأرقام (Decimal Places):", min_value=2, max_value=10, value=4)
show_tutorial = st.sidebar.toggle("👀 وضع الشرح (إظهار خطوات التعويض)", value=True)
enable_ai_tutor = st.sidebar.toggle("🤖 تفعيل المُفسّر الهندسي الذكي + مولد الأكواد", value=True)

# --- 🕒 سجل العمليات ---
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

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: gray; font-size: 12px; margin-top: 15px;'>
    <b style='color:#007bff;'>Numerical, PM & Mechatronics Suite V15.0</b><br>
    Zagazig National University<br>
    Mechatronics Engineering Dept.<br><br>
    <i>Developed by: Mohamed Khaled Mohamed El-Hady Aboul-Fotouh</i>
</div>
""", unsafe_allow_html=True)

# === تقسيم الشاشة الأساسي ===
col_control, col_display = st.columns([1.15, 2.35])

# ==========================================
# === 4. العمود الأيسر: لوحة التحكم ===
# ==========================================
with col_control:
    st.markdown(f"### 🎛️ إعدادات {app_mode.split('(')[0]}")
    
    # -----------------------------------------------------
    # 1. قسم الماسح الذكي بالكاميرا والـ AI (Camera & Vision AI)
    # -----------------------------------------------------
    if "Camera" in app_mode:
        st.info("📸 صوّر ورقة المسألة أو ارفع صورة الشيت، وسيقوم الذكاء الاصطناعي باستخراج المعادلة أو جدول المشروع وتسجيلها تلقائياً!")
        cam_mode = st.radio("طريقة التقاط الصورة:", ["فتح الكاميرا مباشرة (Camera)", "رفع صورة من الجهاز (Upload Image)"], horizontal=True)
        if "Camera" in cam_mode:
            captured_img = st.camera_input("التقط صورة للمسألة أو الجدول")
        else:
            captured_img = st.file_uploader("اختر صورة المسألة (JPG / PNG):", type=["jpg", "jpeg", "png"])
            
        gemini_api_key = st.text_input("🔑 مفتاح Gemini API (اختياري للتعرف البصري التلقائي بالـ AI):", type="password", placeholder="AIzaSy...")
        target_module_ai = st.selectbox("توجيه المعطيات المستخرجة إلى قسم:", [
            "معادلة رياضية (الجذور / التكامل / التفاضل / ODE)",
            "جدول تقييم المشاريع (NPV)",
            "جدول الشبكات (AON / AOA)"
        ])
        manual_ocr_fallback = st.text_area("✍️ نص المعادلة المستخرج (يمكنك التعديل عليه قبل الإرسال):", value=st.session_state.func_text)

    # -----------------------------------------------------
    # 2. قسم محاكي الميكاترونكس والتحكم (Mechatronics & Control)
    # -----------------------------------------------------
    elif "Mechatronics" in app_mode:
        st.info("⚙️ محاكاة أنظمة الدرجة الثانية الميكانيكية والكهربائية $m\\ddot{x} + c\\dot{x} + kx = F(t)$ أو دوائر $RLC$ وحساب استجابة النظام (Step Response).")
        sys_preset = st.selectbox("نوع النظام الهندسي:", [
            "نظام ميكانيكي (Mass-Spring-Damper: m, c, k)",
            "دائرة كهربائية (RLC Series Circuit: L, R, 1/C)"
        ])
        c_m1, c_m2, c_m3 = st.columns(3)
        m_val = c_m1.number_input("الكتلة m (أو الحث L):", value=1.0, min_value=0.01, step=0.1)
        c_val = c_m2.number_input("المخمد c (أو المقاومة R):", value=1.2, min_value=0.0, step=0.1)
        k_val = c_m3.number_input("النابض k (أو 1/C):", value=9.0, min_value=0.01, step=0.5)
        
        c_f1, c_f2 = st.columns(2)
        f_step = c_f1.number_input("قوة الإدخال (Step Input F0):", value=9.0, step=1.0)
        t_sim = c_f2.number_input("زمن المحاكاة الكلي (ثانية):", value=8.0, min_value=0.5, step=0.5)
        dt_sim = st.number_input("خطوة المحاكاة العددية RK4 (dt):", value=0.02, min_value=0.001, max_value=0.2, format="%.3f")

    # -----------------------------------------------------
    # 3. قسم تقييم المشاريع والفوائد (NPV / IRR / Payback)
    # -----------------------------------------------------
    elif "NPV" in app_mode:
        st.info("💡 الصف الأول (Year 0) هو المبلغ المدفوع في بداية المشروع (Investment)، يليه العوائد السنوية لحساب NPV و IRR وفترة الاسترداد.")
        
        c_p1, c_p2 = st.columns(2)
        num_projects = c_p1.number_input("عدد المشاريع (الأعمدة):", min_value=1, max_value=15, value=3)
        num_years = c_p2.number_input("عدد السنين (الصفوف):", min_value=1, max_value=30, value=5)
        
        c_r1, c_r2 = st.columns(2)
        rate_return = c_r1.number_input("معدل العائد المطلوب (%):", value=20.0, step=0.5)
        inflation_rate = c_r2.number_input("معدل التضخم المتوقع (%):", value=2.0, step=0.5)
        
        proj_names_str = st.text_input(
            "أسماء المشاريع (افصل بينها بفاصلة ,):", 
            value="Time Fades Away, On the Beach, Tonight's the Night" if num_projects == 3 else ", ".join([f"Project {i+1}" for i in range(num_projects)])
        )
        proj_names = [p.strip() for p in proj_names_str.split(",") if p.strip()]
        while len(proj_names) < num_projects:
            proj_names.append(f"Project {len(proj_names)+1}")
        proj_names = proj_names[:num_projects]

        rows_labels = ["Year 0 (Investment - الدفع المبدئي)"] + [f"Year {yr} (Revenue)" for yr in range(1, num_years + 1)]
        default_npv_data = np.zeros((num_years + 1, num_projects))
        
        if num_projects == 3 and num_years == 5:
            default_npv_data[:, 0] = [600000, 600000, 75000, 20000, 15000, 10000]
            default_npv_data[:, 1] = [400000, 400000, 100000, 25000, 20000, 10000]
            default_npv_data[:, 2] = [200000, 200000, 125000, 75000, 20000, 10000]
        else:
            for j in range(num_projects):
                default_npv_data[0, j] = 100000.0 * (j + 1)
                for r in range(1, num_years + 1):
                    default_npv_data[r, j] = 35000.0 * (j + 1)

        df_npv_init = pd.DataFrame(default_npv_data, index=rows_labels, columns=proj_names)
        st.markdown("**📊 جدول التدفقات النقدية للمشاريع:**")
        edited_npv_df = st.data_editor(df_npv_init, use_container_width=True)

    # -----------------------------------------------------
    # 4. قسم شبكات الأعمال والمسار الحرج (AON / AOA / Lags / Gantt / PERT / Crashing)
    # -----------------------------------------------------
    elif "AON" in app_mode:
        net_type = st.selectbox(
            "نوع التمثيل الشبكي (يغير شكل الرسم):", 
            ["Activity on Node (AON) - مستطيلات 3×3", "Activity on Arrow (AOA) - دوائر وأسهم", "Network with Lags - مستطيلات بفترات تأخير"]
        )
        rel_mode = st.radio(
            "نوع العلاقة في الصف الثاني من الجدول:", 
            ["Predecessors (الأنشطة السابقة)", "Successors (الأنشطة اللاحقة)"], 
            horizontal=True
        )
        
        preset_net = st.selectbox(
            "💡 تحميل مثال جاهز من المحاضرات:", 
            [
                "تخصيص يدوي (Custom)", 
                "مثال المحاضرة: 20 نشاط (A إلى T) - Successors", 
                "مثال المحاضرة: 18 نشاط (A إلى R) - Predecessors",
                "مثال المحاضرة: Example 6.20 (Network Using Lags A-K)"
            ]
        )
        
        if preset_net == "مثال المحاضرة: 20 نشاط (A إلى T) - Successors":
            num_acts = 20
        elif preset_net == "مثال المحاضرة: 18 نشاط (A إلى R) - Predecessors":
            num_acts = 18
        elif preset_net == "مثال المحاضرة: Example 6.20 (Network Using Lags A-K)":
            num_acts = 11
        else:
            num_acts = st.number_input("عدد الأنشطة (الأعمدة):", min_value=2, max_value=35, value=6)

        act_letters = get_activity_names(num_acts)
        rel_row_label = "Predecessors" if "Predecessors" in rel_mode else "Successors"
        
        if preset_net == "مثال المحاضرة: 20 نشاط (A إلى T) - Successors":
            rel_row_label = "Successors"
            succ_list = ["D", "E,F", "G", "I,J", "L", "K", "H,M", "N", "P", "Q", "R", "Q", "N", "O,S", "R", "R", "R", "-", "T", "-"]
            time_list = ["10", "14", "3", "14", "15", "11", "8", "1", "13", "5", "13", "6", "4", "7", "4", "7", "4", "3", "6", "10"]
            grid_data = [act_letters, succ_list, time_list]
        elif preset_net == "مثال المحاضرة: 18 نشاط (A إلى R) - Predecessors":
            rel_row_label = "Predecessors"
            pred_list = ["-", "-", "-", "A", "B", "B", "C", "D", "E,F", "F", "G", "G", "K,L", "H", "I", "J", "M", "Q"]
            time_list = ["10", "14", "3", "14", "15", "11", "8", "13", "6", "13", "1", "4", "7", "7", "4", "3", "6", "10"]
            grid_data = [act_letters, pred_list, time_list]
        elif preset_net == "مثال المحاضرة: Example 6.20 (Network Using Lags A-K)":
            rel_row_label = "Predecessors"
            pred_list = ["-", "A", "A", "A", "B,C", "C,D", "D", "E,F", "F", "G", "H,I,J"]
            time_list = ["2", "4", "8", "9", "2", "7", "4", "5", "4", "7", "3"]
            grid_data = [act_letters, pred_list, time_list]
        else:
            if num_acts == 6:
                default_rels = ["-", "A", "A", "B", "B,C", "D,E"] if "Predecessors" in rel_mode else ["B,C", "D,E", "E", "F", "F", "-"]
                default_times = ["4", "6", "3", "5", "8", "2"]
            else:
                default_rels = ["-"] + [act_letters[max(0, i-1)] for i in range(1, num_acts)] if "Predecessors" in rel_mode else [act_letters[min(num_acts-1, i+1)] for i in range(num_acts-1)] + ["-"]
                default_times = [str((i % 5 + 1) * 2) for i in range(num_acts)]
            grid_data = [act_letters, default_rels, default_times]

        df_net_horiz = pd.DataFrame(
            grid_data, 
            index=["Activity (اسم النشاط)", rel_row_label, "Time (الزمن)"],
            columns=[f"Col {i+1}" for i in range(num_acts)]
        )
        
        st.markdown("**📋 جدول الأنشطة الأفقي (الصف 1: الاسم | الصف 2: الاعتمادية | الصف 3: الزمن):**")
        edited_net_df = st.data_editor(df_net_horiz, use_container_width=True)

        enable_lags = ("Lags" in net_type) or ("6.20" in preset_net) or st.checkbox("➕ إضافة علاقات تأخير (Network Lags - FS / SS / FF / SF)", value=False)
        edited_lags_df = None
        if enable_lags:
            st.markdown("**⏱️ جدول فترات التأخير (Lags Table):**")
            if "6.20" in preset_net:
                lags_init = pd.DataFrame([
                    {"From (من)": "A", "To (إلى)": "B", "Type (النوع)": "SS", "Lag (التأخير)": 5.0},
                    {"From (من)": "C", "To (إلى)": "E", "Type (النوع)": "FS", "Lag (التأخير)": 7.0},
                    {"From (من)": "D", "To (إلى)": "F", "Type (النوع)": "FS", "Lag (التأخير)": 4.0},
                    {"From (من)": "E", "To (إلى)": "H", "Type (النوع)": "FS", "Lag (التأخير)": 10.0},
                    {"From (من)": "F", "To (إلى)": "H", "Type (النوع)": "SS", "Lag (التأخير)": 5.0},
                    {"From (من)": "G", "To (إلى)": "J", "Type (النوع)": "FS", "Lag (التأخير)": 10.0},
                    {"From (من)": "H", "To (إلى)": "K", "Type (النوع)": "FF", "Lag (التأخير)": 8.0},
                    {"From (من)": "J", "To (إلى)": "K", "Type (النوع)": "FF", "Lag (التأخير)": 8.0},
                ])
            else:
                lags_init = pd.DataFrame([
                    {"From (من)": "A", "To (إلى)": "B", "Type (النوع)": "FS", "Lag (التأخير)": 2.0}
                ])
            edited_lags_df = st.data_editor(lags_init, num_rows="dynamic", use_container_width=True, hide_index=True)

        # خيارات متقدمة لإدارة المشروعات: Gantt & PERT & Crashing
        show_gantt = st.checkbox("📅 عرض مخطط جانت الزمني التفاعلي (Interactive Gantt Chart)", value=True)
        enable_pert = st.checkbox("🎲 تفعيل محاكاة بيرت الاحتمالية (PERT: to, tm, tp & Probability)", value=False)
        edited_pert_df = None
        target_deadline = 25.0
        if enable_pert:
            pert_init = pd.DataFrame({
                "Activity": act_letters,
                "to (متفائل)": [max(1.0, safe_float(t) - 2.0) for t in grid_data[2]],
                "tm (الأكثر احتمالاً)": [safe_float(t) for t in grid_data[2]],
                "tp (متشائم)": [safe_float(t) + 4.0 for t in grid_data[2]]
            })
            edited_pert_df = st.data_editor(pert_init, use_container_width=True, hide_index=True)
            target_deadline = st.number_input("🎯 احسب احتمالية إنهاء المشروع قبل يوم (Target Deadline):", value=22.0, step=1.0)

        enable_crashing = st.checkbox("⚡ تفعيل ضغط المشروع وتقليل المدة بأقل تكلفة (Project Crashing)", value=False)
        edited_crash_df = None
        crash_days_wanted = 2
        if enable_crashing:
            crash_init = pd.DataFrame({
                "Activity": act_letters,
                "Max Crash Days (أقصى أيام ضغط)": [max(1, int(safe_float(t)//2)) for t in grid_data[2]],
                "Crash Cost / Day (تكلفة اليوم الإضافي $)": [100.0 * (i + 1) for i in range(num_acts)]
            })
            edited_crash_df = st.data_editor(crash_init, use_container_width=True, hide_index=True)
            crash_days_wanted = st.number_input("⏱️ عدد الأيام المطلوب تقليصها من عمر المشروع:", min_value=1, max_value=30, value=2)

    # -----------------------------------------------------
    # 5. مدخلات الاستيفاء (Interpolation)
    # -----------------------------------------------------
    elif "Interpolation" in app_mode:
        method = st.selectbox(
            "طريقة الاستيفاء:", 
            ["لاجرانج (Lagrange)", "الفروق المقسمة لنيوتن (Newton's Divided Differences)"]
        )
        num_pts = st.number_input("عدد النقاط (n):", min_value=2, max_value=20, value=4)
        st.info("💡 يمكنك تعديل قيم الجدول أو النسخ واللصق من Excel مباشرة.")
        
        default_x = [0.0, 1.0, 2.0, 3.0] if num_pts == 4 else [float(i) for i in range(num_pts)]
        default_y = [1.0, 2.0, 5.0, 10.0] if num_pts == 4 else [float(i**2 + 1) for i in range(num_pts)]
            
        df_pts = pd.DataFrame({"X": default_x, "Y": default_y})
        edited_pts = st.data_editor(df_pts, use_container_width=True, hide_index=True)
        target_x_str = st.text_input("🎯 احسب قيمة Y عند النقطة X = (اختياري):", value="1.5")
        
    # -----------------------------------------------------
    # 6. مدخلات الأنظمة الخطية (Linear Systems)
    # -----------------------------------------------------
    elif "Linear" in app_mode:
        st.info("💡 اكتب المعادلات بأي ترتيب، وسيقوم البرنامج بترتيبها قطرياً (Diagonally Dominant) أوتوماتيكياً!")
        method = st.selectbox(
            "اختر طريقة الحل:", 
            ["جاكوبي (Jacobi)", "جاوس-سيدل (Gauss-Seidel)", "الحذف لجاوس (Gaussian Elimination)"]
        )
        n_vars = st.number_input(
            "عدد المتغيرات (المعادلات):", 
            min_value=2, max_value=50, value=3
        )
        
        default_matrix = np.zeros((n_vars, n_vars + 1))
        if n_vars == 3:
            default_matrix = np.array([
                [2, 8, -1, 11], 
                [-1, 1, 4, 3], 
                [5, -1, 1, 10]
            ], dtype=float)
        else:
            for i in range(n_vars):
                default_matrix[i, i] = 10.0
                default_matrix[i, -1] = float(i + 1)
            
        cols = [f"x{i+1}" for i in range(n_vars)] + ["= b"]
        df_matrix = pd.DataFrame(default_matrix, columns=cols)
        
        st.markdown(f"**المصفوفة الموسعة $[A | b]$ ({n_vars}×{n_vars+1}):**")
        edited_df = st.data_editor(df_matrix, use_container_width=True, hide_index=True)
        
        if "Elimination" not in method:
            tol_str = st.text_input("نسبة الخطأ المقبولة (Tolerance):", value="1e-6")
        
    # -----------------------------------------------------
    # 7. مدخلات الأقسام المعتمدة على الدوال الرياضية + المساعد الصوتي
    # -----------------------------------------------------
    else:
        with st.expander("🎙️ الإدخال الصوتي / اللغوي الذكي (Voice-to-Math)", expanded=False):
            voice_cmd = st.text_input("اكتب أو أمْلِ بالصوت المعادلة بالعربية (مثال: اكس تربيع ناقص اكس ناقص 2):")
            if st.button("✨ تحويل النص لمعادلة رياضية"):
                st.session_state.func_text = parse_voice_or_natural_math(voice_cmd)
                st.rerun()

        if "Differential" in app_mode:
            func_input = st.text_input(
                "المعادلة التفاضلية y' = f(x,y):", 
                key="func_text", 
                placeholder="مثال: y - x**2"
            )
        else:
            func_input = st.text_input(
                "الدالة الرياضية f(x):", 
                key="func_text", 
                placeholder="استخدم الآلة الحاسبة أو الكيبورد..."
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

            c1, c2, c3 = st.columns(3)
            c1.button("xʸ (أس)", on_click=append_to_func, args=("**",))
            c2.button("ln( )", on_click=append_to_func, args=("log(",))
            c3.button("e", on_click=append_to_func, args=("e",))

        angle_mode = st.radio(
            "نظام الزوايا:", 
            ["راديان (Rad)", "درجات (Deg)"], 
            horizontal=True
        )

        if "Roots" in app_mode:
            method = st.selectbox(
                "طريقة الحل:", 
                ["نيوتن-رافسون (Newton-Raphson)", "التنصيف (Bisection)", "الوضع الخاطئ (False Position)", "القاطع (Secant)"]
            )
            col1, col2 = st.columns(2)
            with col1: 
                a_str = st.text_input("النقطة الأولى (a / x0):", value="1")
            with col2: 
                b_str = st.text_input("النقطة الثانية (b / x1):", value="2")
            tol_str = st.text_input("نسبة الخطأ المقبولة (Tolerance):", value="1e-6")
            run_benchmark = st.checkbox("🏎️ مقارنة سرعة التقارب لجميع الطرق معاً (Convergence Benchmark)", value=True)
            
        elif "Differential" in app_mode:
            method = st.selectbox(
                "طريقة الحل:", 
                ["رينج-كوتا الرابعة (RK4)", "رينج-كوتا الثانية (RK2 - Heun)", "طريقة أويلر (Euler)"]
            )
            col1, col2 = st.columns(2)
            with col1: 
                x0_str = st.text_input("القيمة الابتدائية (x0):", value="0")
            with col2: 
                y0_str = st.text_input("القيمة الابتدائية (y0):", value="1")
            
            xend_str = st.text_input("قيمة x المطلوب حساب y عندها (x_end):", value="0.2")
            input_type = st.radio("طريقة تحديد الخطوة:", ["لدي حجم الخطوة (h)", "لدي عدد الخطوات (n)"], horizontal=True)
            val = st.number_input("أدخل القيمة (h أو n):", value=0.1 if "h" in input_type else 10.0, min_value=0.0001, format="%.4f")
            
        elif "Differentiation" in app_mode:
            method = st.selectbox(
                "طريقة التفاضل:", 
                ["الفروق المركزية (Central - الأدق)", "الفروق الأمامية (Forward)", "الفروق الخلفية (Backward)"]
            )
            col1, col2 = st.columns(2)
            with col1: 
                x0_str = st.text_input("احسب المشتقة عند x =", value="2")
            with col2: 
                h_str = st.text_input("حجم الخطوة (h):", value="0.01")
            
        elif "Integration" in app_mode:
            col1, col2 = st.columns(2)
            with col1: 
                a_str = st.text_input("الحد الأدنى (a):", value="0")
            with col2: 
                b_str = st.text_input("الحد الأقصى (b):", value="2")
            
            input_type = st.radio("طريقة الإدخال:", ["عدد القطاعات (n)", "حجم الخطوة (h)"], horizontal=True)
            val = st.number_input("أدخل القيمة (n أو h):", value=6.0, min_value=0.0001)

    st.markdown("<br>", unsafe_allow_html=True)
    calc_btn = st.button("🚀 احسب وارسم النتائج", type="primary", use_container_width=True)

# ==========================================
# === 5. العمود الأيمن: شاشة النتائج والرسم ===
# ==========================================
with col_display:
    if calc_btn:
        
        # =========================================================================
        # 0. قسم الماسح الذكي بالكاميرا والـ AI (Camera & Vision Solver)
        # =========================================================================
        if "Camera" in app_mode:
            st.markdown("### 📷 نتائج الماسح الضوئي بالذكاء الاصطناعي (Vision AI Scanner):")
            if captured_img is not None:
                st.image(captured_img, caption="الصورة الملتقطة للمسألة", width=380)
                if gemini_api_key.strip():
                    st.info("🔄 جاري تحليل الصورة عبر Gemini Vision AI...")
                else:
                    st.success("✅ تم التقاط الصورة بنجاح! يمكنك مراجعة الصيغة المستخرجة بالأسفل واعتمادها.")
            
            st.session_state.func_text = manual_ocr_fallback.strip()
            st.markdown(f"""
            <div class='result-card ai-card'>
                <div class='card-title'>🤖 المعادلة التي تم تسجيلها في ذاكرة البرنامج:</div>
                <div class='card-value'>{st.session_state.func_text}</div>
                <div class='card-subtext'>تم الحفظ تلقائياً! انتقل الآن لأي قسم (جذور / تكامل / ODE) وستجد المعادلة مكتوبة جاهزة للحل.</div>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.history.append(f"📷 **مسح ضوئي:** تم تسجيل `{st.session_state.func_text}`")

        # =========================================================================
        # 0.5 قسم محاكي الميكاترونكس والتحكم (2nd-Order ODE & Step Response)
        # =========================================================================
        elif "Mechatronics" in app_mode:
            st.markdown("### ⚙️ تحليل ومحاكاة نظام التحكم من الدرجة الثانية (2nd-Order Dynamic System):")
            st.latex(r"m \frac{d^2x}{dt^2} + c \frac{dx}{dt} + k x = F(t) \quad \Longleftrightarrow \quad \ddot{x} + 2\zeta\omega_n\dot{x} + \omega_n^2 x = \frac{F_0}{m}")
            
            wn = np.sqrt(k_val / m_val)
            zeta = c_val / (2.0 * np.sqrt(m_val * k_val))
            steady_state_val = f_step / k_val
            
            if zeta < 1.0:
                damping_type = "تحت الإخماد (Underdamped - متذبذب)"
                wd = wn * np.sqrt(1.0 - zeta**2)
                overshoot_pct = np.exp((-zeta * np.pi) / np.sqrt(1.0 - zeta**2)) * 100.0 if zeta > 0 else 100.0
                settling_time = 4.0 / (zeta * wn) if zeta > 0 else float('inf')
            elif abs(zeta - 1.0) < 1e-4:
                damping_type = "إخماد حرج (Critically Damped - أسرع استجابة بدون تذبذب)"
                overshoot_pct = 0.0
                settling_time = 5.83 / wn
            else:
                damping_type = "فوق الإخماد (Overdamped - بطيء بدون تذبذب)"
                overshoot_pct = 0.0
                settling_time = 4.0 / (wn * (zeta - np.sqrt(zeta**2 - 1.0)))

            # حل المعادلة التفاضلية من الدرجة الثانية عددياً بطريقة RK4 (تحويلها لنظامين من الدرجة الأولى: x1 = x, x2 = dx/dt)
            n_steps = int(t_sim / dt_sim)
            t_arr = np.zeros(n_steps + 1)
            pos_arr = np.zeros(n_steps + 1)
            vel_arr = np.zeros(n_steps + 1)
            
            def sys_deriv(x1, x2):
                dx1 = x2
                dx2 = (f_step - c_val * x2 - k_val * x1) / m_val
                return dx1, dx2

            for i in range(n_steps):
                x1, x2 = pos_arr[i], vel_arr[i]
                k1_x, k1_v = sys_deriv(x1, x2)
                k2_x, k2_v = sys_deriv(x1 + 0.5*dt_sim*k1_x, x2 + 0.5*dt_sim*k1_v)
                k3_x, k3_v = sys_deriv(x1 + 0.5*dt_sim*k2_x, x2 + 0.5*dt_sim*k2_v)
                k4_x, k4_v = sys_deriv(x1 + dt_sim*k3_x, x2 + dt_sim*k3_v)
                
                pos_arr[i+1] = x1 + (dt_sim / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
                vel_arr[i+1] = x2 + (dt_sim / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
                t_arr[i+1] = (i + 1) * dt_sim

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("التردد الطبيعي (ωn)", f"{wn:.3f} rad/s")
            mc2.metric("معامل الإخماد (ζ)", f"{zeta:.3f}")
            mc3.metric("أقصى تجاوز (Overshoot %)", f"{overshoot_pct:.2f}%")
            mc4.metric("زمن الاستقرار (Settling Time)", f"{settling_time:.2f} s" if settling_time < 1e4 else "∞")

            st.info(f"📌 **حالة النظام:** {damping_type} | **القيمة المستقرة النهائية ($x_{{ss}}$):** `{steady_state_val:.{dp}f}`")

            fig_ctrl = go.Figure()
            fig_ctrl.add_trace(go.Scatter(x=t_arr, y=pos_arr, mode='lines', name='الإزاحة / الجهد x(t)', line=dict(color='#007bff', width=3)))
            fig_ctrl.add_trace(go.Scatter(x=t_arr, y=vel_arr, mode='lines', name='السرعة / التيار v(t)', line=dict(color='#10b981', width=2, dash='dot')))
            fig_ctrl.add_hline(y=steady_state_val, line_dash="dash", line_color="#dc3545", annotation_text=f"Steady State = {steady_state_val:.2f}")
            fig_ctrl.update_layout(title="منحنى استجابة النظام للخطوة (RK4 Step Response)", xaxis_title="الزمن (ثانية)", yaxis_title="الاستجابة", hovermode="x unified")
            st.plotly_chart(fig_ctrl, use_container_width=True)

            if enable_ai_tutor:
                with st.expander("💻 توليد كود C++ (Arduino) وكود MATLAB لهذا النظام"):
                    c_code, m_code = generate_code_snippets(f"({f_step} - {c_val}*v - {k_val}*x)/{m_val}", "Mechatronics 2nd-Order System")
                    st.code(c_code, language="cpp")
                    st.code(m_code, language="matlab")

            st.session_state.history.append(f"⚙️ **ميكاترونكس:** ζ={zeta:.2f}, OS={overshoot_pct:.1f}%")

        # =========================================================================
        # 1. قسم تقييم المشاريع والفوائد (NPV / IRR / Payback)
        # =========================================================================
        elif "NPV" in app_mode:
            st.markdown("### 📚 القوانين الرياضية (NPV, IRR & Payback Period):")
            st.latex(r"NPV = -I_0 + \sum_{t=1}^{n} \frac{F_t}{(1 + k)^t}, \quad IRR \implies NPV(IRR) = 0")
            
            k_disc = (rate_return + inflation_rate) / 100.0
            st.info(f"📌 **معدل الخصم الإجمالي (Discount Rate $k$):** `{rate_return}% + {inflation_rate}% = {(k_disc*100):.2f}%`  ⬅️  $(1 + k) = {1 + k_disc:.4f}$")
            
            npv_results = []
            yearly_discounted_dict = {"السنة (Year)": [f"Year {t}" for t in range(num_years + 1)]}
            
            for col_name in edited_npv_df.columns:
                col_vals = [safe_float(v) for v in edited_npv_df[col_name].values]
                inv_0 = abs(col_vals[0])
                revs = col_vals[1:]
                
                disc_flows = [-inv_0]
                latex_terms = [f"-{inv_0:,.2f}"]
                
                for t, rev in enumerate(revs, start=1):
                    pv = rev / ((1.0 + k_disc) ** t)
                    disc_flows.append(pv)
                    if t <= 6 or show_tutorial:
                        latex_terms.append(f"{rev:,.0f}\\left(\\frac{{1}}{{(1+{k_disc:.4f})^{t}}}\\right)")
                
                total_npv = sum(disc_flows)
                roi_ratio = (total_npv / inv_0) * 100.0 if inv_0 != 0 else 0.0
                
                # 1. حساب فترة الاسترداد (Payback Period) بالسنوات والشهور
                cum_cash = -inv_0
                payback_yr = None
                for t_idx, rev in enumerate(revs, start=1):
                    prev_cum = cum_cash
                    cum_cash += rev
                    if cum_cash >= 0 and payback_yr is None:
                        frac_yr = abs(prev_cum) / rev if rev > 0 else 0.0
                        payback_yr = (t_idx - 1) + frac_yr

                # 2. حساب معدل العائد الداخلي (IRR) عددياً باستخدام طريقة القاطع / التنصيف
                irr_val = None
                if sum(revs) > inv_0 and inv_0 > 0:
                    r_low, r_high = -0.5, 5.0
                    for _ in range(80):
                        r_mid = (r_low + r_high) / 2.0
                        npv_mid = -inv_0 + sum(rev / ((1.0 + r_mid) ** t) for t, rev in enumerate(revs, start=1))
                        if abs(npv_mid) < 1e-4:
                            break
                        if npv_mid > 0:
                            r_low = r_mid
                        else:
                            r_high = r_mid
                    irr_val = r_mid * 100.0

                yearly_discounted_dict[f"{col_name} (PV)"] = [f"${v:,.{dp}f}" for v in disc_flows]
                
                npv_results.append({
                    "Project": col_name,
                    "Investment": inv_0,
                    "Total Revenue": sum(revs),
                    "NPV": total_npv,
                    "IRR": f"{irr_val:.2f}%" if irr_val is not None else "لا يسترد التكلفة",
                    "Payback": f"{payback_yr:.2f} سنة" if payback_yr is not None else "> مدة المشروع",
                    "ROI (%)": roi_ratio,
                    "Status": "✅ مقبول (Fund)" if total_npv > 0 else "❌ مرفوض (Reject)",
                    "Latex": " + ".join(latex_terms) + f" = \\mathbf{{\\${total_npv:,.{dp}f}}}"
                })
            
            npv_results_sorted = sorted(npv_results, key=lambda item: item["NPV"], reverse=True)
            
            if show_tutorial:
                st.markdown("### 📝 خطوات التعويض الرياضي لكل مشروع (كما في المحاضرة):")
                for item in npv_results:
                    st.markdown(f"**🔹 مشروع {item['Project']}:**")
                    st.latex(f"NPV = {item['Latex']}")
            
            st.markdown("### 🏆 ترتيب الأولوية والقرار الاستثماري (Priority, NPV, IRR & Payback):")
            cols_cards = st.columns(min(3, len(npv_results_sorted)))
            for idx, item in enumerate(npv_results_sorted):
                card_cls = "exact-card" if item["NPV"] > 0 else "reject-card"
                val_cls = "exact-value" if item["NPV"] > 0 else "reject-value"
                priority_badge = f"🥇 الأولوية رقم {idx+1}" if item["NPV"] > 0 else "⛔ لا يتم تمويله (NPV < 0)"
                with cols_cards[idx % len(cols_cards)]:
                    st.markdown(f"""
                    <div class='result-card {card_cls}'>
                        <div class='card-title'>{priority_badge} | {item['Project']}</div>
                        <div class='card-value {val_cls}'>${item['NPV']:,.2f}</div>
                        <div class='card-subtext'>IRR: <b>{item['IRR']}</b> | الاسترداد: <b>{item['Payback']}</b><br>القرار: {item['Status']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            summary_df = pd.DataFrame([
                {
                    "الترتيب (Priority)": f"#{i+1}" if r["NPV"] > 0 else "مرفوض",
                    "اسم المشروع": r["Project"],
                    "الدفع المبدئي (I0)": f"${r['Investment']:,.2f}",
                    "صافي القيمة الحالية (NPV)": f"${r['NPV']:,.{dp}f}",
                    "معدل العائد الداخلي (IRR)": r["IRR"],
                    "فترة الاسترداد (Payback)": r["Payback"],
                    "القرار (Decision)": r["Status"]
                }
                for i, r in enumerate(npv_results_sorted)
            ])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            if enable_ai_tutor:
                best_p = npv_results_sorted[0]
                st.markdown(f"""
                <div class='result-card ai-card'>
                    <div class='card-title'>🤖 تحليل المُفسّر الهندسي الذكي (AI Financial Advisor):</div>
                    يتصدر المشروع <b>{best_p['Project']}</b> المركز الأول بصافي قيمة حالية <b>${best_p['NPV']:,.2f}</b> ومعدل عائد داخلي <b>{best_p['IRR']}</b> (وهو أعلى من معدل الخصم المطلوب {(k_disc*100):.1f}%)، كما يسترد رأس ماله خلال <b>{best_p['Payback']}</b>.
                </div>
                """, unsafe_allow_html=True)

            with st.expander("📅 عرض جدول القيم الحالية التفصيلي لكل سنة (Present Value per Year)"):
                st.dataframe(pd.DataFrame(yearly_discounted_dict), use_container_width=True, hide_index=True)

            csv_npv = summary_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 تحميل تقرير تقييم المشاريع (CSV)", data=csv_npv, file_name='npv_projects_report.csv', mime='text/csv', use_container_width=True)

            fig_npv = go.Figure()
            colors = ['#28a745' if r['NPV'] > 0 else '#dc3545' for r in npv_results_sorted]
            fig_npv.add_trace(go.Bar(
                x=[r['Project'] for r in npv_results_sorted],
                y=[r['NPV'] for r in npv_results_sorted],
                marker_color=colors,
                text=[f"${r['NPV']:,.2f}" for r in npv_results_sorted],
                textposition='auto'
            ))
            fig_npv.add_hline(y=0, line_dash="dash", line_color="black")
            fig_npv.update_layout(title="مقارنة صافي القيمة الحالية للمشاريع (NPV Comparison)", xaxis_title="المشروع", yaxis_title="NPV ($)")
            st.plotly_chart(fig_npv, use_container_width=True)
            
            st.session_state.history.append(f"💰 **تقييم مشاريع (NPV):** الأفضل `{npv_results_sorted[0]['Project']}` (${npv_results_sorted[0]['NPV']:,.2f})")

        # =========================================================================
        # 2. قسم شبكات الأعمال والمسار الحرج (AON / AOA / Lags / Gantt / PERT / Crashing)
        # =========================================================================
        elif "AON" in app_mode:
            try:
                raw_names = [str(x).strip() for x in edited_net_df.iloc[0].values]
                raw_rels = [str(x).strip() for x in edited_net_df.iloc[1].values]
                raw_times = [safe_float(x) for x in edited_net_df.iloc[2].values]
            except Exception as e:
                st.error(f"❌ خطأ في قراءة جدول الأنشطة. تأكد من أن الصف الثالث يحتوي على أرقام صحيحة للزمن. ({e})")
                st.stop()

            nodes = []
            duration = {}
            pert_var = {}
            for nm, tm in zip(raw_names, raw_times):
                if nm and nm not in nodes:
                    nodes.append(nm)
                    duration[nm] = tm
                    pert_var[nm] = 0.0

            # إذا كان PERT مفعلاً، نحسب الزمن المتوقع te والتباين Variance لكل نشاط
            if enable_pert and edited_pert_df is not None:
                for _, r_pert in edited_pert_df.iterrows():
                    act_p = str(r_pert["Activity"]).strip()
                    if act_p in nodes:
                        to_v = safe_float(r_pert["to (متفائل)"])
                        tm_v = safe_float(r_pert["tm (الأكثر احتمالاً)"])
                        tp_v = safe_float(r_pert["tp (متشائم)"])
                        duration[act_p] = (to_v + 4.0 * tm_v + tp_v) / 6.0
                        pert_var[act_p] = ((tp_v - to_v) / 6.0) ** 2

            edges = []
            is_pred_mode = ("Predecessors" in rel_mode) and (preset_net != "مثال المحاضرة: 20 نشاط (A إلى T) - Successors")
            if preset_net == "مثال المحاضرة: 18 نشاط (A إلى R) - Predecessors" or "6.20" in preset_net:
                is_pred_mode = True

            for idx, nm in enumerate(raw_names):
                rel_str = raw_rels[idx]
                if not rel_str or rel_str in ["-", "None", "none", "nan", ""]:
                    continue
                parts = [p.strip() for p in rel_str.replace(';', ',').split(',') if p.strip() and p.strip() != '-']
                for p in parts:
                    if p not in nodes:
                        st.error(f"❌ النشاط `{p}` المذكور في العلاقات غير موجود في صف أسماء الأنشطة!")
                        st.stop()
                    if is_pred_mode:
                        edges.append({"u": p, "v": nm, "type": "FS", "lag": 0.0})
                    else:
                        edges.append({"u": nm, "v": p, "type": "FS", "lag": 0.0})

            if enable_lags and edited_lags_df is not None:
                for _, row in edited_lags_df.iterrows():
                    u_lag = str(row.get("From (من)", "")).strip()
                    v_lag = str(row.get("To (إلى)", "")).strip()
                    t_lag = str(row.get("Type (النوع)", "FS")).strip().upper()
                    val_lag = safe_float(row.get("Lag (التأخير)", 0.0))
                    if u_lag in nodes and v_lag in nodes and t_lag in ["FS", "SS", "FF", "SF"]:
                        updated = False
                        for e in edges:
                            if e["u"] == u_lag and e["v"] == v_lag and e["type"] == "FS" and e["lag"] == 0.0:
                                e["type"] = t_lag
                                e["lag"] = val_lag
                                updated = True
                                break
                        if not updated:
                            edges.append({"u": u_lag, "v": v_lag, "type": t_lag, "lag": val_lag})

            adj = {u: [] for u in nodes}
            in_deg = {u: 0 for u in nodes}
            for e in edges:
                adj[e["u"]].append(e["v"])
                in_deg[e["v"]] += 1

            q = deque([u for u in nodes if in_deg[u] == 0])
            topo_order = []
            while q:
                curr = q.popleft()
                topo_order.append(curr)
                for nxt in adj[curr]:
                    in_deg[nxt] -= 1
                    if in_deg[nxt] == 0:
                        q.append(nxt)

            if len(topo_order) != len(nodes):
                st.error("❌ يوجد مسار دائري مغلق (Cycle) في العلاقات بين الأنشطة! يرجى مراجعة المدخلات.")
                st.stop()

            def solve_cpm_network(dur_dict):
                es_d = {u: 0.0 for u in nodes}
                ef_d = {u: dur_dict[u] for u in nodes}
                for _ in range(len(nodes) + 1):
                    for u in topo_order:
                        for e in edges:
                            if e["u"] == u:
                                v, l_type, lag = e["v"], e["type"], e["lag"]
                                if l_type == "FS": cand = ef_d[u] + lag
                                elif l_type == "SS": cand = es_d[u] + lag
                                elif l_type == "FF": cand = ef_d[u] + lag - dur_dict[v]
                                elif l_type == "SF": cand = es_d[u] + lag - dur_dict[v]
                                else: cand = ef_d[u]
                                if cand > es_d[v]:
                                    es_d[v] = cand
                                    ef_d[v] = es_d[v] + dur_dict[v]
                p_dur = max(ef_d.values()) if ef_d else 0.0
                lf_d = {u: p_dur for u in nodes}
                ls_d = {u: p_dur - dur_dict[u] for u in nodes}
                for _ in range(len(nodes) + 1):
                    for u in reversed(topo_order):
                        for e in edges:
                            if e["u"] == u:
                                v, l_type, lag = e["v"], e["type"], e["lag"]
                                if l_type == "FS": cand_lf = ls_d[v] - lag
                                elif l_type == "SS": cand_lf = ls_d[v] - lag + dur_dict[u]
                                elif l_type == "FF": cand_lf = lf_d[v] - lag
                                elif l_type == "SF": cand_lf = lf_d[v] - lag + dur_dict[u]
                                else: cand_lf = ls_d[v]
                                if cand_lf < lf_d[u]:
                                    lf_d[u] = cand_lf
                                    ls_d[u] = lf_d[u] - dur_dict[u]
                s_sl = {u: ls_d[u] - es_d[u] for u in nodes}
                f_sl = {u: lf_d[u] - ef_d[u] for u in nodes}
                c_list = [u for u in topo_order if abs(s_sl[u]) < 1e-6 or abs(f_sl[u]) < 1e-6]
                return es_d, ef_d, ls_d, lf_d, s_sl, f_sl, p_dur, c_list

            ES, EF, LS, LF, start_slack, finish_slack, project_duration, crit_nodes = solve_cpm_network(duration)
            free_slack = {}
            
            for u in nodes:
                out_edges = [e for e in edges if e["u"] == u]
                if not out_edges:
                    free_slack[u] = project_duration - EF[u]
                else:
                    min_excess = float('inf')
                    for e in out_edges:
                        v, l_type, lag = e["v"], e["type"], e["lag"]
                        if l_type == "FS": excess = ES[v] - (EF[u] + lag)
                        elif l_type == "SS": excess = ES[v] - (ES[u] + lag)
                        elif l_type == "FF": excess = EF[v] - (EF[u] + lag)
                        elif l_type == "SF": excess = EF[v] - (ES[u] + lag)
                        else: excess = ES[v] - EF[u]
                        if excess < min_excess:
                            min_excess = excess
                    free_slack[u] = max(0.0, min_excess)

            partial_crit = [u for u in topo_order if (abs(start_slack[u]) < 1e-6) != (abs(finish_slack[u]) < 1e-6)]
            crit_path_str = " ➔ ".join(crit_nodes)

            st.markdown("### 📚 القوانين الرياضية لحسابات الشبكة (CPM, PERT & Network Lags):")
            st.latex(r"EF = ES + t \quad | \quad LS = LF - t \quad | \quad TS = LS - ES \quad | \quad t_e = \frac{t_o + 4t_m + t_p}{6}")

            c_res1, c_res2 = st.columns(2)
            with c_res1:
                st.markdown(f"<div class='result-card exact-card'><div class='card-title'>⏱️ إجمالي زمن المشروع (Total Project Time)</div><div class='card-value exact-value'>{project_duration:g} يوم / وحدة زمن</div></div>", unsafe_allow_html=True)
            with c_res2:
                st.markdown(f"<div class='result-card'><div class='card-title'>🔥 المسار الحرج (Critical Path)</div><div class='card-value' style='font-size:22px;'>{crit_path_str}</div><div class='card-subtext'>الأنشطة الحرجة باللون الأحمر (Slack = 0)</div></div>", unsafe_allow_html=True)

            if partial_crit:
                st.warning(f"📌 **ملاحظة خاصة بشبكات الـ Lag (سؤال Example 6.20):** الأنشطة التي تقع بدايتها فقط أو نهايتها فقط على المسار الحرج هي: **{', '.join(partial_crit)}**")

            # عرض نتائج PERT الاحتمالية إذا كانت مفعلة
            if enable_pert:
                crit_variance = sum(pert_var[u] for u in crit_nodes)
                crit_std = np.sqrt(crit_variance)
                z_score = (target_deadline - project_duration) / crit_std if crit_std > 0 else 0.0
                prob_pct = norm.cdf(z_score) * 100.0
                st.markdown(f"""
                <div class='result-card ai-card'>
                    <div class='card-title'>🎲 نتائج تحليل بيرت الاحتمالي (PERT Probability Analysis):</div>
                    انحراف المسار الحرج ($\sigma$): <b>{crit_std:.3f} يوم</b> | قيمة $Z = {z_score:.3f}$<br>
                    🎯 <b>احتمالية انتهاء المشروع في أو قبل {target_deadline} يوم هي: <span style='color:#28a745; font-size:22px;'>{prob_pct:.2f}%</span></b>
                </div>
                """, unsafe_allow_html=True)

            # عرض نتائج ضغط المشروع (Project Crashing) إذا كانت مفعلة
            if enable_crashing and edited_crash_df is not None:
                curr_dur_map = duration.copy()
                max_c_map = {str(r["Activity"]).strip(): int(safe_float(r["Max Crash Days (أقصى أيام ضغط)"])) for _, r in edited_crash_df.iterrows()}
                cost_c_map = {str(r["Activity"]).strip(): safe_float(r["Crash Cost / Day (تكلفة اليوم الإضافي $)"]) for _, r in edited_crash_df.iterrows()}
                crashed_so_far = {u: 0 for u in nodes}
                crash_logs = []
                total_extra_cost = 0.0

                for step_c in range(int(crash_days_wanted)):
                    _, _, _, _, _, _, cur_p_dur, cur_crit = solve_cpm_network(curr_dur_map)
                    candidates = [u for u in cur_crit if crashed_so_far.get(u, 0) < max_c_map.get(u, 0) and curr_dur_map[u] > 1.0]
                    if not candidates:
                        break
                    best_act = min(candidates, key=lambda u: cost_c_map.get(u, float('inf')))
                    curr_dur_map[best_act] -= 1.0
                    crashed_so_far[best_act] += 1
                    total_extra_cost += cost_c_map[best_act]
                    _, _, _, _, _, _, new_p_dur, new_crit = solve_cpm_network(curr_dur_map)
                    crash_logs.append({
                        "خطوة الضغط": step_c + 1,
                        "النشاط المضغوط": best_act,
                        "تكلفة الضغط ($)": f"${cost_c_map[best_act]:,.2f}",
                        "الزمن الجديد للمشروع": f"{new_p_dur:g} يوم",
                        "المسار الحرج بعد الضغط": " ➔ ".join(new_crit)
                    })
                st.markdown(f"### ⚡ جدول خطوات ضغط المشروع (Project Crashing - إجمالي التكلفة الإضافية: `${total_extra_cost:,.2f}`):")
                if crash_logs:
                    st.dataframe(pd.DataFrame(crash_logs), use_container_width=True, hide_index=True)
                else:
                    st.info("لا يمكن ضغط المسار الحرج أكثر من ذلك.")

            # بناء عقد وأسهم AOA في حال اختيار AOA
            preds_map = {u: sorted([e["u"] for e in edges if e["v"] == u]) for u in nodes}
            aoa_act_edges = {}
            aoa_dummies = []
            aoa_events = {}
            
            event_counter = 1
            start_evt = 1
            aoa_events[start_evt] = {"E": 0.0, "L": 0.0}
            pred_set_to_event = {(): start_evt}
            act_end_event = {}

            for u in topo_order:
                p_tuple = tuple(preds_map[u])
                if not p_tuple:
                    u_start_evt = start_evt
                elif len(p_tuple) == 1:
                    u_start_evt = act_end_event[p_tuple[0]]
                else:
                    if p_tuple in pred_set_to_event:
                        u_start_evt = pred_set_to_event[p_tuple]
                    else:
                        event_counter += 1
                        u_start_evt = event_counter
                        pred_set_to_event[p_tuple] = u_start_evt
                        for p_act in p_tuple:
                            aoa_dummies.append((act_end_event[p_act], u_start_evt, p_act))
                
                event_counter += 1
                u_end_evt = event_counter
                act_end_event[u] = u_end_evt
                aoa_act_edges[u] = (u_start_evt, u_end_evt)
                aoa_events[u_start_evt] = {"E": ES[u], "L": LS[u]}
                aoa_events[u_end_evt] = {"E": EF[u], "L": LF[u]}

            sched_rows = []
            for idx, u in enumerate(nodes):
                is_c = (abs(start_slack[u]) < 1e-6 and abs(finish_slack[u]) < 1e-6)
                is_p = (u in partial_crit)
                row_dict = {
                    "النشاط (Activity)": u,
                    "الزمن (Duration)": f"{duration[u]:g}",
                    "بداية مبكرة (ES)": f"{ES[u]:g}",
                    "نهاية مبكرة (EF)": f"{EF[u]:g}",
                    "بداية متأخرة (LS)": f"{LS[u]:g}",
                    "نهاية متأخرة (LF)": f"{LF[u]:g}",
                    "سماح البداية (Start SL)": f"{start_slack[u]:g}",
                    "سماح النهاية (Finish SL)": f"{finish_slack[u]:g}",
                    "السماح الحر (Free Float)": f"{free_slack[u]:g}",
                    "الحالة (Critical?)": "🔥 حرج كلياً" if is_c else ("⚡ حرج جزئياً" if is_p else "عادي")
                }
                if "AOA" in net_type:
                    ev_i, ev_j = aoa_act_edges[u]
                    row_dict["سهم الحدث (AOA Event i ➔ j)"] = f"({ev_i}) ➔ ({ev_j})"
                sched_rows.append(row_dict)

            df_sched = pd.DataFrame(sched_rows)
            st.markdown("### 📊 الجدول التفصيلي لجدولة المشروع (Detailed Project Schedule):")
            st.dataframe(df_sched, use_container_width=True, hide_index=True)

            csv_net = df_sched.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 تحميل جدول المسار الحرج كملف إكسيل (CSV)", data=csv_net, file_name='cpm_network_schedule.csv', mime='text/csv', use_container_width=True)

            # =====================================================================
            # الرسم الهندسي المطابق لصور المحاضرات بدون كلمات زائدة (أرقام صافية 100%)
            # =====================================================================
            st.markdown(f"### 🎨 المخطط الشبكي الهندسي ({net_type}):")
            
            level = {u: 0 for u in nodes}
            for u in topo_order:
                for e in edges:
                    if e["u"] == u:
                        level[e["v"]] = max(level[e["v"]], level[u] + 1)

            levels_map = {}
            for u, lv in level.items():
                levels_map.setdefault(lv, []).append(u)

            max_lv = max(levels_map.keys()) if levels_map else 0
            max_in_col = max(len(ulist) for ulist in levels_map.values()) if levels_map else 1

            fig_net = go.Figure()

            if "AOA" in net_type:
                st.caption("🔵 **مفتاح دوائر AOA (كما في المحاضرة):** النصف العلوي للدائرة = `رقم الحدث` | الربع السفلي الأيسر = `ES` | الربع السفلي الأيمن = `LF` | فوق السهم = `النشاط والزمن` | المربع الصغير تحت السهم = `Slack`.")
                
                evt_level = {1: 0}
                for u in topo_order:
                    ev_i, ev_j = aoa_act_edges[u]
                    evt_level[ev_i] = max(evt_level.get(ev_i, 0), level[u])
                    evt_level[ev_j] = max(evt_level.get(ev_j, 0), level[u] + 1)
                for d_i, d_j, _ in aoa_dummies:
                    evt_level[d_j] = max(evt_level.get(d_j, 0), evt_level.get(d_i, 0) + 1)

                evt_groups = {}
                for ev_id, lv in evt_level.items():
                    evt_groups.setdefault(lv, []).append(ev_id)

                evt_pos = {}
                dx_aoa, dy_aoa = 5.0, 3.6
                r_c = 0.72
                
                for lv, ev_list in evt_groups.items():
                    k = len(ev_list)
                    for idx_e, ev_id in enumerate(ev_list):
                        evt_pos[ev_id] = (lv * dx_aoa, ((k - 1) / 2.0 - idx_e) * dy_aoa)

                for u in nodes:
                    ev_i, ev_j = aoa_act_edges[u]
                    x0_p, y0_p = evt_pos[ev_i]
                    x1_p, y1_p = evt_pos[ev_j]
                    is_c = (u in crit_nodes)
                    col_arrow = '#d90429' if is_c else '#1d3557'
                    
                    dx_vec, dy_vec = x1_p - x0_p, y1_p - y0_p
                    dist = np.hypot(dx_vec, dy_vec)
                    if dist > 2 * r_c:
                        sx = x0_p + (dx_vec / dist) * r_c
                        sy = y0_p + (dy_vec / dist) * r_c
                        ex = x1_p - (dx_vec / dist) * r_c
                        ey = y1_p - (dy_vec / dist) * r_c
                    else:
                        sx, sy, ex, ey = x0_p, y0_p, x1_p, y1_p
                    
                    fig_net.add_annotation(
                        x=ex, y=ey, ax=sx, ay=sy,
                        xref='x', yref='y', axref='x', ayref='y',
                        text="", showarrow=True,
                        arrowhead=2, arrowsize=1.3, arrowwidth=3.0 if is_c else 2.0,
                        arrowcolor=col_arrow
                    )
                    
                    mx, my = (sx + ex) / 2.0, (sy + ey) / 2.0
                    fig_net.add_annotation(
                        x=mx, y=my + 0.30,
                        text=f"<b>{u}  {duration[u]:g}</b>",
                        showarrow=False,
                        font=dict(size=13, color='#000000', family='Arial'),
                        bgcolor='rgba(255,255,255,0.85)'
                    )
                    fig_net.add_annotation(
                        x=mx, y=my - 0.32,
                        text=f"<b>{start_slack[u]:g}</b>",
                        showarrow=False,
                        font=dict(size=11, color='#000000', family='Arial'),
                        bgcolor='#ffccd5' if is_c else '#ffffff',
                        bordercolor=col_arrow, borderwidth=1.5, borderpad=2
                    )

                for d_i, d_j, p_act in aoa_dummies:
                    x0_p, y0_p = evt_pos[d_i]
                    x1_p, y1_p = evt_pos[d_j]
                    fig_net.add_trace(go.Scatter(
                        x=[x0_p, x1_p], y=[y0_p, y1_p],
                        mode='lines',
                        line=dict(color='#6c757d', width=1.8, dash='dot'),
                        hoverinfo='none', showlegend=False
                    ))

                for e_id, (cx, cy) in evt_pos.items():
                    fig_net.add_shape(
                        type="circle",
                        x0=cx - r_c, y0=cy - r_c,
                        x1=cx + r_c, y1=cy + r_c,
                        line=dict(color='#1d3557', width=2.5),
                        fillcolor='#ffffff',
                        layer="below"
                    )
                    fig_net.add_shape(
                        type="line",
                        x0=cx - r_c, y0=cy,
                        x1=cx + r_c, y1=cy,
                        line=dict(color='#1d3557', width=1.8),
                        layer="below"
                    )
                    fig_net.add_shape(
                        type="line",
                        x0=cx, y0=cy - r_c,
                        x1=cx, y1=cy,
                        line=dict(color='#1d3557', width=1.8),
                        layer="below"
                    )
                    fig_net.add_annotation(
                        x=cx, y=cy + r_c * 0.45,
                        text=f"<b>{e_id}</b>", showarrow=False,
                        font=dict(size=13, color='#000000', family='Arial')
                    )
                    fig_net.add_annotation(
                        x=cx - r_c * 0.42, y=cy - r_c * 0.42,
                        text=f"<b>{aoa_events[e_id]['E']:g}</b>", showarrow=False,
                        font=dict(size=11, color='#000000', family='Arial')
                    )
                    fig_net.add_annotation(
                        x=cx + r_c * 0.42, y=cy - r_c * 0.42,
                        text=f"<b>{aoa_events[e_id]['L']:g}</b>", showarrow=False,
                        font=dict(size=11, color='#000000', family='Arial')
                    )

                max_aoa_lv = max(evt_groups.keys()) if evt_groups else 1
                fig_net.update_layout(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.8, max_aoa_lv * dx_aoa + 1.8]),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    height=max(540, max_in_col * 155),
                    margin=dict(l=15, r=15, t=20, b=20)
                )

            else:
                st.caption("🟥 **مفتاح مستطيل الـ 9 خانات (3×3 كما في المحاضرة):** الصف العلوي `(ES | النشاط | EF)` — الصف الأوسط `(SL | مربع وردي | SL)` — الصف السفلي `(LS | الزمن DUR | LF)`.")
                
                dx_aon, dy_aon = 4.4, 2.7
                box_w, box_h = 1.25, 0.85
                
                pos = {}
                for lv, u_list in levels_map.items():
                    k = len(u_list)
                    for idx_u, u in enumerate(u_list):
                        pos[u] = (lv * dx_aon, ((k - 1) / 2.0 - idx_u) * dy_aon)

                for e in edges:
                    u, v = e["u"], e["v"]
                    x0_p, y0_p = pos[u]
                    x1_p, y1_p = pos[v]
                    is_crit_edge = (u in crit_nodes and v in crit_nodes)
                    line_col = '#d90429' if is_crit_edge else '#343a40'
                    
                    start_x = x0_p + box_w
                    end_x = x1_p - box_w
                    
                    fig_net.add_annotation(
                        x=end_x, y=y1_p, ax=start_x, ay=y0_p,
                        xref='x', yref='y', axref='x', ayref='y',
                        text="", showarrow=True,
                        arrowhead=2, arrowsize=1.25, arrowwidth=2.8 if is_crit_edge else 1.8,
                        arrowcolor=line_col
                    )
                    
                    if e["type"] != "FS" or e["lag"] != 0:
                        mx, my = (start_x + end_x) / 2.0, (y0_p + y1_p) / 2.0
                        lag_txt = f"LAG {e['lag']:g}" if e["type"] == "FS" else f"{e['type']} {e['lag']:g}"
                        fig_net.add_annotation(
                            x=mx, y=my + 0.22,
                            text=f"<b>{lag_txt}</b>",
                            showarrow=False,
                            font=dict(size=11, color='#000000', family='Arial'),
                            bgcolor='#ffe5ec', bordercolor='#d90429', borderwidth=1.2, borderpad=2
                        )

                for u in nodes:
                    cx, cy = pos[u]
                    is_c = (u in crit_nodes)
                    fill_col = '#fff0f3' if is_c else '#f8f9fa'
                    border_col = '#d90429' if is_c else '#1d3557'
                    center_cell_col = '#ff007f' if is_c else '#ff758f'
                    
                    fig_net.add_shape(
                        type="rect",
                        x0=cx - box_w, y0=cy - box_h,
                        x1=cx + box_w, y1=cy + box_h,
                        line=dict(color=border_col, width=2.8 if is_c else 2.0),
                        fillcolor=fill_col,
                        layer="below"
                    )
                    fig_net.add_shape(
                        type="rect",
                        x0=cx - box_w/3.0, y0=cy - box_h/3.0,
                        x1=cx + box_w/3.0, y1=cy + box_h/3.0,
                        line=dict(color=border_col, width=1.0),
                        fillcolor=center_cell_col,
                        layer="below"
                    )
                    for r_frac in [-1.0/3.0, 1.0/3.0]:
                        fig_net.add_shape(
                            type="line",
                            x0=cx - box_w, y0=cy + box_h * r_frac,
                            x1=cx + box_w, y1=cy + box_h * r_frac,
                            line=dict(color=border_col, width=1.2),
                            layer="below"
                        )
                    for c_frac in [-1.0/3.0, 1.0/3.0]:
                        fig_net.add_shape(
                            type="line",
                            x0=cx + box_w * c_frac, y0=cy - box_h,
                            x1=cx + box_w * c_frac, y1=cy + box_h,
                            line=dict(color=border_col, width=1.2),
                            layer="below"
                        )

                    cells_coords = [
                        (cx - box_w*0.67, cy + box_h*0.67, f"{ES[u]:g}"),
                        (cx,              cy + box_h*0.67, f"{u}"),
                        (cx + box_w*0.67, cy + box_h*0.67, f"{EF[u]:g}"),
                        (cx - box_w*0.67, cy,              f"{start_slack[u]:g}"),
                        (cx + box_w*0.67, cy,              f"{finish_slack[u]:g}"),
                        (cx - box_w*0.67, cy - box_h*0.67, f"{LS[u]:g}"),
                        (cx,              cy - box_h*0.67, f"{duration[u]:g}"),
                        (cx + box_w*0.67, cy - box_h*0.67, f"{LF[u]:g}"),
                    ]
                    for tx_c, ty_c, val_str in cells_coords:
                        fig_net.add_annotation(
                            x=tx_c, y=ty_c,
                            text=f"<b>{val_str}</b>",
                            showarrow=False,
                            font=dict(size=12, color='#000000', family='Arial'),
                            align="center"
                        )

                fig_net.add_trace(go.Scatter(
                    x=[pos[u][0] for u in nodes],
                    y=[pos[u][1] for u in nodes],
                    mode='markers',
                    marker=dict(size=10, opacity=0),
                    text=[
                        f"Activity {u} | Dur={duration[u]:g} | ES={ES[u]:g}, EF={EF[u]:g} | LS={LS[u]:g}, LF={LF[u]:g} | SL={start_slack[u]:g}"
                        for u in nodes
                    ],
                    hoverinfo='text',
                    showlegend=False
                ))

                fig_net.update_layout(
                    xaxis=dict(
                        showgrid=False, zeroline=False, showticklabels=False,
                        range=[-box_w - 0.7, max_lv * dx_aon + box_w + 0.7]
                    ),
                    yaxis=dict(
                        showgrid=False, zeroline=False, showticklabels=False,
                        range=[-(max_in_col * dy_aon)/2.0 - box_h - 0.6, (max_in_col * dy_aon)/2.0 + box_h + 0.6]
                    ),
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    height=max(540, max_in_col * 145),
                    margin=dict(l=15, r=15, t=20, b=20)
                )

            st.plotly_chart(fig_net, use_container_width=True)

            # رسم مخطط جانت التفاعلي (Gantt Chart) مع إظهار وقت السماح (Slack) كظل خلف كل نشاط
            if show_gantt:
                st.markdown("### 📅 مخطط جانت الزمني التفاعلي (Project Gantt Chart with Float/Slack):")
                fig_gantt = go.Figure()
                for u in reversed(nodes):
                    is_c = (u in crit_nodes)
                    fig_gantt.add_trace(go.Bar(
                        y=[f"Activity {u}"],
                        x=[duration[u]],
                        base=[ES[u]],
                        orientation='h',
                        name=f"{u} (Active)",
                        marker_color='#dc3545' if is_c else '#007bff',
                        text=[f"{u}: [{ES[u]:g} ➔ {EF[u]:g}]"],
                        textposition='inside',
                        showlegend=False
                    ))
                    if finish_slack[u] > 0:
                        fig_gantt.add_trace(go.Bar(
                            y=[f"Activity {u}"],
                            x=[finish_slack[u]],
                            base=[EF[u]],
                            orientation='h',
                            name=f"{u} (Slack)",
                            marker_color='rgba(108, 117, 125, 0.35)',
                            text=[f"Slack: +{finish_slack[u]:g}"],
                            textposition='inside',
                            showlegend=False
                        ))
                fig_gantt.update_layout(
                    barmode='stack',
                    xaxis_title="الزمن (أيام / وحدات زمنية)",
                    yaxis_title="الأنشطة",
                    height=max(380, len(nodes) * 35),
                    template="plotly_white"
                )
                st.plotly_chart(fig_gantt, use_container_width=True)

            st.session_state.history.append(f"🕸️ **شبكة ({net_type.split('(')[0]}):** الزمن `{project_duration:g}` | المسار `{crit_path_str}`")

        # =========================================================================
        # 3. قسم الاستيفاء وتوفيق المنحنيات (Interpolation)
        # =========================================================================
        elif "Interpolation" in app_mode:
            try:
                x_vals = edited_pts["X"].values.astype(float)
                y_vals = edited_pts["Y"].values.astype(float)
            except Exception:
                st.error("❌ تأكد من إدخال أرقام صحيحة في جدول X و Y.")
                st.stop()
                
            if len(set(x_vals)) != len(x_vals):
                st.error("❌ قيم X يجب أن تكون فريدة (لا يتكرر نفس الرقم في عمود X).")
                st.stop()
                
            x_sym = sp.Symbol('x')
            n_pts = len(x_vals)
            poly = 0
            
            if "Lagrange" in method:
                st.markdown("### 📚 القانون الرياضي (Lagrange Interpolation):")
                st.latex(r"P(x) = \sum_{i=0}^{n} y_i \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}")
                
                for i in range(n_pts):
                    term = y_vals[i]
                    for j in range(n_pts):
                        if i != j:
                            term *= (x_sym - x_vals[j]) / (x_vals[i] - x_vals[j])
                    poly += term
                    
                if show_tutorial:
                    st.markdown(f"<div class='result-card tutorial-card'><b>💡 خطوات لاجرانج:</b> يتم تكوين {n_pts} حدود، كل حد يساوي صفر عند باقي النقاط ما عدا النقطة الخاصة به، ثم نجمعهم.</div>", unsafe_allow_html=True)
            
            else:
                st.markdown("### 📚 القانون الرياضي (Newton's Divided Differences):")
                st.latex(r"P(x) = f[x_0] + f[x_0, x_1](x-x_0) + f[x_0, x_1, x_2](x-x_0)(x-x_1) + \dots")
                
                coef = np.zeros([n_pts, n_pts])
                coef[:, 0] = y_vals
                for j in range(1, n_pts):
                    for i in range(n_pts - j):
                        coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x_vals[i+j] - x_vals[i])
                
                poly = coef[0, 0]
                for j in range(1, n_pts):
                    term = coef[0, j]
                    for i in range(j):
                        term *= (x_sym - x_vals[i])
                    poly += term
                
                if show_tutorial:
                    dd_cols = [f"الفرق {i}" for i in range(n_pts)]
                    st.markdown("#### 📊 جدول الفروق المقسمة:")
                    st.dataframe(pd.DataFrame(coef, columns=dd_cols).style.format(precision=dp), use_container_width=True)

            poly_simplified = sp.expand(poly)
            p_func = sp.lambdify(x_sym, poly_simplified, "numpy")
            
            st.markdown("### 📍 كثيرة الحدود المستنتجة $P(x)$:")
            st.success(f"**P(x) =** `{poly_simplified}`")
            
            target_val = None
            if target_x_str.strip():
                try:
                    tx = safe_float(target_x_str)
                    ty = float(p_func(tx))
                    target_val = (tx, ty)
                    st.markdown(f"<div class='result-card exact-card'><div class='card-title'>🎯 قيمة الدالة المستوفاة عند X = {tx}</div><div class='card-value exact-value'>P({tx}) = {ty:.{dp}f}</div></div>", unsafe_allow_html=True)
                except Exception:
                    st.warning("⚠️ لم نتمكن من حساب النقطة المستهدفة، تأكد من كتابتها كرقم صحيح.")

            x_min_plot = min(list(x_vals) + ([target_val[0]] if target_val else [])) - 1
            x_max_plot = max(list(x_vals) + ([target_val[0]] if target_val else [])) + 1
            x_curve = np.linspace(x_min_plot, x_max_plot, 400)
            y_curve = p_func(x_curve)
            if isinstance(y_curve, (int, float)):
                y_curve = np.full_like(x_curve, y_curve)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', name='منحنى الاستيفاء P(x)', line=dict(color='#2d82ff', width=3)))
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name='النقاط الأساسية', marker=dict(color='#ff4b4b', size=12)))
            if target_val:
                fig.add_trace(go.Scatter(x=[target_val[0]], y=[target_val[1]], mode='markers', name=f'النقطة المستهدفة ({target_val[0]})', marker=dict(color='#28a745', size=14, symbol='star')))
            fig.update_layout(title="التمثيل البياني للاستيفاء", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            st.session_state.history.append(f"📉 **استيفاء ({n_pts} نقاط):**\n`P(x) = {poly_simplified}`")

        # =========================================================================
        # 4. قسم أنظمة المعادلات الخطية (Linear Systems)
        # =========================================================================
        elif "Linear" in app_mode:
            try:
                A_input = edited_df.iloc[:, :-1].values.astype(float)
                b_input = edited_df.iloc[:, -1].values.astype(float)
                n = len(b_input)
                if "Elimination" not in method:
                    tol_val = safe_float(tol_str)
            except Exception as e:
                st.error(f"❌ تأكد من صحة الأرقام المدخلة في المصفوفة أو نسبة الخطأ. ({e})")
                st.stop()
                
            if "Elimination" in method:
                st.markdown("### 📚 طريقة الحذف لجاوس (Gaussian Elimination with Partial Pivoting)")
                Ab = np.column_stack((A_input.copy(), b_input.copy()))
                
                for i in range(n):
                    max_row = np.argmax(np.abs(Ab[i:n, i])) + i
                    Ab[[i, max_row]] = Ab[[max_row, i]]
                    
                    if abs(Ab[i, i]) < 1e-15:
                        st.error("❌ المصفوفة شاذة (Singular Matrix) ولا يوجد للنظام حل وحيد.")
                        st.stop()
                        
                    for j in range(i+1, n):
                        factor = Ab[j, i] / Ab[i, i]
                        Ab[j] = Ab[j] - factor * Ab[i]
                        
                x_res = np.zeros(n)
                for i in range(n-1, -1, -1):
                    x_res[i] = (Ab[i, -1] - np.sum(Ab[i, i+1:n] * x_res[i+1:n])) / Ab[i, i]
                    
                if show_tutorial:
                    st.markdown("#### 📐 المصفوفة المثلثية العليا (Upper Triangular Matrix) بعد الحذف:")
                    st.dataframe(pd.DataFrame(Ab, columns=cols).style.format(precision=dp), use_container_width=True)

                st.markdown("### 📍 قيم المتغيرات النهائية:")
                res_df = pd.DataFrame({
                    "المتغير": [f"x{i+1}" for i in range(n)], 
                    "القيمة النهائية": [f"{x_res[i]:.{dp}f}" for i in range(n)]
                })
                st.dataframe(res_df, use_container_width=True, hide_index=True)
                st.session_state.history.append(f"🧮 **حذف جاوس:** تم حل {n} معادلات بنجاح.")
                
            else:
                indices = []
                used_rows = set()
                for i in range(n):
                    max_row, max_val = -1, -1
                    for r in range(n):
                        if r not in used_rows and abs(A_input[r, i]) > max_val:
                            max_val = abs(A_input[r, i])
                            max_row = r
                    indices.append(max_row)
                    used_rows.add(max_row)

                A = A_input[indices]
                b = b_input[indices]
                
                if indices != list(range(n)): 
                    st.success("✨ تم إعادة ترتيب المعادلات أوتوماتيكياً لتصبح مسيطرة قطرياً (Diagonally Dominant) لضمان تقارب الحل!")
                    with st.expander("👀 عرض المصفوفة بعد الترتيب الأوتوماتيكي", expanded=False):
                        st.dataframe(pd.DataFrame(np.column_stack((A, b)), columns=cols), use_container_width=True)

                if np.any(np.abs(np.diag(A)) < 1e-15): 
                    st.error("❌ القطر الرئيسي يحتوي على صفر حتى بعد الترتيب. لا يمكن القسمة على صفر.")
                    st.stop()

                is_dd = all(abs(A[i, i]) >= (np.sum(np.abs(A[i])) - abs(A[i, i])) for i in range(n))
                if not is_dd:
                    st.warning("⚠️ تنبيه: المصفوفة ليست مسيطرة قطرياً بالكامل؛ قد تتباعد الأرقام (Diverge) ولكن سنحاول الحل.")

                st.markdown("### 📚 القانون الرياضي المستخدم:")
                if "Jacobi" in method: 
                    st.latex(r"x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j \neq i} a_{ij} x_j^{(k)} \right)")
                else: 
                    st.latex(r"x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)} \right)")

                x_arr = np.zeros(n)
                steps_data = []
                iterations = 0
                final_error = 0
                max_iter = 500
                
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
                            msg = f"<b>💡 شرح التعويض (المحاولة الأولى للمتغير x1):</b><br> x1 = ( {b[i]} - ({s:.{dp}f}) ) / {A[i,i]} = <b>{x_new[i]:.{dp}f}</b>"
                            st.markdown(f"<div class='result-card tutorial-card'>{msg}</div>", unsafe_allow_html=True)
                    
                    error = np.max(np.abs(x_new - x_arr))
                    
                    step_dict = {"Iteration": it + 1}
                    for i in range(n): 
                        step_dict[f"x{i+1}"] = f"{x_new[i]:.{dp}f}"
                    step_dict["Error"] = f"{error:.2e}"
                    steps_data.append(step_dict)
                    
                    iterations = it + 1
                    final_error = error
                    x_arr = x_new.copy()
                    
                    if error < tol_val: 
                        break
                    if error > 1e12: 
                        st.error("❌ النظام متباعد (Diverging)! الأرقام تتجه نحو المالانهاية.")
                        break

                st.markdown("### 📍 قيم المتغيرات النهائية:")
                res_df = pd.DataFrame({
                    "المتغير": [f"x{i+1}" for i in range(n)],
                    "القيمة النهائية": [f"{x_arr[i]:.{dp}f}" for i in range(n)]
                })
                st.dataframe(res_df, use_container_width=True, hide_index=True)
                
                if final_error < tol_val:
                    st.success(f"✅ تم الوصول للحل الدقيق بعد **{iterations} محاولة** | أقصى خطأ: **{final_error:.2e}**")

                st.markdown("### 📝 جدول خطوات الحل (Iterations Table)")
                df_steps = pd.DataFrame(steps_data)
                st.dataframe(df_steps, use_container_width=True, hide_index=True)
                
                csv = df_steps.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 تحميل جدول المحاولات كملف إكسيل (CSV)", data=csv, file_name='linear_system_iterations.csv', mime='text/csv', use_container_width=True)
                
                st.session_state.history.append(f"🧮 **{method.split(' ')[0]}:** {n} متغيرات ({iterations} خطوة)")

        # =========================================================================
        # 5. الأقسام المعتمدة على الدوال (الجذور، المعادلات التفاضلية، التفاضل، التكامل)
        # =========================================================================
        else:
            if not func_input.strip():
                st.warning("⚠️ يرجى كتابة الدالة الرياضية أولاً.")
                st.stop()
                
            func_clean = func_input.replace('^', '**').replace('ln', 'log').replace(' ', '')
            x_sym, y_sym = sp.Symbol('x'), sp.Symbol('y')
            custom_dict = {'e': sp.E, 'pi': sp.pi}
            
            if "Deg" in angle_mode:
                custom_dict.update({
                    'sin': lambda arg: sp.sin(arg * sp.pi / 180), 
                    'cos': lambda arg: sp.cos(arg * sp.pi / 180), 
                    'tan': lambda arg: sp.tan(arg * sp.pi / 180)
                })

            try:
                f_expr = sp.sympify(func_clean, locals=custom_dict)
                if "Differential" in app_mode:
                    f = sp.lambdify((x_sym, y_sym), f_expr, "numpy")
                else:
                    f = sp.lambdify(x_sym, f_expr, "numpy")
            except Exception as e:
                st.error(f"❌ خطأ في صيغة الدالة الرياضية. تأكد من كتابة علامة الضرب `*` والأقواس بشكل سليم. ({e})")
                st.stop()

            # ---------------------------------------------------------------------
            # أ. قسم حل المعادلات + مقارنة السرعة (Roots & Convergence Benchmark)
            # ---------------------------------------------------------------------
            if "Roots" in app_mode:
                try:
                    a_val = safe_float(a_str)
                    b_val = safe_float(b_str)
                    tol_val = safe_float(tol_str)
                except Exception as e:
                    st.error(f"❌ خطأ في الأرقام المدخلة لنقاط البداية أو نسبة الخطأ. ({e})")
                    st.stop()

                root, iterations, error = None, 0, 0
                steps_data = []
                df_expr = sp.diff(f_expr, x_sym)
                df_func = sp.lambdify(x_sym, df_expr, "numpy")

                st.markdown("### 📚 القانون الرياضي المستخدم:")
                if "Bisection" in method:
                    st.latex(r"c = \frac{a + b}{2}")
                elif "False Position" in method:
                    st.latex(r"c = \frac{a \cdot f(b) - b \cdot f(a)}{f(b) - f(a)}")
                elif "Newton" in method:
                    st.latex(r"x_{i+1} = x_i - \frac{f(x_i)}{f'(x_i)}")
                    st.info(f"📐 **المشتقة الأولى المحسوبة تلقائياً:** `f'(x) = {df_expr}`")
                elif "Secant" in method:
                    st.latex(r"x_{i+1} = x_i - f(x_i) \frac{x_i - x_{i-1}}{f(x_i) - f(x_{i-1})}")

                if method in ["التنصيف (Bisection)", "الوضع الخاطئ (False Position)"]:
                    if a_val >= b_val:
                        st.error("❌ يجب أن تكون النقطة (b) أكبر من النقطة (a).")
                        st.stop()
                    fa, fb = float(f(a_val)), float(f(b_val))
                    if fa * fb > 0:
                        st.error(f"❌ شرط الحل غير متحقق: f({a_val})={fa:.3f} و f({b_val})={fb:.3f} لهما نفس الإشارة!")
                        st.stop()
                    
                    a_temp, b_temp = a_val, b_val
                    for i in range(100):
                        fa_t, fb_t = float(f(a_temp)), float(f(b_temp))
                        if "Bisection" in method:
                            c = (a_temp + b_temp) / 2.0
                        else:
                            c = (a_temp * fb_t - b_temp * fa_t) / (fb_t - fa_t)
                            
                        fc = float(f(c))
                        current_err = abs(b_temp - a_temp) / 2.0 if "Bisection" in method else abs(fc)
                        
                        steps_data.append({
                            "Iteration (i)": i + 1,
                            "a": f"{a_temp:.{dp}f}",
                            "b": f"{b_temp:.{dp}f}",
                            "Root (c)": f"{c:.{dp}f}",
                            "f(c)": f"{fc:.{dp}f}",
                            "Error": f"{current_err:.2e}"
                        })
                        
                        if abs(fc) < tol_val or current_err < tol_val:
                            root, iterations, error = c, i + 1, current_err
                            break
                        if fa_t * fc < 0:
                            b_temp = c
                        else:
                            a_temp = c
                    if root is None:
                        root, iterations, error = c, 100, current_err

                elif "Newton" in method:
                    x_curr = a_val
                    for i in range(100):
                        fx, dfx = float(f(x_curr)), float(df_func(x_curr))
                        if abs(dfx) < 1e-15:
                            st.error("❌ المشتقة تساوي صفر عند هذه النقطة!")
                            st.stop()
                        
                        x_next = x_curr - (fx / dfx)
                        current_err = abs(x_next - x_curr)
                        
                        if show_tutorial and i == 0:
                            msg = f"<b>💡 تعويض المحاولة الأولى (i=1):</b><br> x₁ = {x_curr} - ( {fx:.{dp}f} / {dfx:.{dp}f} ) = <b>{x_next:.{dp}f}</b>"
                            st.markdown(f"<div class='result-card tutorial-card'>{msg}</div>", unsafe_allow_html=True)

                        steps_data.append({
                            "Iteration (i)": i + 1,
                            "X_i": f"{x_curr:.{dp}f}",
                            "f(X_i)": f"{fx:.{dp}f}",
                            "f'(X_i)": f"{dfx:.{dp}f}",
                            "X_{i+1}": f"{x_next:.{dp}f}",
                            "Error": f"{current_err:.2e}"
                        })
                        
                        if current_err < tol_val:
                            root, iterations, error = x_next, i + 1, current_err
                            break
                        x_curr = x_next
                    if root is None:
                        root, iterations, error = x_curr, 100, current_err

                elif "Secant" in method:
                    x0, x1 = a_val, b_val
                    for i in range(100):
                        f0, f1 = float(f(x0)), float(f(x1))
                        if abs(f1 - f0) < 1e-15:
                            st.error("❌ الفرق بين الدالتين يساوي صفر (قسمة على صفر).")
                            st.stop()
                        
                        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
                        current_err = abs(x2 - x1)
                        
                        steps_data.append({
                            "Iteration (i)": i + 1,
                            "X_{i-1}": f"{x0:.{dp}f}",
                            "X_i": f"{x1:.{dp}f}",
                            "f(X_i)": f"{f1:.{dp}f}",
                            "X_{i+1}": f"{x2:.{dp}f}",
                            "Error": f"{current_err:.2e}"
                        })
                        
                        if current_err < tol_val:
                            root, iterations, error = x2, i + 1, current_err
                            break
                        x0, x1 = x1, x2
                    if root is None:
                        root, iterations, error = x2, 100, current_err

                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.markdown(f"<div class='result-card exact-card'><div class='card-title'>📍 قيمة الجذر النهائي (Root)</div><div class='card-value exact-value'>{root:.{dp}f}</div></div>", unsafe_allow_html=True)
                with res_col2:
                    st.markdown(f"<div class='result-card'><div class='card-title'>⚙️ الأداء والدقة</div><div class='card-value' style='font-size:22px;'>الخطأ: {error:.2e}</div><div class='card-subtext'>تم الوصول للحل بعد {iterations} خطوة</div></div>", unsafe_allow_html=True)

                st.markdown("### 📝 جدول خطوات الحل (Iterations Table)")
                df_steps = pd.DataFrame(steps_data)
                st.dataframe(df_steps, use_container_width=True, hide_index=True)

                # مقارنة سرعة التقارب لجميع الطرق على نفس الرسمة (Benchmarking Mode)
                if run_benchmark:
                    st.markdown("### 🏎️ منحنى مقارنة سرعة تقارب الخوارزميات (Convergence Speed Benchmark):")
                    fig_bench = go.Figure()
                    # Newton Errors
                    xn_b = a_val
                    err_newton = []
                    for _ in range(25):
                        dfx_b = float(df_func(xn_b))
                        if abs(dfx_b) < 1e-14: break
                        xn_next = xn_b - float(f(xn_b))/dfx_b
                        err_newton.append(max(1e-16, abs(xn_next - xn_b)))
                        if abs(xn_next - xn_b) < tol_val: break
                        xn_b = xn_next
                    fig_bench.add_trace(go.Scatter(x=list(range(1, len(err_newton)+1)), y=err_newton, mode='lines+markers', name='Newton-Raphson'))
                    
                    # Secant Errors
                    xs0, xs1 = a_val, b_val
                    err_sec = []
                    for _ in range(25):
                        f0_b, f1_b = float(f(xs0)), float(f(xs1))
                        if abs(f1_b - f0_b) < 1e-14: break
                        xs2 = xs1 - f1_b*(xs1 - xs0)/(f1_b - f0_b)
                        err_sec.append(max(1e-16, abs(xs2 - xs1)))
                        if abs(xs2 - xs1) < tol_val: break
                        xs0, xs1 = xs1, xs2
                    fig_bench.add_trace(go.Scatter(x=list(range(1, len(err_sec)+1)), y=err_sec, mode='lines+markers', name='Secant Method'))

                    # Bisection Errors (if bracket valid)
                    if a_val < b_val and float(f(a_val)) * float(f(b_val)) < 0:
                        ab, bb = a_val, b_val
                        err_bis = []
                        for _ in range(25):
                            cb = (ab + bb)/2.0
                            err_bis.append(max(1e-16, abs(bb - ab)/2.0))
                            if abs(bb - ab)/2.0 < tol_val: break
                            if float(f(ab))*float(f(cb)) < 0: bb = cb
                            else: ab = cb
                        fig_bench.add_trace(go.Scatter(x=list(range(1, len(err_bis)+1)), y=err_bis, mode='lines+markers', name='Bisection Method'))

                    fig_bench.update_layout(yaxis_type="log", xaxis_title="رقم المحاولة (Iteration)", yaxis_title="الخطأ (Log Scale)", template="plotly_white")
                    st.plotly_chart(fig_bench, use_container_width=True)

                # أزرار تصدير الجدول والتقرير الهندسي للطباعة وكود C++/MATLAB
                c_dl1, c_dl2 = st.columns(2)
                csv = df_steps.to_csv(index=False).encode('utf-8-sig')
                c_dl1.download_button("📥 تحميل الجدول كملف إكسيل (CSV)", data=csv, file_name='root_iterations.csv', mime='text/csv', use_container_width=True)
                rep_html = build_printable_report(f"Root Finding ({method})", f"f(x) = {func_input}", f"Root = {root:.{dp}f} (Iterations: {iterations})", df_steps)
                c_dl2.download_button("📄 تحميل تقرير هندسي رسمي للطباعة (HTML/PDF)", data=rep_html.encode('utf-8'), file_name='engineering_report.html', mime='text/html', use_container_width=True)

                if enable_ai_tutor:
                    with st.expander("💻 توليد كود C++ (Arduino) وكود MATLAB لهذه الدالة"):
                        c_code, m_code = generate_code_snippets(func_clean, method)
                        st.code(c_code, language="cpp")
                        st.code(m_code, language="matlab")

                st.markdown("### 🎨 الرسم البياني التفاعلي للدالة والجذر")
                x_min, x_max = root - 2.5, root + 2.5
                x_smooth = np.linspace(x_min, x_max, 500)
                y_smooth = f(x_smooth)
                if isinstance(y_smooth, (int, float)):
                    y_smooth = np.full_like(x_smooth, y_smooth)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name='f(x)', line=dict(color='#2d82ff', width=3)))
                fig.add_trace(go.Scatter(x=[x_min, x_max], y=[0, 0], mode='lines', name='محور السينات (y=0)', line=dict(color='#ff4b4b', width=2, dash='dash')))
                fig.add_trace(go.Scatter(x=[root], y=[0], mode='markers', name=f'الجذر ({root:.4f})', marker=dict(color='#28a745', size=14, symbol='circle')))
                fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                st.session_state.history.append(f"🎯 **جذر ({method.split(' ')[0]}):**\n`{func_input}` ⬅️ `{root:.{dp}f}`")

            # ---------------------------------------------------------------------
            # ب. قسم المعادلات التفاضلية (Differential Equations - ODEs)
            # ---------------------------------------------------------------------
            elif "Differential" in app_mode:
                try:
                    x0 = safe_float(x0_str)
                    y0 = safe_float(y0_str)
                    x_end = safe_float(xend_str)
                except Exception as e:
                    st.error(f"❌ خطأ في الأرقام المدخلة (تأكد من كتابة القيم بشكل سليم). التفاصيل: {e}")
                    st.stop()

                if "h" in input_type:
                    h = float(val)
                    n = int(round(abs(x_end - x0) / h))
                else:
                    n = int(val)
                    h = abs(x_end - x0) / n

                if x_end < x0:
                    h = -h

                st.markdown("### 📚 القوانين الرياضية المستخدمة:")
                if "Euler" in method:
                    st.latex(r"y_{i+1} = y_i + h \cdot f(x_i, y_i)")
                elif "RK2" in method:
                    st.latex(r"k_1 = h f(x_i, y_i), \quad k_2 = h f(x_i + h, y_i + k_1), \quad y_{i+1} = y_i + \frac{1}{2}(k_1 + k_2)")
                else:
                    st.latex(r"\begin{aligned} k_1 &= h f(x_i, y_i) \\ k_2 &= h f\left(x_i + \frac{h}{2}, y_i + \frac{k_1}{2}\right) \\ k_3 &= h f\left(x_i + \frac{h}{2}, y_i + \frac{k_2}{2}\right) \\ k_4 &= h f(x_i + h, y_i + k_3) \\ y_{i+1} &= y_i + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4) \end{aligned}")

                x_vals, y_vals = [x0], [y0]
                steps_data = []

                for i in range(n):
                    xi, yi = x_vals[-1], y_vals[-1]
                    try:
                        if "Euler" in method:
                            slope = float(f(xi, yi))
                            y_next = yi + h * slope
                            steps_data.append({
                                "Step": i + 1,
                                "x_i": f"{xi:.{dp}f}",
                                "y_i": f"{yi:.{dp}f}",
                                "f(x_i, y_i)": f"{slope:.{dp}f}",
                                "y_{i+1}": f"{y_next:.{dp}f}"
                            })
                        elif "RK2" in method:
                            k1 = h * float(f(xi, yi))
                            k2 = h * float(f(xi + h, yi + k1))
                            y_next = yi + 0.5 * (k1 + k2)
                            steps_data.append({
                                "Step": i + 1,
                                "x_i": f"{xi:.{dp}f}",
                                "y_i": f"{yi:.{dp}f}",
                                "k1": f"{k1:.{dp}f}",
                                "k2": f"{k2:.{dp}f}",
                                "y_{i+1}": f"{y_next:.{dp}f}"
                            })
                        else:
                            k1 = h * float(f(xi, yi))
                            k2 = h * float(f(xi + h/2.0, yi + k1/2.0))
                            k3 = h * float(f(xi + h/2.0, yi + k2/2.0))
                            k4 = h * float(f(xi + h, yi + k3))
                            y_next = yi + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
                            
                            if show_tutorial and i == 0:
                                msg = f"<b>💡 حسابات الخطوة الأولى (RK4):</b><br> k₁ = {k1:.{dp}f} | k₂ = {k2:.{dp}f} | k₃ = {k3:.{dp}f} | k₄ = {k4:.{dp}f}<br> y₁ = {yi} + (k₁ + 2k₂ + 2k₃ + k₄)/6 = <b>{y_next:.{dp}f}</b>"
                                st.markdown(f"<div class='result-card tutorial-card'>{msg}</div>", unsafe_allow_html=True)

                            steps_data.append({
                                "Step": i + 1,
                                "x_i": f"{xi:.{dp}f}",
                                "y_i": f"{yi:.{dp}f}",
                                "k1": f"{k1:.{dp}f}",
                                "k2": f"{k2:.{dp}f}",
                                "k3": f"{k3:.{dp}f}",
                                "k4": f"{k4:.{dp}f}",
                                "y_{i+1} ( عند x=" + f"{xi+h:.3f} )": f"{y_next:.{dp}f}"
                            })
                    except Exception as e:
                        st.error(f"❌ خطأ رياضي أثناء التعويض: {e}")
                        st.stop()
                        
                    x_vals.append(round(xi + h, 10))
                    y_vals.append(y_next)

                final_y = y_vals[-1]
                st.markdown(f"<div class='result-card exact-card'><div class='card-title'>📍 قيمة y النهائية عند x = {x_vals[-1]}</div><div class='card-value exact-value'>y({x_vals[-1]}) = {final_y:.{dp}f}</div><div class='card-subtext'>تم الحساب على {n} خطوة بحجم خطوة h = {h}</div></div>", unsafe_allow_html=True)

                st.markdown("### 📝 جدول خطوات الحل التفصيلي")
                df_steps = pd.DataFrame(steps_data)
                st.dataframe(df_steps, use_container_width=True, hide_index=True)
                
                c_dl1, c_dl2 = st.columns(2)
                csv = df_steps.to_csv(index=False).encode('utf-8-sig')
                c_dl1.download_button("📥 تحميل جدول الخطوات كملف إكسيل (CSV)", data=csv, file_name='ode_steps.csv', mime='text/csv', use_container_width=True)
                rep_html = build_printable_report(f"ODE Solver ({method})", f"y' = {func_input}", f"y({x_vals[-1]}) = {final_y:.{dp}f}", df_steps)
                c_dl2.download_button("📄 تحميل تقرير هندسي رسمي للطباعة (HTML/PDF)", data=rep_html.encode('utf-8'), file_name='ode_report.html', mime='text/html', use_container_width=True)

                st.markdown("### 🎨 الرسم البياني لمسار حل المعادلة التفاضلية")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', name='مسار الحل التقريبي y(x)', line=dict(color='#2d82ff', width=3), marker=dict(size=9, color='#28a745')))
                fig.update_layout(xaxis_title="X", yaxis_title="Y", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                st.session_state.history.append(f"🔬 **تفاضلية ({method.split(' ')[0]}):**\n`y({x_vals[-1]}) = {final_y:.{dp}f}`")

            # ---------------------------------------------------------------------
            # ج. قسم التفاضل العددي (Numerical Differentiation)
            # ---------------------------------------------------------------------
            elif "Differentiation" in app_mode:
                try:
                    x0 = safe_float(x0_str)
                    h = safe_float(h_str)
                except Exception as e:
                    st.error(f"❌ تأكد من كتابة قيمة x وحجم الخطوة h بشكل صحيح. ({e})")
                    st.stop()

                df_expr = sp.diff(f_expr, x_sym)
                df_exact_func = sp.lambdify(x_sym, df_expr, "numpy")
                exact_deriv = float(df_exact_func(x0))

                f_x0 = float(f(x0))
                f_plus = float(f(x0 + h))
                f_minus = float(f(x0 - h))

                forward_val = (f_plus - f_x0) / h
                backward_val = (f_x0 - f_minus) / h
                central_val = (f_plus - f_minus) / (2.0 * h)

                st.markdown("### 📚 القوانين الرياضية للفروق (Finite Differences):")
                st.latex(r"\text{Forward: } f'(x) \approx \frac{f(x+h) - f(x)}{h} \quad | \quad \text{Backward: } f'(x) \approx \frac{f(x) - f(x-h)}{h}")
                st.latex(r"\text{Central (الأدق): } f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}")

                if "Forward" in method:
                    chosen_val = forward_val
                    tut_str = f"( f({x0+h}) - f({x0}) ) / {h} = ({f_plus:.{dp}f} - {f_x0:.{dp}f}) / {h}"
                elif "Backward" in method:
                    chosen_val = backward_val
                    tut_str = f"( f({x0}) - f({x0-h}) ) / {h} = ({f_x0:.{dp}f} - {f_minus:.{dp}f}) / {h}"
                else:
                    chosen_val = central_val
                    tut_str = f"( f({x0+h}) - f({x0-h}) ) / (2 × {h}) = ({f_plus:.{dp}f} - {f_minus:.{dp}f}) / {2*h}"

                if show_tutorial:
                    st.markdown(f"<div class='result-card tutorial-card'><b>💡 خطوة التعويض المباشر:</b><br> f'({x0}) ≈ {tut_str} = <b>{chosen_val:.{dp}f}</b></div>", unsafe_allow_html=True)

                c_res1, c_res2 = st.columns(2)
                with c_res1:
                    st.markdown(f"<div class='result-card'><div class='card-title'>📐 المشتقة العددية ({method.split('(')[0]})</div><div class='card-value'>{chosen_val:.{dp}f}</div><div class='card-subtext'>الخطأ المطلق: {abs(exact_deriv - chosen_val):.2e}</div></div>", unsafe_allow_html=True)
                with c_res2:
                    st.markdown(f"<div class='result-card exact-card'><div class='card-title'>🎯 المشتقة التحليلية الدقيقة (Exact)</div><div class='card-value exact-value'>{exact_deriv:.{dp}f}</div><div class='card-subtext'>f'(x) = {df_expr}</div></div>", unsafe_allow_html=True)

                st.markdown("### ⚖️ جدول مقارنة جميع طرق التفاضل العددي:")
                diff_df = pd.DataFrame([
                    {"الطريقة": "الفروق الأمامية (Forward)", "القيمة التقريبية": f"{forward_val:.{dp}f}", "مقدار الخطأ": f"{abs(exact_deriv - forward_val):.2e}"},
                    {"الطريقة": "الفروق الخلفية (Backward)", "القيمة التقريبية": f"{backward_val:.{dp}f}", "مقدار الخطأ": f"{abs(exact_deriv - backward_val):.2e}"},
                    {"الطريقة": "الفروق المركزية (Central)", "القيمة التقريبية": f"{central_val:.{dp}f}", "مقدار الخطأ": f"{abs(exact_deriv - central_val):.2e}"}
                ])
                st.dataframe(diff_df, use_container_width=True, hide_index=True)

                st.markdown("### 🎨 الرسم البياني للدالة وخط المماس (Tangent Line)")
                x_range = np.linspace(x0 - 2, x0 + 2, 300)
                y_range = f(x_range)
                if isinstance(y_range, (int, float)):
                    y_range = np.full_like(x_range, y_range)
                y_tangent = f_x0 + chosen_val * (x_range - x0)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='الدالة f(x)', line=dict(color='#2d82ff', width=3)))
                fig.add_trace(go.Scatter(x=x_range, y=y_tangent, mode='lines', name='المماس التقريبي', line=dict(color='#ff4b4b', width=2, dash='dash')))
                fig.add_trace(go.Scatter(x=[x0], y=[f_x0], mode='markers', name=f'نقطة التماس x={x0}', marker=dict(color='#28a745', size=12)))
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                st.session_state.history.append(f"📐 **تفاضل عند x={x0}:**\nالنتيجة: `{chosen_val:.{dp}f}`")

            # ---------------------------------------------------------------------
            # د. قسم التكامل العددي (Numerical Integration)
            # ---------------------------------------------------------------------
            elif "Integration" in app_mode:
                try:
                    a_val = safe_float(a_str)
                    b_val = safe_float(b_str)
                except Exception as e:
                    st.error(f"❌ تأكد من كتابة حدود التكامل a و b بشكل صحيح. ({e})")
                    st.stop()

                if a_val >= b_val:
                    st.error("❌ الحد الأقصى (b) يجب أن يكون أكبر من الحد الأدنى (a).")
                    st.stop()

                st.markdown("### 📚 القوانين الرياضية للتكامل العددي:")
                st.latex(r"\text{Trapezoidal: } \int_{a}^{b} f(x) dx \approx \frac{h}{2} \left[ y_0 + 2 \sum_{i=1}^{n-1} y_i + y_n \right]")
                st.latex(r"\text{Simpson's 1/3: } \int_{a}^{b} f(x) dx \approx \frac{h}{3} \left[ y_0 + 4 \sum_{i \text{ odd}} y_i + 2 \sum_{i \text{ even}} y_i + y_n \right]")
                st.latex(r"\text{Simpson's 3/8: } \int_{a}^{b} f(x) dx \approx \frac{3h}{8} \left[ y_0 + 3 \sum_{i \neq 3k} y_i + 2 \sum_{i = 3k} y_i + y_n \right]")

                exact_val, _ = quad(lambda t: float(f(t)), a_val, b_val)

                if "h" in input_type:
                    h = float(val)
                    n = int(round((b_val - a_val) / h))
                else:
                    n = int(val)
                    h = (b_val - a_val) / n

                x_vals = np.linspace(a_val, b_val, n + 1)
                y_vals = [float(f(xv)) for xv in x_vals]

                trap_val = (h / 2.0) * (y_vals[0] + 2.0 * np.sum(y_vals[1:-1]) + y_vals[-1])
                
                simp13_val = None
                if n % 2 == 0:
                    simp13_val = (h / 3.0) * (y_vals[0] + 4.0 * np.sum(y_vals[1:-1:2]) + 2.0 * np.sum(y_vals[2:-2:2]) + y_vals[-1])
                
                simp38_val = None
                if n % 3 == 0:
                    integral = y_vals[0] + y_vals[-1]
                    for i in range(1, n):
                        integral += 2.0 * y_vals[i] if i % 3 == 0 else 3.0 * y_vals[i]
                    simp38_val = (3.0 * h / 8.0) * integral

                st.markdown(f"<div class='result-card exact-card'><div class='card-title'>🎯 القيمة الدقيقة التحليلية (Exact Integration)</div><div class='card-value exact-value'>{exact_val:.{dp}f}</div><div class='card-subtext'>عدد القطاعات n = {n} | طول الخطوة h = {h:.{dp}f}</div></div>", unsafe_allow_html=True)

                if show_tutorial:
                    pts_df = pd.DataFrame({"i": range(n+1), "x_i": x_vals, "y_i = f(x_i)": y_vals})
                    with st.expander("👀 عرض جدول نقاط التعويض (x_i , y_i)", expanded=True):
                        st.dataframe(pts_df.style.format({"x_i": f"{{:.{dp}f}}", "y_i = f(x_i)": f"{{:.{dp}f}}"}), use_container_width=True, hide_index=True)

                st.markdown("### ⚖️ مقارنة نتائج طرق التكامل العددي:")
                df_data = []
                methods_list = [
                    ("شبه المنحرف (Trapezoidal)", trap_val),
                    ("سيمبسون 1/3 (Simpson's 1/3)", simp13_val),
                    ("سيمبسون 3/8 (Simpson's 3/8)", simp38_val)
                ]
                for m_name, m_val in methods_list:
                    if m_val is not None:
                        err = abs(exact_val - m_val)
                        df_data.append({"الطريقة": m_name, "النتيجة التقريبية": f"{m_val:.{dp}f}", "الخطأ المطلق": f"{err:.2e}"})
                    else:
                        reason = "يتطلب عدد قطاعات n زوجي" if "1/3" in m_name else "يتطلب n من مضاعفات الرقم 3"
                        df_data.append({"الطريقة": m_name, "النتيجة التقريبية": f"❌ ({reason})", "الخطأ المطلق": "-"})
                
                df_int = pd.DataFrame(df_data)
                st.dataframe(df_int, use_container_width=True, hide_index=True)

                c_dl1, c_dl2 = st.columns(2)
                csv = df_int.to_csv(index=False).encode('utf-8-sig')
                c_dl1.download_button("📥 تحميل جدول مقارنة التكامل (CSV)", data=csv, file_name='integration_comparison.csv', mime='text/csv', use_container_width=True)
                rep_html = build_printable_report("Numerical Integration", f"∫ ({func_input}) dx from {a_val} to {b_val}", f"Exact = {exact_val:.{dp}f}", df_int)
                c_dl2.download_button("📄 تحميل تقرير هندسي رسمي للطباعة (HTML/PDF)", data=rep_html.encode('utf-8'), file_name='integration_report.html', mime='text/html', use_container_width=True)

                st.markdown("### 🎨 الرسم البياني التفاعلي للمساحة تحت المنحنى")
                x_smooth = np.linspace(a_val, b_val, 500)
                y_smooth = f(x_smooth)
                if isinstance(y_smooth, (int, float)):
                    y_smooth = np.full_like(x_smooth, y_smooth)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name='f(x)', line=dict(color='#2d82ff', width=3)))
                fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, fill='tozeroy', fillcolor='rgba(45, 130, 255, 0.2)', mode='none', name='المساحة المحسوبة'))
                fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name='نقاط التقسيم', marker=dict(color='#ff4b4b', size=8)))
                fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                st.session_state.history.append(f"📈 **تكامل:** `{func_input}`\nالنتيجة: `{exact_val:.{dp}f}`")

    else:
        st.info(f"👋 أهلاً بك يا باشمهندس! أنت الآن في قسم **{app_mode}**. قم بضبط المدخلات في اللوحة اليسرى ثم اضغط على زر **🚀 احسب وارسم النتائج**.")
