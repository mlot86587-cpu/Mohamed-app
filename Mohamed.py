# ==============================================================================
# 🚀 المنصة الهندسية والرياضية الشاملة (MEGA ALL-IN-ONE ENGINEERING SOLVER)
# تشمل: الجبر + المصفوفات (Gauss-Seidel/Jacobi/LU) + التفاضل والتكامل + الهندسة المستوية
# + الفيزياء والدوائر الكهربية والميكانيكا + البرمجة والمسائل اللفظية + السجل والتقارير
# مع دعم كامل للكاميرا المباشرة ومعالجة الصور (OpenCV) والذكاء الاصطناعي في كل قسم!
# ==============================================================================

import sys
import io
import time
import base64
import datetime
import traceback
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output, Markdown

# تثبيت المكتبات الإضافية تلقائياً في حال عدم توفرها على Colab
try:
    import google.generativeai as genai
except ImportError:
    print("⏳ جاري تثبيت مكتبة Google Generative AI...")
    !pip install -q google-generativeai
    import google.generativeai as genai

try:
    import cv2
except ImportError:
    print("⏳ جاري تثبيت مكتبة الرؤية الحاسوبية OpenCV...")
    !pip install -q opencv-python-headless
    import cv2

try:
    from google.colab.output import eval_js
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


# ==============================================================================
# 1. مدير النظام المركزي، الإعدادات، وسجل العمليات (Core Manager & History)
# ==============================================================================

class MegaSystemManager:
    """النظام المركزي لإدارة إعدادات الذكاء الاصطناعي، وسجل العمليات، وتصدير التقارير"""
    def __init__(self):
        self.api_key = ""
        self.model_name = "gemini-2.5-flash"
        self.temperature = 0.2
        self.detail_level = "شرح هندسي تفصيلي خطوة بخطوة مع القوانين"
        self.history_logs = []

    def configure_api(self, key_str):
        self.api_key = key_str.strip()
        if self.api_key:
            genai.configure(api_key=self.api_key)
            return True
        return False

    def add_to_history(self, category, solver_mode, input_summary, result_summary):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "id": len(self.history_logs) + 1,
            "time": timestamp,
            "category": category,
            "mode": solver_mode,
            "input": input_summary,
            "output": result_summary
        }
        self.history_logs.append(entry)

    def export_history_markdown(self):
        if not self.history_logs:
            return "# لا توجد عمليات مسجلة في السجل حتى الآن."
        md_lines = [
            "# 📑 التقرير الهندسي الشامل لحل المسائل",
            f"**تاريخ التصدير:** `{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
            "---\n"
        ]
        for item in self.history_logs:
            md_lines.append(f"## 🔹 العملية رقم ({item['id']}) - قسم: {item['category']}")
            md_lines.append(f"- **التوقيت:** `{item['time']}`")
            md_lines.append(f"- **طريقة الحل:** `{item['mode']}`")
            md_lines.append(f"### 📥 المدخلات / المسألة:\n```\n{item['input']}\n```")
            md_lines.append(f"### 📤 النتيجة والحل:\n{item['output']}\n")
            md_lines.append("---\n")
        return "\n".join(md_lines)

sys_manager = MegaSystemManager()


# ==============================================================================
# 2. شريط الإعدادات العلوي (Top Control Bar UI)
# ==============================================================================

api_key_widget = widgets.Password(
    placeholder='أدخل مفتاح Gemini API Key هنا لتفعيل الذكاء الاصطناعي والكاميرا...',
    description='🔑 API Key:',
    style={'description_width': '85px'},
    layout=widgets.Layout(width='42%')
)

model_selector = widgets.Dropdown(
    options=[
        ('Gemini 2.5 Flash (سريع ودقيق للمسائل والصور)', 'gemini-2.5-flash'),
        ('Gemini 1.5 Pro (تحليل معمق للمسائل المعقدة)', 'gemini-1.5-pro')
    ],
    value='gemini-2.5-flash',
    description='🤖 النموذج:',
    style={'description_width': '75px'},
    layout=widgets.Layout(width='32%')
)

detail_selector = widgets.Dropdown(
    options=[
        'شرح هندسي تفصيلي خطوة بخطوة مع القوانين',
        'الحل النهائي المختصر مع الخطوات الرئيسية فقط',
        'شرح أكاديمي معمق مع إثبات القوانين والتحقق من الناتج'
    ],
    value='شرح هندسي تفصيلي خطوة بخطوة مع القوانين',
    description='📝 الأسلوب:',
    style={'description_width': '75px'},
    layout=widgets.Layout(width='25%')
)

status_banner = widgets.HTML(
    value="<div style='background:#FFF3E0; color:#E65100; padding:6px 12px; border-radius:6px; border:1px solid #FFB74D;'>"
          "⚠️ <b>حالة الذكاء الاصطناعي:</b> بانتظار إدخال مفتاح API (الحل البرمجي المباشر يعمل بدون مفتاح).</div>"
)

def on_api_key_change(change):
    if sys_manager.configure_api(change['new']):
        status_banner.value = (
            "<div style='background:#E8F5E9; color:#1B5E20; padding:6px 12px; border-radius:6px; border:1px solid #81C784;'>"
            "✅ <b>حالة الذكاء الاصطناعي:</b> متصل وجاهز لتحليل الصور والمسائل في جميع الأقسام!</div>"
        )
    else:
        status_banner.value = (
            "<div style='background:#FFF3E0; color:#E65100; padding:6px 12px; border-radius:6px; border:1px solid #FFB74D;'>"
            "⚠️ <b>حالة الذكاء الاصطناعي:</b> مفتاح API فارغ.</div>"
        )

def on_model_change(change):
    sys_manager.model_name = change['new']

def on_detail_change(change):
    sys_manager.detail_level = change['new']

api_key_widget.observe(on_api_key_change, names='value')
model_selector.observe(on_model_change, names='value')
detail_selector.observe(on_detail_change, names='value')

top_settings_box = widgets.VBox([
    widgets.HTML("<h3 style='margin:0 0 8px 0; color:#0D47A1;'>⚙️ لوحة إعدادات النظام والذكاء الاصطناعي المركزية</h3>"),
    widgets.HBox([api_key_widget, model_selector, detail_selector]),
    status_banner
], layout=widgets.Layout(padding='12px', border='2px solid #1565C0', border_radius='10px', margin='0 0 15px 0', background_color='#FAFAFA'))


# ==============================================================================
# 3. محرك الكاميرا المباشرة، الرؤية الحاسوبية (OpenCV)، واستخراج البيانات بالـ AI
# ==============================================================================

def capture_photo_colab_webcam():
    """تشغيل كاميرا الويب أو كاميرا الهاتف الخلفية داخل Google Colab مع واجهة معاينة احترافية"""
    if not IN_COLAB:
        raise RuntimeError("ميزة الكاميرا المباشرة تتطلب تشغيل الكود داخل Google Colab. استخدم زر (رفع صورة) كبديل.")
    
    js_code = """
    async function captureFrameFromCamera() {
      const modal = document.createElement('div');
      modal.style.cssText = 'position:fixed; top:10%; left:50%; transform:translateX(-50%); background:#ffffff; padding:20px; border:3px solid #1565C0; border-radius:12px; z-index:99999; text-align:center; box-shadow:0 8px 25px rgba(0,0,0,0.4); min-width:360px;';

      const header = document.createElement('h3');
      header.innerText = '📸 كاميرا تصوير المسائل الهندسية والرياضية';
      header.style.cssText = 'margin:0 0 10px 0; color:#0D47A1; font-family:sans-serif;';
      modal.appendChild(header);

      const video = document.createElement('video');
      video.style.cssText = 'width:100%; max-width:480px; border-radius:8px; border:2px solid #90CAF9; background:#000; display:block; margin:0 auto 12px auto;';
      modal.appendChild(video);

      const btnRow = document.createElement('div');
      btnRow.style.cssText = 'display:flex; justify-content:center; gap:12px;';

      const snapBtn = document.createElement('button');
      snapBtn.innerText = '✅ التقاط الصورة الآن';
      snapBtn.style.cssText = 'padding:10px 22px; background:#2E7D32; color:white; border:none; border-radius:6px; font-size:15px; font-weight:bold; cursor:pointer;';

      const cancelBtn = document.createElement('button');
      cancelBtn.innerText = '❌ إلغاء';
      cancelBtn.style.cssText = 'padding:10px 18px; background:#C62828; color:white; border:none; border-radius:6px; font-size:15px; cursor:pointer;';

      btnRow.appendChild(snapBtn);
      btnRow.appendChild(cancelBtn);
      modal.appendChild(btnRow);
      document.body.appendChild(modal);

      let stream = null;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
        video.srcObject = stream;
        await video.play();
      } catch (err) {
        modal.remove();
        throw new Error("تعذر الوصول إلى الكاميرا: " + err.message);
      }

      const result = await new Promise((resolve) => {
        snapBtn.onclick = () => resolve(true);
        cancelBtn.onclick = () => resolve(false);
      });

      if (!result) {
        stream.getVideoTracks()[0].stop();
        modal.remove();
        return null;
      }

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);

      stream.getVideoTracks()[0].stop();
      modal.remove();
      return canvas.toDataURL('image/jpeg', 0.92);
    }
    """
    display(HTML(f"<script>{js_code}</script>"))
    data_url = eval_js("captureFrameFromCamera()")
    if not data_url:
        return None
    binary_bytes = base64.b64decode(data_url.split(',')[1])
    return Image.open(io.BytesIO(binary_bytes))


def apply_opencv_filter(pil_image, filter_mode, rotation_angle=0):
    """تطبيق فلاتر الرؤية الحاسوبية OpenCV لتحسين وضوح المعادلات والرسوم الهندسية"""
    if pil_image is None:
        return None
    try:
        img = pil_image.convert("RGB")
        if rotation_angle != 0:
            img = img.rotate(-rotation_angle, expand=True)

        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        if filter_mode == "صورة أصلية بدون فلتر (Original)":
            return img
        elif filter_mode == "تصفية ذكية للمعادلات والنصوص (Otsu Thresholding)":
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return Image.fromarray(thresh)
        elif filter_mode == "عزل الظلال والإضاءة غير المتساوية (Adaptive Threshold)":
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
            )
            return Image.fromarray(adaptive)
        elif filter_mode == "زيادة الحدة والتباين للرسوم والدوائر (High Contrast & Sharp)":
            enhancer_c = ImageEnhance.Contrast(img)
            img_c = enhancer_c.enhance(1.8)
            enhancer_s = ImageEnhance.Sharpness(img_c)
            return enhancer_s.enhance(2.2)
        elif filter_mode == "إبراز الحواف الهندسية (Canny Edge Enhancement)":
            edges = cv2.Canny(gray, 50, 150)
            inv_edges = cv2.bitwise_not(edges)
            return Image.fromarray(inv_edges)
        else:
            return img
    except Exception:
        return pil_image


def execute_gemini_vision_request(category_title, custom_instructions, user_text, processed_pil_image, mode_type="solve"):
    """إرسال المسألة أو الصورة إلى نموذج Gemini AI سواء للحل الكامل أو لاستخراج المعادلة (OCR)"""
    if not sys_manager.api_key:
        return "❌ **تنبيه هام:** يرجى إدخال مفتاح `Gemini API Key` في الشريط العلوي أولاً لتفعيل الذكاء الاصطناعي والكاميرا الذكية."

    try:
        model = genai.GenerativeModel(sys_manager.model_name)
        
        if mode_type == "ocr_only":
            prompt = (
                f"أنت محرك تعرف ضوئي رياضي وهندسي (Math & Engineering OCR) لقسم ({category_title}).\n"
                "قم بقراءة الصورة المرفقة بدقة واستخرج منها فقط:\n"
                "1. النص الكامل للمسألة كما هو.\n"
                "2. المعادلات الرياضية بصيغة بايثون (SymPy) أو مصفوفة أرقام جاهزة للنسخ.\n"
                "3. المعطيات الرقمية والرموز الموجودة على الرسم.\n"
                "لا تقم بحل المسألة، فقط استخرج محتواها بدقة عالية."
            )
        else:
            prompt = (
                f"أنت بروفيسور ومهندس خبير متخصص في قسم ({category_title}).\n"
                f"أسلوب الشرح المطلوب: ({sys_manager.detail_level}).\n"
                f"تعليمات خاصة بهذا القسم: {custom_instructions}\n\n"
                "قم بتحليل المسألة المرفقة (سواء من الصورة أو النص المكتوب) وقدم إجابة منظمة تشمل:\n"
                "1. **قراءة المسألة وتفريغ المعطيات والمطلوب بدقة.**\n"
                "2. **القوانين والنظريات الهندسية/الرياضية المستخدمة (مكتوبة بتنسيق LaTeX واضح).**\n"
                "3. **خطوات الحل التفصيلية خطوة بخطوة مع التعويض الرقمي.**\n"
                "4. **الناتج النهائي بوضوح مع الوحدات القياسية والتحقق من منطقية الحل.**\n"
            )
            if user_text.strip():
                prompt += f"\n---\nالبيانات أو المعادلات النصية المرفقة من المستخدم:\n{user_text}\n"

        contents = [prompt]
        if processed_pil_image is not None:
            contents.append(processed_pil_image)

        response = model.generate_content(contents)
        return response.text
    except Exception as e:
        return f"❌ **حدث خطأ أثناء معالجة الطلب عبر الذكاء الاصطناعي:**\n`{str(e)}`"


# ==============================================================================
# 4. باني وحدة (الكاميرا + OpenCV + الـ AI) المدمجة في كل تبويب
# ==============================================================================

def build_integrated_camera_ai_panel(category_name, category_ai_hint, get_text_callback, set_text_callback, main_output_widget):
    """
    ينشئ لوحة متكاملة تضم:
    - تصوير مباشر بالكاميرا
    - رفع صورة من الجهاز
    - تدوير الصورة + اختيار فلتر OpenCV مع معاينة الصورة الأصلية والمعالجة جنباً إلى جنب
    - زر استخراج النص والمعادلة من الصورة إلى خانة الإدخال (OCR to Input)
    - زر الحل الشامل بالذكاء الاصطناعي (Vision + Text AI Solver)
    """
    panel_state = {'raw_image': None, 'processed_image': None}

    btn_camera = widgets.Button(
        description='📸 تصوير بالكاميرا',
        button_style='info',
        layout=widgets.Layout(width='175px')
    )
    btn_upload = widgets.FileUpload(
        accept='image/*',
        multiple=False,
        description='📁 رفع صورة',
        layout=widgets.Layout(width='155px')
    )
    filter_selector = widgets.Dropdown(
        options=[
            'صورة أصلية بدون فلتر (Original)',
            'تصفية ذكية للمعادلات والنصوص (Otsu Thresholding)',
            'عزل الظلال والإضاءة غير المتساوية (Adaptive Threshold)',
            'زيادة الحدة والتباين للرسوم والدوائر (High Contrast & Sharp)',
            'إبراز الحواف الهندسية (Canny Edge Enhancement)'
        ],
        value='زيادة الحدة والتباين للرسوم والدوائر (High Contrast & Sharp)',
        description='🔍 فلتر الرؤية:',
        style={'description_width': '80px'},
        layout=widgets.Layout(width='360px')
    )
    rot_selector = widgets.Dropdown(
        options=[('0°', 0), ('90° يمين', 90), ('180°', 180), ('90° يسار', 270)],
        value=0,
        description='🔄 تدوير:',
        style={'description_width': '55px'},
        layout=widgets.Layout(width='145px')
    )
    btn_clear_img = widgets.Button(
        description='🗑️ مسح الصورة',
        button_style='warning',
        layout=widgets.Layout(width='130px')
    )

    btn_ocr_extract = widgets.Button(
        description='📝 استخراج المعادلة/النص من الصورة (OCR)',
        button_style='success',
        layout=widgets.Layout(width='290px')
    )
    btn_ai_solve = widgets.Button(
        description=f'🤖 حل شامل بالذكاء الاصطناعي ({category_name})',
        button_style='primary',
        layout=widgets.Layout(width='360px')
    )

    extra_ai_note = widgets.Text(
        value='',
        placeholder='ملاحظة إضافية للـ AI (مثال: احسب التيار بوحدة ملي أمبير، أو حل بطريقة كرامر)...',
        description='💡 توجيه للـ AI:',
        style={'description_width': '90px'},
        layout=widgets.Layout(width='98%')
    )

    preview_output = widgets.Output(layout=widgets.Layout(margin='6px 0'))

    def refresh_image_preview():
        with preview_output:
            clear_output()
            if panel_state['raw_image'] is None:
                return
            proc_img = apply_opencv_filter(panel_state['raw_image'], filter_selector.value, rot_selector.value)
            panel_state['processed_image'] = proc_img

            fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
            axes[0].imshow(panel_state['raw_image'])
            axes[0].set_title("الصورة الأصلية (Original)", fontsize=10)
            axes[0].axis('off')

            if proc_img.mode == 'L':
                axes[1].imshow(proc_img, cmap='gray')
            else:
                axes[1].imshow(proc_img)
            axes[1].set_title("بعد المعالجة البصرية (Processed for AI)", fontsize=10)
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()

    def on_camera_clicked(b):
        with preview_output:
            clear_output()
            print("⏳ جاري فتح الكاميرا المباشرة... التقط الصورة من النافذة المنبثقة.")
            try:
                captured = capture_photo_colab_webcam()
                if captured is not None:
                    panel_state['raw_image'] = captured
                    refresh_image_preview()
                else:
                    clear_output()
                    print("ℹ️ تم إلغاء الالتقاط بالكاميرا.")
            except Exception as e:
                clear_output()
                print(f"⚠️ تعذر تشغيل الكاميرا: {e}")

    def on_upload_changed(change):
        if not btn_upload.value:
            return
        try:
            val = btn_upload.value
            file_item = list(val.values())[0] if isinstance(val, dict) else val[0]
            img = Image.open(io.BytesIO(file_item['content']))
            panel_state['raw_image'] = img
            refresh_image_preview()
        except Exception as e:
            with preview_output:
                clear_output()
                print(f"⚠️ خطأ أثناء قراءة ملف الصورة: {e}")

    def on_filter_or_rot_change(change):
        if panel_state['raw_image'] is not None:
            refresh_image_preview()

    def on_clear_clicked(b):
        panel_state['raw_image'] = None
        panel_state['processed_image'] = None
        with preview_output:
            clear_output()
            print("ℹ️ تم حذف الصورة المرفقة.")

    def on_ocr_clicked(b):
        with main_output_widget:
            clear_output()
            if panel_state['processed_image'] is None:
                display(Markdown("⚠️ **يرجى تصوير المسألة بالكاميرا أو رفع صورتها أولاً لاستخراج النص والمعادلات منها.**"))
                return
            print("🔍 جاري قراءة الصورة واستخراج المعادلات والنصوص عبر Gemini Vision OCR...")
            ocr_text = execute_gemini_vision_request(
                category_name, category_ai_hint, "", panel_state['processed_image'], mode_type="ocr_only"
            )
            clear_output()
            display(Markdown(f"### 📝 النص والمعادلات المستخرجة من الصورة:\n---\n{ocr_text}"))
            if set_text_callback is not None and not ocr_text.startswith("❌"):
                set_text_callback(ocr_text)

    def on_ai_solve_clicked(b):
        with main_output_widget:
            clear_output()
            combined_text = get_text_callback()
            if extra_ai_note.value.strip():
                combined_text += f"\n[ملاحظة خاصة من المستخدم]: {extra_ai_note.value.strip()}"

            if panel_state['processed_image'] is None and not combined_text.strip():
                display(Markdown("⚠️ **يرجى إدخال المسألة نصياً أو التقاط صورتها بالكاميرا / رفعها أولاً.**"))
                return

            print(f"🤖 جاري تحليل وحل المسألة في قسم ({category_name}) بواسطة الذكاء الاصطناعي...")
            if panel_state['processed_image'] is not None:
                print("📷 يتم الآن فحص الصورة المعالجة بصرياً واستنتاج المعطيات والرموز...")

            ans = execute_gemini_vision_request(
                category_name, category_ai_hint, combined_text, panel_state['processed_image'], mode_type="solve"
            )
            clear_output()
            display(Markdown(f"### 🤖 الحل التفصيلي بالذكاء الاصطناعي — قسم ({category_name}):\n---\n{ans}"))
            sys_manager.add_to_history(category_name, "Gemini Vision & AI Solver", combined_text, ans)

    btn_camera.on_click(on_camera_clicked)
    btn_upload.observe(on_upload_changed, names='value')
    filter_selector.observe(on_filter_or_rot_change, names='value')
    rot_selector.observe(on_filter_or_rot_change, names='value')
    btn_clear_img.on_click(on_clear_clicked)
    btn_ocr_extract.on_click(on_ocr_clicked)
    btn_ai_solve.on_click(on_ai_solve_clicked)

    return widgets.VBox([
        widgets.HTML(
            f"<div style='background:#E3F2FD; padding:6px 10px; border-radius:6px; margin-bottom:6px; color:#0D47A1;'>"
            f"<b>📸🤖 وحدة الكاميرا الذكية والرؤية الحاسوبية والذكاء الاصطناعي — ({category_name})</b></div>"
        ),
        widgets.HBox([btn_camera, btn_upload, filter_selector, rot_selector, btn_clear_img]),
        extra_ai_note,
        widgets.HBox([btn_ai_solve, btn_ocr_extract]),
        preview_output
    ], layout=widgets.Layout(padding='10px', border='2px dashed #1E88E5', border_radius='10px', margin='12px 0', background_color='#F8FDFF'))


# ==============================================================================
# 5. القسم الأول: الجبر، المعادلات، الأعداد المركبة، والكسور الجزئية
# ==============================================================================

def create_algebra_tab():
    alg_mode = widgets.Dropdown(
        options=[
            ('حل معادلة أو نظام معادلات (Solve Equations)', 'solve'),
            ('تحليل المقدار الجبري إلى عوامل (Factorization)', 'factor'),
            ('فك وتبسيط المقادير الجبرية (Expand & Simplify)', 'simplify'),
            ('تفكيك الكسور الجزئية (Partial Fraction Decomposition)', 'apart'),
            ('حساب المجموع والمتسلسلات (Summation Σ)', 'sum')
        ],
        value='solve',
        description='🛠️ العملية:',
        layout=widgets.Layout(width='55%')
    )

    preset_alg = widgets.Dropdown(
        options=[
            ('-- اختر مثالاً جاهزاً للتجربة السريعة --', ''),
            ('معادلة تربيعية: x**2 - 5*x + 6 = 0', 'x**2 - 5*x + 6 = 0 | x'),
            ('نظام معادلتين غير خطي: x**2 + y = 5 , x - y = 1', 'x**2 + y = 5\nx - y = 1 | x, y'),
            ('معادلة درجة ثالثة: x**3 - 6*x**2 + 11*x - 6 = 0', 'x**3 - 6*x**2 + 11*x - 6 = 0 | x'),
            ('كسور جزئية: (2*x + 3) / (x**2 + 3*x + 2)', '(2*x + 3) / (x**2 + 3*x + 2) | x')
        ],
        value='',
        description='📚 أمثلة:',
        layout=widgets.Layout(width='43%')
    )

    eq_input = widgets.Textarea(
        value='x**3 - 6*x**2 + 11*x - 6 = 0',
        placeholder='أدخل المعادلة أو المقدار الجبري (كل معادلة في سطر)...',
        description='المعادلات:',
        layout=widgets.Layout(width='98%', height='95px')
    )

    vars_input = widgets.Text(
        value='x',
        placeholder='مثال: x أو x, y',
        description='المتغيرات:',
        layout=widgets.Layout(width='45%')
    )

    plot_chk = widgets.Checkbox(
        value=True,
        description='📊 رسم بياني للدالة والجذور (إذا كانت في متغير واحد)'
    )

    btn_sympy_alg = widgets.Button(
        description='⚙️ تنفيذ الحل الجبري الدقيق (SymPy)',
        button_style='success',
        layout=widgets.Layout(width='290px')
    )

    out_alg = widgets.Output(layout=widgets.Layout(border='1px solid #BDBDBD', padding='12px', border_radius='8px', margin='10px 0'))

    def on_preset_change(change):
        if change['new']:
            parts = change['new'].split('|')
            eq_input.value = parts[0].strip()
            if len(parts) > 1:
                vars_input.value = parts[1].strip()

    preset_alg.observe(on_preset_change, names='value')

    def run_algebra_solver(b):
        with out_alg:
            clear_output()
            try:
                raw_lines = [l.strip() for l in eq_input.value.strip().split('\n') if l.strip() and not l.startswith('#')]
                sym_vars = [sp.Symbol(v.strip()) for v in vars_input.value.split(',') if v.strip()]
                if not sym_vars:
                    sym_vars = [sp.Symbol('x')]

                mode = alg_mode.value
                display(Markdown(f"### ⚙️ النتائج الجبرية الدقيقة ({alg_mode.label}):"))

                if mode == 'solve':
                    eq_list = []
                    plot_expr = None
                    for line in raw_lines:
                        if '=' in line:
                            lhs, rhs = line.split('=', 1)
                            lhs_s, rhs_s = sp.sympify(lhs), sp.sympify(rhs)
                            eq_list.append(sp.Eq(lhs_s, rhs_s))
                            if plot_expr is None:
                                plot_expr = lhs_s - rhs_s
                        else:
                            expr_s = sp.sympify(line)
                            eq_list.append(sp.Eq(expr_s, 0))
                            if plot_expr is None:
                                plot_expr = expr_s

                    for idx, eq_item in enumerate(eq_list, 1):
                        display(Markdown(f"**المعادلة ({idx}):** $${sp.latex(eq_item)}$$"))

                    sol = sp.solve(eq_list, sym_vars, dict=True)
                    display(Markdown("#### 🎯 الجذور / الحلول الدقيقة:"))
                    if sol:
                        for i, s_dict in enumerate(sol, 1):
                            formatted = ", \\quad ".join([f"{sp.latex(k)} = {sp.latex(v)} \\approx { complex(v.evalf()) if v.is_complex and not v.is_real else float(v.evalf()) if v.is_number else '' }" for k, v in s_dict.items()])
                            display(Markdown(f"- **الحل ({i}):** $${formatted}$$"))
                    else:
                        display(Markdown("⚠️ لم يتم العثور على حلول جبرية صريحة."))

                    if plot_chk.value and len(sym_vars) == 1 and len(eq_list) == 1 and plot_expr is not None:
                        x_var = sym_vars[0]
                        f_num = sp.lambdify(x_var, plot_expr, modules=['numpy'])
                        real_roots = [float(s[x_var].evalf()) for s in sol if x_var in s and s[x_var].is_real]
                        center = np.mean(real_roots) if real_roots else 0.0
                        span = max(5.0, (max(real_roots) - min(real_roots)) * 1.5) if len(real_roots) > 1 else 6.0
                        x_vals = np.linspace(center - span, center + span, 400)
                        y_vals = f_num(x_vals)

                        plt.figure(figsize=(7, 3.5))
                        plt.plot(x_vals, y_vals, label=f'$f({x_var}) = {sp.latex(plot_expr)}$', color='#1565C0', linewidth=2)
                        plt.axhline(0, color='black', linewidth=1, linestyle='--')
                        if real_roots:
                            plt.scatter(real_roots, [0]*len(real_roots), color='#D32F2F', s=70, zorder=5, label='الجذور الحقيقية (Roots)')
                        plt.title("تمثيل الدالة ونقاط التقاطع مع محور السينات (Roots)")
                        plt.xlabel(str(x_var))
                        plt.ylabel(f"f({x_var})")
                        plt.grid(True, linestyle=':', alpha=0.7)
                        plt.legend()
                        plt.show()

                    sys_manager.add_to_history("الجبر والمعادلات", "SymPy Algebra Solver", eq_input.value, str(sol))

                else:
                    expr_str = raw_lines[0].split('=')[0] if '=' in raw_lines[0] else raw_lines[0]
                    expr = sp.sympify(expr_str)
                    display(Markdown(f"**المقدار المدخل:** $${sp.latex(expr)}$$"))

                    if mode == 'factor':
                        res = sp.factor(expr)
                    elif mode == 'simplify':
                        res = sp.simplify(sp.expand(expr))
                    elif mode == 'apart':
                        res = sp.apart(expr, sym_vars[0])
                    elif mode == 'sum':
                        res = sp.summation(expr, (sym_vars[0], 1, 10))

                    display(Markdown(f"#### ✅ النتيجة بعد المعالجة:\n$$ {sp.latex(res)} $$"))
                    sys_manager.add_to_history("الجبر والمعادلات", f"SymPy ({mode})", expr_str, sp.latex(res))

            except Exception as e:
                print(f"❌ خطأ في الصياغة الرياضية المدخلة: {e}")
                print("💡 تلميح: استخدم صيغة بايثون مثل x**2 للأسس و * للضرب، أو اضغط على زر (حل بالذكاء الاصطناعي) بالأسفل ليقرأ أي صيغة تلقائياً.")

    btn_sympy_alg.on_click(run_algebra_solver)

    cam_ai_panel = build_integrated_camera_ai_panel(
        category_name="الجبر والمعادلات الرياضية",
        category_ai_hint="حل المعادلات الجبرية، أنظمة المعادلات، الكسور الجزئية، المتباينات، والأعداد المركبة مع توضيح خطوات التحليل والقانون العام.",
        get_text_callback=lambda: f"العملية: {alg_mode.label}\nالمعادلات:\n{eq_input.value}\nالمتغيرات: {vars_input.value}",
        set_text_callback=lambda txt: setattr(eq_input, 'value', txt),
        main_output_widget=out_alg
    )

    return widgets.VBox([
        widgets.HTML("<h4 style='margin:0 0 8px 0; color:#1565C0;'>📐 القسم الأول: الجبر، المعادلات غير الخطية، والكسور الجزئية</h4>"),
        widgets.HBox([alg_mode, preset_alg]),
        eq_input,
        widgets.HBox([vars_input, plot_chk]),
        btn_sympy_alg,
        cam_ai_panel,
        out_alg
    ])


# ==============================================================================
# 6. القسم الثاني: المصفوفات الموسعة [A|B]، Gauss-Seidel، Jacobi، والترتيب التلقائي
# ==============================================================================

def create_matrix_gauss_seidel_tab():
    dim_selector = widgets.Dropdown(
        options=[('نظام 2×2 (معادلتين)', 2), ('نظام 3×3 (3 معادلات)', 3), ('نظام 4×4 (4 معادلات)', 4), ('نظام 5×5 (5 معادلات)', 5)],
        value=3,
        description='📏 حجم النظام:',
        layout=widgets.Layout(width='28%')
    )

    method_selector = widgets.Dropdown(
        options=[
            ('طريقة Gauss-Seidel التكرارية + الحل الدقيق', 'gauss_seidel'),
            ('طريقة Jacobi التكرارية + الحل الدقيق', 'jacobi'),
            ('مقارنة شاملة (Gauss-Seidel vs Jacobi + LU + Cramer)', 'compare_all'),
            ('خصائص المصفوفة A (المحدد، المعكوس، القيم الذاتية، الرتبة)', 'matrix_props')
        ],
        value='gauss_seidel',
        description='🧮 الخوارزمية:',
        layout=widgets.Layout(width='48%')
    )

    btn_sync_to_grid = widgets.Button(
        description='🔄 تحديث الشبكة من النص',
        button_style='info',
        layout=widgets.Layout(width='20%')
    )

    grid_container = widgets.VBox()
    grid_cells = {'widgets': []}

    text_aug_input = widgets.Textarea(
        value='4 1 2 4\n3 5 1 7\n1 1 3 3',
        placeholder='أدخل المصفوفة الموسعة [A | B] بحيث كل صف في سطر والأرقام مفصولة بمسافة...',
        description='نص [A | B]:',
        layout=widgets.Layout(width='98%', height='90px')
    )

    def build_matrix_grid(n):
        default_data = [
            [4.0, 1.0, 2.0, 4.0, 1.0, 1.0],
            [3.0, 5.0, 1.0, 7.0, 2.0, 1.0],
            [1.0, 1.0, 3.0, 3.0, 1.0, 2.0],
            [1.0, 2.0, 1.0, 8.0, 5.0, 3.0],
            [2.0, 1.0, 1.0, 1.0, 9.0, 12.0]
        ]
        rows_ui = []
        header_cells = [widgets.HTML(f"<div style='width:75px; text-align:center; font-weight:bold; color:#0D47A1;'>x{j+1}</div>") for j in range(n)]
        header_cells.append(widgets.HTML("<div style='width:20px; text-align:center; font-weight:bold;'>|</div>"))
        header_cells.append(widgets.HTML("<div style='width:80px; text-align:center; font-weight:bold; color:#2E7D32;'>الثابت B</div>"))
        rows_ui.append(widgets.HBox(header_cells))

        new_cells = []
        for i in range(n):
            row_w = []
            ui_line = []
            for j in range(n + 1):
                val = default_data[i][j] if i < len(default_data) and j < len(default_data[i]) else (5.0 if i == j else 1.0)
                cell = widgets.FloatText(value=val, layout=widgets.Layout(width='75px'))
                row_w.append(cell)
                ui_line.append(cell)
                if j == n - 1:
                    ui_line.append(widgets.HTML("<b style='margin:0 6px; color:#555;'>|</b>"))
            new_cells.append(row_w)
            rows_ui.append(widgets.HBox(ui_line))

        grid_cells['widgets'] = new_cells
        grid_container.children = [
            widgets.HTML("<b>🔢 شبكة إدخال المصفوفة الموسعة التفاعلية [A | B]:</b>"),
            widgets.VBox(rows_ui, layout=widgets.Layout(padding='6px', border='1px solid #90CAF9', border_radius='6px', background_color='#FFFFFF'))
        ]

    def sync_grid_to_text():
        lines = []
        for row_w in grid_cells['widgets']:
            lines.append(" ".join([str(c.value) for c in row_w]))
        text_aug_input.value = "\n".join(lines)

    def sync_text_to_grid(b=None):
        try:
            raw_rows = [list(map(float, r.strip().split())) for r in text_aug_input.value.strip().split('\n') if r.strip()]
            n = len(raw_rows)
            if 2 <= n <= 5 and all(len(r) == n + 1 for r in raw_rows):
                dim_selector.value = n
                build_matrix_grid(n)
                for i in range(n):
                    for j in range(n + 1):
                        grid_cells['widgets'][i][j].value = raw_rows[i][j]
        except Exception:
            pass

    build_matrix_grid(3)

    def on_dim_change(change):
        build_matrix_grid(change['new'])
        sync_grid_to_text()

    dim_selector.observe(on_dim_change, names='value')
    btn_sync_to_grid.on_click(sync_text_to_grid)

    tol_input = widgets.FloatText(value=0.00001, description='نسبة الخطأ:', layout=widgets.Layout(width='25%'))
    max_iter_input = widgets.IntText(value=25, description='أقصى تكرار:', layout=widgets.Layout(width='25%'))
    relax_omega = widgets.FloatSlider(value=1.0, min=0.5, max=1.9, step=0.05, description='معامل ω (SOR):', layout=widgets.Layout(width='35%'))
    auto_pivot_chk = widgets.Checkbox(value=True, description='تفعيل الترتيب التلقائي لتحقيق القطر المهيمن (Auto-Pivoting)')
    use_grid_chk = widgets.Checkbox(value=False, description='القراءة من الشبكة التفاعلية بدلاً من مربع النص')

    btn_solve_matrix = widgets.Button(
        description='⚙️ تشغيل خوارزمية المصفوفات و Gauss-Seidel',
        button_style='success',
        layout=widgets.Layout(width='330px')
    )

    out_matrix = widgets.Output(layout=widgets.Layout(border='1px solid #BDBDBD', padding='12px', border_radius='8px', margin='10px 0'))

    def run_iterative_method(A, B, method_type, tol, max_iter, omega=1.0):
        n = len(B)
        x = np.zeros(n, dtype=float)
        history = []
        errors = []
        for it in range(1, max_iter + 1):
            x_old = x.copy()
            for i in range(n):
                if method_type == 'gauss_seidel':
                    s1 = np.dot(A[i, :i], x[:i])
                    s2 = np.dot(A[i, i+1:], x_old[i+1:])
                    x_gs = (B[i] - s1 - s2) / A[i, i]
                    x[i] = (1 - omega) * x_old[i] + omega * x_gs
                else:  # Jacobi
                    s = np.dot(A[i, :], x_old) - A[i, i] * x_old[i]
                    x[i] = (B[i] - s) / A[i, i]
            err = float(np.max(np.abs(x - x_old)))
            history.append((it, x.copy(), err))
            errors.append(err)
            if err < tol:
                break
        return x, history, errors

    def execute_matrix_solver(b):
        with out_matrix:
            clear_output()
            try:
                if use_grid_chk.value:
                    sync_grid_to_text()

                rows = [list(map(float, r.strip().split())) for r in text_aug_input.value.strip().split('\n') if r.strip()]
                M = np.array(rows, dtype=float)
                n, m = M.shape
                if m != n + 1:
                    print(f"❌ خطأ في الأبعاد: المصفوفة الموسعة [A|B] يجب أن تكون بأبعاد (n × n+1). الأبعاد المدخلة: ({n} × {m}).")
                    return

                A = M[:, :-1].copy()
                B = M[:, -1].copy()

                # الترتيب التلقائي لتحقيق شرط القطر المهيمن (Diagonal Dominance Pivoting)
                if auto_pivot_chk.value:
                    for i in range(n):
                        max_row = i + np.argmax(np.abs(A[i:, i]))
                        if max_row != i:
                            A[[i, max_row]] = A[[max_row, i]]
                            B[[i, max_row]] = B[[max_row, i]]

                # فحص شرط القطر المهيمن
                diag_vals = np.abs(np.diag(A))
                off_diag_sums = np.sum(np.abs(A), axis=1) - diag_vals
                is_strictly_dominant = np.all(diag_vals >= off_diag_sums)

                display(Markdown("### ⚙️ التقرير الهندسي والعددي للمصفوفة الموسعة $[A | B]$:"))
                dom_status = "✅ **محقق (Strictly/Weakly Diagonally Dominant) — التقارب مضمون**" if is_strictly_dominant else "⚠️ **غير محقق بالكامل — قد يتباطأ التقارب أو يتباعد**"
                display(Markdown(f"- **حالة شرط القطر المهيمن بعد الترتيب:** {dom_status}"))

                aug_after = np.column_stack((A, B))
                display(Markdown(f"**المصفوفة الموسعة $[A|B]$ المعتمدة في الحساب:**\n$$\n{sp.latex(sp.Matrix(np.round(aug_after, 4)))}\n$$"))

                mode = method_selector.value
                if mode == 'matrix_props':
                    det_a = np.linalg.det(A)
                    rank_a = np.linalg.matrix_rank(A)
                    cond_a = np.linalg.cond(A)
                    eig_vals = np.linalg.eigvals(A)
                    display(Markdown(f"- **محدد المصفوفة $\\det(A)$:** `{det_a:.6f}`"))
                    display(Markdown(f"- **رتبة المصفوفة $\\text{{Rank}}(A)$:** `{rank_a}`"))
                    display(Markdown(f"- **رقم الحالة $\\text{{Condition Number}}$:** `{cond_a:.4f}`"))
                    display(Markdown(f"- **القيم الذاتية (Eigenvalues):** `{np.round(eig_vals, 5)}`"))
                    if abs(det_a) > 1e-12:
                        inv_a = np.linalg.inv(A)
                        display(Markdown(f"**معكوس المصفوفة $A^{{-1}}$:**\n$$\n{sp.latex(sp.Matrix(np.round(inv_a, 5)))}\n$$"))
                    return

                # تشغيل Gauss-Seidel أو Jacobi
                active_method = 'jacobi' if mode == 'jacobi' else 'gauss_seidel'
                x_sol, hist, errs = run_iterative_method(A, B, active_method, tol_input.value, max_iter_input.value, relax_omega.value)

                # بناء جدول HTML أنيق للتكرارات
                headers_html = "".join([f"<th style='padding:6px 10px; border:1px solid #90CAF9;'>x<sub>{j+1}</sub></th>" for j in range(n)])
                rows_html = ""
                for it_num, x_vec, err_val in hist:
                    vals_cells = "".join([f"<td style='padding:5px 10px; border:1px solid #E0E0E0; text-align:center;'>{v:.6f}</td>" for v in x_vec])
                    rows_html += (
                        f"<tr>"
                        f"<td style='padding:5px 10px; border:1px solid #E0E0E0; text-align:center; font-weight:bold; background:#F5F5F5;'>{it_num}</td>"
                        f"{vals_cells}"
                        f"<td style='padding:5px 10px; border:1px solid #E0E0E0; text-align:center; color:#D32F2F;'>{err_val:.7f}</td>"
                        f"</tr>"
                    )

                table_html = f"""
                <h4>📋 جدول التكرارات التفصيلي ({'Gauss-Seidel' if active_method=='gauss_seidel' else 'Jacobi'}):</h4>
                <table style='border-collapse:collapse; width:95%; font-family:monospace; font-size:13px;'>
                    <thead style='background:#1565C0; color:white;'>
                        <tr>
                            <th style='padding:6px 10px; border:1px solid #90CAF9;'>التكرار (Iteration)</th>
                            {headers_html}
                            <th style='padding:6px 10px; border:1px solid #90CAF9;'>أقصى خطأ مطلق (Max Error)</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
                """
                display(HTML(table_html))

                exact_sol = np.linalg.solve(A, B)
                display(Markdown(f"#### 🎯 الحل التقريبي النهائي: `{np.round(x_sol, 6)}` (في `{len(hist)}` تكرار)"))
                display(Markdown(f"#### ✅ الحل الجبري الدقيق للمقارنة (Exact LU Solver): `{np.round(exact_sol, 6)}`"))

                # رسم منحنى التقارب
                plt.figure(figsize=(7.5, 3.2))
                plt.semilogy(range(1, len(errs) + 1), errs, marker='o', color='#1565C0', linewidth=2, label=f"{active_method.upper()} Error")
                if mode == 'compare_all':
                    _, _, errs_jac = run_iterative_method(A, B, 'jacobi', tol_input.value, max_iter_input.value, 1.0)
                    plt.semilogy(range(1, len(errs_jac) + 1), errs_jac, marker='s', linestyle='--', color='#E65100', linewidth=2, label="JACOBI Error")
                plt.axhline(tol_input.value, color='green', linestyle=':', label='حد الخطأ المطلوب (Tolerance)')
                plt.title("منحنى سرعة التقارب وتناقص الخطأ (Convergence Rate)")
                plt.xlabel("رقم التكرار (Iteration)")
                plt.ylabel("أقصى خطأ مطلق (Log Scale)")
                plt.grid(True, which="both", linestyle='--', alpha=0.6)
                plt.legend()
                plt.show()

                sys_manager.add_to_history("المصفوفات و Gauss-Seidel", method_selector.label, text_aug_input.value, f"Approx: {np.round(x_sol, 6)} | Exact: {np.round(exact_sol, 6)}")

            except Exception as e:
                print(f"❌ حدث خطأ أثناء حل المصفوفة: {e}")

    btn_solve_matrix.on_click(execute_matrix_solver)

    cam_ai_panel = build_integrated_camera_ai_panel(
        category_name="المصفوفات والأنظمة الخطية و Gauss-Seidel",
        category_ai_hint="حل أنظمة المعادلات الخطية، المصفوفة الموسعة، طريقة جاوس-سيدل، طريقة جاكوبي، قاعدة كرامر، تفكيك LU، وحساب القيم الذاتية ومعكوس المصفوفة خطوة بخطوة.",
        get_text_callback=lambda: f"المصفوفة الموسعة [A|B]:\n{text_aug_input.value}\nالطريقة: {method_selector.label}\nنسبة الخطأ: {tol_input.value}",
        set_text_callback=lambda txt: setattr(text_aug_input, 'value', txt),
        main_output_widget=out_matrix
    )

    return widgets.VBox([
        widgets.HTML("<h4 style='margin:0 0 8px 0; color:#1565C0;'>🔢 القسم الثاني: المصفوفات الموسعة [A|B]، Gauss-Seidel، Jacobi، والتحليل العددي</h4>"),
        widgets.HBox([dim_selector, method_selector, btn_sync_to_grid]),
        grid_container,
        text_aug_input,
        widgets.HBox([tol_input, max_iter_input, relax_omega]),
        widgets.HBox([auto_pivot_chk, use_grid_chk]),
        btn_solve_matrix,
        cam_ai_panel,
        out_matrix
    ])


# ==============================================================================
# 7. القسم الثالث: التفاضل، التكامل، النهايات، متسلسلات تايلور، والمعادلات التفاضلية
# ==============================================================================

def create_calculus_tab():
    calc_op = widgets.Dropdown(
        options=[
            ('التفاضل والمشتقات العليا (Derivative d^n/dx^n)', 'diff'),
            ('التكامل غير المحدود (Indefinite Integral ∫f(x)dx)', 'indef_int'),
            ('التكامل المحدود والمساحة تحت المنحنى (Definite Integral)', 'def_int'),
            ('حساب النهايات (Limits lim x->a)', 'limit'),
            ('مفكوك متسلسلة تايلور وماكلورين (Taylor Series Expansion)', 'taylor'),
            ('حل معادلة تفاضلية عادية من الرتبة الأولى/الثانية (ODE Solver)', 'ode')
        ],
        value='diff',
        description='📈 العملية:',
        layout=widgets.Layout(width='58%')
    )

    func_input = widgets.Text(
        value='x**3 - 3*x + sin(x)',
        placeholder='أدخل الدالة مثال: x**2 * exp(-x) أو للـ ODE: Eq(y(x).diff(x,x) + y(x), sin(x))',
        description='الدالة f(x):',
        layout=widgets.Layout(width='98%')
    )

    var_sym_in = widgets.Text(value='x', description='المتغير:', layout=widgets.Layout(width='22%'))
    order_in = widgets.IntText(value=1, description='الرتبة (n):', layout=widgets.Layout(width='22%'))
    bound_a = widgets.Text(value='0', description='الحد/النقطة a:', layout=widgets.Layout(width='26%'))
    bound_b = widgets.Text(value='pi', description='الحد الأعلى b:', layout=widgets.Layout(width='26%'))

    plot_calc_chk = widgets.Checkbox(value=True, description='📊 رسم الدالة والمشتقة / المساحة التكاملية بيانياً')

    btn_calc_solve = widgets.Button(
        description='⚙️ تنفيذ حساب التفاضل والتكامل (SymPy)',
        button_style='success',
        layout=widgets.Layout(width='310px')
    )

    out_calc = widgets.Output(layout=widgets.Layout(border='1px solid #BDBDBD', padding='12px', border_radius='8px', margin='10px 0'))

    def run_calculus_solver(b):
        with out_calc:
            clear_output()
            try:
                x = sp.Symbol(var_sym_in.value.strip())
                op = calc_op.value

                if op == 'ode':
                    y = sp.Function('y')
                    ode_expr = eval(func_input.value, {"Eq": sp.Eq, "y": y, "x": x, "sin": sp.sin, "cos": sp.cos, "exp": sp.exp})
                    ode_sol = sp.dsolve(ode_expr, y(x))
                    display(Markdown(f"### ⚙️ حل المعادلة التفاضلية (ODE):\n$$ {sp.latex(ode_sol)} $$"))
                    sys_manager.add_to_history("التفاضل والتكامل", "ODE Solver", func_input.value, sp.latex(ode_sol))
                    return

                f_expr = sp.sympify(func_input.value)
                display(Markdown(f"**الدالة المدخلة:** $$ f({sp.latex(x)}) = {sp.latex(f_expr)} $$"))

                if op == 'diff':
                    n_ord = max(1, order_in.value)
                    res = sp.diff(f_expr, x, n_ord)
                    display(Markdown(f"### ✅ المشتقة من الرتبة ({n_ord}):\n$$ \\frac{{d^{n_ord}}}{{d{sp.latex(x)}^{n_ord}}} f({sp.latex(x)}) = {sp.latex(res)} $$"))

                    if plot_calc_chk.value:
                        f_num = sp.lambdify(x, f_expr, 'numpy')
                        df_num = sp.lambdify(x, res, 'numpy')
                        xv = np.linspace(-4, 4, 400)
                        plt.figure(figsize=(7.5, 3.5))
                        plt.plot(xv, f_num(xv), label='$f(x)$ الأصلية', color='#1565C0', linewidth=2)
                        plt.plot(xv, df_num(xv), label=f"$f^{{({n_ord})}}(x)$ المشتقة", color='#E65100', linestyle='--', linewidth=2)
                        plt.axhline(0, color='black', linewidth=0.8)
                        plt.title("مقارنة الدالة الأصلية مع مشتقتها بيانياً")
                        plt.grid(True, linestyle=':', alpha=0.7)
                        plt.legend()
                        plt.show()

                elif op == 'indef_int':
                    res = sp.integrate(f_expr, x)
                    display(Markdown(f"### ✅ التكامل غير المحدود:\n$$ \\int \\left({sp.latex(f_expr)}\\right) d{sp.latex(x)} = {sp.latex(res)} + C $$"))

                elif op == 'def_int':
                    a_v = sp.sympify(bound_a.value)
                    b_v = sp.sympify(bound_b.value)
                    res = sp.integrate(f_expr, (x, a_v, b_v))
                    num_val = float(res.evalf()) if res.is_number else res.evalf()
                    display(Markdown(f"### ✅ التكامل المحدود من ${sp.latex(a_v)}$ إلى ${sp.latex(b_v)}$:\n$$ \\int_{{{sp.latex(a_v)}}}^{{{sp.latex(b_v)}}} \\left({sp.latex(f_expr)}\\right) d{sp.latex(x)} = {sp.latex(res)} \\approx {num_val} $$"))

                    if plot_calc_chk.value and a_v.is_real and b_v.is_real:
                        fa, fb = float(a_v.evalf()), float(b_v.evalf())
                        f_num = sp.lambdify(x, f_expr, 'numpy')
                        margin = max(1.0, abs(fb - fa) * 0.3)
                        xv = np.linspace(min(fa, fb) - margin, max(fa, fb) + margin, 400)
                        x_fill = np.linspace(fa, fb, 250)
                        plt.figure(figsize=(7.5, 3.5))
                        plt.plot(xv, f_num(xv), color='#1565C0', linewidth=2, label='$f(x)$')
                        plt.fill_between(x_fill, f_num(x_fill), color='#90CAF9', alpha=0.5, label=f'المساحة = {num_val:.4f}')
                        plt.axhline(0, color='black', linewidth=0.8)
                        plt.title("تمثيل المساحة تحت المنحنى (Definite Integral Area)")
                        plt.grid(True, linestyle=':', alpha=0.7)
                        plt.legend()
                        plt.show()

                elif op == 'limit':
                    pt = sp.sympify(bound_a.value)
                    res = sp.limit(f_expr, x, pt)
                    display(Markdown(f"### ✅ قيمة النهاية عندما ${sp.latex(x)} \\to {sp.latex(pt)}$:\n$$ \\lim_{{{sp.latex(x)} \\to {sp.latex(pt)}}} \\left({sp.latex(f_expr)}\\right) = {sp.latex(res)} $$"))

                elif op == 'taylor':
                    pt = sp.sympify(bound_a.value)
                    terms = max(2, order_in.value + 1)
                    res = f_expr.series(x, pt, terms).removeO()
                    display(Markdown(f"### ✅ مفكوك تايلور حول النقطة ${sp.latex(pt)}$ (حتى الرتبة {terms-1}):\n$$ {sp.latex(res)} $$"))

                sys_manager.add_to_history("التفاضل والتكامل", calc_op.label, func_input.value, sp.latex(res))

            except Exception as e:
                print(f"❌ تعذر الحساب الرمزي المباشر: {e}\n💡 استخدم زر (حل بالذكاء الاصطناعي) بالأسفل لحل المسألة خطوة بخطوة.")

    btn_calc_solve.on_click(run_calculus_solver)

    cam_ai_panel = build_integrated_camera_ai_panel(
        category_name="التفاضل والتكامل والمعادلات التفاضلية",
        category_ai_hint="حل مسائل الاشتقاق الضمني والبارامتري، التكامل بالتجزيء والتعويض والكسور الجزئية، النهايات وقاعدة لوبيتال، متسلسلات تايلور، والمعادلات التفاضلية وتحويلات لابلاس.",
        get_text_callback=lambda: f"العملية: {calc_op.label}\nالدالة: {func_input.value}\nالحدود/النقطة: a={bound_a.value}, b={bound_b.value}, الرتبة={order_in.value}",
        set_text_callback=lambda txt: setattr(func_input, 'value', txt),
        main_output_widget=out_calc
    )

    return widgets.VBox([
        widgets.HTML("<h4 style='margin:0 0 8px 0; color:#1565C0;'>📈 القسم الثالث: التفاضل، التكامل، النهايات، متسلسلات تايلور، والمعادلات التفاضلية</h4>"),
        calc_op,
        func_input,
        widgets.HBox([var_sym_in, order_in, bound_a, bound_b]),
        plot_calc_chk,
        btn_calc_solve,
        cam_ai_panel,
        out_calc
    ])


# ==============================================================================
# 8. القسم الرابع: الهندسة المستوية، الفراغية، التحليلية، وحساب المثلثات
# ==============================================================================

def create_geometry_tab():
    geom_mode = widgets.Dropdown(
        options=[
            ('حل المثلث العام بمعلومية أضلاع/زوايا (Law of Sines & Cosines)', 'triangle'),
            ('هندسة تحليلية: المسافة، المنتصف، وميل ومعادلة الخط المستقيم', 'analytic_2d'),
            ('المتجهات في الفراغ 3D: الضرب القياسي، الاتجاهي، والزاوية بين متجهين', 'vectors_3d'),
            ('حساب مساحات ومحيطات وحجوم الأشكال الهندسية', 'shapes')
        ],
        value='triangle',
        description='📐 نوع المسألة:',
        layout=widgets.Layout(width='62%')
    )

    geom_input = widgets.Textarea(
        value='a = 7\nb = 10\nC_deg = 60',
        placeholder='أدخل المعطيات الهندسية (كل معطى في سطر)...',
        description='المعطيات:',
        layout=widgets.Layout(width='98%', height='95px')
    )

    def on_geom_mode_change(change):
        m = change['new']
        if m == 'triangle':
            geom_input.value = 'a = 7\nb = 10\nC_deg = 60'
        elif m == 'analytic_2d':
            geom_input.value = 'x1 = 2\ny1 = 3\nx2 = 8\ny2 = 11'
        elif m == 'vectors_3d':
            geom_input.value = 'U = 3, -2, 5\nV = 1, 4, -1'
        elif m == 'shapes':
            geom_input.value = 'shape = cylinder\nradius = 5\nheight = 12'

    geom_mode.observe(on_geom_mode_change, names='value')

    btn_geom_solve = widgets.Button(
        description='⚙️ حساب الخصائص والرسم الهندسي',
        button_style='success',
        layout=widgets.Layout(width='290px')
    )

    out_geom = widgets.Output(layout=widgets.Layout(border='1px solid #BDBDBD', padding='12px', border_radius='8px', margin='10px 0'))

    def run_geometry_solver(b):
        with out_geom:
            clear_output()
            try:
                params = {}
                for line in geom_input.value.strip().split('\n'):
                    if '=' in line:
                        k, v = line.split('=', 1)
                        params[k.strip()] = v.strip()

                mode = geom_mode.value
                display(Markdown(f"### ⚙️ الحل الهندسي الدقيق ({geom_mode.label}):"))

                if mode == 'triangle':
                    a = float(params.get('a', 0))
                    b_side = float(params.get('b', 0))
                    c = float(params.get('c', 0))
                    C_deg = float(params.get('C_deg', 0))

                    if a > 0 and b_side > 0 and C_deg > 0:
                        C_rad = np.radians(C_deg)
                        c = np.sqrt(a**2 + b_side**2 - 2 * a * b_side * np.cos(C_rad))
                    elif a > 0 and b_side > 0 and c > 0:
                        C_rad = np.arccos((a**2 + b_side**2 - c**2) / (2 * a * b_side))
                        C_deg = np.degrees(C_rad)

                    A_rad = np.arccos((b_side**2 + c**2 - a**2) / (2 * b_side * c))
                    B_rad = np.pi - A_rad - np.radians(C_deg)
                    A_deg, B_deg = np.degrees(A_rad), np.degrees(B_rad)
                    s = (a + b_side + c) / 2.0
                    area = np.sqrt(s * (s - a) * (s - b_side) * (s - c))

                    display(Markdown(f"- **أطوال الأضلاع:** $a = {a:.4f}, \\quad b = {b_side:.4f}, \\quad c = {c:.4f}$"))
                    display(Markdown(f"- **قياسات الزوايا:** $\\angle A = {A_deg:.2f}^\\circ, \\quad \\angle B = {B_deg:.2f}^\\circ, \\quad \\angle C = {C_deg:.2f}^\\circ$"))
                    display(Markdown(f"- **المحيط:** `{a + b_side + c:.4f}` | **المساحة (صيغة هيرون):** `{area:.4f}`"))

                    # رسم المثلث هندسياً
                    pts_x = [0, c, b_side * np.cos(A_rad), 0]
                    pts_y = [0, 0, b_side * np.sin(A_rad), 0]
                    plt.figure(figsize=(5.5, 3.8))
                    plt.plot(pts_x, pts_y, marker='o', color='#1565C0', linewidth=2.5)
                    plt.fill(pts_x, pts_y, color='#BBDEFB', alpha=0.4)
                    plt.title("التمثيل الهندسي للمثلث")
                    plt.axis('equal')
                    plt.grid(True, linestyle=':')
                    plt.show()

                elif mode == 'analytic_2d':
                    x1, y1 = float(params['x1']), float(params['y1'])
                    x2, y2 = float(params['x2']), float(params['y2'])
                    dist = np.hypot(x2 - x1, y2 - y1)
                    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
                    intercept = y1 - slope * x1 if x2 != x1 else None

                    display(Markdown(f"- **المسافة بين النقطتين $d$:** `{dist:.4f}`"))
                    display(Markdown(f"- **نقطة المنتصف $M$:** `({mx:.3f}, {my:.3f})`"))
                    if intercept is not None:
                        display(Markdown(f"- **الميل $m$:** `{slope:.4f}` | **معادلة الخط المستقيم:** $$y = {slope:.4f}x + ({intercept:.4f})$$"))

                elif mode == 'vectors_3d':
                    u = np.array(list(map(float, params['U'].split(','))))
                    v = np.array(list(map(float, params['V'].split(','))))
                    dot_p = np.dot(u, v)
                    cross_p = np.cross(u, v)
                    norm_u, norm_v = np.linalg.norm(u), np.linalg.norm(v)
                    angle_deg = np.degrees(np.arccos(np.clip(dot_p / (norm_u * norm_v), -1.0, 1.0)))

                    display(Markdown(f"- **معيار المتجه الأول $\\|\\vec{{U}}\\|$:** `{norm_u:.4f}` | **معيار $\\|\\vec{{V}}\\|$:** `{norm_v:.4f}`"))
                    display(Markdown(f"- **الضرب القياسي $\\vec{{U}} \\cdot \\vec{{V}}$:** `{dot_p:.4f}`"))
                    display(Markdown(f"- **الضرب الاتجاهي $\\vec{{U}} \\times \\vec{{V}}$:** `{np.round(cross_p, 4)}`"))
                    display(Markdown(f"- **الزاوية المحصورة بين المتجهين $\\theta$:** `{angle_deg:.2f}°`"))

                elif mode == 'shapes':
                    r = float(params.get('radius', 5))
                    h = float(params.get('height', 10))
                    vol_cyl = np.pi * r**2 * h
                    surf_cyl = 2 * np.pi * r * (r + h)
                    vol_sph = (4/3) * np.pi * r**3
                    display(Markdown(f"- **الأسطوانة ($r={r}, h={h}$):** الحجم = `{vol_cyl:.4f}` ، المساحة الكلية = `{surf_cyl:.4f}`"))
                    display(Markdown(f"- **الكرة ($r={r}$):** الحجم = `{vol_sph:.4f}` ، مساحة السطح = `{4*np.pi*r**2:.4f}`"))

                sys_manager.add_to_history("الهندسة وحساب المثلثات", geom_mode.label, geom_input.value, "تم الحساب الهندسي بنجاح")

            except Exception as e:
                print(f"❌ خطأ في معطيات المسألة الهندسية: {e}\n💡 يمكنك تصوير الرسم الهندسي بالكاميرا والضغط على (حل بالذكاء الاصطناعي) لتحليله فوراً.")

    btn_geom_solve.on_click(run_geometry_solver)

    cam_ai_panel = build_integrated_camera_ai_panel(
        category_name="الهندسة المستوية والفراغية وحساب المثلثات",
        category_ai_hint="تحليل الأشكال الهندسية المرسومة، استخراج الزوايا والأطوال من الرسم، تطبيق نظريات التطابق والتشابه، فيثاغورس، حساب المثلثات، والهندسة التحليلية والفراغية.",
        get_text_callback=lambda: f"القسم الهندسي: {geom_mode.label}\nالمعطيات:\n{geom_input.value}",
        set_text_callback=lambda txt: setattr(geom_input, 'value', txt),
        main_output_widget=out_geom
    )

    return widgets.VBox([
        widgets.HTML("<h4 style='margin:0 0 8px 0; color:#1565C0;'>📐 القسم الرابع: الهندسة المستوية، الفراغية، المتجهات، وحساب المثلثات</h4>"),
        geom_mode,
        geom_input,
        btn_geom_solve,
        cam_ai_panel,
        out_geom
    ])


# ==============================================================================
# 9. القسم الخامس: الفيزياء، الدوائر الكهربية، الميكانيكا، والأنظمة الهندسية
# ==============================================================================

def create_physics_engineering_tab():
    phys_preset = widgets.Dropdown(
        options=[
            ('-- اختر قانوناً أو نظاماً هندسياً جاهزاً --', ''),
            ('دائرة تيار مستمر وقدرة كهربية: V - I*R = 0 , P - V*I = 0', 'V - I*R = 0\nP - V*I = 0\nV = 24\nR = 8 | I, P'),
            ('ممانعة دائرة تيار متردد RLC توالي: Z = sqrt(R**2 + (XL - XC)**2)', 'Z - sqrt(R**2 + (XL - XC)**2) = 0\nI - V / Z = 0\nV = 220\nR = 30\nXL = 80\nXC = 40 | Z, I'),
            ('الحركة بعجلة منتظمة (نيوتن): v = u + a*t , s = u*t + 0.5*a*t**2', 'v - (u + a*t) = 0\ns - (u*t + 0.5*a*t**2) = 0\nu = 10\na = 2.5\nt = 6 | v, s'),
            ('محول كهربي مثالي: V1/V2 = N1/N2 , V1*I1 = V2*I2', 'V1/V2 - N1/N2 = 0\nV1*I1 - V2*I2 = 0\nV1 = 220\nN1 = 500\nN2 = 50\nI2 = 10 | V2, I1')
        ],
        value='',
        description='⚡ مكتبة القوانين:',
        layout=widgets.Layout(width='68%')
    )

    phys_eq_area = widgets.Textarea(
        value='V - I*R = 0\nP - V*I = 0\nV = 24\nR = 8',
        placeholder='أدخل العلاقات الفيزيائية/الهندسية والمعطيات (كل معادلة أو معطى في سطر)...',
        description='المعادلات:',
        layout=widgets.Layout(width='98%', height='100px')
    )

    phys_targets = widgets.Text(
        value='I, P',
        placeholder='المجاهيل المطلوب حسابها مفصولة بفاصلة مثل: I, P',
        description='المطلوب:',
        layout=widgets.Layout(width='50%')
    )

    def on_phys_preset(change):
        if change['new']:
            parts = change['new'].split('|')
            phys_eq_area.value = parts[0].strip()
            if len(parts) > 1:
                phys_targets.value = parts[1].strip()

    phys_preset.observe(on_phys_preset, names='value')

    btn_phys_solve = widgets.Button(
        description='⚙️ حل معادلات النظام الهندسي/الفيزيائي',
        button_style='success',
        layout=widgets.Layout(width='300px')
    )

    out_phys = widgets.Output(layout=widgets.Layout(border='1px solid #BDBDBD', padding='12px', border_radius='8px', margin='10px 0'))

    def run_physics_solver(b):
        with out_phys:
            clear_output()
            try:
                lines = [l.strip() for l in phys_eq_area.value.strip().split('\n') if l.strip()]
                eqs = []
                for line in lines:
                    if '=' in line:
                        lhs, rhs = line.split('=', 1)
                        eqs.append(sp.Eq(sp.sympify(lhs), sp.sympify(rhs)))
                    else:
                        eqs.append(sp.Eq(sp.sympify(line), 0))

                sol = sp.solve(eqs, dict=True)
                display(Markdown("### ⚙️ نتائج حل النظام الهندسي والفيزيائي:"))
                if sol:
                    target_syms = [sp.Symbol(t.strip()) for t in phys_targets.value.split(',') if t.strip()]
                    for idx, s_dict in enumerate(sol, 1):
                        display(Markdown(f"#### 🔹 مجموعة الحلول ({idx}):"))
                        for k, v in s_dict.items():
                            num_v = float(v.evalf()) if v.is_real else v.evalf()
                            highlight = "🎯 " if k in target_syms else "- "
                            display(Markdown(f"{highlight}**{k} =** `{num_v}`"))
                    sys_manager.add_to_history("الفيزياء والهندسة", "Symbolic Physics Solver", phys_eq_area.value, str(sol))
                else:
                    print("⚠️ لم يتم إيجاد حل مباشر؛ تأكد من عدد المعادلات والمجاهيل أو استخدم زر الذكاء الاصطناعي.")
            except Exception as e:
                print(f"ℹ️ هذه المسألة تحتاج إلى قراءة مخطط الدائرة أو تحليل نصي، اضغط على زر (🤖 حل شامل بالذكاء الاصطناعي) بالأسفل. ({e})")

    btn_phys_solve.on_click(run_physics_solver)

    cam_ai_panel = build_integrated_camera_ai_panel(
        category_name="الفيزياء، الدوائر الكهربية، والميكانيكا الهندسية",
        category_ai_hint="تحليل صور مخططات الدوائر الكهربية (توالي، توازي، قوانين كيرشوف KVL/KCL، نظرية ثيفينين ونورتن، دوائر التيار المتردد والمحولات)، ومخططات الجسم الحر (Free Body Diagrams) في الاستاتيكا والديناميكا.",
        get_text_callback=lambda: f"المعطيات والقوانين الفيزيائية/الهندسية:\n{phys_eq_area.value}\nالمطلوب حسابه: {phys_targets.value}",
        set_text_callback=lambda txt: setattr(phys_eq_area, 'value', txt),
        main_output_widget=out_phys
    )

    return widgets.VBox([
        widgets.HTML("<h4 style='margin:0 0 8px 0; color:#1565C0;'>⚡ القسم الخامس: الفيزياء، الدوائر الكهربية، المحولات، والاستاتيكا/الديناميكا</h4>"),
        phys_preset,
        phys_eq_area,
        phys_targets,
        btn_phys_solve,
        cam_ai_panel,
        out_phys
    ])


# ==============================================================================
# 10. القسم السادس: المسائل اللفظية، البرمجة، الخوارزميات، وتصحيح الأكواد
# ==============================================================================

def create_word_and_coding_tab():
    code_task_mode = widgets.Dropdown(
        options=[
            ('مسألة لفظية / كلامية (تحليل واستنتاج المعادلات والحل)', 'word_problem'),
            ('تصحيح خطأ برمجي وشرح الكود (Code Debugging & Trace)', 'debug_code'),
            ('تحويل فكرة أو Flowchart إلى كود برمجي كامل (Python / C++ / MATLAB)', 'generate_code')
        ],
        value='word_problem',
        description='💻 نوع المهمة:',
        layout=widgets.Layout(width='62%')
    )

    word_code_area = widgets.Textarea(
        value='# يمكنك كتابة مسألة لفظية، أو كتابة كود بايثون لتشغيله مباشرة:\nfor i in range(1, 6):\n    print(f"مربع العدد {i} هو {i**2}")',
        placeholder='اكتب نص المسألة اللفظية، أو الصق كود برمجي (Python / C / Arduino / MATLAB)...',
        description='النص / الكود:',
        layout=widgets.Layout(width='98%', height='120px')
    )

    btn_run_python = widgets.Button(
        description='▶️ تشغيل ككود بايثون مباشر في الخلية',
        button_style='success',
        layout=widgets.Layout(width='290px')
    )

    out_word_code = widgets.Output(layout=widgets.Layout(border='1px solid #BDBDBD', padding='12px', border_radius='8px', margin='10px 0'))

    def run_python_sandbox(b):
        with out_word_code:
            clear_output()
            print("⚙️ مخرجات تنفيذ كود بايثون المباشر:\n" + "=" * 45)
            try:
                sandbox_env = {"np": np, "sp": sp, "plt": plt}
                exec(word_code_area.value, sandbox_env)
                sys_manager.add_to_history("المسائل اللفظية والبرمجة", "Python Direct Execution", word_code_area.value, "Executed Successfully")
            except Exception as e:
                print(f"❌ خطأ برمجي أثناء التنفيذ (Exception): {e}")
                print("💡 اضغط على زر (🤖 حل شامل بالذكاء الاصطناعي) بالأسفل ليقوم باكتشاف الخطأ وتصحيحه لك فوراً!")

    btn_run_python.on_click(run_python_sandbox)

    cam_ai_panel = build_integrated_camera_ai_panel(
        category_name="المسائل اللفظية والبرمجة والخوارزميات",
        category_ai_hint="تحليل المسائل الكلامية وترجمتها إلى معادلات رياضية، أو قراءة صور الأكواد البرمجية ومخططات التدفق (Flowcharts) بلغات Python و C/C++ و MATLAB و Arduino واكتشاف الأخطاء وتصحيحها.",
        get_text_callback=lambda: f"نوع المهمة: {code_task_mode.label}\nالنص أو الكود:\n{word_code_area.value}",
        set_text_callback=lambda txt: setattr(word_code_area, 'value', txt),
        main_output_widget=out_word_code
    )

    return widgets.VBox([
        widgets.HTML("<h4 style='margin:0 0 8px 0; color:#1565C0;'>📝 القسم السادس: المسائل اللفظية، البرمجة، وتصحيح الأكواد (Debugging)</h4>"),
        code_task_mode,
        word_code_area,
        btn_run_python,
        cam_ai_panel,
        out_word_code
    ])


# ==============================================================================
# 11. القسم السابع: سجل العمليات وتصدير التقرير الهندسي (History & Export Tab)
# ==============================================================================

def create_history_tab():
    btn_refresh_hist = widgets.Button(description='🔄 تحديث السجل', button_style='info', layout=widgets.Layout(width='180px'))
    btn_export_md = widgets.Button(description='📥 تحميل التقرير كملف (Markdown)', button_style='success', layout=widgets.Layout(width='260px'))
    btn_clear_hist = widgets.Button(description='🗑️ مسح السجل بالكامل', button_style='danger', layout=widgets.Layout(width='190px'))

    out_hist = widgets.Output(layout=widgets.Layout(border='1px solid #BDBDBD', padding='12px', border_radius='8px', margin='10px 0'))

    def show_history(b=None):
        with out_hist:
            clear_output()
            display(Markdown(sys_manager.export_history_markdown()))

    def download_history(b):
        with out_hist:
            md_content = sys_manager.export_history_markdown()
            filename = "Engineering_Solver_Report.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(md_content)
            print(f"✅ تم حفظ التقرير في الملف: {filename}")
            if IN_COLAB:
                files.download(filename)

    def clear_history_logs(b):
        sys_manager.history_logs.clear()
        show_history()

    btn_refresh_hist.on_click(show_history)
    btn_export_md.on_click(download_history)
    btn_clear_hist.on_click(clear_history_logs)

    return widgets.VBox([
        widgets.HTML("<h4 style='margin:0 0 8px 0; color:#1565C0;'>📑 القسم السابع: سجل العمليات وحفظ التقارير الهندسية</h4>"),
        widgets.HBox([btn_refresh_hist, btn_export_md, btn_clear_hist]),
        out_hist
    ])


# ==============================================================================
# 12. تجميع وتشغيل المنصة الكاملة (Master Application Launcher)
# ==============================================================================

def launch_mega_engineering_platform():
    banner_html = widgets.HTML(
        "<div style='background:linear-gradient(135deg, #0D47A1 0%, #1976D2 50%, #00897B 100%); color:white; padding:18px 22px; border-radius:12px; margin-bottom:14px; box-shadow:0 4px 12px rgba(0,0,0,0.2);'>"
        "<h2 style='margin:0; font-size:22px;'>🚀 المنصة الهندسية والرياضية الشاملة (Mega AI & Camera Problem Solver)</h2>"
        "<p style='margin:6px 0 0 0; font-size:14px; opacity:0.95;'>"
        "نظام موحد يدمج الحل البرمجي والعددي الدقيق (SymPy / NumPy / Gauss-Seidel) + الكاميرا المباشرة + فلاتر الرؤية الحاسوبية (OpenCV) + الذكاء الاصطناعي (Gemini Vision) في جميع أنواع المسائل"
        "</p></div>"
    )

    main_tabs = widgets.Tab(children=[
        create_algebra_tab(),
        create_matrix_gauss_seidel_tab(),
        create_calculus_tab(),
        create_geometry_tab(),
        create_physics_engineering_tab(),
        create_word_and_coding_tab(),
        create_history_tab()
    ])

    tab_titles = [
        '1️⃣ الجبر والمعادلات',
        '2️⃣ المصفوفات و Gauss-Seidel',
        '3️⃣ التفاضل والتكامل',
        '4️⃣ الهندسة والمثلثات',
        '5️⃣ الفيزياء والدوائر',
        '6️⃣ اللفظيات والبرمجة',
        '7️⃣ السجل والتقارير'
    ]
    for i, t in enumerate(tab_titles):
        main_tabs.set_title(i, t)

    display(widgets.VBox([banner_html, top_settings_box, main_tabs]))

# تشغيل المنصة فوراً
launch_mega_engineering_platform()
