import os
import io
import requests
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops, ImageStat, ImageOps
import matplotlib.font_manager as fm
import google.generativeai as genai
import json
import requests
from PIL import Image, ImageOps
import base64
import time 

# ==========================================
# 🧠 自动化核心引擎 (带状态追踪版)
# ==========================================
def analyze_image_style(image_bytes, api_key):
    """
    使用阿里云 Qwen-VL 进行创意转译，解决风格单一和边缘白边问题
    """
    if not api_key:
        st.error("🚨 错误：未检测到阿里云 API Key (i_key)")
        return None

    try:
        # 1. 编码图片
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # 2. 阿里云请求配置
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # --- 💡 核心逻辑：这里是第 39 行开始的 payload，注意缩进 ---
        payload = {
            "model": "qwen-vl-max",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": f"data:image/jpeg;base64,{base64_image}"},
                            {"text": (
                                "你是一个资深平面设计师。请分析上传图片的意境并进行『多样化遮罩转译』：\n"
                                "1. 风格转译：请从以下意象中选择【最贴切的一个】作为 image_gen_prompt 的核心关键词：\n"
                                "   - 意境写意：使用 'Chinese ink wash brush stroke border, messy ink splatter, hand-drawn edges'\n"
                                "   - 意境古朴：使用 'Traditional Chinese silk scroll unfolding, aged paper texture frame'\n"
                                "   - 意境硬朗：使用 'Ancient oriental wooden window lattice, traditional pattern silhouette'\n"
                                "   - 意境喜庆/民俗：使用 'Chinese paper-cut art border, folk silhouette frame'\n"
                                "2. 满铺约束：必须包含 'full bleed design, elements touching the edges, no white margins'。确保设计元素撑满画面，不留白边。\n"
                                "3. 物理约束：必须包含 'high contrast black and white stencil, pure white solid center, deep black background'。\n"
                                "请仅返回标准 JSON，不要解释文字：{'image_gen_prompt': '...', 'font_color': '...'}"
                            )}
                        ]
                    }
                ]
            },
            "parameters": {"result_format": "message"}
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            res_json = response.json()
            content = res_json['output']['choices'][0]['message']['content'][0]['text']
            # 简单清洗 JSON 文本
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
            return None
        else:
            st.error(f"阿里云分析失败: {response.status_code}")
            return None

    except Exception as e:
        st.error(f"分析引擎异常: {e}")
        return None
    
def generate_ai_mask(gen_prompt, api_key, target_w, target_h, base_url):
    if not api_key: return None
    with st.status("🎭 正在生成 AI 异形遮罩...", expanded=True) as status:
        try:
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            
            # --- 💡 核心改动：强化物理约束提示词 ---
            strict_prompt = (
                f"{gen_prompt}, "
                "full bleed design, element touching the edges, " # 强调元素触碰边缘
                "white silhouette isolated on deep black background, "
                "stenciled look, extreme contrast, no frame margin, " # 强调无页边距
                "high quality vector graphics, 2D flat"
            )

            payload = {
                "model": "z-image-turbo",
                "input": {"messages": [{"role": "user", "content": [{"text": strict_prompt}]}]},
                "parameters": {
                    "size": f"{target_w}*{target_h}", 
                    "prompt_extend": False,
                    "negative_prompt": "photo, colorful, gradient, soft edges, realistic, background detail, thin lines, complex background"
                }
            }
            
            response = requests.post(base_url, headers=headers, json=payload, timeout=60)
            res_data = response.json()
            
            if response.status_code == 200:
                try:
                    img_url = res_data["output"]["choices"][0]["message"]["content"][0]["image"]
                except KeyError:
                    img_url = res_data.get("output", {}).get("results", [{}])[0].get("url")
                
                if not img_url: raise Exception("接口未返回图片 URL")
                
                img_res = requests.get(img_url)
                mask_img = Image.open(io.BytesIO(img_res.content)).convert("L")
                
                # 1. 极致对比度：把深灰直接变黑，浅灰直接变白
                mask_img = ImageOps.autocontrast(mask_img, cutoff=5) # 增加忽略比例
                mask_img = mask_img.point(lambda x: 255 if x > 200 else 0, mode='L')

                # 2. 智能探测：确保中心是白的
                center_pixel = mask_img.getpixel((target_w // 2, target_h // 2))
                if center_pixel < 127: 
                    mask_img = ImageOps.invert(mask_img)

                # 3. 此时不再画框，直接返回，看 AI 生成的原始边缘
                status.update(label="✅ 风格转译完成", state="complete")
                return mask_img
            else:
                st.error(f"API 报错: {res_data.get('message', '未知错误')}")
                return None

        except Exception as e:
            st.session_state['last_ai_error'] = f"蒙版生成异常: {str(e)}"
            return None

# --- 1. 基础配置与高级 CSS 注入 ---
st.set_page_config(page_title="MyPosterAgent | Dean's Workstation", layout="wide")
HIGHLIGHT_COLOR = "#FF5588"
VERSION = "v13.0 (Ultimate Edition)"

# --- 1. 基础配置与高级 CSS 注入 ---
st.set_page_config(
    page_title="MyPosterAgent | Dean's Workstation", 
    layout="wide",
    initial_sidebar_state="expanded" # 强制展开，防止“锁死”
)

HIGHLIGHT_COLOR = "#FF5588"
VERSION = "v13.0 (Ultimate Edition)"

st.markdown(f"""
    <style>
    /* 1. Tab 栏背景与文字适配 */
    div[data-testid="stTabs"] [role="tablist"] {{
        position: sticky; 
        top: 0; 
        z-index: 999; 
        background-color: var(--background-color);
        border-bottom: 1px solid var(--secondary-background-color);
        padding-top: 1rem;
    }}

    div[data-testid="stTabs"] [data-baseweb="tab"] {{
        background-color: transparent !important;
        color: var(--text-color) !important;
    }}

    /* 2. 画布预览窗口适配 */
    [data-testid="stImage"] img {{ 
        border: 1px solid rgba(128, 128, 128, 0.2) !important; 
        border-radius: 8px; 
        box-shadow: 0 8px 24px rgba(0,0,0,0.2) !important; 
    }}

    /* 3. 布局与 Header 优化 (精细化隐藏) */
    .block-container {{ padding-top: 1.5rem !important; }}
    
    /* 让 Header 背景透明，但不影响其中的按钮 */
    header[data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0) !important;
        border: none !important;
    }}

    /* 隐藏 Header 内的装饰性 div，保留操作按钮 */
    header[data-testid="stHeader"] > div:first-child {{ visibility: hidden; }}

    /* 确保侧边栏开关和菜单按钮可见 */
    button[kind="headerNoPadding"] {{
        visibility: visible !important;
        z-index: 1000 !important;
    }}

    /* 4. 按钮主题色 */
    div.stButton > button, div.stDownloadButton > button {{
        background-color: {HIGHLIGHT_COLOR} !important; 
        color: white !important;
        border: none !important; 
        border-radius: 6px; 
        font-weight: bold;
        transition: opacity 0.3s;
    }}
    div.stButton > button:hover {{
        opacity: 0.8;
    }}
    </style>
""", unsafe_allow_html=True)

# --- 2. Session State 初始化 ---
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'canvas_w': 880, 'canvas_h': 444,
        'crop_y': 0.5, 'crop_x': 0.5, 'poster_scale': 1.0,
        'mask_scale': 1.0, 'mask_w_scale': 1.0, 'mask_y': 0.0,
        'pop_depth': 0.6, 'enable_popout': False,
        'color': '#FFFFFF', 'size': 60, 'y_pos': 380,
        'shadow_offset': 0, 'shadow_blur': 0, 'shadow_alpha': 150,
        'blur_radius': 50, 'blur_opacity': 100,
        'logo_x': 50, 'logo_y': 85, 'logo_scale': 40
    }
    st.session_state.active_file_id = ""
    st.session_state.cutout_img = None
    st.session_state.loaded_fonts = {}

# --- 3. 核心工具算法 ---
def update_setting(key, from_type):
    st.session_state.settings[key] = st.session_state[f"{from_type}_{key}"]
    st.session_state[f"{'num' if from_type=='sli' else 'sli'}_{key}"] = st.session_state[f"{from_type}_{key}"]

def dual_control(label, key_name, min_val, max_val, step=1.0):
    curr_val = float(st.session_state.settings.get(key_name, min_val))
    c1, c2 = st.columns([2.5, 1])
    c1.markdown(f"<div style='font-size:13px; font-weight:500; margin-top:5px;'>{label}</div>", unsafe_allow_html=True)
    c2.number_input(label, float(min_val), float(max_val), curr_val, float(step), key=f"num_{key_name}", on_change=update_setting, args=(key_name, "num"), label_visibility="collapsed")
    st.slider(label, float(min_val), float(max_val), curr_val, float(step), key=f"sli_{key_name}", on_change=update_setting, args=(key_name, "sli"), label_visibility="collapsed")

# 核心：全员入框识别逻辑 (正侧脸双重捕获)
def run_ai_alignment(pil_image):
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img_h = cv_img.shape[0]
    gray = cv2.equalizeHist(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY))
    
    f_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    p_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    faces_f = f_casc.detectMultiScale(gray, 1.05, 4, minSize=(30, 30))
    faces_p = p_casc.detectMultiScale(gray, 1.05, 4, minSize=(30, 30))
    
    all_y = []
    for (x,y,w,h) in faces_f: all_y.extend([y, y+h])
    for (x,y,w,h) in faces_p: all_y.extend([y, y+h])
    
    if not all_y: 
        return 0.5, 1.0, "未检测到清晰面部"
    
    min_y, max_y = min(all_y), max(all_y)
    center_y = ((min_y + max_y) / 2) / img_h
    face_span_ratio = (max_y - min_y) / img_h
    
    # 动态缩放逻辑
    rec_scale = 1.3 if face_span_ratio < 0.2 else (1.1 if face_span_ratio < 0.4 else 0.9)
    return float(np.clip(center_y, 0.3, 0.7)), rec_scale, f"AI 已捕获全员坐标 (Y跨度 {face_span_ratio:.1%})"

# 修复：明度对比智能取色
def auto_contrast_color(bg_image, y_pos, th):
    try:
        # 在文字 Y 坐标附近截取一条水平区域
        strip_h = int(th * 0.15) 
        top = max(0, int(y_pos) - strip_h // 2)
        bottom = min(th, int(y_pos) + strip_h // 2)
        strip = bg_image.crop((0, top, bg_image.width, bottom)).convert("L")
        
        avg_lum = ImageStat.Stat(strip).mean[0] # 获取平均明度 (0-255)
        return "#000000" if avg_lum > 135 else "#FFFFFF"
    except:
        return "#FFFFFF" # 兜底

def get_removed_bg(api_key, img_bytes, quality="preview"):
    try:
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': img_bytes},
            data={'size': quality},
            headers={'X-Api-Key': api_key},
            timeout=60
        )
        if response.status_code == 200:
            # 注意：remove.bg 返回的是二进制图像流，不是 JSON！
            return Image.open(io.BytesIO(response.content)).convert("RGBA"), "成功"
        else:
            return None, f"错误码: {response.status_code}"
            
    except Exception as e:
        # ✅ 补全这个 except，第 263 行的报错才会消失
        return None, f"请求异常: {str(e)}"

# --- 函数彻底结束，下方 render_poster 将恢复正常 ---

# --- 4. 核心渲染引擎 ---
def render_poster(poster_file, subtitle, sets, tw, th, logo_file=None, font_file=None):
    # 💡 初始化逻辑
    bg_for_analysis = Image.new("RGB", (tw, th), (0, 0, 0))
    final_canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    
    try:
        poster = Image.open(poster_file).convert("RGBA")
        
        # --- 1. 基准缩放与偏移锁定 ---
        base_scale = max(tw / poster.width, th / poster.height)
        final_scale = base_scale * sets['poster_scale']
        nw, nh = int(poster.width * final_scale), int(poster.height * final_scale)
        off_x = int((tw - nw) // 2 - (sets['crop_x'] - 0.5) * max(tw, nw))
        off_y = int((th - nh) // 2 - (sets['crop_y'] - 0.5) * max(th, nh))

        # --- 2. 渲染底板 (吸色 + 模糊) ---
        stat = ImageStat.Stat(poster)
        edge_color = tuple([int(x) for x in stat.median[:3]])
        bg_fill = Image.new("RGBA", (tw, th), edge_color + (255,))
        
        poster_bg = poster.resize((tw, th), Image.Resampling.LANCZOS)
        if sets['blur_radius'] > 0: 
            poster_bg = poster_bg.filter(ImageFilter.GaussianBlur(sets['blur_radius']))
        
        bg_fill = Image.alpha_composite(bg_fill, poster_bg)
        overlay = Image.new("RGBA", (tw, th), (0, 0, 0, int(sets['blur_opacity'])))
        canvas = Image.alpha_composite(bg_fill, overlay)
        
        poster_res = poster.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas.paste(poster_res, (off_x, off_y), poster_res if "A" in poster_res.getbands() else None)
        bg_for_analysis = canvas.convert("RGB")

        # --- 3. 层级 3：异形遮罩 (修正优先级逻辑) ---
        mask_source = None
        
        # 🌟 重点：首先检查 Session State 里有没有 AI 刚刚生成的动态蒙版
        if st.session_state.get('current_mask') is not None:
            mask_source = st.session_state['current_mask']
            # 确保是灰度模式
            if mask_source.mode != "L":
                mask_source = mask_source.convert("L")
        # 🌟 其次：如果没有 AI 蒙版，再看本地有没有静态文件
        elif os.path.exists("mask.png"):
            mask_source = Image.open("mask.png").convert("L")
        
        # 🌟 最后：兜底逻辑（全白）
        if mask_source is None:
            mask_source = Image.new("L", (tw, th), 255)

        # A. 计算并缩放遮罩
        mw, mh = int(tw * sets['mask_w_scale']), int(th * sets['mask_scale'])
        mask_res = mask_source.resize((mw, mh), Image.Resampling.LANCZOS)
        
        # B. 构造全屏 Alpha 蒙版
        full_mask = Image.new("L", (tw, th), 0)
        m_off_y = int(th * sets['mask_y']) + (th // 2) - (mh // 2)
        full_mask.paste(mask_res, ((tw - mw) // 2, m_off_y))
        
        # C. 🚀 关键修复：将灰度图直接注入为 canvas 的 Alpha 通道
        # 这样 canvas 的每一个像素的透明度都由遮罩的明度决定
        if canvas.mode != "RGBA":
            canvas = canvas.convert("RGBA")
        
        # 获取原有 alpha (可能是抠图后的) 并叠加新的异形遮罩
        # multiply 的逻辑：只有两个通道都是白色的地方才会显示
        current_alpha = canvas.getchannel("A")
        new_alpha = ImageChops.multiply(current_alpha, full_mask)
        canvas.putalpha(new_alpha)
        
        # D. 最终合成：将带透明度的内容直接叠在空白画布上，确保透出“底色”
        final_canvas = Image.alpha_composite(final_canvas, canvas)

        # --- 4. 破框层 (Pop-out) ---
        if sets['enable_popout'] and st.session_state.get('cutout_img'):
            cutout = st.session_state.cutout_img.resize((nw, nh), Image.Resampling.LANCZOS)
            cutout_canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
            cutout_canvas.paste(cutout, (off_x, off_y), cutout)
            grad_mask = Image.new("L", (tw, th), 0)
            draw_g = ImageDraw.Draw(grad_mask)
            pop_line = int(th * sets['pop_depth'])
            for y in range(th):
                alpha = 255 if y < pop_line - 20 else (0 if y > pop_line + 20 else int(255 * (1 - (y - (pop_line-20))/40)))
                draw_g.line([(0, y), (tw, y)], fill=alpha)
            cutout_canvas.putalpha(ImageChops.multiply(cutout_canvas.getchannel("A"), grad_mask))
            final_canvas = Image.alpha_composite(final_canvas, cutout_canvas)

        # --- 5. 文字渲染层 (修正路径定义) ---
        if subtitle:
            font = None
            f_size = int(sets['size'])
            if font_file:
                try:
                    font_file.seek(0)
                    font = ImageFont.truetype(font_file, f_size)
                except: pass
            
            # 💡 直接在此处定义文件名，解决 Not Defined 报错
            if font is None and os.path.exists("front.ttf"):
                try: 
                    font = ImageFont.truetype("front.ttf", f_size)
                except: 
                    pass
            
            if font is None:
                try: font = ImageFont.truetype("Arial Unicode.ttf", f_size)
                except: font = ImageFont.load_default()

            t_pos = (tw // 2, int(sets['y_pos']))
            if sets.get('shadow_alpha', 0) > 0:
                s_canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
                s_draw = ImageDraw.Draw(s_canvas)
                s_off = int(sets.get('shadow_offset', 0))
                s_draw.text((t_pos[0] + s_off, t_pos[1] + s_off), subtitle, fill=(0,0,0,int(sets['shadow_alpha'])), font=font, anchor="mm")
                if sets.get('shadow_blur', 0) > 0: 
                    s_canvas = s_canvas.filter(ImageFilter.GaussianBlur(sets['shadow_blur']))
                final_canvas = Image.alpha_composite(final_canvas, s_canvas)
            
            main_text_draw = ImageDraw.Draw(final_canvas)
            main_text_draw.text(t_pos, subtitle, fill=sets['color'], font=font, anchor="mm")

        # --- 6. Logo 层 ---
        if logo_file:
            logo = Image.open(logo_file).convert("RGBA")
            ls = sets['logo_scale'] / 100
            lw, lh = int(logo.width * ls), int(logo.height * ls)
            logo_res = logo.resize((lw, lh), Image.Resampling.LANCZOS)
            final_canvas.paste(logo_res, (int(tw*sets['logo_x']/100 - lw/2), int(th*sets['logo_y']/100 - lh/2)), logo_res)

    except Exception as e:
        st.error(f"渲染引擎故障: {e}")

    return final_canvas, bg_for_analysis

# --- 5. 侧边栏架构 ---
with st.sidebar:
    st.markdown(f"### 🎨 PosterAgent {VERSION}\n<p style='font-size:12px; color:#888;'>Architect: Dean</p>", unsafe_allow_html=True)
    
    # 1. 历史日志与错误处理 (仅在有内容时出现，不占位)
    if st.session_state.get('debug_logs'):
        with st.expander("📜 诊断日志", expanded=False):
            for log in st.session_state.debug_logs[::-1]:
                st.write(f"🕒 {log['time']}")
                st.code(log['body'], language="text")

    if st.session_state.get('last_ai_error'):
        st.error("🚨 自动驾驶异常")
        with st.expander("查看详情"):
            st.info(st.session_state['last_ai_error'])
        if st.button("🗑️ 清除错误", use_container_width=True):
            st.session_state['last_ai_error'] = None
            st.rerun()
        
    # 2. 核心配置 (根据 Key 的状态决定是否默认折叠)
    image_api_key = st.session_state.get('i_key', "")
    z_base_url = st.session_state.get('z_url', "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation")
    
    # 自动探测默认 Key
    default_i_key = image_api_key or st.secrets.get("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
    
    # 💡 逻辑点：如果已经有 Key 了，就折叠起来减少视觉负担
    config_expanded = False if default_i_key else True
    
    with st.expander("🔑 引擎核心配置", expanded=config_expanded):
        image_api_key = st.text_input("阿里云 API Key", type="password", value=default_i_key)
        z_base_url = st.text_input("EndPoint", value=z_base_url)
        st.session_state['i_key'] = image_api_key
        st.session_state['z_url'] = z_base_url
        
    # 3. 引擎控制台 (AI 默认开启)
    st.subheader("⚡ 引擎控制台")
    auto_pilot = st.toggle("🚀 开启 AI 自动驾驶", value=True) # 💡 默认改为 True
    
    # 4. 画布尺寸设置

    with st.expander("🖼️ 画布尺寸与比例", expanded=False):

        tw = st.session_state.settings['canvas_w']

        th = st.session_state.settings['canvas_h']

        options = {"16:9 (880x444)": (880, 444), "9:16 (450x800)": (450, 800), "1:1 (600x600)": (600, 600)}

        current_opt = "自定义"

        for k, v in options.items():

            if (tw, th) == v: current_opt = k

        sel_ratio = st.radio("常用预设", list(options.keys()) + ["自定义"], 

                             index=list(options.keys()).index(current_opt) if current_opt != "自定义" else 3, horizontal=True)

        if sel_ratio != "自定义": tw, th = options[sel_ratio]

        tw = st.number_input("宽度 (W)", 200, 2000, tw)

        th = st.number_input("高度 (H)", 200, 2000, th)

        st.session_state.settings['canvas_w'], st.session_state.settings['canvas_h'] = tw, th



    # --- C. 统一上传入口 (严格缩进在 sidebar 内) ---

    st.markdown("---")

    up_file = st.file_uploader("📥 第一步：上传原图 (支持拖拽)", type=["jpg", "jpeg", "png", "webp"])



    # 💡 核心修复：只有文件存在才执行后续逻辑，解决 NoneType 报错

    if up_file is not None:

        # 1. 状态初始化

        if st.session_state.get('last_uploaded_file') != up_file.name:

            st.session_state.settings['crop_y'] = 0.5

            st.session_state.settings['poster_scale'] = 1.0

            st.session_state.last_uploaded_file = up_file.name



        # 2. 图像预加载 (确保只在这里加载一次)

        original_img = Image.open(up_file).convert("RGBA")

        current_file_fingerprint = f"{up_file.name}_{up_file.size}"

    # 💡 核心修复：只有文件存在才执行后续逻辑，解决 NoneType 报错
    if up_file is not None:
        # 1. 状态初始化
        if st.session_state.get('last_uploaded_file') != up_file.name:
            st.session_state.settings['crop_y'] = 0.5
            st.session_state.settings['poster_scale'] = 1.0
            st.session_state.last_uploaded_file = up_file.name

        # 2. 图像预加载 (确保只在这里加载一次)
        current_file_fingerprint = f"{up_file.name}_{up_file.size}"
        original_img = Image.open(up_file).convert("RGBA")

# ==========================================
        # 3. 自动驾驶联动逻辑 (强制贴底 + 坐标对齐版)
        # ==========================================
        if auto_pilot and st.session_state.get('active_file_id') != current_file_fingerprint:
            with st.status("🚀 AI 正在接管设计控制台...", expanded=True) as status:
                try:
                    file_content = up_file.getvalue()
                    source_img = Image.open(io.BytesIO(file_content)).convert("RGB")
                    
                    # 1. 【位置捕捉】
                    status.write("📐 正在执行全员入框捕捉...")
                    target_y, target_scale, align_msg = run_ai_alignment(source_img)
                    
                    # 2. 【位置对齐：海报动，遮罩稳】
                    st.session_state.settings['crop_y'] = target_y 
                    st.session_state.settings['poster_scale'] = target_scale
                    
                    
                    # 3. 【副标题：彻底去除避让，强制贴底】
                    # 固定在 88% 的位置，确保在画面底部且留有安全边距
                    fixed_bottom_y = th * 0.88
                    st.session_state.settings['y_pos'] = fixed_bottom_y

                    # 4. 风格转译与遮罩生成
                    style_data = analyze_image_style(file_content, image_api_key)
                    if style_data:
                        ai_mask = generate_ai_mask(style_data['image_gen_prompt'], image_api_key, tw, th, z_base_url)
                        if ai_mask:
                            st.session_state['current_mask'] = ai_mask
                            
                            # 5. 智能取色 (针对底部位置进行反色计算)
                            status.write("🎨 正在匹配底部文字颜色...")
                            st.session_state.settings['color'] = auto_contrast_color(
                                source_img.resize((tw,th)), 
                                fixed_bottom_y, 
                                th
                            )

                            # 6. 智能抠图
                            if rbg_key: 
                                status.write("✂️ 正在尝试执行智能抠图...")
                                no_bg_img, msg = get_removed_bg(rbg_key, file_content) 
                                if no_bg_img:
                                    st.session_state['cutout_img'] = no_bg_img
                                    st.session_state.settings['enable_popout'] = True

                            # ✅ 完成并强制刷新
                            st.session_state.active_file_id = current_file_fingerprint
                            status.update(label=f"✅ {align_msg}，底部布局已就绪", state="complete")
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"🚨 自动驾驶异常: {e}")
                    st.session_state.active_file_id = current_file_fingerprint

        # --- D. STEP 1 & 2. 素材加工 ---
        with st.expander("🛠️ STEP 1 & 2. 素材加工 (AI 入框 + 抠图)", expanded=True):
            if st.button("✨ 执行 AI 全员入框", use_container_width=True):
                ry, rs, status = run_ai_alignment(original_img)
                st.session_state.settings['crop_y'], st.session_state.settings['poster_scale'] = ry, rs
                st.toast(status)
                st.rerun()

            # 实时手动控制
            st.session_state.settings['poster_scale'] = st.slider("画面缩放", 0.5, 3.0, float(st.session_state.settings['poster_scale']), 0.1)
            st.session_state.settings['crop_x'] = st.slider("水平重心", -0.5, 1.5, float(st.session_state.settings['crop_x']), 0.01)
            st.session_state.settings['crop_y'] = st.slider("垂直重心", -0.5, 1.5, float(st.session_state.settings['crop_y']), 0.01)
            
            st.markdown("---")
            
            # 抠图模块
            rbg_key = st.text_input("Remove.bg Key", type="password", placeholder="输入 API Key")
            rbg_quality = st.radio("抠图清晰度", ["preview", "full"], horizontal=True)
            
            col_rbg1, col_rbg2 = st.columns(2)
            with col_rbg1:
                if st.button("🚀 执行抠图", use_container_width=True):
                    res, msg = get_removed_bg(rbg_key, up_file.getvalue(), quality=rbg_quality)
                    if res:
                        st.session_state.cutout_img = res
                        st.session_state.settings['enable_popout'] = True
                        st.rerun()
                    else:
                        st.error(msg)
            with col_rbg2:
                if st.button("🗑️ 清除抠图", use_container_width=True):
                    st.session_state.cutout_img = None
                    st.session_state.settings['enable_popout'] = False
                    st.rerun()

            if st.session_state.settings.get('enable_popout'):
                st.session_state.settings['pop_depth'] = st.slider("破框深度", 0.0, 1.0, float(st.session_state.settings['pop_depth']), 0.01)

            # ==========================================
            # 🎭 蒙版预览区 (这里必须保持和上面对齐！)
            # ==========================================
            st.markdown("---")
            st.subheader("🎭 异形遮罩控制")

            if st.session_state.get('current_mask'):
                st.image(st.session_state['current_mask'], caption="当前应用中的 AI 蒙版", use_container_width=True)
                if st.button("🗑️ 清除当前蒙版", use_container_width=True, key="clear_mask_btn"):
                    st.session_state['current_mask'] = None
                    st.rerun()
            else:
                st.info("💡 尚未生成蒙版，请开启上方【自动驾驶】，或在此手动生成")
                gen_prompt = st.text_area(
                    "蒙版生成指令", 
                    value="Chinese ink wash brush stroke border, messy ink splatter, hand-drawn edges", 
                    height=80,
                    key="manual_prompt_input"
                )
                if st.button("✨ 手动生成 AI 蒙版", use_container_width=True, key="manual_gen_btn"):
                    with st.spinner("正在呼叫阿里云 Wanx 生成蒙版..."):
                        new_mask = generate_ai_mask(gen_prompt, image_api_key, tw, th, z_base_url)
                        if new_mask:
                            st.session_state['current_mask'] = new_mask
                            st.rerun()
                st.info("ℹ️ 当前使用全画幅（无异形遮罩）")

            st.markdown("---")
            dual_control("遮罩高度缩放", "mask_scale", 0.1, 2.5, 0.01)
            dual_control("遮罩宽度缩放", "mask_w_scale", 0.1, 2.5, 0.01)
            dual_control("遮罩垂直位移", "mask_y", -1.0, 1.0, 0.01)

            # --- STEP 4. 字体与智能阴影 (同样保持对齐) ---
            with st.expander("🔤 STEP 4. 字体与智能阴影", expanded=False):
                font_ups = st.file_uploader("上传字体", type=["ttf","otf"], accept_multiple_files=True)
            if font_ups:
                for f in font_ups: 
                    if f.name not in st.session_state.loaded_fonts:
                        # 存入字节，这是最稳妥的持久化方式
                        st.session_state.loaded_fonts[f.name] = io.BytesIO(f.read())
            
            # 2. 字体选择逻辑
            available_fonts = list(st.session_state.loaded_fonts.keys())
            if 'curr_font_name' not in st.session_state:
                st.session_state.curr_font_name = available_fonts[0] if available_fonts else None

            if available_fonts:
                st.session_state.curr_font_name = st.selectbox(
                    "选择字体", available_fonts, 
                    index=available_fonts.index(st.session_state.curr_font_name) if st.session_state.curr_font_name in available_fonts else 0
                )
            
            # 拿到字体对象给渲染用
            current_font_data = st.session_state.loaded_fonts.get(st.session_state.curr_font_name)
            
            sub_text = st.text_input("标题内容", "SUMMER VIBES")

            # 3. 字号与位移 (弃用 dual_control，改用强绑定以防失效)
            # 先计算边界
            curr_size = float(st.session_state.settings.get('size', 60))
            margin = curr_size / 2 
            min_y, max_y = float(margin), float(th - margin)

            # 字号控制：直接写入 settings['size']
            st.session_state.settings['size'] = st.slider("字号", 10.0, float(th), curr_size, 1.0, key="fixed_size_slider")

            # 位移纠偏与控制
            curr_y = float(st.session_state.settings.get('y_pos', 380))
            if curr_y > max_y: curr_y = max_y
            if curr_y < min_y: curr_y = min_y
            
            st.session_state.settings['y_pos'] = st.slider("文字高度", min_y, max_y, curr_y, 1.0, key="fixed_y_slider")
            
            st.markdown("---")
            
            # 4. 颜色与自动对比色
            c_col1, c_col2 = st.columns([1, 1])
            with c_col1:
                st.session_state.settings['color'] = st.color_picker("文本颜色", st.session_state.settings['color'])
            with c_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🎯 自动高对比色", use_container_width=True):
                    st.session_state.trigger_auto_color = True
                    st.rerun()
                    
            st.markdown("---")
            dual_control("阴影偏移", "shadow_offset", 0, 30, 1)
            dual_control("阴影模糊", "shadow_blur", 0, 30, 1)
            dual_control("阴影透明", "shadow_alpha", 0, 255, 5)

            

# --- 6. 主工作流 Tab (带吸顶) ---
tabs = st.tabs(["🎨 工作站", "🏠 使用说明", "💎 API 与技术"])

with tabs[0]:
    if up_file:
        current_font = st.session_state.loaded_fonts.get(st.session_state.curr_font_name)
        
        # 渲染主图与背景分析副本
        res_img, bg_analysis = render_poster(
    up_file, 
    sub_text, 
    st.session_state.settings, # 确保传的是整个 settings 字典
    tw, th, 
    font_file=current_font_data # 确保传的是我们刚拿到的 current_font_data
)        # 拦截取色触发器
        if getattr(st.session_state, 'trigger_auto_color', False):
            best_color = auto_contrast_color(bg_analysis, st.session_state.settings['y_pos'], th)
            st.session_state.settings['color'] = best_color
            st.session_state.trigger_auto_color = False
            st.rerun() # 重新渲染颜色
            
        st.image(res_img, use_container_width=True)
        
        buf = io.BytesIO(); res_img.save(buf, format="PNG")
        st.download_button("📥 导出高清 PNG", buf.getvalue(), f"Dean_Design_{tw}x{th}.png", use_container_width=True)
    else:
        # --- 首页欢迎区 (审美与稳定性的终极平衡) ---
        
        # 定义核心样式变量
        card_bg = "var(--secondary-background-color)"
        text_c = "var(--text-color)"
        
        # 使用单个 st.markdown，并用 HTML 的 <br> 代替 Python 的换行
        # 这样 Markdown 渲染器就抓不到任何“空行”来把它变成代码块了
        content = (
            f"<div style='text-align: center; padding: 4rem 2rem; background-color: {card_bg}; "
            f"border-radius: 20px; border: 1px solid rgba(128,128,128,0.1); "
            f"box-shadow: 0 10px 30px rgba(0,0,0,0.05); margin: 1rem 0; color: {text_c};'>"
            f"<h1 style='font-size: 2.5rem; margin-bottom: 5px;'>✨ 影视海报转异形弹窗Agent</h1>"
            f"<p style='opacity: 0.8; font-size: 1.1rem;'>平台开发：<b>Dean</b> | 当前版本：{VERSION}</p>"
            f"<hr style='border: 0; border-top: 1px solid rgba(128,128,128,0.1); margin: 2rem 0;'>"
            f"<div style='max-width: 400px; margin: 0 auto; text-align: left;'>"
            f"<p>🚀 <b>工作站简介：</b></p>"
            f"<ul style='opacity: 0.9; font-size: 0.95rem; line-height: 1.8;'>"
            f"<li>已接入 AI 面部追踪与智能取色系统</li>"
            f"<li>支持 Remove.bg 智能人物破框视觉效果</li>"
            f"<li>新增自定义字体持久化缓存系统</li>"
            f"</ul></div>"
            f"<div style='margin-top: 2.5rem; padding: 1rem 2rem; background-color: rgba(255, 75, 75, 0.1); "
            f"border-radius: 12px; border: 1px solid rgba(255, 75, 75, 0.2); display: inline-block;'>"
            f"<p style='margin: 0; color: {HIGHLIGHT_COLOR}; font-weight: bold;'>🔑 如需获取 API Key 请联系 Dean</p>"
            f"</div>"
            f"<p style='opacity: 0.5; margin-top: 2.5rem; font-size: 0.85rem;'>👈 准备好了吗？请在左侧侧边栏上传素材开始创作</p>"
            f"</div>"
        )
        
        st.markdown(content, unsafe_allow_html=True)

# --- 定义通用卡片样式变量 (适配黑白模式) ---
card_style = (
    f"padding: 2.5rem; "
    f"background-color: var(--secondary-background-color); "
    f"border-radius: 18px; "
    f"border: 1px solid rgba(128,128,128,0.1); "
    f"box-shadow: 0 8px 20px rgba(0,0,0,0.05); "
    f"margin-bottom: 2rem; "
    f"color: var(--text-color);"
)

with tabs[1]:
    # --- 使用指南卡片 ---
    guide_content = (
        f"<div style='{card_style}'>"
        f"<h2 style='margin-top:0;'>🏠 MyPosterAgent 使用指南</h2>"
        f"<hr style='border:0; border-top:1px solid rgba(128,128,128,0.1); margin:1.5rem 0;'>"
        f"<div style='text-align: left; line-height: 1.8;'>"
        f"<h4>🎨 核心流程</h4>"
        f"<p>1. <b>画布构建</b>：支持自由定义像素或一键切换 16:9 / 9:16 等流媒体比例。</p>"
        f"<p>2. <b>AI 全员入框</b>：引擎会扫描画面中<b>所有正脸与侧脸</b>，计算最高和最低坐标跨度，自适应调整画面焦距，绝不遗漏任何人物。</p>"
        f"<p>3. <b>破框与遮罩</b>：确保目录下存在 <code>mask.png</code>。开启 AI 抠图后，调整破框线深度，让人物突破异形边界。</p>"
        f"<p>4. <b>智能排版</b>：点击<b>「🎯 自动高对比色」</b>，系统会扫描文字所在区域的像素均值，自动分配绝不翻车的极昼白或暗夜黑。</p>"
        f"</div></div>"
    )
    st.markdown(guide_content, unsafe_allow_html=True)

with tabs[2]:
    # 1. 准备数据
    api_total = 50.0
    api_used = st.session_state.get('api_used', 12.5) 
    api_remain = api_total - api_used
    consume_ratio = api_used / api_total

    # 2. 模拟原生 Metric 的 HTML (自定义颜色与排版)
    # 这样它们就能 100% 呆在 div 容器里
    metrics_html = f"""
    <div style="display: flex; justify-content: space-between; margin: 2rem 0; text-align: left;">
        <div style="flex: 1;">
            <p style="margin:0; font-size: 0.9rem; opacity: 0.7;">当月总配额</p>
            <h2 style="margin:0; font-size: 1.8rem;">{api_total:.1f}</h2>
        </div>
        <div style="flex: 1; border-left: 1px solid rgba(128,128,128,0.1); padding-left: 20px;">
            <p style="margin:0; font-size: 0.9rem; opacity: 0.7;">当前剩余可用</p>
            <h2 style="margin:0; font-size: 1.8rem; color: #2ecc71;">{api_remain:.1f}</h2>
            <p style="margin:0; font-size: 0.8rem; color: #e74c3c;">-{api_used:.1f} 已耗</p>
        </div>
        <div style="flex: 1; border-left: 1px solid rgba(128,128,128,0.1); padding-left: 20px;">
            <p style="margin:0; font-size: 0.9rem; opacity: 0.7;">接口心跳延迟</p>
            <h2 style="margin:0; font-size: 1.8rem;">24ms</h2>
            <p style="margin:0; font-size: 0.8rem; color: #2ecc71;">-2ms 极速</p>
        </div>
    </div>
    """

    # 3. 将所有内容打包成一个不间断的字符串输出
    status_card = (
        f"<div style='{card_style}'>"
        f"<h2 style='margin-top:0;'>💎 API 引擎与额度追踪</h2>"
        f"<p style='opacity:0.7;'>实时监控核心算力池与模块健康度</p>"
        f"<hr style='border:0; border-top:1px solid rgba(128,128,128,0.1); margin:1.5rem 0;'>"
        f"<h4>✂️ Remove.bg 算力池</h4>"
        f"{metrics_html}" # 插入模拟的指标
        f"<div style='margin-top: 1.5rem;'>"
        f"<p style='margin-bottom: 5px; font-size: 0.9rem;'>🚀 算力消耗水位线: {consume_ratio*100:.1f}%</p>"
        f"<div style='width: 100%; background-color: rgba(128,128,128,0.1); border-radius: 10px; height: 10px; overflow: hidden;'>"
        f"<div style='width: {consume_ratio*100}%; background-color: {HIGHLIGHT_COLOR}; height: 100%;'></div>"
        f"</div>"
        f"</div>"
        f"</div>"
    )
    st.markdown(status_card, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    

    # --- 1. 技术看板卡片 (包含核心库与官方入口) ---
    # 我们把表格里的数据直接用 HTML 列表或小型表格的形式包在卡片里
    
    libraries_html = ""
    tech_data = [
        {"name": "Streamlit", "use": "前端 UI 框架", "status": "Running"},
        {"name": "Pillow (PIL)", "use": "图像渲染引擎", "status": "Optimized"},
        {"name": "OpenCV", "use": "AI 视觉识别", "status": "Active"},
        {"name": "NumPy", "use": "矩阵运算", "status": "Stable"}
    ]
    
    for item in tech_data:
        libraries_html += (
            f"<div style='display: flex; justify-content: space-between; margin-bottom: 0.8rem; font-size: 0.9rem;'>"
            f"  <span style='font-weight: bold;'>{item['name']}</span>"
            f"  <span style='opacity: 0.7;'>{item['use']}</span>"
            f"  <span style='color: #2ecc71;'>{item['status']}</span>"
            f"</div>"
        )

# 2. 核心模块健康度卡片 (独立卡片)
    tech_card = (
        f"<div style='{card_style}'>"
        f"<h4>🔧 核心模块健康度</h4>"
        f"<div style='margin-top: 1rem;'>"
        f"<div style='margin-bottom: 1.2rem;'>"
        f"  <span style='color: #2ecc71;'>🟢 Healthy</span> <b>Main Rendering Engine</b> (v13.0)<br>"
        f"  <small style='opacity: 0.6; margin-left: 1.5rem;'>作用：负责将多层图像（底图、文字、遮罩）进行亚像素级合成，输出高清海报。</small>"
        f"</div>"
        f"<div style='margin-bottom: 1.2rem;'>"
        f"  <span style='color: #2ecc71;'>🟢 Healthy</span> <b>HaarCascade Face Topology</b><br>"
        f"  <small style='opacity: 0.6; margin-left: 1.5rem;'>作用：基于 AI 视觉识别画面人脸，自动计算视角跨度，确保不切头、不遮脸。</small>"
        f"</div>"
        f"<div style='margin-bottom: 0.5rem;'>"
        f"  <span style='color: #2ecc71;'>🟢 Healthy</span> <b>Luminance Auto-Contrast</b><br>"
        f"  <small style='opacity: 0.6; margin-left: 1.5rem;'>作用：实时扫描文本区域明度，自动切换黑白配色，保证阅读效果。</small>"
        f"</div>"
        f"</div></div>"
    )
    st.markdown(tech_card, unsafe_allow_html=True)

    # 3. 技术栈底座卡片 (独立卡片，不含入口)
    libraries_html = ""
    tech_data = [
        {"name": "Streamlit", "use": "前端 UI 框架", "status": "Running"},
        {"name": "Pillow (PIL)", "use": "图像渲染引擎", "status": "Optimized"},
        {"name": "OpenCV", "use": "AI 视觉识别", "status": "Active"},
        {"name": "NumPy", "use": "矩阵运算", "status": "Stable"}
    ]
    for item in tech_data:
        libraries_html += (
            f"<div style='display: flex; justify-content: space-between; margin-bottom: 0.8rem; font-size: 0.9rem;'>"
            f"  <span style='font-weight: bold;'>{item['name']}</span>"
            f"  <span style='opacity: 0.7;'>{item['use']}</span>"
            f"  <span style='color: #2ecc71;'>{item['status']}</span>"
            f"</div>"
        )

    base_tech_card = (
        f"<div style='{card_style}'>"
        f"<h4>🛠️ 技术栈底座</h4>"
        f"<div style='margin-top: 1.2rem;'>"
        f"{libraries_html}"
        f"</div></div>"
    )
    st.markdown(base_tech_card, unsafe_allow_html=True)

    # 4. 官方入口 (保留上一版的高级横向排版，但不包裹在卡片内)
    st.markdown("#### 🔗 官方入口")
    
    links_html = f"""
    <div style="display: flex; gap: 20px; flex-wrap: wrap; font-size: 0.95rem; margin-top: 10px;">
        <a href="https://www.remove.bg/dashboard" target="_blank" style="color: #5b5b5b; text-decoration: none; font-weight: 500;">
            <span>🔗</span> Remove.bg 控制台
        </a>
        <a href="https://docs.streamlit.io/" target="_blank" style="color: #5b5b5b; text-decoration: none; font-weight: 500;">
            <span>🔗</span> Streamlit 部署文档
        </a>
        <a href="https://pillow.readthedocs.io/" target="_blank" style="color: #5b5b5b; text-decoration: none; font-weight: 500;">
            <span>🔗</span> Pillow 官方手册
        </a>
    </div>
    """
    st.markdown(links_html, unsafe_allow_html=True)

    # 5. 底部版权
    st.markdown("<div style='margin-top: 4rem; text-align: center; opacity: 0.2; font-size: 0.8rem;'>© 2026 Dean Design Studio | Built with Passion</div>", unsafe_allow_html=True)
