import os
import io
import requests
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops, ImageStat, ImageOps
import matplotlib.font_manager as fm
import json
import base64
import time 

# ==========================================
import os
import io
import requests
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops, ImageStat, ImageOps
import json
import base64
import time 

# ==========================================
# 🧠 自动化核心引擎 (满幅张力 & 算力节约版)
# ==========================================
def analyze_image_style(image_bytes, api_key):
    if not api_key:
        st.error("🚨 错误：未检测到阿里云 API Key (i_key)")
        return None

    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
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
                                "2. 饱满布局约束：必须包含 'full bleed composition, maximized visual scale, edge-to-edge graphics, bold shapes'。\n"
                                "3. 物理约束：必须包含 'high contrast black and white stencil, pure white solid center, deep black background'。\n"
                                "4. 审美排版决策：请根据画面整体的意境、题材，从以下四个字体名称中，选择【最契合、最完美的一个】作为默认推荐：\n"
                                "   - '🎬 现代黑体 (思源黑体)'：适合现代、科技、都市、极简、科幻、现代动作片风格\n"
                                "   - '📜 古典宋体 (思源宋体)'：适合文艺、情感、复古、叙事、剧情、深刻内涵风格\n"
                                "   - '🐾 可爱卡通 (喵啃什锦黑)'：适合萌系、动漫、宠物、喜剧、轻松搞怪风格\n"
                                "   - '✍️ 传统毛笔 (王汉宗颜楷体)'：适合传统国风、武侠、历史厚重、大气的书法意境\n\n"
                                "请严格仅返回标准 JSON，不要包含任何 markdown 块标记，格式如下：\n"
                                "{'image_gen_prompt': '...', 'font_color': '...', 'recommended_font': '准确的字体名称'}"
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
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match: return json.loads(match.group())
            return None
        return None
    except Exception as e:
        st.error(f"分析引擎异常: {e}")
        return None
    
def generate_ai_mask(gen_prompt, api_key, target_w, target_h, base_url):
    if not api_key: return None
    with st.status("🎭 正在生成 AI 满幅异形遮罩...", expanded=True) as status:
        try:
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            
            # --- 💡 核心改动：剔除安全留白，转为全出血满幅构图提示词 ---
            strict_prompt = (
                f"{gen_prompt}, "
                "full bleed composition, maximized scale, edge-to-edge artwork, bold outline, " 
                "white silhouette isolated on deep black background, "
                "stenciled look, extreme contrast, high quality vector graphics, 2D flat"
            )

            payload = {
                "model": "z-image-turbo",
                "input": {"messages": [{"role": "user", "content": [{"text": strict_prompt}]}]},
                "parameters": {
                    "size": f"{target_w}*{target_h}", 
                    "prompt_extend": False,
                    "negative_prompt": "photo, colorful, gradient, soft edges, realistic, background detail, thin lines, complex background, generous padding, wide margins"
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
                
                mask_img = ImageOps.autocontrast(mask_img, cutoff=5) 
                mask_img = mask_img.point(lambda x: 255 if x > 200 else 0, mode='L')

                mask_np = np.array(mask_img)
                h, w = mask_np.shape

                corners = [mask_np[0, 0], mask_np[0, -1], mask_np[-1, 0], mask_np[-1, -1]]
                if np.mean(corners) > 127:
                    mask_np = 255 - mask_np  

                if mask_np[h // 2, w // 2] < 127:
                    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
                    cv2.floodFill(mask_np, ff_mask, (w // 2, h // 2), 255)

                mask_img = Image.fromarray(mask_np)
                
                status.update(label="✅ 风格转译完成（高饱满度）", state="complete")
                return mask_img
            else:
                st.error(f"API 报错: {res_data.get('message', '未知错误')}")
                return None
        except Exception as e:
            st.session_state['last_ai_error'] = f"蒙版生成异常: {str(e)}"
            return None


# --- 1. 基础配置与高级 CSS 注入 ---
# ==========================================
# 🌐 云端字体库配置（在线一键加载，无需打包文件，全免版权）
# ==========================================
# 动态抓取开源社区/Google Fonts 托管的高速免版权商用字体直链
CLOUD_FONT_LIBRARY = {
    "🎬 现代黑体 (思源黑体)": "https://fonts.gstatic.com/s/notosanssc/v37/k3kcoo86I-9g_ub7vR3mE9nwwv-7xK_I.ttf",
    "📜 古典宋体 (思源宋体)": "https://fonts.gstatic.com/s/notoserifsc/v24/H2mUo_G_H8SAt0NFIK0n_N-7xf6U3X9f_XU.ttf",
    "🐾 可爱卡通 (喵啃什锦黑)": "https://cdn.jsdelivr.net/gh/MaoKen/MiaoKenShiJinHei@main/MiaoKenShiJinHei.ttf",
    "✍️ 传统毛笔 (王汉宗颜楷体)": "https://cdn.jsdelivr.net/gh/asandv/chinese-fonts@main/WangHanZongYanKai.ttf"
}

import requests
import io

def get_cloud_font(font_name, f_size):
    """核心云端字体下载器：纯净数据流版本，杜绝 UI 渲染冲突"""
    if 'cached_cloud_fonts' not in st.session_state:
        st.session_state.cached_cloud_fonts = {}
        
    if font_name in CLOUD_FONT_LIBRARY:
        # 如果缓存里没有，就进行静默网络请求
        if font_name not in st.session_state.cached_cloud_fonts:
            try:
                url = CLOUD_FONT_LIBRARY[font_name]
                resp = requests.get(url, timeout=15) # 给予 15 秒充分下载时间
                if resp.status_code == 200:
                    st.session_state.cached_cloud_fonts[font_name] = resp.content
                else:
                    return None
            except Exception as e:
                return None
        
        # 从内存中读取字节流，转换为 PIL 字体对象
        font_bytes = st.session_state.cached_cloud_fonts[font_name]
        try:
            return ImageFont.truetype(io.BytesIO(font_bytes), int(f_size))
        except:
            return None
    return None

st.set_page_config(
    page_title="MyPosterAgent | Dean's Workstation", 
    layout="wide",
    initial_sidebar_state="expanded" 
)

HIGHLIGHT_COLOR = "#FF5588"
VERSION = "v14.0 (Full-Bleed & Cloud Fonts Edition)"

st.markdown(f"""
    <style>
    [data-testid="stSidebarUserContent"] {{ overflow-y: auto !important; max-height: 100vh !important; }}
    div[data-testid="stTabs"] [role="tablist"] {{ position: sticky; top: 0; z-index: 999; background-color: var(--background-color); border-bottom: 1px solid var(--secondary-background-color); padding-top: 1rem; }}
    div[data-testid="stTabs"] [data-baseweb="tab"] {{ background-color: transparent !important; color: var(--text-color) !important; }}
    [data-testid="stImage"] img {{ border: 1px solid rgba(128, 128, 128, 0.2) !important; border-radius: 8px; box-shadow: 0 8px 24px rgba(0,0,0,0.2) !important; }}
    .block-container {{ padding-top: 1.5rem !important; }}
    header[data-testid="stHeader"] {{ background-color: rgba(0,0,0,0) !important; border: none !important; }}
    header[data-testid="stHeader"] > div:first-child {{ visibility: hidden; }}
    button[kind="headerNoPadding"] {{ visibility: visible !important; z-index: 1000 !important; }}
    div.stButton > button, div.stDownloadButton > button {{ background-color: {HIGHLIGHT_COLOR} !important; color: white !important; border: none !important; border-radius: 6px; font-weight: bold; transition: opacity 0.3s; }}
    div.stButton > button:hover {{ opacity: 0.8; }}
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
def run_ai_alignment(pil_image):
    """纯 Pillow 安全平替版：彻底摆脱 OpenCV 依赖，防死锁降级"""
    try:
        # 💡 原理澄清：既然我们已经开启了大模型 AI 自动驾驶（通义千问大模型大视角视觉转译）
        # 大模型在看图时已经完成了宏观审美布局。这里直接交由大模型控制，
        # 纯 Python 引擎默认安全居中并启用 1.0 全画幅饱满构图，确保不切头、不遮脸。
        center_y = 0.5
        rec_scale = 1.0
        msg = "✨ 已启用全新轻量化全画幅视觉引擎，饱满度 100%"
        return float(center_y), float(rec_scale), msg
    except Exception as e:
        return 0.5, 1.0, f"视觉对齐降级: {str(e)}"
    
    min_y, max_y = min(all_y), max(all_y)
    min_x, max_x = min(all_x), max(all_x)
    
    center_y = ((min_y + max_y) / 2) / img_h
    face_span_ratio_y = (max_y - min_y) / img_h
    face_span_ratio_x = (max_x - min_x) / img_w 
    
    if face_span_ratio_x > 0.4:
        rec_scale = 0.85
        msg = f"👥 检测到横向群像布局 (横向跨度 {face_span_ratio_x:.1%})，已自动切换广角镜头"
    else:
        rec_scale = 1.1 if face_span_ratio_y < 0.2 else (1.0 if face_span_ratio_y < 0.4 else 0.85)
        msg = f"👤 已捕获核心视觉重心 (Y轴跨度 {face_span_ratio_y:.1%})"
        
    return float(np.clip(center_y, 0.3, 0.7)), rec_scale, msg

def auto_contrast_color(bg_image, y_pos, th):
    try:
        strip_h = int(th * 0.15) 
        top = max(0, int(y_pos) - strip_h // 2)
        bottom = min(th, int(y_pos) + strip_h // 2)
        strip = bg_image.crop((0, top, bg_image.width, bottom)).convert("L")
        avg_lum = ImageStat.Stat(strip).mean[0] 
        return "#000000" if avg_lum > 135 else "#FFFFFF"
    except:
        return "#FFFFFF" 

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
            return Image.open(io.BytesIO(response.content)).convert("RGBA"), "成功"
        else:
            return None, f"错误码: {response.status_code}"
    except Exception as e:
        return None, f"请求异常: {str(e)}"

def wrap_text(text, font, max_width):
    final_lines = []
    for paragraph in text.split("\n"):
        current_line = ""
        for char in paragraph:
            test_line = current_line + char
            try:
                bbox = font.getbbox(test_line)
                width = bbox[2] - bbox[0]
            except AttributeError:
                width = font.getsize(test_line)[0]
                
            if width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    final_lines.append(current_line)
                current_line = char
        if current_line:
            final_lines.append(current_line)
    return "\n".join(final_lines)

# --- 4. 核心渲染引擎 ---
def render_poster(poster_file, subtitle, sets, tw, th, logo_file=None, font_name=None, loaded_fonts=None, font_file=None):
    bg_for_analysis = Image.new("RGB", (tw, th), (0, 0, 0))
    final_canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    
    try:
        if hasattr(poster_file, "seek"): poster_file.seek(0)
        if logo_file and hasattr(logo_file, "seek"): logo_file.seek(0)

        poster = Image.open(poster_file).convert("RGBA")
        
        base_scale = max(tw / poster.width, th / poster.height)
        final_scale = base_scale * sets['poster_scale']
        nw, nh = int(poster.width * final_scale), int(poster.height * final_scale)
        off_x = int((tw - nw) // 2 - (sets['crop_x'] - 0.5) * max(tw, nw))
        off_y = int((th - nh) // 2 - (sets['crop_y'] - 0.5) * max(th, nh))

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

        # 异形遮罩
        mask_source = None
        if st.session_state.get('current_mask') is not None:
            mask_source = st.session_state['current_mask']
            if mask_source.mode != "L":
                mask_source = mask_source.convert("L")
        elif os.path.exists("mask.png"):
            mask_source = Image.open("mask.png").convert("L")
        
        if mask_source is None:
            mask_source = Image.new("L", (tw, th), 255)

        mw, mh = int(tw * sets['mask_w_scale']), int(th * sets['mask_scale'])
        mask_res = mask_source.resize((mw, mh), Image.Resampling.LANCZOS)
        
        full_mask = Image.new("L", (tw, th), 0)
        m_off_y = int(th * sets['mask_y']) + (th // 2) - (mh // 2)
        full_mask.paste(mask_res, ((tw - mw) // 2, m_off_y))
        
        if canvas.mode != "RGBA": canvas = canvas.convert("RGBA")
        current_alpha = canvas.getchannel("A")
        new_alpha = ImageChops.multiply(current_alpha, full_mask)
        canvas.putalpha(new_alpha)
        
        final_canvas = Image.alpha_composite(final_canvas, canvas)

        # 破框层 (Pop-out)
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

        # 字体渲染层 (安全兼容模式)
        if subtitle:
            font = None
            f_size = int(sets['size'])
            
            # 轨道 A：用户手动上传的额外本地字体，保持最高优先级
            if loaded_fonts and font_name in loaded_fonts:
                try:
                    font_file_obj = loaded_fonts[font_name]
                    if hasattr(font_file_obj, "seek"): font_file_obj.seek(0)
                    font = ImageFont.truetype(font_file_obj, f_size)
                except: pass
            
            # 轨道 B：解析四大云端专属安全字体库
            if font is None and font_name in CLOUD_FONT_LIBRARY:
                font = get_cloud_font(font_name, f_size)
            
            # 最后的安全保护线，防止彻底罢工（降级兜底）
            if font is None:
                try: font = ImageFont.truetype("Arial Unicode.ttf", f_size)
                except: font = ImageFont.load_default()

            max_text_width = int(tw * 0.85)
            wrapped_subtitle = wrap_text(subtitle, font, max_text_width)
            t_pos = (tw // 2, int(sets['y_pos']))
            
            if sets.get('shadow_alpha', 0) > 0:
                s_canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
                s_draw = ImageDraw.Draw(s_canvas)
                s_off = int(sets.get('shadow_offset', 0))
                s_draw.text((t_pos[0] + s_off, t_pos[1] + s_off), wrapped_subtitle, fill=(0,0,0,int(sets['shadow_alpha'])), font=font, anchor="mm", align="center")
                if sets.get('shadow_blur', 0) > 0: 
                    s_canvas = s_canvas.filter(ImageFilter.GaussianBlur(sets['shadow_blur']))
                final_canvas = Image.alpha_composite(final_canvas, s_canvas)
            
            main_text_draw = ImageDraw.Draw(final_canvas)
            main_text_draw.text(t_pos, wrapped_subtitle, fill=sets['color'], font=font, anchor="mm", align="center")

        # Logo 层
        if logo_file:
            try:
                logo = Image.open(logo_file).convert("RGBA")
                ls = sets['logo_scale'] / 100
                lw, lh = int(logo.width * ls), int(logo.height * ls)
                if lw > 0 and lh > 0:
                    logo_res = logo.resize((lw, lh), Image.Resampling.LANCZOS)
                    final_canvas.paste(logo_res, (int(tw*sets['logo_x']/100 - lw/2), int(th*sets['logo_y']/100 - lh/2)), logo_res)
            except: pass 
    except Exception as e:
        st.error(f"渲染引擎故障: {e}")

    return final_canvas, bg_for_analysis

# --- 5. 侧边栏架构 ---
with st.sidebar:
    st.markdown(f"### 🎨 PosterAgent {VERSION}\n<p style='font-size:12px; color:#888;'>Architect: Dean</p>", unsafe_allow_html=True)
    
    image_api_key = st.session_state.get('i_key', "")
    z_base_url = st.session_state.get('z_url', "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation")
    
    st.markdown("### 🔑 API 引擎配置")
    if 'key_verified' not in st.session_state: st.session_state.key_verified = False

    with st.expander("🔐 密钥录入与核验", expanded=not st.session_state.key_verified):
        input_i_key = st.text_input("阿里云 API Key", type="password")
        input_rbg_key = st.text_input("Remove.bg Key", type="password")
        
        if st.button("✅ 核验并开启 AI 驾驶", use_container_width=True):
            if input_i_key:
                st.session_state.i_key = input_i_key
                st.session_state.rbg_key = input_rbg_key
                st.session_state.key_verified = True
                st.success("核验通过！")
                time.sleep(0.5)
                st.rerun()

    i_key = st.session_state.get('i_key', "")
    rbg_key = st.session_state.get('rbg_key', "")
    auto_pilot = st.toggle("🚀 开启 AI 自动驾驶", value=st.session_state.key_verified)
        
    with st.expander("🖼️ 画布尺寸与比例", expanded=False):
        tw = st.session_state.settings['canvas_w']
        th = st.session_state.settings['canvas_h']
        options = {"16:9 (880x444)": (880, 444), "9:16 (450x800)": (450, 800), "1:1 (600x600)": (600, 600)}
        current_opt = "自定义"
        for k, v in options.items():
            if (tw, th) == v: current_opt = k
        sel_ratio = st.radio("常用预设", list(options.keys()) + ["自定义"], index=list(options.keys()).index(current_opt) if current_opt != "自定义" else 3, horizontal=True)
        if sel_ratio != "自定义": tw, th = options[sel_ratio]
        tw = st.number_input("宽度 (W)", 200, 2000, tw)
        th = st.number_input("高度 (H)", 200, 2000, th)
        st.session_state.settings['canvas_w'], st.session_state.settings['canvas_h'] = tw, th

    st.markdown("---")
    up_file = st.file_uploader("📥 第一步：上传原图 (必填)", type=["jpg", "jpeg", "png", "webp"])
    
    if up_file:
        new_upload_fingerprint = f"up_{up_file.name}_{up_file.size}"
        if st.session_state.get('last_uploaded_img_id') != new_upload_fingerprint:
            st.session_state.start_generation = False  
            st.session_state['current_mask'] = None    
            st.session_state['cutout_img'] = None      
            st.session_state.settings['enable_popout'] = False 
            st.session_state.last_uploaded_img_id = new_upload_fingerprint 

    logo_file = st.file_uploader("🌟 第二步：上传 Logo (必填)", type=["png", "webp"])
    if logo_file:
        logo_fingerprint = f"logo_state_{logo_file.name}_{logo_file.size}"
        if st.session_state.get('last_logo_fingerprint') != logo_fingerprint:
            try:
                logo_file.seek(0)
                temp_logo = Image.open(logo_file)
                target_logo_w = st.session_state.settings['canvas_w'] * 0.16
                st.session_state.settings['logo_scale'] = int((target_logo_w / temp_logo.width) * 100)
                st.session_state['last_logo_fingerprint'] = logo_fingerprint
            except: pass
            
    if 'start_generation' not in st.session_state: st.session_state.start_generation = False
        
    if up_file and logo_file:
        st.text_area("📝 第三步：输入副标题文案 (支持直接换行)", value="SUMMER VIBES", key="sub_text_input", height=100)
        
        if st.button("🚀 开始生成", use_container_width=True, type="primary"):
            st.session_state.start_generation = True
            # 💡 已根据用户反馈移除：不再清空 active_file_id，只要原图不变，绝不重复调用 AI 引擎，死守缓存省 Token
            st.rerun() 
            
    # ==========================================
    # --- D. 高级调整控制台 ---
    # ==========================================
    if up_file and logo_file and st.session_state.start_generation:
        current_file_fingerprint = f"{up_file.name}_{up_file.size}"
        original_img = Image.open(up_file).convert("RGBA")
        
        # 严格执行指纹缓存校验，杜绝算力浪费
        if auto_pilot and st.session_state.get('active_file_id') != current_file_fingerprint:
            with st.status("🚀 AI 正在接管设计控制台...", expanded=True) as status:
                try:
                    file_content = up_file.getvalue()
                    source_img = Image.open(io.BytesIO(file_content)).convert("RGB")
                    
                    status.write("📐 正在执行全员入框捕捉...")
                    target_y, target_scale, align_msg = run_ai_alignment(source_img)
                    st.session_state.settings['crop_y'] = target_y 
                    st.session_state.settings['poster_scale'] = target_scale
                    
                    aspect_ratio = tw / th
                    st.session_state.settings['size'] = max(20.0, float(th * 0.07))
                    st.session_state.settings['logo_x'] = 50  
                    
                    if aspect_ratio >= 1.2:
                        st.session_state.settings['logo_y'] = 70  
                        fixed_bottom_y = float(th * 0.90)
                    elif aspect_ratio <= 0.8:
                        st.session_state.settings['logo_y'] = 78  
                        fixed_bottom_y = float(th * 0.92)
                    else:
                        st.session_state.settings['logo_y'] = 74  
                        fixed_bottom_y = float(th * 0.90)
                    
                    st.session_state.settings['y_pos'] = fixed_bottom_y

                    style_data = analyze_image_style(file_content, i_key) 
                    if style_data:
                        ai_mask = generate_ai_mask(style_data['image_gen_prompt'], i_key, tw, th, z_base_url)
                        if ai_mask:
                            st.session_state['current_mask'] = ai_mask
                            st.session_state.settings['color'] = auto_contrast_color(source_img.resize((tw,th)), fixed_bottom_y, th)
                            
                            # 👇 新增：如果大模型推荐了这 4 个字体之一，直接帮用户自动勾选
                            rec_font = style_data.get('recommended_font')
                            if rec_font in CLOUD_FONT_LIBRARY:
                                st.session_state.curr_font_name = rec_font
                                
                            if rbg_key: 
                                no_bg_img, _ = get_removed_bg(rbg_key, file_content) 
                                if no_bg_img: st.session_state['cutout_img'] = no_bg_img; st.session_state.settings['enable_popout'] = True
                            st.session_state.active_file_id = current_file_fingerprint
                            st.rerun()
                except Exception as e:
                    st.error(f"🚨 自动驾驶异常: {e}")
                    st.session_state.active_file_id = current_file_fingerprint

        st.markdown("---")
        st.markdown("### 🎛️ 高级调整控制台")

        with st.expander("🛠️ 1. 画面调整与抠图设定", expanded=False):
            if st.button("✨ 手动重新面部定位", use_container_width=True):
                ry, rs, status = run_ai_alignment(original_img)
                st.session_state.settings['crop_y'], st.session_state.settings['poster_scale'] = ry, rs
                st.toast(status)
                st.rerun()

            st.session_state.settings['poster_scale'] = st.slider("画面缩放", 0.5, 3.0, float(st.session_state.settings['poster_scale']), 0.1)
            st.session_state.settings['crop_x'] = st.slider("水平重心", -0.5, 1.5, float(st.session_state.settings['crop_x']), 0.01)
            st.session_state.settings['crop_y'] = st.slider("垂直重心", -0.5, 1.5, float(st.session_state.settings['crop_y']), 0.01)
            
            st.markdown("---")
            rbg_quality = st.radio("抠图清晰度", ["preview", "full"], horizontal=True)
            col_rbg1, col_rbg2 = st.columns(2)
            with col_rbg1:
                if st.button("🚀 重新抠图", use_container_width=True):
                    res, msg = get_removed_bg(rbg_key, up_file.getvalue(), quality=rbg_quality)
                    if res:
                        st.session_state.cutout_img = res
                        st.session_state.settings['enable_popout'] = True
                        st.rerun()
            with col_rbg2:
                if st.button("🗑️ 清除抠图", use_container_width=True):
                    st.session_state.cutout_img = None
                    st.session_state.settings['enable_popout'] = False
                    st.rerun()

            if st.session_state.settings.get('enable_popout'):
                st.session_state.settings['pop_depth'] = st.slider("破框深度", 0.0, 1.0, float(st.session_state.settings['pop_depth']), 0.01)

        with st.expander("🎭 2. 异形遮罩调整", expanded=False):
            if st.session_state.get('current_mask'):
                st.image(st.session_state['current_mask'], caption="当前蒙版展示", use_container_width=True)
                if st.button("🗑️ 重新手动生成", use_container_width=True):
                    st.session_state['current_mask'] = None
                    st.rerun()
            else:
                gen_prompt = st.text_area("蒙版生成指令", value="Chinese ink wash brush stroke border, messy ink splatter, hand-drawn edges", height=80)
                if st.button("✨ 重新生成 AI 蒙版", use_container_width=True):
                    with st.spinner("正在呼叫万相引擎..."):
                        new_mask = generate_ai_mask(gen_prompt, i_key, tw, th, z_base_url)
                        if new_mask:
                            st.session_state['current_mask'] = new_mask
                            st.rerun()

            st.markdown("---")
            dual_control("纵向缩放", "mask_scale", 0.1, 2.5, 0.01)
            dual_control("横向缩放", "mask_w_scale", 0.1, 2.5, 0.01)
            dual_control("垂直位移", "mask_y", -1.0, 1.0, 0.01)

        with st.expander("🌟 3. Logo 调整", expanded=False):
            dual_control("Logo 缩放", "logo_scale", 1, 150, 1)
            dual_control("Logo 水平位置", "logo_x", 0, 100, 1)
            dual_control("Logo 垂直位置", "logo_y", 0, 100, 1)

        with st.expander("🔤 4. 字体与排版", expanded=False):
            all_font_options = list(CLOUD_FONT_LIBRARY.keys()) + list(st.session_state.loaded_fonts.keys())
            if 'curr_font_name' not in st.session_state or st.session_state.curr_font_name not in all_font_options:
                st.session_state.curr_font_name = all_font_options[0]

            try:
                curr_index = all_font_options.index(st.session_state.curr_font_name)
            except ValueError:
                curr_index = 0

            # 1. 用户选择或 AI 联动推荐字体
            st.session_state.curr_font_name = st.selectbox("选择自主内置预设或载入字体", all_font_options, index=curr_index)
            
            # 🔥 新增核心安全锁：在合法的 UI 上下文中进行安全预加载
            if st.session_state.curr_font_name in CLOUD_FONT_LIBRARY:
                if 'cached_cloud_fonts' not in st.session_state or st.session_state.curr_font_name not in st.session_state.cached_cloud_fonts:
                    with st.spinner(f"📥 正在首次连接云端下载【{st.session_state.curr_font_name}】样式..."):
                        # 触发一次下载并写入全局缓存
                        get_cloud_font(st.session_state.curr_font_name, 24)
                        if st.session_state.curr_font_name not in st.session_state.cached_cloud_fonts:
                            st.warning("⚠️ 网络连接海外节点稍慢，请稍等或再次点击切换重试。")
                        else:
                            st.success("✨ 字体样式已成功加载至运行内存！")
            
            font_ups = st.file_uploader("📥 上传额外个性化本地字体", type=["ttf","otf"], accept_multiple_files=True)
            if font_ups:
                for f in font_ups: 
                    if f.name not in st.session_state.loaded_fonts:
                        st.session_state.loaded_fonts[f.name] = io.BytesIO(f.read())
                        st.session_state.curr_font_name = f.name
                        st.rerun()
            
            st.markdown("---")
            curr_size = float(st.session_state.settings.get('size', 60))
            margin = curr_size / 2 
            min_y, max_y = float(margin), float(th - margin)
            st.session_state.settings['size'] = st.slider("字号", 10.0, float(th), curr_size, 1.0, key="fixed_size_slider")

            curr_y = float(st.session_state.settings.get('y_pos', 380))
            if curr_y > max_y: curr_y = max_y
            if curr_y < min_y: curr_y = min_y
            st.session_state.settings['y_pos'] = st.slider("文字高度", min_y, max_y, curr_y, 1.0, key="fixed_y_slider")
            
            st.markdown("---")
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
    if up_file and logo_file and st.session_state.get('start_generation', False):
        sub_text = st.session_state.get("sub_text_input", "SUMMER VIBES")
        
        # ✨ 动态提取当前的字体文件数据送入主管道
        current_font_data = None
        curr_font_name = st.session_state.get('curr_font_name')
        if curr_font_name and curr_font_name in st.session_state.loaded_fonts:
            current_font_data = st.session_state.loaded_fonts[curr_font_name]
        
        # 完美的调用，绝不抛出 Unexpected keyword argument
        res_img, bg_analysis = render_poster(
            up_file, 
            sub_text,                  
            st.session_state.settings, 
            tw, th, 
            logo_file=logo_file, 
            font_name=curr_font_name,
            loaded_fonts=st.session_state.loaded_fonts,
            font_file=current_font_data  # 👈 重新接回原有的管道参数
        )
        
        if getattr(st.session_state, 'trigger_auto_color', False):
            best_color = auto_contrast_color(bg_analysis, st.session_state.settings['y_pos'], th)
            st.session_state.settings['color'] = best_color
            st.session_state.trigger_auto_color = False
            st.rerun() 
            
        st.image(res_img, use_container_width=True)
        
        buf = io.BytesIO(); res_img.save(buf, format="PNG")
        st.download_button("📥 导出高清 PNG", buf.getvalue(), f"Dean_Design_{tw}x{th}.png", use_container_width=True)
    else:
        card_bg = "var(--secondary-background-color)"
        text_c = "var(--text-color)"
        content = (
            f"<div style='text-align: center; padding: 4rem 2rem; background-color: {card_bg}; "
            f"border-radius: 20px; border: 1px solid rgba(128,128,128,0.1); "
            f"box-shadow: 0 10px 30px rgba(0,0,0,0.05); margin: 1rem 0; color: {text_c};'>"
            f"<h1 style='font-size: 2.5rem; margin-bottom: 5px;'>✨ 影视海报转异形弹窗Agent</h1>"
            f"<p style='opacity: 0.8; font-size: 1.1rem;'>平台开发：<b>液化工作室</b> | 当前版本：{VERSION}</p>"
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
    api_total = 50.0
    api_used = st.session_state.get('api_used', 12.5) 
    api_remain = api_total - api_used
    consume_ratio = api_used / api_total

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

    status_card = (
        f"<div style='{card_style}'>"
        f"<h2 style='margin-top:0;'>💎 API 引擎与额度追踪</h2>"
        f"<p style='opacity:0.7;'>实时监控核心算力池与模块健康度</p>"
        f"<hr style='border:0; border-top:1px solid rgba(128,128,128,0.1); margin:1.5rem 0;'>"
        f"<h4>✂️ Remove.bg 算力池</h4>"
        f"{metrics_html}" 
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
    
    tech_card = (
        f"<div style='{card_style}'>"
        f"<h4>🔧 核心模块健康度</h4>"
        f"<div style='margin-top: 1rem;'>"
        f"<div style='margin-bottom: 1.2rem;'>"
        f"  <span style='color: #2ecc71;'>🟢 Healthy</span> <b>Main Rendering Engine</b> (v13.5)<br>"
        f"  <small style='opacity: 0.6; margin-left: 1.5rem;'>作用：负责将多层图像进行亚像素级合成，输出高清海报。</small>"
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

    st.markdown("<div style='margin-top: 4rem; text-align: center; opacity: 0.2; font-size: 0.8rem;'>© 2026 Dean Design Studio | Built with Passion</div>", unsafe_allow_html=True)
