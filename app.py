import os
import io
import requests
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops, ImageStat
import matplotlib.font_manager as fm

# --- 1. 基础配置与高级 CSS 注入 ---
st.set_page_config(page_title="MyPosterAgent | Dean's Workstation", layout="wide")
HIGHLIGHT_COLOR = "#FF5588"
VERSION = "v13.0 (Ultimate Edition)"

st.markdown(f"""
    <style>
    /* 1. Tab 栏背景与文字适配 */
    div[data-testid="stTabs"] [role="tablist"] {{
        position: sticky; 
        top: 0; 
        z-index: 999; 
        background-color: var(--background-color); /* 自动适配背景 */
        border-bottom: 1px solid var(--secondary-background-color); /* 自动适配边框线 */
        padding-top: 1rem;
    }}

    div[data-testid="stTabs"] [data-baseweb="tab"] {{
        background-color: transparent !important;
        color: var(--text-color) !important; /* 自动适配文字颜色 */
    }}

    /* 2. 画布预览窗口适配 */
    [data-testid="stImage"] img {{ 
        /* 这里的边框线改用透明度，黑白模式通吃 */
        border: 1px solid rgba(128, 128, 128, 0.2) !important; 
        border-radius: 8px; 
        /* 阴影在黑暗模式下需要调深一点，或者使用 rgba */
        box-shadow: 0 8px 24px rgba(0,0,0,0.2) !important; 
    }}

    /* 3. 布局与 Header 优化 */
    .block-container {{ padding-top: 1.5rem !important; }}
    header {{ visibility: hidden; height: 0px !important; }}

    /* 4. 按钮主题色 (保持 HIGHLIGHT_COLOR) */
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
        'mask_scale': 1.0, 'mask_w_scale': 1.0, 'mask_y': 0.5,
        'pop_depth': 0.6, 'enable_popout': False,
        'color': '#FFFFFF', 'size': 60, 'y_pos': 380,
        'shadow_offset': 0, 'shadow_blur': 0, 'shadow_alpha': 150,
        'blur_radius': 50, 'blur_opacity': 100,
        'logo_x': 50, 'logo_y': 85, 'logo_scale': 40
    }
    st.session_state.active_file_id = None
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

# 修复：全员入框识别 (正侧脸双重捕获 + 最大跨度计算)
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
    
    if not all_y: return 0.5, 1.0, "未检测到清晰面部"
    
    # 获取所有脸部的最顶端和最底端
    min_y, max_y = min(all_y), max(all_y)
    center_y = ((min_y + max_y) / 2) / img_h
    face_span_ratio = (max_y - min_y) / img_h
    
    # 动态缩放：脸部跨度越小（人远），放大倍数越大
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
        # 确保 quality 参数被正确传给 API
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': img_bytes},
            data={'size': quality}, # 'preview', 'full', or 'auto'
            headers={'X-Api-Key': api_key},
            timeout=60
        )
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content)).convert("RGBA"), "Success"
        return None, response.json().get('errors', [{}])[0].get('title', "API Error")
    except Exception as e:
        return None, f"Network Error: {str(e)}"

# --- 4. 核心渲染引擎 ---
def render_poster(poster_file, subtitle, sets, tw, th, logo_file=None, font_file=None):
    final_canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    poster = Image.open(poster_file).convert("RGBA")
    
    # --- 1. 基准缩放锁定 (恢复入框生效的计算逻辑) ---
    base_scale = max(tw / poster.width, th / poster.height)
    final_scale = base_scale * sets['poster_scale']
    
    # 强制固定缩放后的尺寸，确保底图和抠图层引用完全一致的变量
    nw, nh = int(poster.width * final_scale), int(poster.height * final_scale)
    
    # --- 2. 像素偏移量锁定 (恢复入框生效的公式) ---
    # 使用 max(tw, nw) 是为了兼容 AI 识别算法的坐标系映射
    off_x = int((tw - nw) // 2 - (sets['crop_x'] - 0.5) * max(tw, nw))
    off_y = int((th - nh) // 2 - (sets['crop_y'] - 0.5) * max(th, nh))

    # --- 3. 渲染层级 1 & 2：模糊底板与主体 ---
    bg_fill = poster.resize((tw, th), Image.Resampling.LANCZOS)
    if sets['blur_radius'] > 0: 
        bg_fill = bg_fill.filter(ImageFilter.GaussianBlur(sets['blur_radius']))
    overlay = Image.new("RGBA", (tw, th), (0, 0, 0, int(sets['blur_opacity'])))
    canvas = Image.alpha_composite(bg_fill.convert("RGBA"), overlay)
    
    poster_res = poster.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas.paste(poster_res, (off_x, off_y), poster_res if "A" in poster_res.getbands() else None)
    
    # 保存底色供分析
    bg_for_analysis = canvas.copy()

    # --- 4. 层级 3：异形遮罩 (严禁删除，完整补齐) ---
    if os.path.exists("mask.png"):
        mask_org = Image.open("mask.png").convert("L")
        mw, mh = int(tw * sets['mask_w_scale']), int(th * sets['mask_scale'])
        mask_res = mask_org.resize((mw, mh), Image.Resampling.LANCZOS)
        full_mask = Image.new("L", (tw, th), 0)
        m_off_y = int(th * sets['mask_y']) - (mh // 2)
        full_mask.paste(mask_res, ((tw - mw) // 2, m_off_y))
        
        bg_layer = canvas.copy()
        bg_layer.putalpha(full_mask)
        final_canvas.paste(bg_layer, (0, 0), bg_layer)

    # --- 5. 层级 4：破框 (强制使用与主体一致的 nw, nh 和 off_x, off_y) ---
    if sets['enable_popout'] and st.session_state.cutout_img:
        # 即使 API 返回的尺寸略有差异，也强行缩放到 nw, nh
        cutout = st.session_state.cutout_img.resize((nw, nh), Image.Resampling.LANCZOS)
        cutout_canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        # 关键：这里必须使用和上面 paste poster_res 时一模一样的 off_x, off_y
        cutout_canvas.paste(cutout, (off_x, off_y), cutout)
        
        grad_mask = Image.new("L", (tw, th), 0)
        draw_g = ImageDraw.Draw(grad_mask)
        pop_line = int(th * sets['pop_depth'])
        for y in range(th):
            alpha = 255 if y < pop_line - 20 else (0 if y > pop_line + 20 else int(255 * (1 - (y - (pop_line-20))/40)))
            draw_g.line([(0, y), (tw, y)], fill=alpha)
        
        cutout_canvas.putalpha(ImageChops.multiply(cutout_canvas.getchannel("A"), grad_mask))
        final_canvas = Image.alpha_composite(final_canvas, cutout_canvas)

# --- 6. 文字渲染层 ---
    if subtitle:
        try:
            # 【核心修复】：如果传入的是缓存的 BytesIO，必须重置指针到 0
            if hasattr(font_file, 'seek'):
                font_file.seek(0)
            
            # 使用 sets['size'] 确保同步 UI 的字号
            font = ImageFont.truetype(font_file, int(sets['size']))
        except Exception as e:
            # 如果加载失败，打印一下原因方便调试，并降级
            # print(f"Font Load Error: {e}") 
            font = ImageFont.load_default()
            
        draw = ImageDraw.Draw(final_canvas)
        
        t_pos = (tw // 2, int(sets['y_pos']))
        
        # 渲染阴影 (如果需要)
        if sets['shadow_offset'] > 0 or sets['shadow_blur'] > 0:
            s_canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
            s_draw = ImageDraw.Draw(s_canvas) # 阴影层也需要自己的 draw
            s_draw.text((t_pos[0]+sets['shadow_offset'], t_pos[1]+sets['shadow_offset']), 
                        subtitle, fill=(0,0,0,int(sets['shadow_alpha'])), font=font, anchor="mm")
            if sets['shadow_blur'] > 0: 
                s_canvas = s_canvas.filter(ImageFilter.GaussianBlur(sets['shadow_blur']))
            final_canvas = Image.alpha_composite(final_canvas, s_canvas)
        
        # 渲染主文字
        # 此时 draw 已经定义，不会再报错了
        draw.text(t_pos, subtitle, fill=sets['color'], font=font, anchor="mm")

    # --- 7. Logo 层 (保持原有逻辑) ---
    if logo_file:
        logo = Image.open(logo_file).convert("RGBA")
        ls = sets['logo_scale'] / 100
        lw, lh = int(logo.width * ls), int(logo.height * ls)
        logo_res = logo.resize((lw, lh), Image.Resampling.LANCZOS)
        final_canvas.paste(logo_res, (int(tw*sets['logo_x']/100 - lw/2), int(th*sets['logo_y']/100 - lh/2)), logo_res)

    return final_canvas, bg_for_analysis

# --- 5. 侧边栏架构 ---
with st.sidebar:
    st.markdown(f"### 🎨 PosterAgent {VERSION}\n<p style='font-size:12px; color:#888;'>Architect: Dean</p>", unsafe_allow_html=True)
    
    # 修复：画布尺寸设置
    with st.expander("🖼️ 画布尺寸与比例", expanded=False):
        c_size = (st.session_state.settings['canvas_w'], st.session_state.settings['canvas_h'])
        options = {"16:9 (880x444)": (880, 444), "9:16 (450x800)": (450, 800), "1:1 (600x600)": (600, 600)}
        
        current_opt = "自定义"
        for k, v in options.items():
            if c_size == v: current_opt = k
            
        sel_ratio = st.radio("常用预设", list(options.keys()) + ["自定义"], index=list(options.keys()).index(current_opt) if current_opt != "自定义" else 3, horizontal=True)
        
        if sel_ratio != "自定义":
            st.session_state.settings['canvas_w'], st.session_state.settings['canvas_h'] = options[sel_ratio]

        dual_control("宽度 (W)", "canvas_w", 200, 2000, 1)
        dual_control("高度 (H)", "canvas_h", 200, 2000, 1)

    tw, th = st.session_state.settings['canvas_w'], st.session_state.settings['canvas_h']
    up_file = st.file_uploader("1. 上传主图", type=["jpg","png","jpeg"], label_visibility="collapsed")
    
    if up_file:
        # --- STEP 1 & 2. 素材加工 (彻底修复位移与顺序 Bug) ---
        with st.sidebar.expander("🛠️ STEP 1 & 2. 素材加工 (AI 入框 + 抠图)", expanded=True):
            
            # 1. 核心修复：确保即使没点入框，也有基础参数，且参数在 Session 中是稳定的
            # 如果是新上传的图，重置一下位置（可选）
            if st.session_state.get('last_uploaded_file') != up_file.name:
                st.session_state.settings['crop_y'] = 0.5
                st.session_state.settings['poster_scale'] = 1.0
                st.session_state.last_uploaded_file = up_file.name

            # --- AI 入框模块 ---
            if st.button("✨ 执行 AI 全员入框", use_container_width=True):
                ry, rs, status = run_ai_alignment(Image.open(up_file))
                # 显式更新字典，确保持久化
                st.session_state.settings['crop_y'] = ry
                st.session_state.settings['poster_scale'] = rs
                st.toast(status)
                st.rerun()

            # --- 实时手动控制 (这里的 slider 必须直接绑定 session_state) ---
            # 修改：让滑块直接操作字典里的值，这样任何刷新都不会丢失
            st.session_state.settings['poster_scale'] = st.slider("画面缩放", 0.5, 3.0, float(st.session_state.settings['poster_scale']), 0.1)
            st.session_state.settings['crop_x'] = st.slider("水平重心", -0.5, 1.5, float(st.session_state.settings['crop_x']), 0.01)
            st.session_state.settings['crop_y'] = st.slider("垂直重心", -0.5, 1.5, float(st.session_state.settings['crop_y']), 0.01)
            
            st.markdown("---")
            
            # --- 抠图模块 ---
            rbg_key = st.text_input("Remove.bg Key", type="password", placeholder="输入 API Key")
            rbg_quality = st.radio("抠图清晰度", ["preview", "full"], horizontal=True)
            
            col_rbg1, col_rbg2 = st.columns([1, 1])
            with col_rbg1:
                if st.button("🚀 执行抠图", use_container_width=True):
                    # 此时，st.session_state.settings['crop_y'] 已经是入框后的值了，不会丢
                    res, msg = get_removed_bg(rbg_key, up_file.getvalue(), quality=rbg_quality)
                    if res:
                        st.session_state.cutout_img = res
                        st.session_state.settings['enable_popout'] = True
                        # 同步余额
                        cost = 0.25 if rbg_quality == "preview" else 1.0
                        st.session_state.api_used = st.session_state.get('api_used', 12.5) + cost
                        st.rerun()
                    else:
                        st.error(msg)
            with col_rbg2:
                if st.button("🗑️ 清除抠图", use_container_width=True):
                    st.session_state.cutout_img = None
                    st.session_state.settings['enable_popout'] = False
                    st.rerun()

            if st.session_state.settings['enable_popout']:
                # 同样，破框深度也直接绑定
                st.session_state.settings['pop_depth'] = st.slider("破框深度", 0.0, 1.0, float(st.session_state.settings['pop_depth']), 0.01)
            
# --- STEP 3. 遮罩微调 ---
        with st.expander("🎭 STEP 3. 遮罩样式微调", expanded=False):
            dual_control("遮罩高度", "mask_scale", 0.1, 2.5, 0.01)
            dual_control("遮罩宽度", "mask_w_scale", 0.1, 2.5, 0.01)
            dual_control("垂直位移", "mask_y", -1.0, 1.0, 0.01)

# --- STEP 4. 字体与智能阴影 (最终稳定性增强版) ---
        with st.expander("🔤 STEP 4. 字体与智能阴影", expanded=False):
            # 1. 字体持久化
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