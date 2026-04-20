import os
import io
import requests
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageChops
import matplotlib.font_manager as fm

# --- 1. 基础配置与初始化 ---
st.set_page_config(page_title="异形海报处理工作站", layout="wide")
HIGHLIGHT_COLOR = "#FF5588"
VERSION = "v11.6 (Final Stable)"

# 默认参数模版
DEFAULT_SETTINGS = {
    "size": 60, "y_pos": 300, "color": "#FFFFFF", "crop_y": 0.5,
    "poster_scale": 1.0, 
    "logo_x": 50, "logo_y": 85, "logo_scale": 40,
    "mask_scale": 1.0, "mask_w_scale": 1.0, 
    "mask_x": 0.5, "mask_y": 0.5,
    "enable_popout": False, "pop_depth": 0.6, "font_path": None,
    "shadow_offset": 0, "shadow_blur": 0, "shadow_alpha": 150 
}

# --- 1. 初始化 Session State (脚本最顶端) ---
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'canvas_w': 880,
        'canvas_h': 444,
        'crop_y': 0.5,
        'crop_x': 0.5,
        'poster_scale': 1.0,
        'mask_scale': 1.0,
        'mask_w_scale': 1.0,
        'mask_y': 0.5,
        'pop_depth': 0.5,
        'enable_popout': False,
        'color': '#FFFFFF',
        'size': 60,
        'y_pos': 380,
        'shadow_offset': 0,
        'shadow_blur': 0,
        'shadow_alpha': 150, 
        'blur_radius': 50,    # 模糊强度
        'blur_opacity': 100,  # 背景暗度 (0-255)
    }
    # 核心修复：在这里初始化所有可能被“读取”的变量
    st.session_state.active_file_id = None  # 解决你刚才的报错
    st.session_state.cutout_img = None      # 确保抠图缓存也有地基
if 'loaded_fonts' not in st.session_state:
    st.session_state.loaded_fonts = {} # 格式: {"字体名": 文件对象}

# --- 2. AI 自动对齐引擎 ---
def run_ai_alignment(pil_image):
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img_h, img_w = cv_img.shape[:2]
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray) 
    f_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    p_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    faces_f = f_cascade.detectMultiScale(gray, 1.05, 4, minSize=(40, 40))
    faces_p = p_cascade.detectMultiScale(gray, 1.05, 4, minSize=(40, 40))
    all_y = []
    for (x, y, w, h) in faces_f: all_y.extend([y, y + h])
    for (x, y, w, h) in faces_p: all_y.extend([y, y + h])
    if not all_y: return 0.5, 1.0, "未捕捉到清晰面部"
    y_min_raw, y_max_raw = min(all_y), max(all_y)
    face_center_y = ((y_min_raw + y_max_raw) / 2) / img_h
    recommended_crop_y = np.clip(face_center_y, 0.3, 0.6)
    face_height_pct = (y_max_raw - y_min_raw) / img_h
    final_scale = 1.3 if face_height_pct < 0.2 else (1.1 if face_height_pct < 0.4 else 0.9)
    return float(recommended_crop_y), float(final_scale), f"AI 已捕获全员坐标"

# --- 3. 强效吸顶样式与精准滑块主题 ---
st.markdown(f"""
    <style>
    .block-container {{ padding-top: 0rem !important; margin-top: -10px !important; }}
    header {{ visibility: hidden; height: 0px !important; }}
    
    /* 按钮样式 */
    div.stButton > button, div.stDownloadButton > button {{
        background-color: {HIGHLIGHT_COLOR} !important; color: white !important;
        border: 1px solid {HIGHLIGHT_COLOR} !important; border-radius: 4px;
    }}

    /* --- 精准控制滑块颜色 --- */
    /* 1. 只有已填充的进度条颜色 (粉色) */
    div[data-baseweb="slider"] > div:first-child > div:first-child {{
        background: {HIGHLIGHT_COLOR} !important;
    }}
    
    /* 2. 圆形操作点颜色 (粉色) */
    div[data-baseweb="slider"] [role="slider"] {{
        background-color: {HIGHLIGHT_COLOR} !important;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2) !important;
    }}

    /* 3. 未填充的底槽颜色 (恢复原生浅灰色，增加质感) */
    div[data-baseweb="slider"] > div {{
        background-color: #f0f2f6; 
    }}

    /* 预览图描边 */
    [data-testid="stImage"] img {{ border: 1px solid #EEEEEE !important; border-radius: 4px; }}
    </style>
""", unsafe_allow_html=True)

# --- 4. 工具函数 ---
@st.cache_data
def get_system_fonts():
    fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    return dict(sorted({os.path.basename(f): f for f in fonts if f.lower().endswith(('.ttf', '.otf'))}.items()))

def get_removed_bg(api_key, img_bytes, quality="preview"):
    """
    支持画质选择的抠图函数
    quality: "preview" (0.25 credits) or "full" (1 credit)
    """
    try:
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': img_bytes}, 
            data={'size': quality},
            headers={'X-Api-Key': api_key}, 
            timeout=60 # 抠原图可能较慢，延长超时
        )
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content)).convert("RGBA"), "Success"
        # 错误处理优化
        err_msg = response.json().get('errors', [{}])[0].get('title', f"HTTP {response.status_code}")
        return None, err_msg
    except Exception as e: 
        return None, "网络连接超时，请检查梯子或网络环境"

def update_setting(key, from_type):
    """
    核心：双向同步逻辑
    无论从 slider 还是 number_input 改变，都同步到 settings 并更新对方的 state
    """
    source_key = f"{from_type}_{key}"
    target_type = "num" if from_type == "sli" else "sli"
    target_key = f"{target_type}_{key}"
    
    # 1. 获取当前操作的值
    new_val = st.session_state[source_key]
    
    # 2. 同步到全局配置
    st.session_state.settings[key] = new_val
    
    # 3. 强制同步另一个组件的状态
    st.session_state[target_key] = new_val

def dual_control(label, key_name, min_val, max_val, step=1.0):
    # 确保初始化
    if key_name not in st.session_state.settings:
        st.session_state.settings[key_name] = min_val
    
    curr_val = float(st.session_state.settings[key_name])
    
    # 布局：左边标题，右边数值输入
    c1, c2 = st.columns([2.5, 1])
    with c1:
        st.markdown(f"<div style='font-size:13px; font-weight:500; margin-top:5px;'>{label}</div>", unsafe_allow_html=True)
    with c2:
        # 数值输入框 (绑定 key 并设置回调)
        st.number_input(
            label=f"n_{key_name}", min_value=float(min_val), max_value=float(max_val),
            value=curr_val, step=float(step), label_visibility="collapsed",
            key=f"num_{key_name}", on_change=update_setting, args=(key_name, "num")
        )
    
    # 下方：滑块 (绑定 key 并设置回调)
    st.slider(
        label=label, min_value=float(min_val), max_value=float(max_val),
        value=curr_val, step=float(step), label_visibility="collapsed",
        key=f"sli_{key_name}", on_change=update_setting, args=(key_name, "sli")
    )

def render_poster(poster_file, subtitle, sets, tw, th, logo_file=None):
    final_canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    poster = Image.open(poster_file).convert("RGBA")
    
# --- 2. 渲染背景 (确保全屏填充 + 颜色同步) ---
    img_w, img_h = poster.size
    blur_r = int(sets.get('blur_radius', 50))
    blur_alpha = int(sets.get('blur_opacity', 100))

    # A. 生成【强制铺满画布】的模糊底色层
    # 无论原图多小，先强行拉伸到画布大小，确保没有黑边
    bg_fill = poster.resize((tw, th), Image.Resampling.LANCZOS)
    if blur_r > 0:
        bg_fill = bg_fill.filter(ImageFilter.GaussianBlur(radius=blur_r))
    
    # B. 准备【同步位移】的模糊层 (作为主体的投影/延伸)
    base_scale = max(tw / img_w, th / img_h)
    final_scale = base_scale * sets.get('poster_scale', 1.0)
    new_w, new_h = int(img_w * final_scale), int(img_h * final_scale)
    
    poster_resized = poster.resize((new_w, new_h), Image.Resampling.LANCZOS)
    bg_move = poster_resized.copy()
    if blur_r > 0:
        bg_move = bg_move.filter(ImageFilter.GaussianBlur(radius=blur_r))

    # C. 计算位移 (使用之前的自由位移公式)
    center_x = (tw - new_w) // 2
    center_y = (th - new_h) // 2
    move_range_x = max(tw, new_w)
    move_range_y = max(th, new_h)
    
    off_x = int(center_x - (sets.get('crop_x', 0.5) - 0.5) * move_range_x)
    off_y = int(center_y - (sets.get('crop_y', 0.5) - 0.5) * move_range_y)

    # D. 多层合成 (像三明治一样)
    # 第一层：铺满全屏的模糊背景 (底座，保证永远不黑)
    canvas = bg_fill.convert("RGBA")
    
    # 第二层：跟随主体一起动的模糊层 (增加氛围感)
    # 注意：如果不需要双重模糊，这一步可以跳过，直接去第三层
    canvas.paste(bg_move, (off_x, off_y), bg_move if "A" in bg_move.getbands() else None)
    
    # 第三层：半透明黑色遮罩 (压暗背景)
    overlay = Image.new("RGBA", (tw, th), (0, 0, 0, blur_alpha))
    canvas = Image.alpha_composite(canvas, overlay)
    
    # 第四层：清晰的主体海报
    canvas.paste(poster_resized, (off_x, off_y), poster_resized if "A" in poster_resized.getbands() else None)
    
    poster_fitted = canvas

    # --- B. 遮罩层处理 (重新找回遮罩) ---
    if os.path.exists("mask.png"):
        mask_org = Image.open("mask.png").convert("L")
        mw = int(tw * sets.get('mask_w_scale', 1.0))
        mh = int(th * sets.get('mask_scale', 1.0))
        mask_resized = mask_org.resize((mw, mh), Image.Resampling.LANCZOS)
        
        full_mask = Image.new("L", (tw, th), 0)
        off_x = (tw - mw) // 2
        off_y = int(th * sets.get('mask_y', 0.5)) - (mh // 2)
        full_mask.paste(mask_resized, (off_x, off_y))
        
        # 应用遮罩：把背景层贴上去
        bg_layer = poster_fitted.copy()
        bg_layer.putalpha(full_mask)
        final_canvas.paste(bg_layer, (0, 0), bg_layer)

    # --- C. 破框处理 (保持比例一致) ---
    if sets.get('enable_popout') and st.session_state.cutout_img:
        cutout = st.session_state.cutout_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        cutout_fitted = cutout.crop((left, top, left + tw, top + th))
        
        grad_mask = Image.new("L", (tw, th), 0)
        draw_g = ImageDraw.Draw(grad_mask)
        ts, te = int(th*(sets['pop_depth']-0.12)), int(th*(sets['pop_depth']+0.12))
        for y in range(th):
            a = 255 if y < ts else (0 if y > te else int(255*(1-(y-ts)/(te-ts))))
            draw_g.line([(0, y), (tw, y)], fill=a)
        
        cutout_fitted.putalpha(ImageChops.multiply(cutout_fitted.getchannel("A"), grad_mask))
        final_canvas.paste(cutout_fitted, (0, 0), cutout_fitted)

    # 5. 渲染标题与阴影
    try:
        font_path = sets.get('font_path')
        font = ImageFont.truetype(font_path, int(sets['size'])) if font_path else ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(final_canvas)
    text_pos = (tw // 2, int(sets['y_pos']))

# 动态加载选择的字体
    if current_font_file:
        # PIL 可以直接读取上传的文件流 (BytesIO)
        font = ImageFont.truetype(current_font_file, size=int(sets.get('size', 60)))
    else:
        # 兜底：如果没传字体，使用系统路径下可能存在的字体或报错提醒
        font = ImageFont.load_default()

    
    # 阴影逻辑
    if sets.get('shadow_offset', 0) > 0 or sets.get('shadow_blur', 0) > 0:
        s_canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        s_draw = ImageDraw.Draw(s_canvas)
        off = int(sets['shadow_offset'])
        s_draw.text((text_pos[0]+off, text_pos[1]+off), subtitle, fill=(0,0,0,int(sets['shadow_alpha'])), font=font, anchor="mm")
        if sets['shadow_blur'] > 0:
            s_canvas = s_canvas.filter(ImageFilter.GaussianBlur(sets['shadow_blur']))
        final_canvas = Image.alpha_composite(final_canvas, s_canvas)

    ImageDraw.Draw(final_canvas).text(text_pos, subtitle, fill=sets['color'], font=font, anchor="mm")

    # 6. Logo
    if logo_file:
        logo = Image.open(logo_file).convert("RGBA")
        l_scale = sets.get('logo_scale', 80) / 100
        lw, lh = int(logo.width * l_scale), int(logo.height * l_scale)
        logo = logo.resize((lw, lh), Image.Resampling.LANCZOS)
        lx = int(tw * sets['logo_x'] / 100 - lw / 2)
        ly = int(th * sets['logo_y'] / 100 - lh / 2)
        final_canvas.paste(logo, (lx, ly), logo)
    
    return final_canvas


# --- 5. 侧边栏 ---
all_fonts = get_system_fonts()

with st.sidebar:
    st.markdown(f"### 🎨 PosterAgent {VERSION}\n<p style='font-size:12px; color:#888;'>Created by Dean</p>", unsafe_allow_html=True)
    
# --- 全局画布尺寸设置 ---
    with st.expander("🖼️ 画布尺寸设置", expanded=False):
        # 定义预设选项
        options = ["自定义", "16:9 (880x444)", "9:16 (450x800)", "1:1 (600x600)"]
        
        # 逻辑：根据当前 settings 的值自动判断应该勾选哪一个
        current_size = (st.session_state.settings['canvas_w'], st.session_state.settings['canvas_h'])
        default_index = 0 # 默认为自定义
        if current_size == (880, 444): default_index = 1
        elif current_size == (450, 800): default_index = 2
        elif current_size == (600, 600): default_index = 3

        ratio = st.radio("常用比例", options, index=default_index, horizontal=True)
        
        # 只有在点击对应的选项时，才覆盖 settings
        if ratio == "16:9 (880x444)":
            st.session_state.settings['canvas_w'], st.session_state.settings['canvas_h'] = 880, 444
        elif ratio == "9:16 (450x800)":
            st.session_state.settings['canvas_w'], st.session_state.settings['canvas_h'] = 450, 800
        elif ratio == "1:1 (600x600)":
            st.session_state.settings['canvas_w'], st.session_state.settings['canvas_h'] = 600, 600
        
        # 双路控制保持不变，它们会自动读取/写入 settings
        dual_control("画布宽度 (W)", "canvas_w", 100, 2000, 1)
        dual_control("画布高度 (H)", "canvas_h", 100, 2000, 1)

    # 提取当前值，确保下方渲染逻辑永远有 tw 和 th 可用
    tw = st.session_state.settings.get('canvas_w', 880)
    th = st.session_state.settings.get('canvas_h', 444)

    up_file = st.file_uploader("海报上传", type=["jpg","png","jpeg"], label_visibility="collapsed")
    
    if up_file:
        file_id = f"{up_file.name}_{up_file.size}"
        if st.session_state.active_file_id != file_id:
            st.session_state.active_file_id = file_id
            st.session_state.cutout_img = None
            
        # --- STEP 1: AI 构图 ---
        with st.expander("📐 STEP 1. 构图与智能对齐", expanded=True):
            if st.button("✨ 执行 AI 一键全员入框", use_container_width=True):
                with st.spinner("AI 正在根据新画布计算..."):
                    # 传入动态的 tw, th 给 AI 适配逻辑
                    rec_y, rec_scale, status = run_ai_alignment(Image.open(up_file))
                    
                    st.session_state.settings['crop_y'] = rec_y
                    st.session_state.settings['poster_scale'] = rec_scale
                    st.session_state.settings['crop_x'] = 0.5

                    # 同步刷新滑块状态
                    st.session_state[f"num_crop_y"] = float(rec_y)
                    st.session_state[f"sli_crop_y"] = float(rec_y)
                    st.session_state[f"num_crop_x"] = 0.5
                    st.session_state[f"sli_crop_x"] = 0.5
                    st.session_state[f"num_poster_scale"] = float(rec_scale)
                    st.session_state[f"sli_poster_scale"] = float(rec_scale)
                    
                    st.toast(status)
                    st.rerun() 
            
            dual_control("背景缩放", "poster_scale", 0.5, 3.0, 0.1)
            dual_control("水平重心", "crop_x", -0.5, 1.5, 0.01) # 扩大范围允许移出画布
            dual_control("垂直重心", "crop_y", -0.5, 1.5, 0.01)
            
            st.markdown("---") # 分割线
            st.markdown("<div style='font-size:12px; color:#888;'>模糊氛围感</div>", unsafe_allow_html=True)
            dual_control("模糊强度", "blur_radius", 0, 100, 1)
            dual_control("背景暗度", "blur_opacity", 0, 255, 5)

        # --- STEP 2: AI 抠图 (增强版) ---
        with st.expander("✂️ STEP 2. AI 抠图与画质控制", expanded=st.session_state.settings['enable_popout']):
            rbg_key = st.text_input("Remove.bg Key", type="password", help="需填入有效 API Key")
            
            # 1. 补齐画质选项
            target_quality = st.selectbox(
                "输出画质", 
                options=["preview", "full"], 
                format_func=lambda x: "预览版 (低分/免费)" if x=="preview" else "高清版 (高分/消耗点数)",
                help="preview 消耗 0.25 点数，full 消耗 1 点数"
            )
            
            # 2. 抠图动作与反馈
            if st.button("🚀 执行 AI 抠图", use_container_width=True):
                if not rbg_key:
                    st.error("请先输入 API Key")
                else:
                    # 使用 st.status 提供动态反馈
                    with st.status("正在处理图像...", expanded=True) as status_box:
                        st.write("正在上传至 Remove.bg...")
                        res, msg = get_removed_bg(rbg_key, up_file.getvalue(), quality=target_quality)
                        
                        if res:
                            st.session_state.cutout_img = res
                            st.session_state.settings['enable_popout'] = True
                            status_box.update(label="✅ 抠图成功！已应用破框效果", state="complete")
                            st.rerun()
                        else:
                            status_box.update(label=f"❌ 抠图失败: {msg}", state="error")

            # 3. 结果反馈预览
            if st.session_state.cutout_img:
                st.info("检测到已存在抠图缓存")
                c1, c2 = st.columns(2)
                with c1:
                    st.image(st.session_state.cutout_img, caption="已抠图层", use_container_width=True)
                with c2:
                    if st.button("🗑️ 清除抠图", use_container_width=True):
                        st.session_state.cutout_img = None
                        st.session_state.settings['enable_popout'] = False
                        st.rerun()
            
            dual_control("破框深度", "pop_depth", 0.1, 1.0, 0.01)

        # --- STEP 3: 遮罩调整 ---
        with st.expander("🎭 STEP 3. 遮罩样式调整", expanded=False):
            dual_control("遮罩高度", "mask_scale", 0.1, 2.5, 0.01)
            dual_control("遮罩宽度", "mask_w_scale", 0.1, 2.5, 0.01)
            dual_control("垂直位移", "mask_y", -1.0, 1.0, 0.01)

        # --- STEP 4: Logo ---
        logo_f = st.file_uploader("🏷️ STEP 4. Logo (可选)", type=["png","jpg"])
        if logo_f:
            with st.expander("🏷️ Logo 细节微调", expanded=True):
                dual_control("Logo 缩放", "logo_scale", 15, 100, 1)
                dual_control("横向位置 %", "logo_x", 50, 100, 1)
                dual_control("纵向位置 %", "logo_y", 70, 100, 1)

# --- STEP 5: 文本与阴影 ---
        with st.sidebar:
            st.markdown("### 🔤 字体库管理")
    
            # 支持批量上传
            new_fonts = st.file_uploader("添加本地字体", type=["ttf", "otf"], accept_multiple_files=True)
            
            if new_fonts:
                for f in new_fonts:
                    st.session_state.loaded_fonts[f.name] = f
        
            current_font_file = None
            if st.session_state.loaded_fonts:
                font_names = list(st.session_state.loaded_fonts.keys())
                # 让用户从已上传的列表中选择一个
                selected_font_name = st.selectbox("当前使用的字体", font_names)
                current_font_file = st.session_state.loaded_fonts[selected_font_name]
                
                # 预览框
                st.markdown(f"""
                    <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; 
                         text-align: center; background: white; color: black; font-family: sans-serif;">
                        <span style="font-size: 12px; color: #666;">预览字体名：</span><br>
                        <strong>{selected_font_name}</strong>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("请上传字体以开始设计")

            # 3. 详细配置
            with st.expander("✍️ STEP 5. 标题与阴影配置", expanded=False):
                subtitle_text = st.text_input("标题内容", "SUMMER VIBES")
                
                # 这里我们根据是否有上传字体来决定渲染逻辑
                dual_control("字号", "size", 10, 300, 1)
                dual_control("垂直位置", "y_pos", 0, th, 1)
                st.session_state.settings['color'] = st.color_picker("颜色", st.session_state.settings['color'])
                
                st.markdown("---")
                dual_control("阴影偏移", "shadow_offset", 0, 30, 1)
                dual_control("阴影模糊", "shadow_blur", 0, 20, 1)
                dual_control("阴影透明", "shadow_alpha", 0, 255, 5)

# --- 6. 主区域 ---
tabs = st.tabs(["🏠 使用说明", "🎨 工作站"]) if not up_file else st.tabs(["🎨 工作站", "🏠 使用说明"])
with tabs[0 if not up_file else 1]:
    st.markdown(f"### 🎨 关于 PosterAgent {VERSION}\n**Developer:** Dean")
    st.markdown("""
    这款工具专为**异形海报与流媒体封面**设计，集成了 AI 视觉算法，帮助开发者快速完成构图。
    #### 🚀 核心功能：
    * **AI 智能全员入框**：自动识别人物位置，一键计算最佳构图。
    * **智能破框效果**：实现人物超越遮罩边界的立体视觉感。
    * **双路精准控制**：支持“数字微调 + 滑块拖拽”双模式。
    * **动态字体阴影**：支持实时模糊阴影渲染。
    """)
with tabs[1 if not up_file else 0]:
    if up_file:
        res = render_poster(up_file, subtitle_text, st.session_state.settings, 880, 444, logo_f)
        st.image(res, use_container_width=True)
        buf = io.BytesIO(); res.save(buf, format="PNG")
        st.download_button("📥 导出 PNG", buf.getvalue(), "poster.png", "image/png", use_container_width=True)