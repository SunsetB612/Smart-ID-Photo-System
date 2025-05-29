from flask import Flask, render_template, request, send_file, jsonify,url_for
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import numpy as np
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
import logging

# MODNet相关导入
try:
    from src.models.modnet import MODNet
    MODNET_AVAILABLE = True
except ImportError:
    print("警告: MODNet模型未找到，将使用rembg作为备选方案")
    from rembg import remove
    MODNET_AVAILABLE = False

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_FOLDER'] = 'pretrained/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 全局变量存储模型
modnet_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_modnet_model():
    """加载MODNet模型"""
    global modnet_model
    
    if not MODNET_AVAILABLE:
        return None
        
    if modnet_model is not None:
        return modnet_model
    
    try:
        ckpt_path = os.path.join(app.config['MODEL_FOLDER'], 'modnet_photographic_portrait_matting.ckpt')
        
        if not os.path.exists(ckpt_path):
            print(f"MODNet模型文件不存在: {ckpt_path}")
            return None
        
        # 初始化模型
        modnet = MODNet(backbone_pretrained=False)
        modnet = nn.DataParallel(modnet)
        
        # 加载预训练权重
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        
        # 处理不同的checkpoint格式
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        
        # 处理DataParallel的键名前缀问题
        if any(k.startswith('module.') for k in state_dict.keys()):
            modnet.load_state_dict(state_dict)
        else:
            # 如果没有module前缀，需要添加
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[f'module.{k}'] = v
            modnet.load_state_dict(new_state_dict)
        
        modnet.eval()
        modnet_model = modnet
        print("MODNet模型加载成功")
        return modnet_model
        
    except Exception as e:
        print(f"MODNet模型加载失败: {e}")
        return None

def preprocess_image_for_modnet(image_path, ref_size=512):
    """为MODNet预处理图像"""
    try:
        im = Image.open(image_path)
        
        # 转换为RGB
        if im.mode != 'RGB':
            im = im.convert('RGB')
        
        # 调整大小，保持宽高比
        im_size = im.size
        ratio = min(ref_size / max(im_size), 1.0)
        new_size = tuple([int(x * ratio) for x in im_size])
        
        im = im.resize(new_size, Image.LANCZOS)
        
        # 转换为tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        return transform(im).unsqueeze(0), im_size
    except Exception as e:
        print(f"图像预处理失败: {e}")
        return None, None

def modnet_remove_background(image_path):
    """使用MODNet移除背景"""
    model = load_modnet_model()
    if model is None:
        return None
    
    try:
        # 预处理图像
        im_tensor, im_size = preprocess_image_for_modnet(image_path)
        if im_tensor is None:
            return None
        
        # 推理
        with torch.no_grad():
            _, _, matte = model(im_tensor, True)
        
        # 后处理遮罩
        matte = matte[0][0].data.cpu().numpy()
        matte = np.clip(matte, 0, 1)
        
        # 调整回原始尺寸
        matte_resized = Image.fromarray((matte * 255).astype(np.uint8)).resize(im_size, Image.LANCZOS)
        matte_np = np.array(matte_resized) / 255.0
        
        # 加载原始图像
        original_img = Image.open(image_path).convert('RGB')
        original_np = np.array(original_img)
        
        # 创建RGBA图像
        rgba_img = np.zeros((original_np.shape[0], original_np.shape[1], 4), dtype=np.uint8)
        rgba_img[:, :, :3] = original_np  # RGB通道
        rgba_img[:, :, 3] = (matte_np * 255).astype(np.uint8)  # Alpha通道
        
        # 转换为PIL图像
        result_img = Image.fromarray(rgba_img, 'RGBA')
        return result_img
        
    except Exception as e:
        print(f"MODNet背景移除失败: {e}")
        return None

def remove_background(image_path):
    """移除背景的统一接口"""
    # 优先使用MODNet
    if MODNET_AVAILABLE:
        result = modnet_remove_background(image_path)
        if result is not None:
            return result
        print("MODNet处理失败，回退到rembg")
    
    # 备选方案：使用rembg
    try:
        with open(image_path, 'rb') as f:
            input_data = f.read()
        output_data = remove(input_data)
        return Image.open(io.BytesIO(output_data)).convert('RGBA')
    except Exception as e:
        print(f"rembg处理也失败: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/background', methods=['GET', 'POST'])
def background():
    original_img = None
    removed_bg_img = None
    final_img = None
    processing_method = None

    if request.method == 'POST':
        file = request.files.get('file')
        bg_color = request.form.get('bg_color', '#ffffff')

        if file and allowed_file(file.filename):
            try:
                # 1. 保存原图
                filename = secure_filename(file.filename)
                original_path = os.path.join(app.config['UPLOAD_FOLDER'], f'original_{filename}')
                file.save(original_path)

                # 2. 去除背景（使用MODNet或rembg）
                removed_img = remove_background(original_path)
                if removed_img is None:
                    return render_template('background.html', error="背景移除失败，请尝试其他图片")
                
                processing_method = "MODNet" if MODNET_AVAILABLE and modnet_model is not None else "rembg"
                
                # 保存去背景图片（用于显示）
                removed_path = os.path.join(app.config['UPLOAD_FOLDER'], f'removed_{filename}')
                if removed_img.mode == 'RGBA':
                    # 创建白色背景用于预览
                    preview_bg = Image.new("RGB", removed_img.size, (255, 255, 255))
                    preview_bg.paste(removed_img, mask=removed_img.split()[3])
                    preview_bg.save(removed_path, 'JPEG')

                # 3. 应用新背景色
                # 将十六进制颜色转换为RGB
                if bg_color.startswith('#'):
                    bg_color = bg_color[1:]
                bg_rgb = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))
                
                # 创建新背景
                final_bg = Image.new('RGBA', removed_img.size, bg_rgb + (255,))
                final_bg.paste(removed_img, mask=removed_img.split()[3])
                
                # 保存最终结果
                final_path = os.path.join(app.config['UPLOAD_FOLDER'], f'final_{filename}')
                final_bg.convert('RGB').save(final_path, 'JPEG')

                # 4. 设置前端显示路径
                original_img = '/' + original_path
                removed_bg_img = '/' + removed_path
                final_img = '/' + final_path

            except Exception as e:
                print(f"处理过程中出错: {e}")
                return render_template('background.html', error=f"处理失败: {str(e)}")

    return render_template(
        'background.html',
        original_img=original_img,
        removed_bg_img=removed_bg_img,
        final_img=final_img,
        processing_method=processing_method
    )

@app.route('/crop', methods=['GET', 'POST'])
def crop():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = "crop_" + file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            output = crop_image(filepath)
            return send_file(output, mimetype='image/png', as_attachment=True, download_name='cropped.png')
    return render_template('crop.html')

@app.route('/watermark', methods=['GET', 'POST'])
def watermark():
    if request.method == 'POST':
        file = request.files['file']
        watermark_text = request.form.get('watermark', '')
        if file and allowed_file(file.filename):
            filename = "watermark_" + file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            output = add_watermark(filepath, watermark_text)
            return send_file(output, mimetype='image/png', as_attachment=True, download_name='watermarked.png')
    return render_template('watermark.html')

@app.route('/api/model_status')
def model_status():
    """API接口：返回模型状态"""
    status = {
        'modnet_available': MODNET_AVAILABLE,
        'modnet_loaded': modnet_model is not None,
        'fallback': 'rembg'
    }
    return jsonify(status)

@app.route('/portrait_cutout', methods=['GET', 'POST'])
def portrait_cutout():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # 这里直接用static/uploads目录，不要重复加uploads
            upload_folder = app.config['UPLOAD_FOLDER']  # static/uploads/
            cutout_folder = os.path.join('static', 'cutouts')  # static/cutouts
            os.makedirs(upload_folder, exist_ok=True)
            os.makedirs(cutout_folder, exist_ok=True)

            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)

            result_img = remove_background(filepath)
            if result_img is None:
                return render_template('portrait_cutout.html', error='背景去除失败')

            name, ext = os.path.splitext(filename)
            if ext.lower() not in ['.png', '.jpg', '.jpeg']:
                ext = '.png'

            result_filename = f'cutout{ext}'
            result_path = os.path.join(cutout_folder, result_filename)

            pil_format = 'JPEG' if ext.lower() in ['.jpg', '.jpeg'] else 'PNG'
            result_img.save(result_path, format=pil_format)

            original_img_url = url_for('static', filename=f'uploads/{filename}')
            cutout_img_url = url_for('static', filename=f'cutouts/{result_filename}')
            download_url = url_for('download_cutout', filename=result_filename)

            return render_template(
                'portrait_cutout.html',
                original_img=original_img_url,
                cutout_img=cutout_img_url,
                download_url=download_url
            )

    return render_template('portrait_cutout.html')


@app.route('/download_cutout/<filename>')
def download_cutout(filename):
    filename = secure_filename(filename)
    cutout_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'cutouts')
    file_path = os.path.join(cutout_folder, filename)
    if os.path.exists(file_path):
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'png'
        mime_type = 'image/jpeg' if ext in ['jpg', 'jpeg'] else f'image/{ext}'
        return send_file(file_path, mimetype=mime_type, as_attachment=True, download_name=filename)
    else:
        return "文件不存在", 404


def change_background(filepath, bg_color):
    """更改背景颜色（保持兼容性）"""
    removed_img = remove_background(filepath)
    if removed_img is None:
        return None
    
    bg = Image.new('RGBA', removed_img.size, bg_color)
    bg.paste(removed_img, mask=removed_img.split()[3])
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'background_changed.png')
    bg.convert('RGB').save(output_path)
    return output_path

def crop_image(filepath):
    """裁剪图像"""
    img = Image.open(filepath)
    width, height = img.size
    left = width * 0.1
    top = height * 0.1
    right = width * 0.9
    bottom = height * 0.9
    cropped_img = img.crop((left, top, right, bottom))
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cropped.png')
    cropped_img.save(output_path)
    return output_path

def add_watermark(filepath, watermark):
    """添加水印"""
    img = Image.open(filepath).convert("RGBA")
    txt = Image.new('RGBA', img.size, (255,255,255,0))
    draw = ImageDraw.Draw(txt)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), watermark, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = img.width - text_width - 10
    y = img.height - text_height - 10
    draw.text((x, y), watermark, fill=(255, 255, 255, 128), font=font)
    combined = Image.alpha_composite(img, txt)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'watermarked.png')
    combined.convert('RGB').save(output_path)
    return output_path

if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    
    # 预加载MODNet模型
    if MODNET_AVAILABLE:
        print("正在预加载MODNet模型...")
        load_modnet_model()
    
    app.run(debug=True, port=5001)