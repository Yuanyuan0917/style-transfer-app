import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ✅ 禁用 GPU，强制使用 CPU

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # ✅ 允许 WebGL 访问

# ✅ 允许使用 Render 环境变量指定模型 URL（可选）
MODEL_URL = os.getenv("MODEL_URL", "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")
print(f"🔄 加载风格迁移模型: {MODEL_URL}")
model = hub.load(MODEL_URL)
print("✅ 模型加载完成！")

# ✅ 解析 Base64 图片
def decode_base64_image(base64_str):
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((256, 256))
        img_tensor = tf.image.convert_image_dtype(np.array(image), tf.float32)[tf.newaxis, ...]
        return img_tensor
    except Exception as e:
        raise ValueError(f"❌ 解码 Base64 失败: {str(e)}")  # ✅ 具体报错

# ✅ 编码风格化图片
def encode_tensor_to_base64(tensor):
    try:
        output_image = tf.image.convert_image_dtype(tensor[0], tf.uint8).numpy()
        img = Image.fromarray(output_image)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        raise ValueError(f"❌ 编码 PNG 失败: {str(e)}")  # ✅ 具体报错

# ✅ API 路由
@app.route('/style_transfer', methods=['POST'])
def style_transfer():
    data = request.get_json()
    try:
        print("📸 接收到风格迁移请求...")

        content_img = decode_base64_image(data['content'])
        style_img = decode_base64_image(data['style'])

        print("🔄 开始风格迁移...")
        stylized = model(content_img, style_img)[0]
        
        print("✅ 风格迁移成功，正在编码...")
        encoded_result = encode_tensor_to_base64(stylized)

        print("✅ 处理完成，返回结果")
        return jsonify({"stylized": encoded_result})
    
    except Exception as e:
        print(f"❌ 服务器处理失败: {str(e)}")  # ✅ 让 Render 显示错误日志
        return jsonify({"error": str(e)}), 500

# ✅ 运行 Flask 服务（Render 需要 `host='0.0.0.0'` 并读取 `$PORT`）
if __name__ == '__main__':
    PORT = int(os.getenv("PORT", 10000))  # Render 会提供 $PORT
    app.run(host='0.0.0.0', port=PORT)
