from flask import Flask, request, jsonify
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os  # ✅ 读取环境变量
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # ✅ 允许 WebGL 访问

# ✅ 允许使用 Render 环境变量指定模型 URL（可选）
MODEL_URL = os.getenv("MODEL_URL", "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")
print(f"🔄 加载风格迁移模型: {MODEL_URL}")
model = hub.load(MODEL_URL)

# ✅ 解析 Base64 图片
def decode_base64_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((256, 256))
    img_tensor = tf.image.convert_image_dtype(np.array(image), tf.float32)[tf.newaxis, ...]
    return img_tensor

# ✅ 编码风格化图片
def encode_tensor_to_base64(tensor):
    output_image = tf.image.convert_image_dtype(tensor[0], tf.uint8).numpy()
    img = Image.fromarray(output_image)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ✅ API 路由
@app.route('/style_transfer', methods=['POST'])
def style_transfer():
    data = request.get_json()
    try:
        content_img = decode_base64_image(data['content'])
        style_img = decode_base64_image(data['style'])
        stylized = model(content_img, style_img)[0]
        encoded_result = encode_tensor_to_base64(stylized)
        return jsonify({"stylized": encoded_result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ 运行 Flask 服务（Render 需要 `host='0.0.0.0'` 并读取 `$PORT`）
if __name__ == '__main__':
    PORT = int(os.getenv("PORT", 10000))  # Render 会提供 $PORT
    app.run(host='0.0.0.0', port=PORT)
