import time

from flask import Flask, request, send_file, jsonify
from io import BytesIO
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from facefusion import logger
from facefusion.uis.components.webcam import process_stream_frame

executor = None


def create_app(max_workers):
	app = Flask(__name__)

	executor = ThreadPoolExecutor(max_workers=max_workers if max_workers else 4)  # 控制最大线程数

	@app.route('/process_image', methods=['POST'])
	def process_image():
		if 'image' not in request.files:
			return jsonify({'error': 'No image file provided'}), 400

		# 获取图像文件
		image_file = request.files['image']
		image_bytes = image_file.read()  # 读取图像字节

		np_img = np.frombuffer(image_bytes, np.uint8)
		capture_frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

		start_time = time.time()

		source_face = None
		future = executor.submit(process_stream_frame, source_face, capture_frame)  # , source_face
		processed_frame = future.result()

		end_time = time.time()  # 记录结束时间
		logger.info(f"Processing time: {end_time - start_time:.4f} seconds", __name__)  # 打印处理时间

		_, img_encoded = cv2.imencode('.jpg', processed_frame)
		return send_file(BytesIO(img_encoded.tobytes()), mimetype='image/jpeg')

	return app
