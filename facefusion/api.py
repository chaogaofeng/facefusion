import time

from flask import Flask, request, send_file, jsonify
from io import BytesIO
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from facefusion import logger
from facefusion.audio import create_empty_audio_frame
from facefusion.face_analyser import get_many_faces, get_average_face
from facefusion.processors.core import get_processors_modules
from facefusion.typing import Face, VisionFrame

executor = None


def process_frame(source_face: Face, target_vision_frame: VisionFrame, processors: list[str]) -> VisionFrame:
	source_audio_frame = create_empty_audio_frame()
	for processor_module in get_processors_modules(processors):
		logger.disable()
		if processor_module.pre_process('stream'):
			target_vision_frame = processor_module.process_frame(
				{
					'source_face': source_face,
					'source_audio_frame': source_audio_frame,
					'target_vision_frame': target_vision_frame
				})
		logger.enable()
	return target_vision_frame


def merge_images(frame, background_image):
	# 确保背景图像和主图像都是 RGB 格式
	if len(frame.shape) == 2:  # 如果主图像是灰度图
		frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

	if len(background_image.shape) == 2:  # 如果背景图像是灰度图
		background_image = cv2.cvtColor(background_image, cv2.COLOR_GRAY2BGR)

	# 将背景图像调整为与主图像相同的大小
	background_resized = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))

	# 确保两幅图像都具有相同的通道数
	if frame.shape[2] != background_resized.shape[2]:
		raise ValueError("The number of channels in the frame and background image must match.")

	# 使用 addWeighted 函数合并两幅图像
	alpha = 0.7  # 主图像的透明度
	beta = 0.3  # 背景图像的透明度
	combined_image = cv2.addWeighted(frame, alpha, background_resized, beta, 0)

	return combined_image


def create_app(max_workers):
	app = Flask(__name__)

	global executor
	executor = ThreadPoolExecutor(max_workers=max_workers if max_workers else 4)  # 控制最大线程数

	@app.route('/process_image', methods=['POST'])
	def process_image():
		watermark_file = request.files.get('water', None)
		swap_file = request.files.get('swap', None)
		beautify = request.form.get('beautify', 'true').lower() == 'true'  # 获取是否美颜的参数

		processors = []
		if beautify:
			processors.append('face_enhancer')

		if 'image' not in request.files:
			return jsonify({'error': 'no image file provided'}), 400
		# 获取待处理图像
		image_file = request.files['image']
		image_bytes = image_file.read()  # 读取图像字节
		np_img = np.frombuffer(image_bytes, np.uint8)
		capture_frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

		# 获取待替换人脸图像
		source_face = None
		if swap_file:
			processors.append('face_swapper')
			image_bytes = swap_file.read()  # 读取图像字节
			np_img = np.frombuffer(image_bytes, np.uint8)
			source_frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
			source_faces = get_many_faces([source_frame])
			source_face = get_average_face(source_faces)

		# 获取背景图像
		background_frame = None
		if watermark_file:
			image_bytes = watermark_file.read()
			np_img = np.frombuffer(image_bytes, np.uint8)
			background_frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

		start_time = time.time()
		future = executor.submit(process_frame, source_face, capture_frame)
		processed_frame = future.result()
		end_time = time.time()
		logger.info(f"Processing time: {end_time - start_time:.4f} seconds", __name__)  # 打印处理时间

		if background_frame:
			processed_frame = merge_images(processed_frame, background_frame)

		_, img_encoded = cv2.imencode('.jpg', processed_frame)
		return send_file(BytesIO(img_encoded.tobytes()), mimetype='image/jpeg')

	return app
