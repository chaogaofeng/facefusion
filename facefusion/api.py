import asyncio
import time
from collections import OrderedDict

from fastapi import FastAPI, File, UploadFile, Form, WebSocket
from fastapi.responses import StreamingResponse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import cv2
import numpy as np
from starlette.websockets import WebSocketDisconnect

from facefusion import logger, state_manager
from facefusion.audio import create_empty_audio_frame
from facefusion.face_analyser import get_many_faces, get_average_face
from facefusion.processors.core import get_processors_modules

executor = None


def process_frame(frame_data, source_face=None, background_frame=None, beautify=True):
	start_time = time.time()

	print(state_manager.get_item('face_selector_mode'), "====")

	processors = []
	if beautify:
		processors.append('face_enhancer')
	if source_face is not None:
		processors.append('face_swapper')

	# frameIndex:int    //帧序
	# width:int         //图像宽
	# height:int        //图像高
	# format:String     //图像格式
	# length:int        //图像数据大小
	# data:byte[]       //图像数据

	# 常见格式处理：
	# RGBA_8888 = 1
	# RGBX_8888 = 2
	# RGB_888     = 3
	# RGB_565 = 4
	# NV21
	# JPEG
	# YUV_420_888
	# YUV_422_888
	# YUV_444_888
	frame_index = frame_data.get('frameIndex', 0)
	image_bytes = frame_data['data']
	np_img = np.frombuffer(image_bytes, np.uint8)
	target_vision_frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

	# 异步执行图像处理
	source_audio_frame = create_empty_audio_frame()
	i = 0
	for processor_module in get_processors_modules(processors):
		t = time.time()
		logger.disable()
		if processor_module.pre_process('stream'):
			target_vision_frame = processor_module.process_frame(
				{
					'source_face': source_face,
					'source_audio_frame': source_audio_frame,
					'target_vision_frame': target_vision_frame
				})
		logger.enable()
		e = time.time()
		logger.info(f"{frame_index}: processor {processors[i]}, processing time: {e - t:.4f} seconds",
					__name__)  # 打印处理时间
		i += 1

	if background_frame is not None:
		t = time.time()
		target_vision_frame = merge_images(target_vision_frame, background_frame)
		e = time.time()
		logger.info(f"{frame_index}: processor background, processing time: {e - t:.4f} seconds",
					__name__)  # 打印处理时间

	_, img_encoded = cv2.imencode('.jpg', target_vision_frame)

	end_time = time.time()
	processing_time = end_time - start_time
	logger.info(f"{frame_index}: processors {processors}, processing time: {end_time - start_time:.4f} seconds",
				__name__)  # 打印处理时间
	return BytesIO(img_encoded.tobytes()), processing_time


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
	app = FastAPI()

	global executor
	executor = ThreadPoolExecutor(max_workers=max_workers if max_workers else 4)  # 控制最大线程数

	state_manager.set_item('face_selector_mode', 'one')

	@app.post('/process_image')
	async def process_image(
		image: UploadFile = File(...),
		water: UploadFile = File(None),
		swap: UploadFile = File(None),
		beautify: bool = Form(True)
	):
		frame_data = {}

		# 获取待处理图像
		image_bytes = await image.read()  # 读取图像字节
		frame_data['data'] = image_bytes

		# 获取待替换人脸图像
		source_face = None
		if swap:
			image_bytes = await swap.read()  # 读取图像字节
			np_img = np.frombuffer(image_bytes, np.uint8)
			source_frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
			source_faces = get_many_faces([source_frame])
			source_face = get_average_face(source_faces)

		# 获取背景图像
		background_frame = None
		if water:
			image_bytes = await water.read()
			np_img = np.frombuffer(image_bytes, np.uint8)
			background_frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

		future = executor.submit(process_frame, frame_data, source_face, background_frame, beautify)
		processed_frame, processing_time = await asyncio.wrap_future(future)
		return StreamingResponse(processed_frame, media_type='image/jpeg')

	@app.websocket("/ws")
	async def websocket_endpoint(websocket: WebSocket):
		await websocket.accept()

		# 有序字典用于保存处理后的帧数据
		results = OrderedDict()
		next_id_to_send = 1

		# 设置初始参数
		initial_params_set = False
		background_frame = None
		source_face = None
		try:
			while True:
				# 等待客户端请求
				frame_data = await websocket.receive_json()

				# 首次设置初始参数
				if not initial_params_set:
					# 设置是否美化
					beautify = frame_data.get("beautify", True)

					# 设置背景图像
					water_file = frame_data.get('water')
					if water_file:
						water_bytes = await water_file.read()
						np_img = np.frombuffer(water_bytes, np.uint8)
						background_frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

					# 设置换脸图像
					swap_file = frame_data.get('swap')
					if swap_file:
						swap_bytes = await swap_file.read()
						np_img = np.frombuffer(swap_bytes, np.uint8)
						source_frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
						source_faces = get_many_faces([source_frame])
						source_face = get_average_face(source_faces)

					initial_params_set = True  # 标记初始化完成

				future = executor.submit(process_frame, frame_data, source_face, background_frame, beautify)
				frameIndex = frame_data['frameIndex']
				results[frameIndex] = future  # 将 Future 按 frame_id 存入字典

				# 检查是否有按顺序完成的结果可以返回
				while next_id_to_send in results and results[next_id_to_send].done():
					processed_image, processing_time = await asyncio.wrap_future(results[next_id_to_send])
					# 发送处理结果
					await websocket.send_json({
						"frame_id": next_id_to_send,
						"image": processed_image.getvalue(),
						"processing_time": processing_time
					})
					# 移除已发送的结果，并更新下一个待发送的帧编号
					del results[next_id_to_send]
					next_id_to_send += 1
		except WebSocketDisconnect:
			logger.info("WebSocket disconnected", __name__)
		except Exception as e:
			logger.error(f"Error in WebSocket connection: {e}", __name__)
		finally:
			await websocket.close()

	return app
