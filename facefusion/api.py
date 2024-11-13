import asyncio
import struct
import time
import zlib
import uvicorn
import traceback
from collections import OrderedDict
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.websockets import WebSocketState
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import ffmpeg
from starlette.websockets import WebSocketDisconnect

from facefusion import logger, state_manager
from facefusion.audio import create_empty_audio_frame
from facefusion.face_analyser import get_many_faces, get_average_face
from facefusion.processors.core import get_processors_modules


def identify_image_format(image_bytes):
	# 检查文件头
	if image_bytes[:2] == b'\xff\xd8':
		return 'JPEG'
	elif image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
		return 'PNG'
	elif image_bytes[:3] == b'GIF':
		return 'GIF'
	elif image_bytes[:2] == b'BM':
		return 'BMP'
	elif image_bytes[:4] == b'\x52\x49\x46\x46':
		return 'WEBP'
	else:
		raise ValueError(f"不支持的图像格式 {image_bytes[:20]}")


def encode_h265(image, fps=30, bitrate=800000, i_frame_interval = 2):
	"""
	将图像数据压缩为 H.265 格式并返回字节流。
	:param i_frame_interval:
	:param bitrate:
	:param image: 输入图像（NumPy 数组）
	:param fps: 视频帧率
	:return: H.265 压缩数据的字节流
	"""

	try:
		height, width, _ = image.shape
		# 使用 FFmpeg 压缩图像为 H.265 格式，并将输出流重定向到内存
		out, _ = (
			ffmpeg
			.input('pipe:0', format='rawvideo', pix_fmt='yuv420p', s=f'{width}x{height}')
			.output('pipe:1', vcodec='libx265', r=fps, video_bitrate=bitrate, g=i_frame_interval * fps, pix_fmt='yuv420p')  # 设置像素格式为 yuv420p
			.run(input=image.tobytes(), quiet=True)
		)
		logger.debug("压缩后字节流长度:", len(out))
		return out  # 返回压缩后的字节流
	except Exception as e:
		raise ValueError(f"H.265 压缩失败: {e}")


def decode_h265(h265_bytes, width, height):
	"""
	从 H.265 压缩字节流解码并返回图像（BGR 格式）。
	:param h265_bytes: H.265 压缩字节流
	:param width: 图像宽度
	:param height: 图像高度
	:return: 解码后的图像（NumPy 数组）
	"""
	with open('encode.txt', 'wb') as f:
		f.write(h265_bytes)
	try:
		# 使用 FFmpeg 解码 H.265 数据
		out, _ = (
			ffmpeg
			.input('pipe:0', vcodec='hevc', s=f'{width}x{height}')  # 输入数据是 H.265 编码的视频流
			.output('pipe:1', format='rawvideo', pix_fmt='yuv420p')  # 输出为 yuv420p 格式
			.run(input=h265_bytes)
		)
		logger.debug("解码后字节流长度:", len(out))
		return out
	except Exception as e:
		raise ValueError(f"H.265 解码失败: {e}")


def convert_to_bitmap(width, height, format_type, data):
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
	if data is None or len(data) == 0:
		raise ValueError("数据为空或无效")

	if isinstance(format_type, bytearray):
		format_type = format_type.decode('utf-8')

	if format_type == "RGBA_8888":
		if len(data) != width * height * 4:
			raise ValueError("数据长度与图像尺寸不匹配")
		image_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
		image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
	elif format_type == "RGBX_8888":
		if len(data) != width * height * 4:
			raise ValueError("数据长度与图像尺寸不匹配")
		image_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
		image = cv2.cvtColor(image_array[:, :, :3], cv2.COLOR_RGB2BGR)
	elif format_type == "RGB_888":
		if len(data) != width * height * 3:
			raise ValueError("数据长度与图像尺寸不匹配")
		image_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
		image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
	elif format_type == "RGB_565":
		if len(data) != width * height * 2:
			raise ValueError("数据长度与图像尺寸不匹配")
		image_array = np.frombuffer(data, dtype=np.uint16).reshape((height, width))
		image = cv2.cvtColor(image_array, cv2.COLOR_BGR5652BGR)
	elif format_type == "NV21":
		if len(data) != width * (height + height // 2):
			raise ValueError("数据长度与图像尺寸不匹配")
		image_array = np.frombuffer(data, dtype=np.uint8).reshape((height + height // 2, width))
		image = cv2.cvtColor(image_array, cv2.COLOR_YUV2BGR_NV21)
	elif format_type == "JPEG":
		image_array = np.frombuffer(data, dtype=np.uint8)
		image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
	elif format_type == "PNG":
		image_array = np.frombuffer(data, dtype=np.uint8)
		image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
	elif format_type == "YUV_420_888":
		data = decode_h265(data, width, height)
		if len(data) != width * height * 3 // 2:
			raise ValueError("数据长度与图像尺寸不匹配")
		image_array = np.frombuffer(data, dtype=np.uint8).reshape((height * 3 // 2, width))
		image = cv2.cvtColor(image_array, cv2.COLOR_YUV2BGR_I420)
	elif format_type == "YUV_422_888":
		if len(data) != width * height * 2:
			raise ValueError("数据长度与图像尺寸不匹配")
		image_array = np.frombuffer(data, dtype=np.uint8).reshape((height * 2, width))
		image = cv2.cvtColor(image_array, cv2.COLOR_YUV2BGR_Y422)
	elif format_type == "YUV_444_888":
		if len(data) != width * height * 3:
			raise ValueError("数据长度与图像尺寸不匹配")
		image_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
		image = cv2.cvtColor(image_array, cv2.COLOR_YUV2BGR)
	else:
		raise ValueError(f"不支持的图像格式 {format_type}")

	if image is None:
		raise ValueError("Cannot convert image to bitmap")
	return image


def bitmap_to_data(image, width, height, format_type):
	if image is None:
		raise ValueError("图像为空")

	if isinstance(format_type, bytearray):
		format_type = format_type.decode('utf-8')

	if format_type == "RGBA_8888":
		image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
		return image_array.tobytes()
	elif format_type == "RGBX_8888":
		image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return np.concatenate((image_array, np.full((*image_array.shape[:-1], 1), 255, dtype=np.uint8)),
							  axis=-1).tobytes()
	elif format_type == "RGB_888":
		image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return image_array.tobytes()
	elif format_type == "RGB_565":
		image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return image_array.astype(np.uint16).tobytes()
	elif format_type == "NV21":
		height, width = image.shape[:2]

		# 转换到 NV21 格式
		yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
		y_plane = yuv_image[:height, :]
		uv_plane = yuv_image[height:, :].reshape((height // 2, width))

		# NV21 格式要求 UV 平面交替排列
		vu_plane = uv_plane[:, 1::2]
		vu_plane = np.dstack((vu_plane, uv_plane[:, ::2])).reshape(-1)

		return np.concatenate((y_plane.flatten(), vu_plane)).tobytes()
	elif format_type == "JPEG":
		_, jpeg_data = cv2.imencode('.jpg', image)
		return jpeg_data.tobytes()
	elif format_type == "PNG":
		_, png_data = cv2.imencode('.png', image)
		return png_data.tobytes()
	elif format_type == "YUV_420_888":
		return encode_h265(image)
		# yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
		# return yuv_image.tobytes()
	elif format_type == "YUV_422_888":
		yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_Y422)
		return yuv_image.tobytes()
	elif format_type == "YUV_444_888":
		yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
		return yuv_image.tobytes()
	else:
		raise ValueError(f"不支持的图像格式 {format_type}")


def process_frame(frame_data, source_face=None, background_frame=None, beautify=True):
	start_time = time.time()

	processors = []
	if source_face is not None:
		processors.append('face_swapper')
	if beautify:
		processors.append('face_enhancer')

	# frameIndex:int    //帧序
	# width:int         //图像宽
	# height:int        //图像高
	# format:String     //图像格式
	# length:int        //图像数据大小
	# data:byte[]       //图像数据

	frame_index = frame_data.get('frameIndex', 0)
	data = frame_data.get('data')
	width = frame_data['width']
	height = frame_data['height']
	format_type = frame_data['format']
	target_vision_frame = convert_to_bitmap(width, height, format_type, data)

	# 异步执行图像处理
	source_audio_frame = create_empty_audio_frame()
	i = 0
	for processor_module in get_processors_modules(processors):
		t = time.time()
		# logger.disable()
		# if processor_module.pre_process('stream'):
		target_vision_frame = processor_module.process_frame(
			{
				'source_face': source_face,
				'source_audio_frame': source_audio_frame,
				'target_vision_frame': target_vision_frame
			})
		logger.enable()
		e = time.time()
		logger.debug(
			f"Processed frame: index {frame_index}, processor {processors[i]}, processing time: {e - t:.4f} seconds",
			__name__)  # 打印处理时间
		i += 1

	if background_frame is not None:
		try:
			processors.append('background')
			t = time.time()
			target_vision_frame = merge_images(target_vision_frame, background_frame)
			e = time.time()
			logger.debug(
				f"Processed frame: index {frame_index}, processor background, processing time: {e - t:.4f} seconds",
				__name__)  # 打印处理时间
		except Exception as e:
			logger.error(f"background_frame error: {e}", __name__)

	end_time = time.time()
	logger.info(
		f"Processed frame: index {frame_index}, processors {processors}, processing time: {end_time - start_time:.4f} seconds",
		__name__)  # 打印处理时间

	# 获取图像的宽度和高度
	height, width, channels = target_vision_frame.shape

	# 将图像数据转换为字节数组
	image_data = bitmap_to_data(target_vision_frame, width, height, format_type)
	return {
		"frameIndex": frame_index,
		"width": width,
		"height": height,
		"data": image_data,
		"format": format_type,
		"length": len(image_data),
		"processing_time": end_time - start_time,
		"start": frame_data.get("start")
	}


def merge_images(frame, background_frame, alpha=0.5, beta=0.5):
	# 调整背景图像大小与前景图像一致
	height, width = frame.shape[:2]
	background_resized = cv2.resize(background_frame, (width, height))

	# 确保通道数一致
	if len(frame.shape) != len(background_resized.shape):
		if len(frame.shape) == 3 and len(background_resized.shape) == 2:
			background_resized = cv2.cvtColor(background_resized, cv2.COLOR_GRAY2BGR)
		elif len(frame.shape) == 2 and len(background_resized.shape) == 3:
			frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

	# 确保数据类型一致
	frame = frame.astype(np.uint8)
	background_resized = background_resized.astype(np.uint8)

	# 合并图像
	combined_image = cv2.addWeighted(frame, alpha, background_resized, beta, 0)
	return combined_image


def create_app():
	app = FastAPI()

	max_workers = state_manager.get_item('execution_thread_count')
	executor = ThreadPoolExecutor(max_workers=max_workers if max_workers else 4)  # 控制最大线程数
	logger.info(f"{max_workers} thread workers", __name__)

	@app.post('/process_image')
	async def process_image(
		image: UploadFile = File(...),
		water: UploadFile = File(None),
		swap: UploadFile = File(None),
		beautify: bool = Form(False)
	):
		frame_data = {}

		# 获取待处理图像
		image_bytes = await image.read()  # 读取图像字节
		frame_format = identify_image_format(image_bytes)
		frame_data['frameIndex'] = 0
		frame_data['width'] = 0
		frame_data['height'] = 0
		frame_data['format'] = frame_format
		frame_data['data'] = image_bytes

		# 获取待替换人脸图像
		source_face = None
		if swap:
			image_bytes = await swap.read()  # 读取图像字节
			try:
				image_format = identify_image_format(image_bytes)
				source_frame = convert_to_bitmap(0, 0, image_format, image_bytes)
				source_faces = get_many_faces([source_frame])
				source_face = get_average_face(source_faces)
			except Exception as e:
				traceback.print_exc()
				logger.error(f"Error processing swap image: {e}", __name__)

		# 获取背景图像
		background_frame = None
		if water:
			image_bytes = await water.read()
			try:
				image_format = identify_image_format(image_bytes)
				background_frame = convert_to_bitmap(0, 0, image_format, image_bytes)
			except Exception as e:
				traceback.print_exc()
				logger.error(f"Error processing water image: {e}", __name__)

		future = executor.submit(process_frame, frame_data, source_face, background_frame, beautify)
		processed_frame = await asyncio.wrap_future(future)

		return StreamingResponse(BytesIO(processed_frame['data']), media_type=image.content_type)

	@app.websocket("/ws")
	async def websocket_endpoint(websocket: WebSocket):
		await websocket.accept()

		# 创建一个缓冲区
		buffer = bytearray()
		# 有序字典用于保存处理后的帧数据
		results = OrderedDict()
		next_id_to_send = None
		send_queue = asyncio.Queue(maxsize=100)  # 用于存储待发送的数据帧
		stop_flag = False
		MAX_CHUNK_SIZE = 512 * 1024  # 512KB

		async def send_loop():
			"""异步发送队列中的数据帧"""
			while not stop_flag:
				try:
					processed_t = await asyncio.wait_for(send_queue.get(), timeout=5)
					if processed_t is None:
						break  # 若接收到 None，跳出循环
					# 创建完整数据包
					data_content = (
						struct.pack('!I', processed_t['frameIndex']) +
						struct.pack('!I', processed_t['width']) +
						struct.pack('!I', processed_t['height']) +
						struct.pack('!I', processed_t['length']) + processed_t['data']
					)

					packet_t = struct.pack('!II', 1, len(data_content)) + data_content
					checksum_t = zlib.crc32(data_content) & 0xFFFFFFFF
					packet_t += struct.pack('!Q', checksum_t)

					# 发送处理结果
					s_t = time.time()
					# for i in range(0, len(packet_t), MAX_CHUNK_SIZE):
					# 	chunk = packet_t[i:i + MAX_CHUNK_SIZE]
					# 	await websocket.send_bytes(chunk)
					await websocket.send_bytes(packet_t)
					e_t = time.time()
					total = e_t - processed_t['start']
					logger.info(
						f"Sent frame, index: {processed_t['frameIndex']}, w*h: {processed_t['width']}x{processed_t['height']},"
						f"length: {processed_t['length']}, format: {str(processed_t['format'])}, send time: {e_t - s_t}, total time: {total}",
						__name__)
					send_queue.task_done()
				# await asyncio.sleep(0.01)  # 添加小延时缓解缓冲区负载
				except asyncio.TimeoutError:
					# logger.debug(f"send_loop exit: timeout", __name__)
					continue
				except WebSocketDisconnect:
					# logger.debug(f"send_loop exit: disconnect", __name__)
					break
				except Exception:
					traceback.print_exc()
					# logger.debug(f"send_loop exit: exception", __name__)
					break
			logger.info(f"send_loop exit", __name__)

		# 启动发送循环
		send_task = asyncio.create_task(send_loop())

		# 设置初始参数
		background_frame = None
		source_face = None
		beautify = False
		try:
			while True:
				# 检查是否有按顺序完成的结果可以返回
				while next_id_to_send in results and results[next_id_to_send].done():
					processed_t = await asyncio.wrap_future(results[next_id_to_send])
					# 移除已发送的结果，并更新下一个待发送的帧编号
					del results[next_id_to_send]
					next_id_to_send += 1
					await send_queue.put(processed_t)

				# # 发送处理结果
				# s_t = time.time()
				# data_content = (
				# 	struct.pack('!I', processed_t['frameIndex']) +
				# 	struct.pack('!I', processed_t['width']) +
				# 	struct.pack('!I', processed_t['height']) +
				# 	struct.pack('!I', processed_t['length']) + processed_t['data']
				# )
				#
				# packet_t = struct.pack('!II', 1, len(data_content)) + data_content
				# checksum_t = zlib.crc32(data_content) & 0xFFFFFFFF
				# packet_t += struct.pack('!Q', checksum_t)
				#
				# # for i in range(0, len(packet_t), MAX_CHUNK_SIZE):
				# # 	chunk = packet_t[i:i + MAX_CHUNK_SIZE]
				# # 	await websocket.send_bytes(chunk)
				# await websocket.send_bytes(packet_t)
				# e_t = time.time()
				# total = e_t - processed_t['start']
				# logger.info(
				# 	f"Sent frame, index: {processed_t['frameIndex']}, w*h: {processed_t['width']}x{processed_t['height']},"
				# 	f"length: {processed_t['length']}, format: {str(processed_t['format'])}, send time: {e_t - s_t}, total time: {total}",
				# 	__name__)

				# 等待客户端请求
				try:
					# data = await websocket.receive_bytes()
					# 设置超时，单位为秒
					data = await asyncio.wait_for(websocket.receive_bytes(), 0.1)
					if not data:
						continue
					buffer.extend(data)  # 将接收到的数据添加到缓冲区
					logger.debug(f"Received data:  recv {len(data)}, total {len(buffer)}", __name__)
				except asyncio.TimeoutError:
					pass
				# logger.info(f"Timeout reached while waiting for recv data", __name__)

				while len(buffer) >= 8:  # 至少需要 8 字节来读取包类型和数据长度
					packet_type, data_length = struct.unpack('!II', buffer[:8])

					crc_len = 8
					# 检查缓冲区是否包含完整的数据包
					if len(buffer) < 8 + data_length + crc_len:  # +4 是校验和的长度
						break  # 缓冲区不够，等待下次接收

					logger.debug(f"Received data: packet type {packet_type}, data length {data_length}", __name__)

					# 提取完整的数据包
					packet = buffer[:8 + data_length + crc_len]
					buffer = buffer[8 + data_length + crc_len:]  # 移除已处理的数据包

					# 解析数据包内容
					content = packet[8:-crc_len]  # 去掉包头和校验和
					checksum = struct.unpack('!Q', packet[-crc_len:])[0]

					# 校验 CRC32
					calculated_checksum = zlib.crc32(content) & 0xFFFFFFFF
					if calculated_checksum != checksum:
						logger.error(
							f"CRC32 checksum mismatch {packet_type}, {checksum}, calculated: {calculated_checksum}",
							__name__)
					# continue

					# 根据包类型进行处理
					if packet_type == 0:  # 心跳包
						offset = 0

						device_id_length = struct.unpack('!I', content[offset:offset + 4])[0]
						offset += 4

						device_id = content[offset:offset + device_id_length].decode('utf-8')
						offset += device_id_length

						await websocket.send_bytes(packet)
						logger.debug(f"Sent heartbeat: device {device_id}", __name__)
					elif packet_type == 2:  # 参数更新包
						offset = 0

						beautify_flag = struct.unpack('!B', content[offset:offset + 1])[0]
						offset += 1

						background_image_length = struct.unpack('!I', content[offset:offset + 4])[0]
						offset += 4

						if background_image_length > 0:
							background_image_data = content[offset: offset + background_image_length]
							offset += background_image_length
							try:
								image_format = identify_image_format(background_image_data)
								background_frame = convert_to_bitmap(0, 0, image_format, background_image_data)
							except Exception as e:
								traceback.print_exc()
								logger.error(f'Error occurred while loading background image: {e}', __name__)

						swap_image_length = struct.unpack('!I', content[offset:offset + 4])[0]
						offset += 4

						if swap_image_length > 0:
							swap_image_data = content[offset:offset + swap_image_length]
							offset += swap_image_length
							try:
								image_format = identify_image_format(swap_image_data)
								source_frame = convert_to_bitmap(0, 0, image_format, swap_image_data)
								source_faces = get_many_faces([source_frame])
								source_face = get_average_face(source_faces)
							except Exception as e:
								traceback.print_exc()
								logger.error(f'Error occurred while loading swap image: {e}', __name__)
						beautify = beautify_flag == 1

					elif packet_type == 1:  # 相机帧
						start = time.time()
						offset = 0

						device_id_length = struct.unpack('!I', content[offset:offset + 4])[0]
						offset += 4

						device_id = content[offset:offset + device_id_length].decode('utf-8')
						offset += device_id_length

						user_length = struct.unpack('!I', content[offset:offset + 4])[0]
						offset += 4

						user = content[offset:offset + user_length].decode('utf-8')
						offset += user_length

						frame_index = struct.unpack('!I', content[offset:offset + 4])[0]
						offset += 4

						format_data_length = struct.unpack('!I', content[offset:offset + 4])[0]
						offset += 4

						format_type = ""
						if format_data_length > 0:
							format_type = content[offset:offset + format_data_length]
							offset += format_data_length

						width = struct.unpack('!I', content[offset:offset + 4])[0]
						offset += 4

						height = struct.unpack('!I', content[offset:offset + 4])[0]
						offset += 4

						image_data_length = struct.unpack('!I', content[offset:offset + 4])[0]
						offset += 4

						image_data = None
						if image_data_length > 0:
							image_data = content[offset:offset + image_data_length]
							offset += image_data_length

						logger.info(f"Received frame, index: {frame_index}, w*h: {width}x{height},"
									f"length: {image_data_length}, format: {str(format_type)}, time: {time.time() - start} ",
									__name__)

						if image_data_length == 0:
							continue

						frame_data = {
							'frameIndex': frame_index,
							'width': width,
							'height': height,
							'data': image_data,
							'format': format_type,
							'start': start,
						}

						future = executor.submit(process_frame, frame_data, source_face, background_frame, beautify)
						results[frame_index] = future  # 将 Future 按 frame_id 存入字典
						if next_id_to_send is None:
							next_id_to_send = frame_index
					else:
						logger.warn(f"Received unknown packet type {packet_type}", __name__)

		except WebSocketDisconnect as e:
			logger.info(f"WebSocket disconnected: {e.code}", __name__)
			stop_flag = True
			await send_queue.put(None)
			await send_task
		except Exception as e:
			traceback.print_exc()
			logger.error(f"Error in WebSocket connection: {e}", __name__)
		finally:
			if not stop_flag:
				logger.info(f"webSocket exit", __name__)
				await websocket.close()

	return app


app = create_app()


def start_app():
	port = 8005
	if state_manager.get_item('execution_queue_count') > 1:
		import subprocess
		subprocess.run([
			"uvicorn",
			f"facefusion.api:app",
			"--host", "0.0.0.0",
			"--port", str(port),
			"--workers", str(state_manager.get_item('execution_queue_count'))
		])
	else:
		app = create_app()
		uvicorn.run(app, host="0.0.0.0", port=port, ws_max_size=33554432, ws_ping_interval=60, ws_ping_timeout=60)
