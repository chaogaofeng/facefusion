import asyncio
import os
import struct
import subprocess
import time
import zlib

import cv2
import numpy as np
import psutil
import websockets

from facefusion.api import decode_h265, convert_to_bitmap, encode_h265
from facefusion.uis.components.webcam import get_webcam_capture


async def send_packet(websocket, packet_type, data):
	# 创建包头
	data_length = len(data)
	header = struct.pack('!II', packet_type, data_length)
	packet = header + data

	# 计算 CRC32 校验和并添加到包尾
	checksum = zlib.crc32(data) & 0xFFFFFFFF
	packet += struct.pack('!Q', checksum)

	await websocket.send(packet)
	print(f"Sent packet type {packet_type} with length {data_length}")


async def heartbeat(websocket):
	device_id = "TestDevice"
	device_id_bytes = device_id.encode('utf-8')
	device_id_length = struct.pack('!I', len(device_id_bytes))

	content = device_id_length + device_id_bytes
	await send_packet(websocket, packet_type=0, data=content)


async def send_camera_frame(websocket, frame_index, h265_stream, width, height):
	# 将图像帧编码为 JPEG 格式
	# _, image_data = cv2.imencode('.jpg', frame)
	# image_data = image_data.tobytes()

	# image_data = b'\x09' * 1024 * 1024 * 4
	image_data = h265_stream

	device_id = "TestDevice"
	user = "User1"
	format_type = "YUV_420_888"

	# 打包数据
	device_id_bytes = device_id.encode('utf-8')
	user_bytes = user.encode('utf-8')
	format_bytes = format_type.encode('utf-8')
	compress_bytes = 'h265'.encode('utf-8')

	packet_data = (
		struct.pack('!I', len(device_id_bytes)) + device_id_bytes +
		struct.pack('!I', len(user_bytes)) + user_bytes +
		struct.pack('!I', frame_index) +  # frame index
		struct.pack('!Q', int(time.time()*1000)) +  # frame index
		struct.pack('!I', len(compress_bytes)) + compress_bytes +
		struct.pack('!I', len(format_bytes)) + format_bytes +
		struct.pack('!I', width) +  # width
		struct.pack('!I', height) +  # height
		struct.pack('!I', len(image_data)) + image_data
	)

	await send_packet(websocket, packet_type=1, data=packet_data)


# Receiving frames and saving them
async def receive_frame(websocket):
	try:
		# Receive a packet from the WebSocket
		packet = await websocket.recv()

		# Extract the header information
		packet_type, data_length = struct.unpack('!II', packet[:8])
		data = packet[8:-8]  # Exclude header and checksum
		checksum_received = struct.unpack('!Q', packet[-8:])[0]

		# Validate checksum
		checksum_calculated = zlib.crc32(data) & 0xFFFFFFFF
		if checksum_received != checksum_calculated:
			print("Checksum mismatch!")
			return

		# Process the received data if it's an image frame
		if packet_type == 1:
			# Unpack the data
			offset = 0
			frame_index = struct.unpack('!I', data[offset:offset + 4])[0]
			offset += 4
			timestamp = struct.unpack('!Q', data[offset:offset + 8])[0]
			offset += 8
			width = struct.unpack('!I', data[offset:offset + 4])[0]
			offset += 4
			height = struct.unpack('!I', data[offset:offset + 4])[0]
			offset += 4
			image_len = struct.unpack('!I', data[offset:offset + 4])[0]
			offset += 4
			image_data = data[offset:offset + image_len]

			print(f"Recv Frame index: {frame_index}, diff {time.time() - timestamp/1000}")

			# Save the received frame to disk
			# frame_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
			# os.makedirs('images', exist_ok=True)
			# if frame_image is not None:
			# 	filename = f"images/frame_{frame_index}.jpg"
			# 	cv2.imwrite(filename, frame_image)
			# 	print(f"Saved frame {frame_index} to {filename}, length: {len(image_data)}")
			# else:
			# 	print(f"Failed to decode frame {frame_index}")
		else:
			print(f"Received packet type {packet_type}, but it's not an image frame.")
	except websockets.exceptions.ConnectionClosed:
		print("WebSocket connection closed.")

	except Exception as e:
		print(f"Error receiving frame: {e}")


async def main():
	# webcam_width = 1920
	# webcam_height = 1080
	# webcam_capture = get_webcam_capture()
	# if webcam_capture and webcam_capture.isOpened():
	# 	# webcam_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # type:ignore[attr-defined]
	# 	webcam_capture.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
	# 	webcam_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
	# 	webcam_capture.set(cv2.CAP_PROP_FPS, 30)


	jpg_path = 'test.jpg'
	# compressed_data = compress_jpg_to_h265(jpg_path)
	# print("compressed data:", len(compressed_data))
	# uncompressed_data = decode_h265_to_frame(compressed_data)
	# print("uncompressed data:", len(uncompressed_data))

	bgr_image = cv2.imread(jpg_path)
	# 获取图像尺寸
	height, width = bgr_image.shape[:2]

	# 转换为 YUV_I420 格式
	yuv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YUV_I420)

	# YUV_420_888 格式的内存布局
	# - Y 分量为单独的平面
	# - U 和 V 分量按交错方式存储
	y_plane = yuv_image[:height, :]
	u_plane = yuv_image[height:height + height // 4, :].reshape((height // 2, width // 2))
	v_plane = yuv_image[height + height // 4:, :].reshape((height // 2, width // 2))

	# 将 U 和 V 平面交错存储
	uv_interleaved = np.empty((height // 2, width), dtype=np.uint8)
	uv_interleaved[:, 0::2] = u_plane
	uv_interleaved[:, 1::2] = v_plane

	# 最终 YUV_420_888 格式
	yuv_frame = np.vstack((y_plane, uv_interleaved))

	# # 配置 FFmpeg 子进程进行 H.265 编码（接受原始 YUV 数据）
	# ffmpeg_command = [
	# 	'ffmpeg',
	# 	'-y',  # 覆盖输出
	# 	'-f', 'rawvideo',  # 原始视频流
	# 	'-pixel_format', 'yuv420p',  # YUV 4:2:0 格式
	# 	'-video_size', f'{width}x{height}',  # 视频分辨率
	# 	'-framerate', str(30),  # 帧率
	# 	'-i', '-',  # 输入来自标准输入
	# 	'-c:v', 'libx265',  # H.265 编码
	# 	'-preset', 'fast',  # 编码速度
	# 	'-f', 'hevc',  # 输出为原始 H.265 流
	# 	'-'  # 输出到标准输出
	# ]
	# ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

	uri = "ws://36.133.28.180:8005/ws"  # 替换为你的 WebSocket 服务器地址
	async with websockets.connect(uri) as websocket:
		frameIndex = 0
		while True:
			# if capture_frame is None:
			# _, capture_frame = webcam_capture.read()
			# 转换为 YUV_420 格式
			# yuv_frame = cv2.cvtColor(capture_frame, cv2.COLOR_BGR2YUV_I420)

			# # 将帧写入 FFmpeg 的标准输入
			# ffmpeg_process.stdin.write(yuv_frame.tobytes())
			#
			# # 从标准输出获取 H.265 字节流
			# h265_stream = ffmpeg_process.stdout.read()

			h265_stream = encode_h265(yuv_frame.tobytes(), width, height)
			print(f"encode: {len(h265_stream)}")

			frameIndex += 1
			t = time.time()
			await send_camera_frame(websocket, frameIndex, h265_stream, width, height)
			e = time.time()
			print("capture_frame", "frameIndex", frameIndex, "send time", e - t)

			if frameIndex > 1:
				t = time.time()
				await receive_frame(websocket)
				e = time.time()
				print("capture_frame", "frameIndex", frameIndex, "recv time", e - t)

		# process = psutil.Process()
		# # 获取内存信息
		# mem_info = process.memory_info()
		# print(f"RSS: {mem_info.rss / 1024 / 1024} M")
		# print(f"VMS: {mem_info.vms / 1024 / 1024} M")
		#
		# time.sleep(0.1)


if __name__ == '__main__':
	asyncio.run(main())
