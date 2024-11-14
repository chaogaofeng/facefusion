import asyncio
import os
import struct
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


async def send_camera_frame(websocket, frame_index, frame):
	# 将图像帧编码为 JPEG 格式
	_, image_data = cv2.imencode('.jpg', frame)
	image_data = image_data.tobytes()

	# image_data = b'\x09' * 1024 * 1024 * 4

	device_id = "TestDevice"
	user = "User1"
	format_type = "JPEG"

	# 打包数据
	device_id_bytes = device_id.encode('utf-8')
	user_bytes = user.encode('utf-8')
	format_bytes = format_type.encode('utf-8')

	packet_data = (
		struct.pack('!I', len(device_id_bytes)) + device_id_bytes +
		struct.pack('!I', len(user_bytes)) + user_bytes +
		struct.pack('!I', frame_index) +  # frame index
		struct.pack('!I', len(format_bytes)) + format_bytes +
		struct.pack('!I', 1920) +  # width
		struct.pack('!I', 1080) +  # height
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
			width = struct.unpack('!I', data[offset:offset + 4])[0]
			offset += 4
			height = struct.unpack('!I', data[offset:offset + 4])[0]
			offset += 4
			image_len = struct.unpack('!I', data[offset:offset + 4])[0]
			offset += 4
			image_data = data[offset:offset + image_len]

			# Save the received frame to disk
			frame_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
			os.makedirs('images', exist_ok=True)
			if frame_image is not None:
				filename = f"images/frame_{frame_index}.jpg"
				cv2.imwrite(filename, frame_image)
				print(f"Saved frame {frame_index} to {filename}, length: {len(image_data)}")
			else:
				print(f"Failed to decode frame {frame_index}")
		else:
			print(f"Received packet type {packet_type}, but it's not an image frame.")
	except websockets.exceptions.ConnectionClosed:
		print("WebSocket connection closed.")

	except Exception as e:
		print(f"Error receiving frame: {e}")


async def main():
	webcam_width = 1920
	webcam_height = 1080
	webcam_capture = get_webcam_capture()
	if webcam_capture and webcam_capture.isOpened():
		webcam_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # type:ignore[attr-defined]
		webcam_capture.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
		webcam_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
		webcam_capture.set(cv2.CAP_PROP_FPS, 30)

	uri = "ws://36.133.28.180:8005/ws"  # 替换为你的 WebSocket 服务器地址
	async with websockets.connect(uri) as websocket:
		frameIndex = 0
		while True:
			# if capture_frame is None:
			_, capture_frame = webcam_capture.read()
			frameIndex += 1
			t = time.time()
			await send_camera_frame(websocket, frameIndex, capture_frame)
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
	with open('encode.txt', 'rb') as f:
		h265_bytes = f.read()
		data = decode_h265(h265_bytes, 144,176)
		image = convert_to_bitmap(144,176,'YUV_420_888',data)
		encode_h265(image)
	# asyncio.run(main())
