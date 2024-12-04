import asyncio
import struct
import time
import zlib

import cv2
import numpy as np
import websockets

from app.utils import encode_h265


async def main():
	jpg_path = 'test.jpg'

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

	async def send_messages(websocket):
		"""
		并发发送消息
		"""
		frameIndex = 1
		try:
			while True:
				for i in range(30):
					t = time.perf_counter()
					image_data = encode_h265(yuv_frame.tobytes(), width, height)
					device_id = "TestDevice"
					user = "User1"
					format_type = "YUV_420_888"

					# 打包数据
					device_id_bytes = device_id.encode('utf-8')
					user_bytes = user.encode('utf-8')
					format_bytes = format_type.encode('utf-8')
					compress_bytes = 'video/hevc'.encode('utf-8')

					packet_data = (
						struct.pack('!I', len(device_id_bytes)) + device_id_bytes +
						struct.pack('!I', len(user_bytes)) + user_bytes +
						struct.pack('!I', frameIndex) +  # frame index
						struct.pack('!Q', int(time.time() * 1000)) +  # frame index
						struct.pack('!I', len(compress_bytes)) + compress_bytes +
						struct.pack('!I', len(format_bytes)) + format_bytes +
						struct.pack('!I', width) +  # width
						struct.pack('!I', height) +  # height
						struct.pack('!I', len(image_data)) + image_data
					)

					data_length = len(packet_data)
					header = struct.pack('!II', 1, data_length)
					packet = header + packet_data

					# 计算 CRC32 校验和并添加到包尾
					checksum = zlib.crc32(packet_data) & 0xFFFFFFFF
					packet += struct.pack('!Q', checksum)

					await websocket.send(packet)
					print(f"Sent: frame {frameIndex}, w*h {width}x{height}, length {data_length}, time {time.perf_counter() - t:.3f}")
					frameIndex += 1
					await asyncio.sleep(1 / 30)  # 模拟发送间隔
		except websockets.ConnectionClosed as e:
			print(f"Connection closed: {e}")
		except Exception as e:
			print(f"Error sending message: {e}")

	async def receive_messages(websocket):
		"""
		并发接收消息
		"""
		try:
			while True:
				t = time.perf_counter()
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
					continue
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

					print(f"Received: frame {frame_index}, w*h {width}x{height}, length {image_len}, time {time.perf_counter() - t:.3f}")

		except websockets.ConnectionClosed as e:
			print(f"Connection closed: {e}")
		except Exception as e:
			print(f"Error receiving message: {e}")

	uri = "ws://127.0.0.1:8005/ws"  # 替换为你的 WebSocket 服务器地址
	async with websockets.connect(uri) as websocket:
		# 并发任务
		send_task = asyncio.create_task(send_messages(websocket))
		receive_task = asyncio.create_task(receive_messages(websocket))

		# 等待任务完成
		await asyncio.gather(send_task, receive_task)


if __name__ == '__main__':
	asyncio.run(main())
