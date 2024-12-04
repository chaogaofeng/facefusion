import struct
import time
import traceback
import zlib

import uvicorn
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from facefusion import logger
from facefusion.face_analyser import get_many_faces, get_average_face
from facefusion.audio import create_empty_audio_frame
from facefusion.processors.core import get_processors_modules
from app import utils

def process_frame(target_vision_frame, processors, source_face=None, background_frame=None):
	i = 0
	source_audio_frame = create_empty_audio_frame()
	for processor_module in get_processors_modules(processors):
		start_time = time.perf_counter()
		logger.disable()
		if processor_module.pre_process('stream'):
			target_vision_frame = processor_module.process_frame(
				{
					'source_face': source_face,
					'source_audio_frame': source_audio_frame,
					'target_vision_frame': target_vision_frame
				})
		logger.enable()
		logger.debug(
			f"Processed frame: processor {processors[i]}, processing time: {time.perf_counter() - start_time:.6f} seconds",
			__name__)  # 打印处理时间
		i += 1
	return target_vision_frame


try:
	import uvloop

	asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
	pass

app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
	try:
		await websocket.accept()
		buffer = bytearray()  # 创建一个缓冲区
		input_queue = asyncio.Queue(maxsize=600)  # 输入队列
		output_queue = asyncio.PriorityQueue(maxsize=600)  # 输出队列
		stop_event = asyncio.Event()  # 是否正在运行
		max_concurrent_tasks = 10  # 最大并发任务数
		task_timeout = 0.5  # 任务超时时间

		device_id = ''
		# 设置参数
		background_frame = None
		source_face = None
		beautify = False

		async def message_reader():
			"""从客户端读取消息并放入输入队列"""
			try:
				while not stop_event.is_set():
					data = await websocket.receive_bytes()
					buffer.extend(data)  # 将接收到的数据添加到缓冲区
					while len(buffer) >= 8:  # 至少需要 8 字节来读取包类型和数据长度
						packet_type, data_length = struct.unpack('!II', buffer[:8])
						crc_len = 8
						# 检查缓冲区是否包含完整的数据包
						if len(buffer) < 8 + data_length + crc_len:  # +8 是校验和的长度
							break  # 缓冲区不够，等待下次接收
						logger.debug(f"Received: packet type {packet_type}, data length {data_length}", __name__)

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
								f"CRC32 mismatch: packet type {packet_type}, checksum {checksum}, calculated {calculated_checksum}",
								__name__)
							continue

						# 根据包类型进行处理
						if packet_type == 0:  # 心跳包
							offset = 0

							device_id_length = struct.unpack('!I', content[offset:offset + 4])[0]
							offset += 4

							device_id = content[offset:offset + device_id_length].decode('utf-8')
							offset += device_id_length

							logger.debug(f"Received: heartbeat, device id {device_id}", __name__)
							await websocket.send_bytes(packet)
							logger.debug(f"Sent: heartbeat, device id {device_id}", __name__)
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
									image_format = utils.identify_image_format(background_image_data)
									background_frame = asyncio.run(
										utils.convert_to_bitmap(0, 0, image_format, background_image_data))
								except Exception as e:
									traceback.print_exc()
									logger.error(f'Error occurred while loading background image: {e}', __name__)

							swap_image_length = struct.unpack('!I', content[offset:offset + 4])[0]
							offset += 4

							if swap_image_length > 0:
								swap_image_data = content[offset:offset + swap_image_length]
								offset += swap_image_length
								try:
									image_format = utils.identify_image_format(swap_image_data)
									source_frame = asyncio.run(
										utils.convert_to_bitmap(0, 0, image_format, swap_image_data))
									source_faces = get_many_faces([source_frame])
									source_face = get_average_face(source_faces)
								except Exception as e:
									traceback.print_exc()
									logger.error(f'Error occurred while loading swap image: {e}', __name__)
							beautify = beautify_flag == 1

							logger.debug(
								f'Received: parameter update, beautify: {beautify}, background image: {background_frame is not None}, swap image: {source_face is not None}',
								__name__)

						elif packet_type == 1:  # 相机帧
							start_time = time.perf_counter()
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

							timestamp = struct.unpack('!Q', content[offset:offset + 8])[0]
							offset += 8

							compressed_length = struct.unpack('!I', content[offset:offset + 4])[0]
							offset += 4

							compressed = content[offset:offset + compressed_length].decode('utf-8')
							offset += compressed_length

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

							end_time = time.perf_counter()
							logger.info(
								f"Received: frame {frame_index}, w*h {width}x{height}, format {str(format_type)}, compress {compressed},"
								f"length {image_data_length}, unpack time {end_time - start_time:.6f}", __name__)

							if image_data_length == 0:
								continue

							frame_data = {
								'frameIndex': frame_index,
								'compressed': compressed,
								'format': format_type,
								'width': width,
								'height': height,
								'data': image_data,
								'data_length': image_data_length,
								'start_time': start_time,
							}
							try:
								await input_queue.put(frame_data)
							except asyncio.QueueFull:
								logger.warn(f"Input queue full, dropping frame {frame_index}", __name__)
						else:
							logger.warn(f"Received: packet type {packet_type} unknown", __name__)
					await input_queue.put(data)

			except WebSocketDisconnect as e:
				logger.warn(f"WebSocket disconnected, closing {e}", __name__)
			except Exception as e:
				traceback.print_exc()
				logger.error(f"Error occurred while message_reader: {e}", __name__)
			finally:
				stop_event.set()  # 标志停止

		async def message_writer():
			"""从输出队列按顺序发送消息到客户端"""
			try:
				while not stop_event.is_set():
					try:
						frame_data = await asyncio.wait_for(output_queue.get(), timeout=0.1)
						start_time = time.perf_counter()
						data_content = (
							struct.pack('!I', frame_data['frameIndex']) +
							struct.pack('!Q', int(time.time() * 1000)) +
							struct.pack('!I', frame_data['width']) +
							struct.pack('!I', frame_data['height']) +
							struct.pack('!I', frame_data['data_length']) + frame_data['data']
						)

						packet_t = struct.pack('!II', 1, len(data_content)) + data_content
						checksum_t = zlib.crc32(data_content) & 0xFFFFFFFF
						packet_t += struct.pack('!Q', checksum_t)
						await websocket.send_bytes(packet_t)
						end_time = time.perf_counter()
						logger.info(
							f"Sent: frame {frame_data['frameIndex']}, w*h {frame_data['width']}x{frame_data['height']}, length {frame_data['data_length']},"
							f"send time: {end_time - start_time:.6f}, total time: {end_time - frame_data['start_time']:.6f}",
							__name__)
						output_queue.task_done()
					except asyncio.TimeoutError:
						continue
			except WebSocketDisconnect as e:
				logger.info(f"WebSocket disconnected, closing {e}", __name__)
			except Exception as e:
				traceback.print_exc()
				logger.error(f"Error occurred while message_writer: {e}", __name__)
			finally:
				stop_event.set()  # 标志停止

		async def message_processor():
			"""处理输入队列中的消息并将结果放入输出队列"""

			async def process_message(frame_data):
				"""处理消息"""
				processors = []
				if source_face is not None:
					processors.append('face_swapper')
				if beautify:
					processors.append('face_enhancer')
				if not processors:
					return frame_data

				try:
					start_time = time.perf_counter()

					frame_index = frame_data.get('frameIndex', 0)
					compressed = frame_data.get('compressed', '')
					data = frame_data.get('data')
					width = frame_data['width']
					height = frame_data['height']
					format_type = frame_data['format']
					target_vision_frame = await utils.convert_to_bitmap(width, height, format_type, data, compressed)

					target_vision_frame = await asyncio.to_thread(
						process_frame,
						target_vision_frame,
						processors,
						source_face,
						background_frame
					)

					image_data = await utils.bitmap_to_data(target_vision_frame, width, height, format_type, compressed)

					logger.info(
						f"Processing: frame {frame_index}, total process time {time.perf_counter() - start_time:.6f}",
						__name__)

					height, width, channels = target_vision_frame.shape
					frame_data['width'] = width
					frame_data['height'] = height
					frame_data['data'] = image_data
					frame_data['data_length'] = len(image_data)
					return frame_data
				except Exception as e:
					traceback.print_exc()
					logger.error(f"Error occurred while processing message: {e}", __name__)

			try:
				while not stop_event.is_set():
					tasks = []
					for _ in range(max_concurrent_tasks):
						try:
							frame_data = await asyncio.wait_for(input_queue.get(), timeout=0.1)
							input_queue.task_done()
							task = asyncio.create_task(
								asyncio.wait_for(process_message(frame_data), timeout=task_timeout)
							)
							tasks.append(task)
						except asyncio.TimeoutError:
							continue
					if tasks:
						done, pending = await asyncio.wait(tasks,
														   return_when=asyncio.ALL_COMPLETED)  # 等待所有任务完成，捕获超时或其他异常
						for task in done:
							try:
								frame_data = task.result()  # 获取结果，可能抛出异常
								if frame_data is not None:
									await output_queue.put((frame_data['frameIndex'], frame_data))
							except asyncio.QueueFull:
								logger.error(f"Output queue full. Dropping frame.", __name__)
							except asyncio.TimeoutError:
								logger.error(f"Task timeout. Dropping frame.", __name__)
							except Exception as e:
								traceback.print_exc()
								logger.error(f"Error occurred while processing task: {e}. Dropping frame", __name__)
						for task in pending:
							task.cancel()
							with asyncio.suppress(asyncio.CancelledError):
								await task
			except Exception as e:
				traceback.print_exc()
				logger.error(f"Error occurred while message_processor: {e}", __name__)
			finally:
				stop_event.set()

		reader_task = asyncio.create_task(message_reader())
		processor_task = asyncio.create_task(message_processor())
		writer_task = asyncio.create_task(message_writer())

		try:
			await asyncio.gather(reader_task, processor_task, writer_task)
		except asyncio.CancelledError:
			pass
		finally:
			# 确保任务被取消
			reader_task.cancel()
			processor_task.cancel()
			writer_task.cancel()
			await asyncio.gather(reader_task, processor_task, writer_task, return_exceptions=True)
	except Exception as e:
		logger.error(f"Error occurred while processing websocket: {e}", __name__)
	finally:
		await websocket.close()


if __name__ == '__main__':
	port = 8000
	uvicorn.run(app, host="0.0.0.0", port=port, ws_max_size=33554432, ws_ping_interval=60, ws_ping_timeout=60)
