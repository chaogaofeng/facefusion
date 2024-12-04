import asyncio
import time

import cv2
import ffmpeg
import numpy as np

from facefusion import logger


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


async def encode_h265(data, width, height, format='hevc', pix_fmt='yuv420p', vcodec='libx265'):
	"""
	将图像数据压缩为 H.265 格式并返回字节流。

	:param data: 原始图像数据
	:param width: 图像宽度
	:param height: 图像高度
	:param format: 视频解码器，默认 'hevc'
	:param pix_fmt: 输入图像像素格式，默认 'yuv420p'
	:param vcodec: 输出视频编码器，默认 'libx265'
	:return: 压缩后的字节流
	"""
	try:
		start_time = time.perf_counter()

		# 构造 FFmpeg 命令
		command = (
			ffmpeg
			.input('pipe:0', format='rawvideo', pix_fmt=pix_fmt, s=f'{width}x{height}')
			.output('pipe:1', vcodec=vcodec, format=format, pix_fmt='yuv420p', preset='ultrafast')
			.compile()
		)

		# 启动子进程
		process = await asyncio.create_subprocess_exec(
			*command,
			stdin=asyncio.subprocess.PIPE,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE
		)

		# 传递数据并获取输出
		stdout, stderr = await process.communicate(input=data)
		# 检查子进程返回码
		if process.returncode != 0:
			raise ValueError(
				f"{format} 压缩失败: \nSTDOUT: {stdout.decode('utf-8', errors='ignore')}\n"
				f"STDERR: {stderr.decode('utf-8', errors='ignore')}"
			)

		# 输出调试信息
		elapsed_time = time.perf_counter() - start_time
		logger.debug(f"{format} 压缩完成: 输入 {len(data)} ==> 输出 {len(stdout)}, 耗时 {elapsed_time:.6f}", __name__)

		return stdout  # 返回压缩后的字节流
	except asyncio.SubprocessError as e:
		raise ValueError(f"{format} 编码过程失败（子进程错误）: {e}")
	except Exception as e:
		raise ValueError(f"{format} 编码过程失败（未知错误）: {e}")


async def decode_h265(data, width, height, pix_fmt='yuv420p', vcodec='hevc'):
	"""
	从 H.265 压缩字节流解码并返回解压字节流。

	:param data: H.265 编码的字节流
	:param width: 解码后视频的宽度
	:param height: 解码后视频的高度
	:param pix_fmt: 解码后视频的像素格式，默认 'yuv420p'
	:param vcodec: 视频解码器，默认 'hevc'
	:return: 解码后的字节流
	"""
	try:
		start_time = time.perf_counter()

		# 构造 FFmpeg 解码命令
		command = (
			ffmpeg
			.input('pipe:0', vcodec=vcodec)  # 输入 H.265 流
			.output('pipe:1', format='rawvideo', pix_fmt=pix_fmt, s=f'{width}x{height}')
			.compile()
		)

		# 启动异步子进程
		process = await asyncio.create_subprocess_exec(
			*command,
			stdin=asyncio.subprocess.PIPE,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE
		)

		# 向子进程写入数据并获取输出
		stdout, stderr = await process.communicate(input=data)

		# 检查子进程的返回码
		if process.returncode != 0:
			raise ValueError(
				f"{vcodec} 解码失败: \nSTDOUT: {stdout.decode('utf-8', errors='ignore')}\n"
				f"STDERR: {stderr.decode('utf-8', errors='ignore')}"
			)

		# 记录调试信息
		elapsed_time = time.perf_counter() - start_time
		print(f"{vcodec} 解码完成: 输入 {len(data)} ==> {len(stdout)} 字节，耗时 {elapsed_time:.3f} 秒")

		return stdout  # 返回解码后的字节流

	except asyncio.SubprocessError as e:
		raise ValueError(f"{vcodec} 解码过程失败（子进程错误）: {e}")
	except Exception as e:
		raise ValueError(f"{vcodec} 解码过程失败（未知错误）: {e}")


async def encode_video(data, width, height, format='video/hevc', pix_fmt='yuv420p'):
	# format h265的是video/hevc h264的是video/avc
	if format == 'video/avc':
		return await encode_h265(data, width, height, format='h264', pix_fmt=pix_fmt, vcodec='libx264')
	else:
		return await encode_h265(data, width, height, format='hevc', pix_fmt=pix_fmt, vcodec='libx265')


async def decode_video(data, width, height, format='video/hevc', pix_fmt='yuv420p'):
	if format == 'video/avc':
		return await decode_h265(data, width, height, pix_fmt, vcodec='h264')
	else:
		return await decode_h265(data, width, height, pix_fmt, vcodec='hevc')


async def convert_to_bitmap(width, height, format_type, data, compressed_format=None):
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
		if compressed_format:
			data = await decode_video(data, width, height, pix_fmt='nv21', format=compressed_format)
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
		if compressed_format:
			data = await decode_video(data, width, height, pix_fmt='yuv420p', format=compressed_format)
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


async def bitmap_to_data(image, width, height, format_type, compressed_format=None):
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
		if compressed_format:
			return encode_video(np.concatenate((y_plane.flatten(), vu_plane)).tobytes(), width, height, pix_fmt='nv21',
								format=compressed_format)
		return np.concatenate((y_plane.flatten(), vu_plane)).tobytes()
	elif format_type == "JPEG":
		_, jpeg_data = cv2.imencode('.jpg', image)
		return jpeg_data.tobytes()
	elif format_type == "PNG":
		_, png_data = cv2.imencode('.png', image)
		return png_data.tobytes()
	elif format_type == "YUV_420_888":
		yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
		if compressed_format:
			return encode_video(yuv_image.tobytes(), width, height, pix_fmt='yuv420p', format=compressed_format)
		return yuv_image.tobytes()
	# return yuv_image.tobytes()
	elif format_type == "YUV_422_888":
		yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_Y422)
		return yuv_image.tobytes()
	elif format_type == "YUV_444_888":
		yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
		return yuv_image.tobytes()
	else:
		raise ValueError(f"不支持的图像格式 {format_type}")


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
	yuv_420_888 = np.vstack((y_plane, uv_interleaved))

	tasks = []
	for i in range(1, 100):
		# 异步调用 encode_video
		tasks.append(encode_video(yuv_420_888.tobytes(), width, height))

	# 并发执行所有编码任务
	await asyncio.gather(*tasks)


if __name__ == '__main__':
	asyncio.run(main())
