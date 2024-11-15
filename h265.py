import cv2
import ffmpeg
import numpy as np

from facefusion.api import encode_h265, decode_h265


def compress_jpg_to_h265(jpg_path):
	with open(jpg_path, 'rb') as f:
		jpg_data = f.read()

	# 使用 FFmpeg 进行 H.265 压缩并转换为字节流
	out, err = (
		ffmpeg
		.input('pipe:0')  # 从标准输入读取
		.output('pipe:1', vcodec='libx265', format='hevc', pix_fmt='yuv420p')  # 输出 H.265 格式，YUV420p
		.run(input=jpg_data, capture_stdout=True, capture_stderr=True)
	)

	print('compress Output message:', len(out) if out else 0)
	print('compress Error message:', err)
	return out  # 返回压缩后的 H.265 字节流


def decode_h265_to_frame(compressed_data):
	# 解码 H.265 字节流
	out, err = (
		ffmpeg
		.input('pipe:0', format='hevc')  # 指定输入格式为 H.265 (HEVC)
		.output('pipe:1', format='rawvideo', pix_fmt='bgr24', s='1920x1080')  # 指定输出格式和尺寸
		.run(input=compressed_data, capture_stdout=True, capture_stderr=True)
	)

	print('decode Output message:', len(out) if out else 0)
	print('decode Error message:', err)
	return out


if __name__ == '__main__':
	jpg_path = 'test.jpg'
	compressed_data = compress_jpg_to_h265(jpg_path)
	print("compressed data:", len(compressed_data))
	uncompressed_data = decode_h265_to_frame(compressed_data)
	print("uncompressed data:", len(uncompressed_data))

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

	decode_h265(yuv_420_888.tobytes(), width, height)


