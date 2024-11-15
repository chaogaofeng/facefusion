import ffmpeg


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
	decode_h265_to_frame(compressed_data)
