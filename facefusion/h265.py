import fcntl
import threading
import time
import os
import ffmpeg
import atexit


class VideoTranscoder:
	def __init__(self, width, height, vcodec='libx265', format='hevc', preset='ultrafast', pix_fmt='yuv420p'):
		"""
		:param vcodec: libx265、hevc_nvenc、libx264、h264_nvenc
		:param format: hevc、h264
		:param preset: ultrafast
		"""
		self.width = width
		self.height = height
		self.vcodec = vcodec
		self.format = format
		self.preset = preset
		self.pix_fmt = pix_fmt
		self.decode_process = None
		self.encode_process = None
		if self.pix_fmt == 'yuv420p':
			self.frame_size = width * height * 3 // 2  # yuv420p 每帧大小
		self.decode_lock = threading.Lock()  # 解码专用锁
		self.encode_lock = threading.Lock()  # 编码专用锁
		atexit.register(self.close)

	def start_decode_process(self):
		self.close_decode_process()
		self.decode_process = (
			ffmpeg
			.input('pipe:0', vcodec=self.vcodec)
			.output('pipe:1', format='rawvideo', pix_fmt=self.pix_fmt, s=f'{self.width}x{self.height}')
			.run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
		)

	def start_encode_process(self, gop_size=1):
		self.close_encode_process()
		self.encode_process = (
			ffmpeg
			.input('pipe:0', format='rawvideo', pix_fmt=self.pix_fmt, s=f'{self.width}x{self.height}')
			.output('pipe:1', vcodec=self.vcodec, format=self.format, pix_fmt=self.pix_fmt, preset=self.preset, g=gop_size)
			.run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
		)
		# 设置 stdout 为非阻塞模式
		flags = fcntl.fcntl(self.encode_process.stdout, fcntl.F_GETFL)
		fcntl.fcntl(self.encode_process.stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)

	def decode(self, data):
		with self.decode_lock:
			if not self.decode_process or self.decode_process.poll() is not None:
				if not self.decode_process or self.decode_process.poll() is not None:
					print("start decode process...")
					self.start_decode_process()
			try:
				self.decode_process.stdin.write(data)
				self.decode_process.stdin.flush()
				raw_frame = self.decode_process.stdout.read(self.frame_size)
				return raw_frame
			except BrokenPipeError:
				self.start_decode_process()
				return None
			except Exception as e:
				print(f"video decode: {e}")
				return None

	def encode(self, data):
		"""encode"""
		with self.encode_lock:
			if not self.encode_process or self.encode_process.poll() is not None:
				if not self.encode_process or self.encode_process.poll() is not None:
					print("start encode process...")
					self.start_encode_process()
			try:
				if len(data) != self.frame_size:
					print(f"Invalid frame size: {len(data)}. Expected: {self.frame_size}")
					return None
				self.encode_process.stdin.write(data)
				self.encode_process.stdin.flush()
				encoded_frame = self.encode_process.stdout.read(self.frame_size)  # H.265 格式数据
				return encoded_frame
			except BrokenPipeError:
				self.start_encode_process()
				return None
			except Exception as e:
				print(f"video encode: {e}")
				return None

	def close(self):
		"""关闭解码和编码进程"""
		self.close_decode_process()
		self.close_encode_process()

	def close_decode_process(self):
		if self.decode_process:
			self.decode_process.stdin.close()
			self.decode_process.stdout.close()
			self.decode_process.stderr.close()
			self.decode_process.wait()
			self.decode_process = None

	def close_encode_process(self):
		if self.encode_process:
			self.encode_process.stdin.close()
			self.encode_process.stdout.close()
			self.encode_process.stderr.close()
			self.encode_process.wait()
			self.encode_process = None


if __name__ == '__main__':
	width = 1920
	height = 1080

	transcoder = VideoTranscoder(width, height)

	import os

	input_data = os.urandom(width * height * 3 // 2)

	for i in range(100):
		t = time.time()
		decoded_frame = transcoder.decode(input_data)
		if decoded_frame is not None:
			print("decode success!", time.time() - t)
		else:
			print("decode fail!")

		t = time.time()
		encoded_frame = transcoder.encode(decoded_frame if decoded_frame else input_data)
		if encoded_frame is not None:
			print("encode success!", time.time() - t)
		else:
			print("encode fail!!")

	transcoder.close()
