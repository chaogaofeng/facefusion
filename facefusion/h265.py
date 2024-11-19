import threading
import time

import ffmpeg


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
		self.decode_lock = threading.Lock()  # 解码专用锁
		self.encode_lock = threading.Lock()  # 编码专用锁

	def start_decode_process(self):
		self.decode_process = (
			ffmpeg
			.input('pipe:0', vcodec=self.vcodec)
			.output('pipe:1', format='rawvideo', pix_fmt=self.pix_fmt, s=f'{self.width}x{self.height}')
			.run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
		)

	def start_encode_process(self):
		self.encode_process = (
			ffmpeg
			.input('pipe:0', format='rawvideo', pix_fmt=self.pix_fmt, s=f'{self.width}x{self.height}')
			.output('pipe:1', vcodec=self.vcodec, format=self.format, pix_fmt=self.pix_fmt, preset=self.preset)
			.run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
		)

	def decode(self, data):
		"""decode"""
		if not self.decode_process:
			self.start_decode_process()
		with self.decode_lock:
			try:
				self.decode_process.stdin.write(data)
				self.encode_process.stdin.flush()
				raw_frame = self.decode_process.stdout.read()
				return raw_frame
			except BrokenPipeError:
				self.start_decode_process()
				return None
			except Exception as e:
				print(f"video decode: {e}")
				return None

	def encode(self, data):
		"""encode"""
		if not self.encode_process:
			self.start_encode_process()
		print("====0")
		with self.encode_lock:
			try:
				print("====1")
				self.encode_process.stdin.write(data)
				self.encode_process.stdin.flush()
				print("====2")
				encoded_frame = self.encode_process.stdout.read()  # H.265 格式数据
				print("====3")
				return encoded_frame
			except BrokenPipeError:
				self.start_encode_process()
				return None
			except Exception as e:
				print(f"video encode: {e}")
				return None

	def close(self):
		"""关闭解码和编码进程"""
		if self.decode_process:
			self.decode_process.stdin.close()
			self.decode_process.stdout.close()
			self.decode_process.stderr.close()
			self.decode_process.wait()
		if self.encode_process:
			self.encode_process.stdin.close()
			self.encode_process.stdout.close()
			self.encode_process.stderr.close()
			self.encode_process.wait()


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
