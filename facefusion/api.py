import asyncio
import time
from collections import OrderedDict

from fastapi import FastAPI, File, UploadFile, Form, WebSocket
from fastapi.responses import StreamingResponse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import cv2
import numpy as np
from starlette.websockets import WebSocketDisconnect

from facefusion import logger, state_manager
from facefusion.audio import create_empty_audio_frame
from facefusion.face_analyser import get_many_faces, get_average_face
from facefusion.processors.core import get_processors_modules

executor = None

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
        raise ValueError("不支持的图像格式")

def convert_to_bitmap(width, height, format, data):
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

    if format == "RGBA_8888":
        image_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
        image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
    elif format == "RGBX_8888":
        image_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
        image = cv2.cvtColor(image_array[:, :, :3], cv2.COLOR_RGB2BGR)
    elif format == "RGB_888":
        image_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        image =  cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    elif format == "RGB_565":
        image_array = np.frombuffer(data, dtype=np.uint16).reshape((height, width))
        image =  cv2.cvtColor(image_array, cv2.COLOR_BGR5652BGR)
    elif format == "NV21":
        image_array = np.frombuffer(data, dtype=np.uint8).reshape((height + height // 2, width))
        image =  cv2.cvtColor(image_array, cv2.COLOR_YUV2BGR_NV21)
    elif format == "JPEG":
        image_array = np.frombuffer(data, dtype=np.uint8)
        image =  cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    elif format == "PNG":
        image_array = np.frombuffer(data, dtype=np.uint8)
        image =  cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
    elif format == "YUV_420_888":
        image_array = np.frombuffer(data, dtype=np.uint8).reshape((height * 3 // 2, width))
        image =  cv2.cvtColor(image_array, cv2.COLOR_YUV2BGR_I420)
    elif format == "YUV_422_888":
        image_array = np.frombuffer(data, dtype=np.uint8).reshape((height * 2, width))
        image =  cv2.cvtColor(image_array, cv2.COLOR_YUV2BGR_Y422)
    elif format == "YUV_444_888":
        image_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        image =  cv2.cvtColor(image_array, cv2.COLOR_YUV2BGR)
    else:
        raise ValueError("不支持的图像格式")

    if image is None:
        raise ValueError("Cannot convert image to bitmap")
    return image

def process_frame(frame_data, source_face=None, background_frame=None, beautify=True):
    start_time = time.time()

    processors = []
    if beautify:
        processors.append('face_enhancer')
    if source_face is not None:
        processors.append('face_swapper')

    # frameIndex:int    //帧序
    # width:int         //图像宽
    # height:int        //图像高
    # format:String     //图像格式
    # length:int        //图像数据大小
    # data:byte[]       //图像数据

    frame_index = frame_data.get('frameIndex', 0)
    data = frame_data['data']
    width = frame_data['width']
    height = frame_data['height']
    format =frame_data['format']
    target_vision_frame = convert_to_bitmap(width, height, data, format)

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
        logger.info(f"{frame_index}: processor {processors[i]}, processing time: {e - t:.4f} seconds",
                    __name__)  # 打印处理时间
        i += 1

    if background_frame is not None:
        t = time.time()
        target_vision_frame = merge_images(target_vision_frame, background_frame)
        e = time.time()
        logger.info(f"{frame_index}: processor background, processing time: {e - t:.4f} seconds",
                    __name__)  # 打印处理时间

    end_time = time.time()
    logger.info(f"{frame_index}: processors {processors}, processing time: {end_time - start_time:.4f} seconds",
                __name__)  # 打印处理时间

    # 获取图像的宽度和高度
    height, width, channels = target_vision_frame.shape

    # 将图像数据转换为字节数组
    image_data = bitmap_image.tobytes()
    return {
        "width": width,
        "height": height,
        "data": image_data,
        "format": format,
        "processing_time": end_time - start_time
    }


def merge_images(frame, background_image):
    # 确保背景图像和主图像都是 RGB 格式
    if len(frame.shape) == 2:  # 如果主图像是灰度图
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if len(background_image.shape) == 2:  # 如果背景图像是灰度图
        background_image = cv2.cvtColor(background_image, cv2.COLOR_GRAY2BGR)

    # 将背景图像调整为与主图像相同的大小
    background_resized = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))

    # 确保两幅图像都具有相同的通道数
    if frame.shape[2] != background_resized.shape[2]:
        raise ValueError("The number of channels in the frame and background image must match.")

    # 使用 addWeighted 函数合并两幅图像
    alpha = 0.7  # 主图像的透明度
    beta = 0.3  # 背景图像的透明度
    combined_image = cv2.addWeighted(frame, alpha, background_resized, beta, 0)

    return combined_image


def create_app(max_workers):
    app = FastAPI()

    global executor
    executor = ThreadPoolExecutor(max_workers=max_workers if max_workers else 4)  # 控制最大线程数

    @app.post('/process_image')
    async def process_image(
        image: UploadFile = File(...),
        water: UploadFile = File(None),
        swap: UploadFile = File(None),
        beautify: bool = Form(True)
    ):
        frame_data = {}

        # 获取待处理图像
        image_bytes = await image.read()  # 读取图像字节
        frame_data['data'] = image_bytes

        # 获取待替换人脸图像
        source_face = None
        if swap:
            image_bytes = await swap.read()  # 读取图像字节
            image_format = identify_image_format(image_bytes)
            source_frame = convert_to_bitmap(0, 0, image_format, image_bytes)
            source_faces = get_many_faces([source_frame])
            source_face = get_average_face(source_faces)

        # 获取背景图像
        background_frame = None
        if water:
            image_bytes = await water.read()
            image_format = identify_image_format(image_bytes)
            background_frame = convert_to_bitmap(0, 0, image_format, image_bytes)

        future = executor.submit(process_frame, frame_data, source_face, background_frame, beautify)
        processed_frame, processing_time = await asyncio.wrap_future(future)


        return StreamingResponse(processed_frame, media_type='image/jpeg')

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()

        # 有序字典用于保存处理后的帧数据
        results = OrderedDict()
        next_id_to_send = 1

        # 设置初始参数
        initial_params_set = False
        background_frame = None
        source_face = None
        try:
            while True:
                # 等待客户端请求
                frame_data = await websocket.receive_json()

                # 首次设置初始参数
                if not initial_params_set:
                    # 设置是否美化
                    beautify = frame_data.get("beautify", True)

                    # 设置背景图像
                    water_file = frame_data.get('water')
                    if water_file:
                        image_bytes = await water_file.read()
                        image_format = identify_image_format(image_bytes)
                        background_frame = convert_to_bitmap(0, 0, image_format, image_bytes)

                    # 设置换脸图像
                    swap_file = frame_data.get('swap')
                    if swap_file:
                        image_bytes = await swap_file.read()
                        image_format = identify_image_format(image_bytes)
                        source_frame = convert_to_bitmap(0, 0, image_format, image_bytes)
                        source_faces = get_many_faces([source_frame])
                        source_face = get_average_face(source_faces)

                    initial_params_set = True  # 标记初始化完成

                future = executor.submit(process_frame, frame_data, source_face, background_frame, beautify)
                frame_index = frame_data['frameIndex']
                results[frame_index] = future  # 将 Future 按 frame_id 存入字典

                # 检查是否有按顺序完成的结果可以返回
                while next_id_to_send in results and results[next_id_to_send].done():
                    processed = await asyncio.wrap_future(results[next_id_to_send])
                    processed['frameIndex'] = next_id_to_send
                    # 发送处理结果
                    await websocket.send_json(processed)
                    # 移除已发送的结果，并更新下一个待发送的帧编号
                    del results[next_id_to_send]
                    next_id_to_send += 1
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected", __name__)
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}", __name__)
        finally:
            await websocket.close()

    return app
