# modules/vision_system.py
import pyrealsense2 as rs
import cv2

def initialize_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    
    # Create an align object to align depth frames to color frames
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    return pipeline, align

def get_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    # Convert images to numpy arrays
    color_image = cv2.cvtColor(cv2.imdecode(cv2.imencode('.jpg', color_frame.get_data())[1], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if False else None
    color_image =  cv2.cvtColor( cv2.UMat(color_frame.get_data()).get(), cv2.COLOR_BGR2RGB) if color_frame else None
    # Alternative using numpy:
    import numpy as np
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    
    return {'color': color_image, 'depth': depth_image}

def cleanup(pipeline):
    pipeline.stop()
