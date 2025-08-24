import cv2
import os

def extract_first_100_frames(video_path, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 读取视频
    video = cv2.VideoCapture(video_path)
    
    # 获取视频的基本信息
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video FPS: {fps}")
    print(f"Total frames in video: {frame_count}")
    
    # 读取前100帧
    for frame_idx in range(1):
        ret, frame = video.read()
        
        # 如果视频帧数不足100或读取失败，提前退出
        if not ret:
            print(f"Only extracted {frame_idx} frames (video ended)")
            break
            
        # 保存帧
        frame_path = os.path.join(output_folder, f"frame_{frame_idx:03d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        # 打印进度
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx} frames")
    
    # 释放资源
    video.release()
    print("Finished extracting frames")

def extract_first_100_frames_to_video(input_video_path, output_video_path):
    # 读取输入视频
    video = cv2.VideoCapture(input_video_path)
    
    # 获取视频的基本信息
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或使用 'XVID'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 读取并保存前100帧
    for frame_idx in range(100):
        ret, frame = video.read()
        
        # 如果视频帧数不足100或读取失败，提前退出
        if not ret:
            print(f"Only extracted {frame_idx} frames (video ended)")
            break
            
        # 写入帧到输出视频
        out.write(frame)
        
        # 打印进度
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx} frames")
    
    # 释放资源
    video.release()
    out.release()
    print("Finished extracting frames to video")


# 使用示例
video_path = "/cto_studio/vistring/luoxiangyang/LivePortrait_adain/assets/myexamples/f7da403a9277045363d72429f9c38400.mov"  # 替换为你的视频路径
output_folder = "/cto_studio/vistring/luoxiangyang/LivePortrait_adain/assets/myexamples/test.mp4"  # 输出文件夹名称
extract_first_100_frames_to_video(video_path, output_folder)