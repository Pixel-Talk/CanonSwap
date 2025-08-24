import cv2
import face_alignment
import torch
import numpy as np
from PIL import Image
def extract_face_mask(image_path, output_path, target_size=256):
    """
    从单张图片中提取人脸，创建透明背景的面部图像
    
    参数:
    image_path: 输入图片路径
    output_path: 输出图片路径（应该是.png格式以支持透明）
    target_size: 输出图片的大小
    """
    # 初始化人脸检测器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device=device,
        face_detector='sfd'
    )
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 检测人脸关键点
    landmarks = fa.get_landmarks(image_rgb)
    if landmarks is None or len(landmarks) == 0:
        raise ValueError("未检测到人脸")
    
    # 获取第一个人脸的关键点
    landmark = landmarks[0]
    
    # 创建面部轮廓蒙版
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 定义面部轮廓点的索引
    # 面部轮廓（下巴到额头）
    jaw_points = landmark[0:17]
    # 左眉毛
    left_eyebrow = landmark[17:22]
    # 右眉毛
    right_eyebrow = landmark[22:27]
    # 鼻梁
    nose_bridge = landmark[27:31]
    # 鼻子上部
    nose_tip = landmark[31:36]
    
    # 创建完整的面部轮廓
    face_contour = np.vstack([
        jaw_points,
        np.flip(right_eyebrow, axis=0),
        np.flip(nose_bridge, axis=0),
        np.flip(left_eyebrow, axis=0)
    ])
    
    # 将轮廓点转换为整数
    face_contour = face_contour.astype(np.int32)
    
    # 填充面部区域
    cv2.fillPoly(mask, [face_contour], 255)
    
    # 使用高斯模糊软化边缘
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # 创建4通道图像（RGBA）
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # 设置alpha通道
    rgba[:, :, 3] = mask
    
    # 调整大小
    rgba_resized = cv2.resize(rgba, (target_size, target_size))
    
    # 转换为PIL图像并保存
    pil_image = Image.fromarray(cv2.cvtColor(rgba_resized, cv2.COLOR_BGRA2RGBA))
    pil_image.save(output_path, 'PNG')
    
    print(f"人脸已保存到: {output_path}")

# 使用示例
if __name__ == "__main__":
    input_path = "/cto_studio/vistring/luoxiangyang/LivePortrait_adain/assets/myexamples/10/10_c.png"  # 输入图片路径
    output_path = "/cto_studio/vistring/luoxiangyang/LivePortrait_adain/assets/myexamples/10/10_crop.png"  # 输出图片路径
    
    # try:
    extract_face_mask(input_path, output_path, target_size=256)
    # except Exception as e:
    #     print(f"错误: {str(e)}")