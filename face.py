from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import time
from PIL import Image


def draw_landmarks_on_image(rgb_image, detection_result):
  cap = cv2.VideoCapture(0)
  face_landmarks_list = detection_result.face_landmarks #将检测到的人脸关键点（face_landmarks）赋值给变量 face_landmarks_list
  annotated_image = np.copy(rgb_image) #将原始图像（rgb_image）进行复制并赋值给变量 annotated_image

  # 循环遍历检测到的人脸以可视化
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # 绘制人脸特征点
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList() #可以访问和操作 NormalizedLandmarkList 中存储的人脸关键点数据
    # 将关键点数据从face_landmarks的数据结构转换为face_landmarks_proto的Protobuf格式，以便进行序列化、传输或与其他支持Protobuf格式的系统进行交互
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    # 在图像上绘制人脸关键点
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    # 绘制人脸关键点和连接线
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    # 绘制人脸关键点和连接线
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image



def plot_face_blendshapes_bar_graph(face_blendshapes):
  # 提取人脸混合形状类别名称和分数
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # 提取人脸混合形状类别名称和分数
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  # 创建一个包含单个子图的图形对象，其中子图的大小为 12x12 英寸
  fig, ax = plt.subplots(figsize=(12, 12))
  # 创建一个水平条形图，显示面部混合形状类别的分数和排名
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  # 设置子图ax的刻度和反转y轴的操作
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # 用值标记每个条形
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()