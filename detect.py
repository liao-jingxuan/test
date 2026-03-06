#!/home/dashuai/anaconda3/envs/py38/bin/python3
import rospy
import cv2
import torch
from ultralytics import YOLO
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge, CvBridgeError
import os
import atexit  # 新增：用于注册退出清理函数

class YoloDetectorNode:
    def __init__(self):
        rospy.init_node('yolo_detector_node', anonymous=True)
        self.bridge = CvBridge()
        self.latest_image = None
        # 结果文件路径（需和导航节点的路径保持一致）
        self.result_file = "/home/dashuai/exam_ws/result.txt"

        # 加载模型并强制使用CPU
        self.model = YOLO('yolov5-master/runs/detect/dect_model2/weights/best.pt')
        self.model.to('cpu')
        rospy.loginfo("YOLO model loaded on CPU.")

        # 订阅相机话题
        self.image_sub = rospy.Subscriber(
            "/camera/rgb/image_raw", 
            Image, 
            self.image_callback, 
            queue_size=1
        )
        rospy.loginfo("YOLO Detector Node is ready.")
        rospy.loginfo("YOLO model loaded on CPU.")

        # 订阅导航节点发送的区域编号（如"A1"、"B1_正面"）
        self.area_sub = rospy.Subscriber(
            "/current_detection_area",
            String,
            self.area_detection_callback,
            queue_size=10
        )
        rospy.loginfo("Subscribed to area topic: /current_detection_area")

        # 仅保留识别完成信号发布
        self.done_pub = rospy.Publisher(
            "/yolo_detection_done",
            Bool,
            queue_size=10
        )
        rospy.loginfo("Publisher ready for: /yolo_detection_done")

        # 新增：注册退出清理函数
        atexit.register(self.cleanup)

    def cleanup(self):
        """程序退出时自动清理资源"""
        cv2.destroyAllWindows()
        rospy.loginfo("YOLO Detector Node: Windows closed and resources released.")

    def image_callback(self, msg):
        """持续接收摄像头图像，存储最新帧"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # 格式统一（灰度/RGBA→BGR）
        if len(cv_image.shape) == 2:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        elif cv_image.shape[2] == 4:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGR)
        
        self.latest_image = cv_image

    def classify_flower_gender(self, cls_name):
        """根据模型类别名（A_female/A_male/male/female）映射花卉性别"""
        # 适配你的模型类别：A_female=雌, A_male=雄, male=雄, female=雌
        if cls_name == "A_female" or cls_name == "female":
            return "雌"
        elif cls_name == "A_male" or cls_name == "male":
            return "雄"
        else:
            return "未知"

    def area_detection_callback(self, msg):
        """收到区域编号后执行识别，写入txt并发布完成信号"""
        area_name = msg.data.strip()
        if not area_name or self.latest_image is None:
            rospy.logwarn(f"Invalid area name ({area_name}) or no image available!")
            self.done_pub.publish(Bool(data=False))
            return

        rospy.loginfo(f"Start detection for area: {area_name}")
        
        # 执行YOLO检测
        rgb_image = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB)
        results = self.model(rgb_image, conf=0.1, iou=0.3)

        # 解析检测结果（提取所有目标，按置信度排序）
        flower_genders = []
        if results[0].boxes:
            # 按置信度降序排列所有检测到的目标
            boxes = results[0].boxes
            sorted_indices = torch.argsort(boxes.conf, descending=True).cpu().numpy()
            
            for idx in sorted_indices:
                cls_id = int(boxes.cls[idx].cpu().numpy())
                cls_name = self.model.names[cls_id]
                conf = float(boxes.conf[idx].cpu().numpy())
                gender = self.classify_flower_gender(cls_name)
                
                flower_genders.append(gender)
                rospy.loginfo(f"Detected: {cls_name} (conf: {conf:.2f}) → Gender: {gender}")

            # 绘制检测框并显示
            rendered = results[0].plot()
            out_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            cv2.imshow("YOLO Real-time Detection", out_bgr)
            cv2.waitKey(1)
        else:
            rospy.loginfo(f"No objects detected in area {area_name}.")
            flower_genders = ["未知"]

        # 写入结果到txt文件（格式：区域编号: 性别1性别2性别3）
        try:
            # 确保每个区域固定3个结果，不足补"未知"
            while len(flower_genders) < 3:
                flower_genders.append("未知")
            flower_genders = flower_genders[:3]  # 最多取前3个
            result_line = f"{area_name}: {''.join(flower_genders)}"
            
            with open(self.result_file, "a", encoding="utf-8") as f:
                f.write(result_line + "\n")
            rospy.loginfo(f"Result written to {self.result_file}: {result_line}")
        except Exception as e:
            rospy.logerr(f"Failed to write result to file: {e}")
            self.done_pub.publish(Bool(data=False))
            return

        # 发布识别完成信号（核心：通知导航继续下一步）
        self.done_pub.publish(Bool(data=True))
        rospy.loginfo(f"Detection for {area_name} completed, signal sent to navigation.")

if __name__ == '__main__':
    try:
        node = YoloDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    # 移除原来的 cv2.destroyAllWindows()，改由 atexit 自动处理
