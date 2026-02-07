#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge

# pip install ultralytics opencv-python
from ultralytics import YOLO

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.declare_parameter('image_topic', '/rgbd_camera/image')
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('conf', 0.5)

        self.image_topic = self.get_parameter('image_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.conf = float(self.get_parameter('conf').value)

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        self.pub = self.create_publisher(Detection2DArray, '/yolo/detections', 10)
        self.sub = self.create_subscription(Image, self.image_topic, self.cb, qos_profile_sensor_data)

        self.get_logger().info(f"YOLO running: model={self.model_path} topic={self.image_topic}")

    def cb(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        res = self.model.predict(img, conf=self.conf, verbose=False)[0]

        out = Detection2DArray()
        out.header = msg.header

        # Ultralytics boxes: xyxy in pixels
        if res.boxes is None:
            self.pub.publish(out)
            return

        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls_id = int(b.cls[0].item())
            score = float(b.conf[0].item())
            name = self.model.names.get(cls_id, str(cls_id))

            det = Detection2D()
            det.header = out.header
            det.bbox.center.position.x = (x1 + x2) / 2.0
            det.bbox.center.position.y = (y1 + y2) / 2.0
            det.bbox.size_x = (x2 - x1)
            det.bbox.size_y = (y2 - y1)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = name
            hyp.hypothesis.score = score
            det.results.append(hyp)

            out.detections.append(det)

        self.pub.publish(out)

def main():
    rclpy.init()
    node = YoloDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
