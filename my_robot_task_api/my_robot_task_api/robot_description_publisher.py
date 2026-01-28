import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class RobotDescriptionPublisher(Node):
    def __init__(self):
        super().__init__('robot_description_publisher')
        self.declare_parameter('robot_description', '')
        urdf = self.get_parameter('robot_description').get_parameter_value().string_value

        self.pub = self.create_publisher(String, 'robot_description', 10)
        msg = String()
        msg.data = urdf

        # publish a few times so late subscribers get it
        for _ in range(10):
            self.pub.publish(msg)

        self.timer = self.create_timer(1.0, lambda: self.pub.publish(msg))

def main():
    rclpy.init()
    node = RobotDescriptionPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
