#!/usr/bin/env python3
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class LLMPlanner(Node):
    def __init__(self):
        super().__init__('llm_planner')
        self.scene = {"objects": []}

        self.sub_scene = self.create_subscription(String, '/scene/objects_json', self.cb_scene, 10)
        self.sub_cmd   = self.create_subscription(String, '/task_command', self.cb_cmd, 10)
        self.pub_plan  = self.create_publisher(String, '/task_plan', 10)

        self.get_logger().info("LLM planner online (scene + task_command -> task_plan)")

    def cb_scene(self, msg: String):
        try:
            self.scene = json.loads(msg.data)
        except Exception:
            pass

    def cb_cmd(self, msg: String):
        user_cmd = msg.data.strip()
        plan = self.make_plan(user_cmd, self.scene)
        self.pub_plan.publish(String(data=json.dumps(plan)))
        self.get_logger().info(f"Published plan: {plan}")

    def make_plan(self, user_cmd: str, scene: dict):
        cmd = user_cmd.lower()
        # For now: simple rules that emulate what the LLM should output.
        objs = scene.get("objects", [])
        def has(cls_name: str) -> bool:
            return any(o.get("class") == cls_name for o in objs)

    # 1) Stack cups
        if "stack" in cmd and "cup" in cmd:
            return {"intent":"stack",
                "objects":{"pick":{"class":"cup","index":0},
                           "place_on":{"class":"cup","index":1}}}

    # 2) Pick box and place on table
        if "pick" in cmd and "box" in cmd and ("table" in cmd or "on the table" in cmd):
            return {"intent":"pick_and_place",
                "objects":{"pick":{"class":"box","index":0},
                           "place_on":{"class":"table","index":0}}}
        #pick and place sports ball
        if "pick" in cmd and ("ball" in cmd or "sports ball" in cmd) and ("table" in cmd):
            return {"intent":"pick_and_place",
            "objects":{"pick":{"class":"sports ball","index":0},
                       "place_on":{"class":"table","index":0}}}


    # 3) Pick cup and place on table
        if "pick" in cmd and "cup" in cmd and ("table" in cmd or "on the table" in cmd):
            return {"intent":"pick_and_place",
                "objects":{"pick":{"class":"cup","index":0},
                           "place_on":{"class":"table","index":0}}}

    # 4) Pick box and place on plate (fallback to table if no plate detected)
        if "pick" in cmd and "box" in cmd and ("plate" in cmd or "on a plate" in cmd):
            target = "plate" if has("plate") else "table"
            return {"intent":"pick_and_place",
                "objects":{"pick":{"class":"box","index":0},
                           "place_on":{"class":target,"index":0}}}
        
        if "pick" in cmd and "box" in cmd and "table" in cmd:
    # fallback for demo
            return {"intent":"pick_and_place",
            "objects":{"pick":{"class":"sports ball","index":0},
                       "place_on":{"class":"table","index":0}}}


        return {"intent":"unknown", "objects":{}}

def main():
    rclpy.init()
    node = LLMPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
