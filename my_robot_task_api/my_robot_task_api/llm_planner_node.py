#!/usr/bin/env python3
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from openai import OpenAI


class LLMPlanner(Node):
    def __init__(self):
        super().__init__('llm_planner')
        self.scene = {"objects": []}

        self.sub_scene = self.create_subscription(String, '/scene/objects_json', self.cb_scene, 10)
        self.sub_cmd   = self.create_subscription(String, '/task_command', self.cb_cmd, 10)
        self.pub_plan  = self.create_publisher(String, '/task_plan', 10)

        self.client = OpenAI()
        self.model = self.declare_parameter("model", "gpt-4o-mini").value
        self.get_logger().info("LLM planner online (OpenAI Structured Outputs)")

    def cb_scene(self, msg: String):
        
        try:
            self.scene = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Bad scene JSON: {e}")
            return

        objs = self.scene.get("objects", [])
        has_table = any(o.get("class") == "table" for o in objs)

        if not has_table:
            objs.append({
                "class": "table",
                "pose": {
                    "frame_id": "world",  # <-- IMPORTANT: static model pose is in world
                    "position": {"x": 0.6, "y": 0.0, "z": 0.15},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                },
                "source": "fixed_sdf"
            })
            self.scene["objects"] = objs

    def cb_cmd(self, msg: String):
        user_cmd = msg.data.strip()
        if not user_cmd:
            return

        try:
            plan = self.make_plan_openai(user_cmd, self.scene)
        except Exception as e:
            self.get_logger().error(f"OpenAI call failed: {e}")
            plan = {"intent": "unknown", "objects": {"pick": None, "place_on": None}, "notes": "planner_error"}

        self.pub_plan.publish(String(data=json.dumps(plan)))
        self.get_logger().info(f"Published plan: {plan}")

    def _schema(self) -> dict:
        return {
            "name": "robot_task_plan",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "intent": {"type": "string", "enum": ["pick_and_place", "stack", "unknown"]},
                    "objects": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "pick": {
                                "anyOf": [
                                    {"type": "null"},
                                    {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "class": {"type": "string"},
                                            "index": {"type": "integer", "minimum": 0},
                                        },
                                        "required": ["class", "index"],
                                    },
                                ]
                            },
                            "place_on": {
                                "anyOf": [
                                    {"type": "null"},
                                    {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "class": {"type": "string"},
                                            "index": {"type": "integer", "minimum": 0},
                                        },
                                        "required": ["class", "index"],
                                    },
                                ]
                            },
                        },
                        "required": ["pick", "place_on"],
                    },
                    "notes": {"type": "string"},
                },
                "required": ["intent", "objects", "notes"],
            },
            "strict": True,
        }

    def _build_prompt(self, user_cmd: str, scene: dict) -> str:
        objs = scene.get("objects", [])
        return (
            "You are a robot task planner. Produce ONE plan matching the given JSON schema.\n"
            "Rules:\n"
            "- Use only objects that exist in scene.objects.\n"
            "- Choose the best matching object by class name.\n"
            "- 'index' is the 0-based index among objects of the same class in scene.objects order.\n"
            "- If required objects are missing, intent=unknown and pick/place_on must be null.\n"
            "- If user says 'stack cups', return intent=stack with pick=cup index 0, place_on=cup index 1 (needs >=2 cups).\n"
            "- If user says 'pick X and place on Y', return intent=pick_and_place.\n\n"
            f"User command: {user_cmd}\n"
            f"Scene objects: {json.dumps(objs)}\n"
        )

    def _extract_plan_str(self, resp) -> str:
        # 1) Best case: SDK provides output_text
        out_text = getattr(resp, "output_text", None)
        if out_text:
            return out_text

        # 2) Walk output items and find text or json payload
        # Different SDK/model combos may store the structured result as JSON instead of text.
        for item in (getattr(resp, "output", None) or []):
            for c in (getattr(item, "content", None) or []):
                ctype = getattr(c, "type", None)

                # If there's direct JSON payload
                if ctype in ("output_json", "json") and getattr(c, "json", None) is not None:
                    return json.dumps(c.json)

                # If there's text payload
                if getattr(c, "text", None):
                    return c.text

                # Some variants store it in "data"
                if getattr(c, "data", None):
                    return str(c.data)

        # 3) Last resort: stringify and try to find a JSON object
        # (kept minimal; avoids crashing silently)
        self.get_logger().error(f"resp.output = {getattr(resp,'output',None)}")

        raise RuntimeError("No plan content found in response (no output_text / json / text).")

    def make_plan_openai(self, user_cmd: str, scene: dict) -> dict:
        prompt = self._build_prompt(user_cmd, scene)
        schema = self._schema()

        last_err = None
        for attempt in range(3):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    max_output_tokens=200,
                    temperature = 0,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": schema["name"],
                            "strict": True,
                            "schema": schema["schema"],
                        }
                    },
                )
                plan_str = self._extract_plan_str(resp)
                plan = json.loads(plan_str)
                return self._validate_against_scene(plan, scene)
            except Exception as e:
                last_err = e
                # backoff then retry
                time.sleep(1.5 * (attempt + 1))

        raise last_err

    def _validate_against_scene(self, plan: dict, scene: dict) -> dict:
        objs = scene.get("objects", [])

        def exists(cls: str, idx: int) -> bool:
            matches = [o for o in objs if (o.get("class") == cls)]
            return 0 <= idx < len(matches)

        intent = plan.get("intent", "unknown")
        if intent in ("pick_and_place", "stack"):
            pick = plan.get("objects", {}).get("pick")
            place_on = plan.get("objects", {}).get("place_on")

            if pick is None or place_on is None:
                return {"intent": "unknown", "objects": {"pick": None, "place_on": None}, "notes": "missing_fields"}

            if not exists(pick["class"], int(pick["index"])) or not exists(place_on["class"], int(place_on["index"])):
                return {"intent": "unknown", "objects": {"pick": None, "place_on": None}, "notes": "object_not_in_scene"}

        return plan


def main():
    rclpy.init()
    node = LLMPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
