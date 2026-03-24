#!/usr/bin/env python3

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from openai import OpenAI


class BaselineLLMPlanner(Node):
    def __init__(self):
        super().__init__("llm_planner_baseline")

        self.scene = {"objects": []}
        self.client = OpenAI()
        self.model = self.declare_parameter("model", "gpt-4o-mini").value

        self.create_subscription(String, "/scene/objects_json", self.cb_scene, 10)
        self.create_subscription(String, "/task_command", self.cb_cmd, 10)
        self.pub_plan = self.create_publisher(String, "/task_plan", 10)

        self.get_logger().info("Baseline LLM planner online")

    def cb_scene(self, msg: String):
        try:
            self.scene = json.loads(msg.data)
        except Exception:
            pass

    def cb_cmd(self, msg: String):
        cmd = msg.data.strip()
        if not cmd:
            return

        try:
            plan = self.make_plan(cmd)
        except Exception as e:
            self.get_logger().error(f"Baseline planner failed: {e}")
            plan = {
                "intent": "unknown",
                "goal": cmd,
                "objects": {"pick": None, "place_on": None},
                "steps": [],
                "fallback": "abort",
                "notes": "planner_error",
                "reasoning_flags": ["planning_failed"]
            }

        self.pub_plan.publish(String(data=json.dumps(plan)))
        self.get_logger().info(f"Published baseline plan: {json.dumps(plan)}")

    def schema(self):
        return {
            "name": "robot_task_plan",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["pick_and_place", "pick_and_hold", "move_home", "unknown"]
                    },
                    "goal": {"type": "string"},
                    "objects": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "pick": {
                                "type": ["object", "null"],
                                "additionalProperties": False,
                                "properties": {
                                    "class": {"type": "string"},
                                    "index": {"type": "integer"},
                                    "id": {"type": "string"}
                                },
                                "required": ["class", "index", "id"]
                            },
                            "place_on": {
                                "type": ["object", "null"],
                                "additionalProperties": False,
                                "properties": {
                                    "class": {"type": "string"},
                                    "index": {"type": "integer"},
                                    "id": {"type": "string"}
                                },
                                "required": ["class", "index", "id"]
                            }
                        },
                        "required": ["pick", "place_on"]
                    },
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "skill": {"type": "string"},
                                "args": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "object_id": {"type": ["string", "null"]},
                                        "target_id": {"type": ["string", "null"]},
                                        "class": {"type": ["string", "null"]},
                                        "index": {"type": ["integer", "null"]}
                                    },
                                    "required": ["object_id", "target_id", "class", "index"]
                                }
                            },
                            "required": ["skill", "args"]
                        }
                    },
                    "fallback": {"type": "string"},
                    "notes": {"type": "string"},
                    "reasoning_flags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["intent", "goal", "objects", "steps", "fallback", "notes", "reasoning_flags"]
            }
        }

    def extract_text(self, resp):
        if getattr(resp, "output_text", None):
            return resp.output_text
        for item in getattr(resp, "output", []):
            for c in getattr(item, "content", []):
                if getattr(c, "text", None):
                    return c.text
                if getattr(c, "json", None):
                    return json.dumps(c.json)
        raise RuntimeError("No response text found")

    def make_plan(self, cmd: str):
        prompt = f"""
You are a robot planner.
Return a JSON plan for the command using only the scene objects.

Command:
{cmd}

Scene:
{json.dumps(self.scene, indent=2)}
"""
        schema = self.schema()

        last = None
        for attempt in range(3):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    max_output_tokens=400,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": schema["name"],
                            "schema": schema["schema"],
                            "strict": True
                        }
                    }
                )
                txt = self.extract_text(resp)
                return json.loads(txt)
            except Exception as e:
                last = e
                time.sleep(1.0 + attempt)
        raise last


def main():
    rclpy.init()
    node = BaselineLLMPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()