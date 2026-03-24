#!/usr/bin/env python3

import json
import re
import time
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from openai import OpenAI


class LLMPlanner(Node):
    def __init__(self):
        super().__init__("llm_planner")

        self.scene = {"objects": []}
        self.last_command = None
        self.last_plan = None

        self.create_subscription(
            String,
            "/scene/objects_json",
            self.cb_scene,
            10
        )

        self.create_subscription(
            String,
            "/task_command",
            self.cb_cmd,
            10
        )

        self.pub_plan = self.create_publisher(
            String,
            "/task_plan",
            10
        )

        self.client = OpenAI()

        self.model = self.declare_parameter("model", "gpt-4o-mini").value
        self.max_scene_objects = int(self.declare_parameter("max_scene_objects", 12).value)
        self.inject_virtual_table = bool(self.declare_parameter("inject_virtual_table", True).value)
        self.allow_stack = bool(self.declare_parameter("allow_stack", False).value)

        self.class_synonyms = {
            "mug": "cup",
            "glass": "cup",
            "container": "box",
            "bin": "tray",
            "basket": "tray",
            "plate": "tray",
            "surface": "table",
            "desk": "table",
        }

        self.verb_synonyms = {
            "grab": "pick",
            "pickup": "pick",
            "pick up": "pick",
            "take": "pick",
            "put": "place",
            "drop": "place",
            "set": "place",
            "move": "place",
        }

        self.get_logger().info("Improved hybrid LLM planner online")

    # Scene handling

    def cb_scene(self, msg: String):
        try:
            scene = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Bad scene JSON: {e}")
            return

        objs = scene.get("objects", [])
        cleaned = []

        for i, obj in enumerate(objs):
            cls = str(obj.get("class", "obj")).strip().lower()
            cls = self.normalize_class(cls)

            pose = obj.get("pose", {})
            if not isinstance(pose, dict):
                continue

            try:
                x = float(pose.get("x"))
                y = float(pose.get("y"))
                z = float(pose.get("z"))
            except Exception:
                continue

            score = float(obj.get("score", 0.0))
            frame = pose.get("frame", "base_link")

            cleaned.append({
                "id": f"{cls}_{i}",
                "class": cls,
                "score": score,
                "pose": {
                    "frame": frame,
                    "x": x,
                    "y": y,
                    "z": z
                }
            })

        self.scene = {"objects": cleaned}

    # Command callback

    def cb_cmd(self, msg: String):
        self.get_logger().info(f"Received task command: {msg.data}")
        cmd = msg.data.strip()
        if not cmd:
            return

        self.last_command = cmd

        try:
            plan = self.make_plan(cmd)
        except Exception as e:
            self.get_logger().error(f"Planner failed: {e}")
            plan = self.make_failure_plan(
                cmd,
                notes="planner_error",
                fallback="abort"
            )

        self.last_plan = plan
        self.pub_plan.publish(String(data=json.dumps(plan)))
        self.get_logger().info(f"Published plan: {json.dumps(plan)}")

    # Normalization / scene utilities

    def normalize_text(self, text: str) -> str:
        t = text.lower().strip()

        for k, v in self.verb_synonyms.items():
            t = re.sub(rf"\b{re.escape(k)}\b", v, t)

        for k, v in self.class_synonyms.items():
            t = re.sub(rf"\b{re.escape(k)}\b", v, t)

        return t

    def normalize_class(self, cls: str) -> str:
        return self.class_synonyms.get(cls.lower(), cls.lower())

    def get_scene_objects(self) -> List[Dict]:
        objs = list(self.scene.get("objects", []))

        if self.inject_virtual_table and not any(o["class"] == "table" for o in objs):
            objs.append({
                "id": "table_virtual_0",
                "class": "table",
                "score": 1.0,
                "pose": {
                    "frame": "base_link",
                    "x": 0.60,
                    "y": 0.00,
                    "z": 0.60
                },
                "virtual": True
            })

        return objs[:self.max_scene_objects]

    def group_by_class(self, objects: List[Dict]) -> Dict[str, List[Dict]]:
        out = {}
        for obj in objects:
            out.setdefault(obj["class"], []).append(obj)

        for cls in out:
            out[cls] = sorted(
                out[cls],
                key=lambda o: (
                    -float(o.get("score", 0.0)),
                    abs(float(o["pose"]["y"])),
                    float(o["pose"]["x"])
                )
            )
        return out

    def scene_summary(self, objects: List[Dict]) -> Dict:
        grouped = self.group_by_class(objects)
        summary = {}
        for cls, arr in grouped.items():
            summary[cls] = [
                {
                    "id": o["id"],
                    "rank": i,
                    "score": round(float(o.get("score", 0.0)), 3),
                    "pose": o["pose"],
                    "virtual": bool(o.get("virtual", False))
                }
                for i, o in enumerate(arr)
            ]
        return summary

    def find_best_object(self, cls: str, grouped: Dict[str, List[Dict]]) -> Optional[Dict]:
        arr = grouped.get(cls, [])
        return arr[0] if arr else None

    def resolve_ordinal(self, text: str) -> int:
        text = text.lower()
        if "second" in text:
            return 1
        if "third" in text:
            return 2
        return 0

    # Heuristic parser before LLM

    def heuristic_guess(self, cmd: str, objects: List[Dict]) -> Dict:
        text = self.normalize_text(cmd)
        grouped = self.group_by_class(objects)

        known_classes = list(grouped.keys())
        pick_cls = None
        place_cls = None

        for cls in known_classes:
            if re.search(rf"\b{re.escape(cls)}\b", text):
                if pick_cls is None:
                    pick_cls = cls
                elif place_cls is None and cls != pick_cls:
                    place_cls = cls

        # destination hints
        if any(p in text for p in ["on table", "onto table", "on the table"]):
            place_cls = "table"

        if any(p in text for p in ["on tray", "onto tray", "in tray", "on the tray"]):
            place_cls = "tray"

        intent = "unknown"
        if "hold" in text and pick_cls:
            intent = "pick_and_hold"
        elif pick_cls and place_cls:
            intent = "pick_and_place"
        elif "stack" in text and pick_cls and place_cls:
            intent = "stack"
        elif "move home" in text or "go home" in text or text == "home":
            intent = "move_home"

        pick_index = self.resolve_ordinal(text)
        place_index = 0

        return {
            "intent": intent,
            "pick_class": pick_cls,
            "place_class": place_cls,
            "pick_index": pick_index,
            "place_index": place_index,
        }

    # Schema

    def schema(self):
        return {
            "name": "robot_task_plan",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["pick_and_place", "pick_and_hold", "stack", "inspect", "move_home", "unknown"]
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
                                "skill": {
                                    "type": "string",
                                    "enum": [
                                        "pick",
                                        "place",
                                        "stack",
                                        "hold",
                                        "inspect",
                                        "move_home",
                                        "open_gripper",
                                        "close_gripper"
                                    ]
                                },
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
                    "reasoning_flags": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": [
                    "intent",
                    "goal",
                    "objects",
                    "steps",
                    "fallback",
                    "notes",
                    "reasoning_flags"
                ]
            }
        }
    # Prompt


    def build_prompt(self, cmd: str, objects: List[Dict], heuristic: Dict) -> str:
        grouped_summary = self.scene_summary(objects)

        return f"""
You are a robot task planner for a ROS2 robot arm.

You must return STRICT JSON only.

Robot capability level:
- The current executor can reliably execute pick-and-place.
- Use intent="pick_and_place" for commands like "put the cup on the tray".
- Use intent="unknown" if the task cannot be done safely.
- stack is only allowed if the task explicitly asks to stack AND valid objects exist.
- If the user says 'hold', 'pick and hold', or 'grab and keep holding', return steps with only a pick skill.
- If the user says 'hold and move home', return steps [pick, move_home]
- move_home and inspect are allowed as high-level descriptions but may not yet be executed by the current executor.

IMPORTANT COMPATIBILITY RULE:
The current executor still consumes:
objects.pick.class
objects.pick.index
objects.place_on.class
objects.place_on.index

So always fill those when intent is pick_and_place or stack.

Planning rules:
- Only use objects that appear in the scene summary.
- Never invent classes or IDs.
- Prefer the highest-ranked instance of a class unless the user explicitly asks for second/third.
- If the user says table and no real table is detected, a virtual table may be available.
- If the command is ambiguous, choose the safest reasonable interpretation and mention ambiguity in notes.
- If impossible, return intent="unknown", empty steps, null pick/place_on, and a useful fallback.

User command:
{cmd}

Normalized hint:
{self.normalize_text(cmd)}

Heuristic guess:
{json.dumps(heuristic, indent=2)}

Scene summary:
{json.dumps(grouped_summary, indent=2)}

Output requirements:
- "goal": short description
- "objects.pick": chosen object or null
- "objects.place_on": chosen place target or null
- "steps": symbolic skills for future executor use
- "fallback": one of: abort, retry_pick, inspect_scene, move_home
- "reasoning_flags": short tags like "ambiguous_target", "used_virtual_table", "best_score_choice"
"""

    # OpenAI response parsing

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

    # Planning

    def make_plan(self, cmd: str) -> Dict:
        objects = self.get_scene_objects()
        heuristic = self.heuristic_guess(cmd, objects)

        # deterministic fast-path for simple pick-and-place
        deterministic = self.try_deterministic_plan(cmd, objects, heuristic)
        if deterministic is not None:
            return deterministic

        prompt = self.build_prompt(cmd, objects, heuristic)
        schema = self.schema()

        last = None
        for attempt in range(3):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    max_output_tokens=500,
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
                plan = json.loads(txt)
                return self.validate_and_finalize(plan, objects, cmd)

            except Exception as e:
                last = e
                time.sleep(1.0 + attempt)

        raise last

    def try_deterministic_plan(self, cmd: str, objects: List[Dict], heuristic: Dict) -> Optional[Dict]:
        grouped = self.group_by_class(objects)
        intent = heuristic["intent"]

        if intent == "move_home":
            return {
                "intent": "move_home",
                "goal": "move home",
                "objects": {"pick": None, "place_on": None},
                "steps": [
                    {
                        "skill": "move_home",
                        "args": {
                            "object_id": None,
                            "target_id": None,
                            "class": None,
                            "index": None
                        }
                    }
                ],
                "fallback": "abort",
                "notes": "deterministic_move_home",
                "reasoning_flags": []
            }

        if intent == "pick_and_hold":
            pick_cls = heuristic["pick_class"]
            if not pick_cls:
                return self.make_failure_plan(cmd, "missing_pick_class", "inspect_scene")

            pick_arr = grouped.get(pick_cls, [])
            if not pick_arr:
                return self.make_failure_plan(cmd, f"missing_pick_object:{pick_cls}", "inspect_scene")

            pick_index = min(heuristic["pick_index"], max(len(pick_arr) - 1, 0))
            pick_obj = pick_arr[pick_index]

            return {
                "intent": "pick_and_hold",
                "goal": f"pick {pick_cls} and hold it",
                "objects": {
                    "pick": {
                        "class": pick_obj["class"],
                        "index": pick_index,
                        "id": pick_obj["id"]
                    },
                    "place_on": None
                },
                "steps": [
                    {
                        "skill": "pick",
                        "args": {
                            "object_id": pick_obj["id"],
                            "target_id": None,
                            "class": pick_obj["class"],
                            "index": pick_index
                        }
                    },
                    

                ],
                "fallback": "retry_pick",
                "notes": "deterministic_pick_and_hold",
                "reasoning_flags": ["hold_after_pick", "best_score_choice"]
            }

        if intent != "pick_and_place":
            return None

        pick_cls = heuristic["pick_class"]
        place_cls = heuristic["place_class"]

        if not pick_cls or not place_cls:
            return None

        pick_arr = grouped.get(pick_cls, [])
        place_arr = grouped.get(place_cls, [])

        if not pick_arr:
            return self.make_failure_plan(
                cmd,
                notes=f"missing_pick_object:{pick_cls}",
                fallback="inspect_scene"
            )

        if not place_arr and place_cls != "table":
            return self.make_failure_plan(
                cmd,
                notes=f"missing_place_target:{place_cls}",
                fallback="inspect_scene"
            )

        pick_index = min(heuristic["pick_index"], max(len(pick_arr) - 1, 0))
        place_index = min(heuristic["place_index"], max(len(place_arr) - 1, 0)) if place_arr else 0

        pick_obj = pick_arr[pick_index]
        place_obj = {"id": "table_virtual_0", "class": "table"} if place_cls == "table" else place_arr[place_index]

        plan = {
            "intent": "pick_and_place",
            "goal": f"pick {pick_cls} and place on {place_cls}",
            "objects": {
                "pick": {
                    "class": pick_obj["class"],
                    "index": pick_index,
                    "id": pick_obj["id"]
                },
                "place_on": {
                    "class": place_obj["class"],
                    "index": place_index,
                    "id": place_obj["id"]
                }
            },
            "steps": [
                {
                    "skill": "pick",
                    "args": {
                        "object_id": pick_obj["id"],
                        "target_id": None,
                        "class": pick_obj["class"],
                        "index": pick_index
                    }
                },
                {
                    "skill": "place",
                    "args": {
                        "object_id": None,
                        "target_id": place_obj["id"],
                        "class": place_obj["class"],
                        "index": place_index
                    }
                }
            ],
            "fallback": "retry_pick",
            "notes": "deterministic_plan",
            "reasoning_flags": [
                "best_score_choice"
            ] + (["used_virtual_table"] if place_cls == "table" else [])
        }

        return self.validate_and_finalize(plan, objects, cmd)

    # ------------------------------------------------------------------
    # Validation and final cleanup
    # ------------------------------------------------------------------

    def validate_and_finalize(self, plan: Dict, objects: List[Dict], cmd: str) -> Dict:
        grouped = self.group_by_class(objects)

        if "intent" not in plan:
            raise RuntimeError("missing intent")

        if plan["intent"] not in ["pick_and_place", "pick_and_hold", "stack", "inspect", "move_home", "unknown"]:
            raise RuntimeError("invalid intent")

        if plan["intent"] == "stack" and not self.allow_stack:
            plan["intent"] = "unknown"
            plan["objects"] = {"pick": None, "place_on": None}
            plan["steps"] = []
            plan["fallback"] = "abort"
            plan["notes"] = "stack_not_enabled_in_executor"
            plan["reasoning_flags"] = ["unsupported_executor_skill"]
            return plan

        plan.setdefault("goal", "unknown")
        plan.setdefault("objects", {"pick": None, "place_on": None})
        plan.setdefault("steps", [])
        plan.setdefault("fallback", "abort")
        plan.setdefault("notes", "")
        plan.setdefault("reasoning_flags", [])

        if plan["fallback"] not in ["abort", "retry_pick", "inspect_scene", "move_home"]:
            plan["fallback"] = "abort"

        if plan["intent"] in ["pick_and_place", "stack"]:
            pick = plan["objects"].get("pick")
            place_on = plan["objects"].get("place_on")

            if not pick or not place_on:
                raise RuntimeError("pick/place missing")

            pick_cls = self.normalize_class(str(pick["class"]))
            place_cls = self.normalize_class(str(place_on["class"]))
            pick_idx = int(pick.get("index", 0))
            place_idx = int(place_on.get("index", 0))

            if pick_cls not in grouped:
                raise RuntimeError(f"invalid pick class: {pick_cls}")

            if place_cls != "table" and place_cls not in grouped:
                raise RuntimeError(f"invalid place class: {place_cls}")

            pick_arr = grouped[pick_cls]
            if pick_idx >= len(pick_arr):
                pick_idx = 0

            pick_obj = pick_arr[pick_idx]

            if place_cls == "table":
                place_obj = next((o for o in objects if o["class"] == "table"), None)
                if place_obj is None:
                    raise RuntimeError("table requested but unavailable")
                place_idx = 0
            else:
                place_arr = grouped[place_cls]
                if place_idx >= len(place_arr):
                    place_idx = 0
                place_obj = place_arr[place_idx]

            if pick_obj["id"] == place_obj["id"]:
                raise RuntimeError("pick and place target are same object")

            plan["objects"]["pick"] = {
                "class": pick_obj["class"],
                "index": pick_idx,
                "id": pick_obj["id"]
            }
            plan["objects"]["place_on"] = {
                "class": place_obj["class"],
                "index": place_idx,
                "id": place_obj["id"]
            }

            # rewrite steps to be consistent
            if plan["intent"] == "pick_and_place":
                plan["steps"] = [
                    {
                        "skill": "pick",
                        "args": {
                            "object_id": pick_obj["id"],
                            "target_id": None,
                            "class": pick_obj["class"],
                            "index": pick_idx
                        }
                    },
                    {
                        "skill": "place",
                        "args": {
                            "object_id": None,
                            "target_id": place_obj["id"],
                            "class": place_obj["class"],
                            "index": place_idx
                        }
                    }
                ]
        if plan["intent"] == "pick_and_hold":
            pick = plan["objects"].get("pick")
            if not pick:
                raise RuntimeError("pick missing for hold")

            pick_cls = self.normalize_class(str(pick["class"]))
            pick_idx = int(pick.get("index", 0))

            if pick_cls not in grouped:
                raise RuntimeError(f"invalid pick class: {pick_cls}")

            pick_arr = grouped[pick_cls]
            if pick_idx >= len(pick_arr):
                pick_idx = 0

            pick_obj = pick_arr[pick_idx]

            plan["objects"]["pick"] = {
                "class": pick_obj["class"],
                "index": pick_idx,
                "id": pick_obj["id"]
            }
            plan["objects"]["place_on"] = None

            plan["steps"] = [
                {
                    "skill": "pick",
                    "args": {
                        "object_id": pick_obj["id"],
                        "target_id": None,
                        "class": pick_obj["class"],
                        "index": pick_idx
                    }
                }
            ]

        if plan["intent"] == "unknown":
            plan["objects"] = {"pick": None, "place_on": None}
            plan["steps"] = plan.get("steps", []) if isinstance(plan.get("steps"), list) else []

        return plan

    def make_failure_plan(self, cmd: str, notes: str, fallback: str = "abort") -> Dict:
        return {
            "intent": "unknown",
            "goal": cmd,
            "objects": {
                "pick": None,
                "place_on": None
            },
            "steps": [],
            "fallback": fallback,
            "notes": notes,
            "reasoning_flags": ["planning_failed"]
        }


def main():
    rclpy.init()
    node = LLMPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()