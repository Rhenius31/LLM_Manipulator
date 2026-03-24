#!/usr/bin/env python3

import csv
import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class BenchmarkRunner(Node):
    def __init__(self):
        super().__init__("benchmark_runner")

        self.declare_parameter("commands_file", "commands.json")
        self.declare_parameter("results_file", "benchmark_results.csv")
        self.declare_parameter("timeout_sec", 360.0)
        self.declare_parameter("recovery_command", "move home")
        self.declare_parameter("recovery_wait_sec", 15.0)
        self.declare_parameter("planner_label", "improved")

        self.commands_file = self.get_parameter("commands_file").value
        self.results_file = self.get_parameter("results_file").value
        self.timeout_sec = float(self.get_parameter("timeout_sec").value)
        self.recovery_command = self.get_parameter("recovery_command").value
        self.recovery_wait_sec = float(self.get_parameter("recovery_wait_sec").value)
        self.planner_label = self.get_parameter("planner_label").value
        self.pub_cmd = self.create_publisher(String, "/task_command", 10)
        self.sub_result = self.create_subscription(String, "/task_result", self.cb_result, 10)
        for _ in range(20):
            if self.pub_cmd.get_subscription_count() > 0:
                break
            self.get_logger().info("Waiting for subscriber on /task_command...")
            time.sleep(0.5)


        self.commands = self.load_commands(self.commands_file)
        self.results = []

        self.current_idx = -1
        self.waiting = False
        self.recovering = False
        self.current_start = None
        self.recovery_start = None

        self.timer = self.create_timer(0.2, self.tick)

        self.get_logger().info(f"Loaded {len(self.commands)} commands")

    def load_commands(self, path):
        p = Path(path)
        data = json.loads(p.read_text())
        if not isinstance(data, list):
            raise RuntimeError("commands file must contain a JSON list")
        return [str(x) for x in data]

    def cb_result(self, msg: String):
        try:
            result = json.loads(msg.data)
        except Exception:
            result = {"status": "failed", "reason": "bad_result_json"}

        if self.recovering:
            self.recovering = False
            return

        if not self.waiting:
            return

        result["benchmark_index"] = self.current_idx
        result["expected_command"] = self.commands[self.current_idx]
        result["planner_label"] = self.planner_label
        result["measured_duration_sec"] = time.time() - self.current_start
        self.results.append(result)
        self.waiting = False

        if result.get("status") != "success":
            self.pub_cmd.publish(String(data=self.recovery_command))
            self.recovering = True
            self.recovery_start = time.time()

    def send_next(self):
        self.current_idx += 1
        if self.current_idx >= len(self.commands):
            self.finish()
            return

        cmd = self.commands[self.current_idx]
        self.get_logger().info(f"Sending {self.current_idx+1}/{len(self.commands)}: {cmd}")
        self.pub_cmd.publish(String(data=cmd))
        self.current_start = time.time()
        self.waiting = True

    def tick(self):
        if self.recovering:
            if (time.time() - self.recovery_start) >= self.recovery_wait_sec:
                self.recovering = False
            return

        if self.current_idx == -1 and not self.waiting:
            self.send_next()
            return

        if self.waiting and (time.time() - self.current_start > self.timeout_sec):
            self.get_logger().warn("Task timeout")
            self.results.append({
                "benchmark_index": self.current_idx,
                "expected_command": self.commands[self.current_idx],
                "command": self.commands[self.current_idx],
                "status": "failed",
                "reason": "timeout",
                "planner_label": self.planner_label,
                "measured_duration_sec": self.timeout_sec
            })
            self.waiting = False

            self.pub_cmd.publish(String(data=self.recovery_command))
            self.recovering = True
            self.recovery_start = time.time()
            return

        if not self.waiting and self.current_idx < len(self.commands):
            self.send_next()

    def finish(self):
        out = Path(self.results_file)
        keys = sorted({k for row in self.results for k in row.keys()})
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.results)

        success = sum(1 for r in self.results if r.get("status") == "success")
        total = len(self.results)
        self.get_logger().info(f"Done: {success}/{total} success")
        self.get_logger().info(f"Saved results to {out}")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = BenchmarkRunner()
    rclpy.spin(node)


if __name__ == "__main__":
    main()