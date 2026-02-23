#!/usr/bin/env python3
import math
import random
import subprocess
import time

WORLD = "default"
SETPOSE_SRV = f"/world/{WORLD}/set_pose"

# model names must match your actual spawned model names
MODELS = ["cup", "box", "tray"]

# table center region (tune these)
X_RANGE = (0.35, 0.60)
Y_RANGE = (-0.22, 0.22)

# spawn a little above table so physics settles them
Z_SPAWN = 0.70

def run(cmd):
    subprocess.run(cmd, check=True)

def set_pose(model_name: str, x: float, y: float, z: float, yaw: float):
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)

    req = f"""
name: "{model_name}"
position: {{x: {x}, y: {y}, z: {z}}}
orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}
""".strip()

    run([
        "gz", "service", "-s", SETPOSE_SRV,
        "--reqtype", "gz.msgs.Pose",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "2000",
        "--req", req
    ])

def random_xy_yaw():
    x = random.uniform(*X_RANGE)
    y = random.uniform(*Y_RANGE)
    yaw = random.uniform(-math.pi, math.pi)
    return x, y, yaw

def main():
    random.seed()

    for k in range(200):  # number of random scenes
        # place each object
        for i, m in enumerate(MODELS):
            x, y, yaw = random_xy_yaw()
            y += (i - 1) * 0.06  # small separation to reduce collisions
            set_pose(m, x, y, Z_SPAWN, yaw)

        # let it settle + allow camera to capture frames
        time.sleep(0.6)

    print("Done.")

if __name__ == "__main__":
    main()
