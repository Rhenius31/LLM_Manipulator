# LLM-Guided Control for Industrial Manipulator”

This project presents a hybrid robotic task-planning system that enables a robotic manipulator to understand natural language commands and execute them autonomously using perception, planning, and control.

The system integrates **Large Language Models (LLMs)**, **YOLOv8-based perception**, and **MoveIt motion planning** within a **ROS 2 framework** to perform pick-and-place tasks in a simulated environment.

---

##  Features

-  Natural language task understanding using LLM (OpenAI API)
-  Real-time object detection using YOLOv8
-  3D object localization using RGB-D camera
-  Hybrid task planning (fast deterministic + LLM reasoning)
-  Motion planning using MoveIt (collision-free trajectories)
-  Gripper control with impact detection feedback
-  Closed-loop execution for improved reliability

---

##  System Architecture
User Input → LLM Planner → Task Plan → Executor → MoveIt → Robot
↑
Scene Representation
↑
YOLOv8 + RGB-D Perception

---

##  Technologies Used

- **ROS 2 (Jazzy)** – middleware and communication
- **Gazebo / Rviz** – simulation environment
- **MoveIt** – motion planning and control
- **YOLOv8 (Ultralytics)** – object detection
- **OpenAI API (LLM)** – task planning
- **Python** – implementation

---

##  How It Works

1. **User Input**
   - User provides command:  
     `"Pick the box and place it on the tray"`

2. **Perception**
   - YOLOv8 detects objects
   - Depth data used to compute 3D positions
   - Scene published as `/scene/objects_json`

3. **Task Planning**
   - LLM interprets command
   - Hybrid approach:
     - Simple → deterministic (fast)
     - Complex → LLM reasoning

4. **Execution**
   - Executor generates motion steps
   - MoveIt plans collision-free trajectory
   - Robot performs pick-and-place

5. **Feedback**
   - Impact detection verifies grasp success
   - Retries if needed

---
