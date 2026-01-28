import math
from geometry_msgs.msg import Quaternion

def quat_from_rpy(roll, pitch, yaw):
    # Standard RPY->Quat
    cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)

    q = Quaternion()
    q.w = cr*cp*cy + sr*sp*sy
    q.x = sr*cp*cy - cr*sp*sy
    q.y = cr*sp*cy + sr*cp*sy
    q.z = cr*cp*sy - sr*sp*cy
    return q

# Example presets (tune these for your real tool direction)
ORIENTATIONS = {
    "carry": quat_from_rpy(0.0, 0.0, 0.0),
    "grasp_down": quat_from_rpy(0.0, math.pi, 0.0),   # tool pointing down (often pitch=pi)
    "place_down": quat_from_rpy(0.0, math.pi, 0.0),
}
