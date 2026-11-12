# build_scene.py — Project 3
# Loads the warehouse, adds a ROS-ready Nova Carter, adds a /clock publisher,
# switches off ONLY the camera render nodes (so the lidar gets the GPU),
# saves the scene, and keeps the window open.

# --- 1. Start Isaac Sim FIRST. No omni/isaacsim import works before this line. ---
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})   # False = open the GUI window

# --- 2. Now the Isaac Sim / Omniverse modules exist and can be imported ---
import os
import omni.usd                                       # access to the USD stage (the scene)
import omni.graph.core as og                          # OmniGraph Python API (for the clock graph)
from pxr import Usd                                   # core USD API (for walking the prim tree)
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import open_stage, add_reference_to_stage, is_stage_loading

# --- 3. Turn on the ROS 2 bridge so OmniGraph ROS nodes can publish ---
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()                               # one app tick so the extension finishes loading

# --- 4. Find NVIDIA's asset server (warehouse + robots live here) ---
assets_root = get_assets_root_path()                  # returns None if assets can't be reached
if assets_root is None:
    raise RuntimeError("Could not reach Isaac assets. Check internet connection.")

warehouse_usd = assets_root + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
carter_usd    = assets_root + "/Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd"  # Carter WITH ROS graphs

# --- 5. Open the warehouse as the main stage ---
open_stage(warehouse_usd)
while is_stage_loading():                             # wait for all warehouse meshes/textures
    simulation_app.update()

# --- 6. Drop the Nova Carter into the scene (a reference = linked asset, like a CAD part in an assembly) ---
add_reference_to_stage(usd_path=carter_usd, prim_path="/World/NovaCarter")
while is_stage_loading():
    simulation_app.update()

# --- 6b. Clock publisher graph: every physics tick -> read sim time -> publish on /clock ---
og.Controller.edit(
    {"graph_path": "/World/ClockGraph",               # where the graph lives in the Stage tree
     "evaluator_name": "execution"},                  # "execution" = an Action Graph (event-driven)
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),            # fires every step while playing
            ("ReadSimTime",    "isaacsim.core.nodes.IsaacReadSimulationTime"), # current sim time (seconds)
            ("PublishClock",   "isaacsim.ros2.bridge.ROS2PublishClock"),       # publishes rosgraph_msgs/Clock
        ],
        og.Controller.Keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick",        "PublishClock.inputs:execIn"),     # tick -> run publisher
            ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),  # sim time -> message stamp
        ],
    },
)
simulation_app.update()                               # let the graph register before saving

# --- 6c. Switch off camera rendering, node by node (lidar/drive/odom/TF untouched) ---
stage = omni.usd.get_context().get_stage()            # handle to the open scene
carter_root = stage.GetPrimAtPath("/World/NovaCarter")

def node_type(prim):
    """Return an OmniGraph node's type string, e.g. 'isaacsim.ros2.bridge.ROS2CameraHelper'."""
    attr = prim.GetAttribute("node:type")
    return attr.Get() if attr and attr.IsValid() else ""

def disable_node(prim):
    """Set the node's 'inputs:enabled' to False. Returns True if the node supports it."""
    attr = prim.GetAttribute("inputs:enabled")
    if attr and attr.IsValid():
        attr.Set(False)
        return True
    return False

def render_target(prim):
    """Return the prim a render-product node points at (a Camera or a lidar)."""
    rel = prim.GetRelationship("inputs:cameraPrim")
    if not rel:
        return None
    targets = rel.GetTargets()
    return stage.GetPrimAtPath(targets[0]) if targets else None

def is_lidar(target):
    """Lidar prims are type OmniLidar; older assets use a Camera prim with 'lidar' in the path."""
    if target is None:
        return False
    path = str(target.GetPath()).lower()
    return target.GetTypeName() == "OmniLidar" or "lidar" in path or "xt_32" in path

disabled, kept, no_switch = [], [], []

print("\n[P3] Graph inventory on the Carter:")
for prim in Usd.PrimRange(carter_root):               # walk every prim under the robot
    if prim.GetTypeName() == "OmniGraph":
        print(f"   GRAPH  {prim.GetPath()}")          # list each graph so we learn the structure
        continue

    t = node_type(prim)
    if not t:
        continue                                      # not an OmniGraph node

    # Case 1: render-product node -> switch off only if it renders a camera (not a lidar)
    if t.endswith("IsaacCreateRenderProduct"):
        target = render_target(prim)
        if is_lidar(target):
            kept.append(f"{prim.GetPath()}  (renders lidar {target.GetPath()})")
        elif disable_node(prim):
            disabled.append(f"{prim.GetPath()}  (camera render)")
        else:
            no_switch.append(str(prim.GetPath()))

    # Case 2: camera ROS publishers (images + camera_info) -> switch off
    elif t.endswith("ROS2CameraHelper") or t.endswith("ROS2CameraInfoHelper"):
        if disable_node(prim):
            disabled.append(f"{prim.GetPath()}  (camera publisher)")
        else:
            no_switch.append(str(prim.GetPath()))

print("\n[P3] Switched OFF:")
for line in disabled:
    print(f"   - {line}")
print("[P3] Kept ON (lidar):")
for line in kept:
    print(f"   + {line}")
if no_switch:
    print("[P3] WARNING: camera nodes with no 'enabled' switch (left running):")
    for line in no_switch:
        print(f"   ? {line}")
print("")
simulation_app.update()

# --- 7. Save our own copy so later milestones can just open this file ---
out_path = os.path.expanduser("~/p3_warehouse/warehouse_carter.usd")
omni.usd.get_context().save_as_stage(out_path)
print(f"[P3] Scene saved to {out_path}. Press PLAY in the window.")

# --- 8. Keep the app running until you close the window ---
while simulation_app.is_running():
    simulation_app.update()                           # renders a frame + steps physics when playing

simulation_app.close()
