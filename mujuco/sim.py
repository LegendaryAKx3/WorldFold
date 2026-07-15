'''
Verify installaion
'''

# import so101_nexus.mujoco
# import gymnasium as gym

# env = gym.make("MuJoCoPickLift-v1")
# print("Installation OK")
# env.close()

'''

'''

###########################################################################################
# old code
# import time
# import gymnasium as gym
# import so101_nexus.mujoco
# import mujoco.viewer
# from so101_nexus import PickConfig, YCBObject

# config = PickConfig(objects=YCBObject(model_id="011_banana"))
# env = gym.make("MuJoCoPickLift-v1", config=config, render_mode="human")
# obs, info = env.reset()

# for _ in range(1000):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()
#     time.sleep(1/50)
#     print("the sim is running btw")
#     if terminated or truncated:
#         obs, info = env.reset()

# env.close()
###########################################################################################


import os
import time
import gymnasium as gym
import so101_nexus
import so101_nexus.mujoco
import mujoco.viewer
from so101_nexus import PickConfig, YCBObject
import numpy as np
import mjviser

TASK                  = "drag"           # the options are: drop, push, drag
TABLE_TOP_Z           = 0.42
CLOTH_COUNT           = 7
CLOTH_SPACING         = 0.03
CLOTH_RADIUS          = 0.005
CLOTH_MASS            = 0.05
CLOTH_HALF            = (CLOTH_COUNT-1) * CLOTH_SPACING / 2
CLOTH_SPAWN_Z         = TABLE_TOP_Z + 0.10
HOVER_Z               = TABLE_TOP_Z + 0.10
GRIP_Z                = 0.45

# SO101 arm model
ARM_XML_PATH     = os.path.join(os.path.dirname(so101_nexus.__file__), "assets", "SO101", "so101_new_calib.xml")
ARM_BASE_POS     = (-0.25, 0, TABLE_TOP_Z)
ARM_JOINTS       = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_CLOSED   = -0.1

PUSH_Z           = TABLE_TOP_Z + 0.010   # fingertip height while pushing; tune +-2mm
DRAG_Z           = TABLE_TOP_Z + 0.006   # fingertip height while dragging; tune +-2mm
ARM_TIMESTEP     = 0.0005                # push/drag need 0.5ms; 1ms explodes on contact
MAX_STEP_MOVE    = 0.0002                # m per step = 0.4 m/s at 0.5ms steps
MAX_JOINT_STEP   = 0.005
CLOTH_DAMPING    = 0.02                  # viscous damping per cloth vertex DOF; calms jitter/explosions

def build_xml_file(timestep, mode, task):

    # drop spawns the cloth high and tilted; push/drag spawn it flat on the table
    if task == "drop":
        spawn = f'pos="0 0 {CLOTH_SPAWN_Z}" euler="25 15 0"'
    else:
        spawn = f'pos="0 0 {TABLE_TOP_Z + 0.02}"'

    if mode == "native":
        top = ""
        edge = '<edge equality="true" damping="0.002"/>'
        block = ""
    else:
        top = '<extension><plugin plugin="mujoco.elasticity.shell"/></extension>'
        edge = '<edge equality="true" damping="0.002"/>'
        block = """
        <plugin plugin="mujoco.elasticity.shell">
            <config key="young" value="3e4"/>
            <config key="poisson" value="0.0"/>
            <config key="thickness" value="1e-3"/>
        </plugin>
        """

    xml = f"""

    <mujoco model="cloth_{task}">
        <option timestep="{timestep}" integrator="implicitfast"/>
        <visual><global offwidth="1280" offheight="720"/></visual>
        {top}
        <worldbody>
        <light pos="0 0 2" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
        <light pos="1 -1 1.5" dir="-0.5 0.5 -1" diffuse="0.4 0.4 0.4"/>
        <geom name="floor" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.35 1"/>
        <geom name="table" type="box" size="0.30 0.30 {TABLE_TOP_Z / 2}"
                pos="0 0 {TABLE_TOP_Z / 2}" friction="0.8 0.005 0.0001"
                rgba="0.55 0.4 0.25 1"/>
        <camera name="main" pos="0.75 -0.75 0.75" xyaxes="0.707 0.707 0 -0.19 0.19 0.96"/>

        <flexcomp name="cloth" type="grid" count="{CLOTH_COUNT} {CLOTH_COUNT} 1"
                    spacing="{CLOTH_SPACING} {CLOTH_SPACING} {CLOTH_SPACING}"
                    {spawn} radius="{CLOTH_RADIUS}" mass="{CLOTH_MASS}"
                    dim="2" rgba="0.8 0.2 0.2 1">
            <contact condim="3" solref="0.02 1" solimp="0.8 0.9 0.01"
                    friction="1.0 0.005 0.0001" selfcollide="none" internal="false"/>
            {edge}
            {block}
        </flexcomp>
        </worldbody>
    </mujoco>

    """

    return xml

def compile_model(timestep, task):
    xml = build_xml_file(timestep, mode="native", task=task)
    spec = mujoco.MjSpec.from_string(xml)

    # push/drag get the real SO101 arm mounted on the table
    if task == "push" or task == "drag":
        arm_spec = mujoco.MjSpec.from_file(ARM_XML_PATH)
        frame = spec.worldbody.add_frame(pos=ARM_BASE_POS)
        frame.attach_body(arm_spec.body("base"), "", "")

    model = spec.compile()

    # calm the cloth: light viscous damping on every cloth vertex DOF
    # (the cloth's joints are the unnamed ones; the arm's joints have names)
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name is None:
            model.dof_damping[model.jnt_dofadr[i]] = CLOTH_DAMPING

    return model

def cloth_body_ids(model):
    ids = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name is not None and name.startswith("cloth_"):
            ids.append(i)
    return ids

def cloth_centroid(model, data):
    ids = cloth_body_ids(model)
    return data.xpos[ids].mean(axis=0)

_max_qacc = 0.0

def tracked_step(model, data):
    # one physics step + health tracking (this is what mjviser calls each step)
    global _max_qacc
    mujoco.mj_step(model, data)

    step_qacc = float(np.max(np.abs(data.qacc)))
    if step_qacc > _max_qacc:
        _max_qacc = step_qacc

def step(model, data, viewer):
    step_start = time.perf_counter()
    tracked_step(model, data)

    # viewer=None runs headless (for tests); skip drawing and realtime pacing
    if viewer is not None:
        viewer.sync()

        # makes it so the sim doesnt finish instantly
        elapsed = time.perf_counter() - step_start
        sleep_time = model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

def viewer_running(viewer):
    if viewer is None:
        return True
    return viewer.is_running()

def settle(model, data, viewer, seconds):
    end_time = data.time + seconds
    while data.time < end_time and viewer_running(viewer):
        step(model, data, viewer)

def ik_step(model, data, target, tol=0.008):
    # ONE nudge of the arm's joint targets toward a cartesian fingertip target.
    # returns True once the fingertip is within tol. call this once per physics step.
    site_id = model.site("gripperframe").id
    error = np.asarray(target, dtype=float) - data.site_xpos[site_id]
    dist = float(np.linalg.norm(error))
    if dist < tol:
        return True

    step_vec = error
    if dist > MAX_STEP_MOVE:
        step_vec = error * (MAX_STEP_MOVE / dist)

    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)
    dof_ids = []
    for name in ARM_JOINTS:
        dof_ids.append(model.joint(name).dofadr[0])
    J = jacp[:, dof_ids]

    # solve J * dq = step_vec for the smallest joint motion dq
    dq = np.linalg.lstsq(J, step_vec, rcond=None)[0]
    biggest = float(np.max(np.abs(dq)))
    if biggest > MAX_JOINT_STEP:
        dq = dq * (MAX_JOINT_STEP / biggest)

    for k in range(len(ARM_JOINTS)):
        qpos_addr = model.joint(ARM_JOINTS[k]).qposadr[0]
        act_id = model.actuator(ARM_JOINTS[k]).id
        new_target = data.qpos[qpos_addr] + dq[k]
        low = model.actuator_ctrlrange[act_id][0]
        high = model.actuator_ctrlrange[act_id][1]
        if new_target < low:
            new_target = low
        if new_target > high:
            new_target = high
        data.ctrl[act_id] = new_target

    return False

def move_gripper_to(model, data, viewer, target, tol=0.008, timeout=8.0):
    # blocking version of ik_step, for headless runs and tests
    end_time = data.time + timeout
    while viewer_running(viewer) and data.time < end_time:
        if ik_step(model, data, target, tol):
            return True
        step(model, data, viewer)

    print(f"warning: gripper stopped short of target {target}")
    return False

def close_gripper(model, data):
    gripper_act = model.actuator("gripper").id
    data.ctrl[gripper_act] = GRIPPER_CLOSED

def make_step_fn(model, data, task):
    end_x = 0.02
    if task == "drag":
        z = DRAG_Z
        start_x = -CLOTH_HALF + 0.02   # drag starts ON the cloth
        waypoints = [
            (start_x, 0, HOVER_Z),
            (start_x, 0, z),         # descend onto the cloth
            (end_x, 0, z),           # drag in +x
            (end_x, 0, HOVER_Z),     # lift
        ]
    else:
        z = PUSH_Z
        # the arm can't hover directly above a point this close to its own base,
        # so pull back high first, then descend just behind the cloth edge
        waypoints = [
            (-CLOTH_HALF - 0.01, 0, TABLE_TOP_Z + 0.08),
            (-CLOTH_HALF - 0.03, 0, z),   # descend behind the cloth edge
            (end_x, 0, z),                # push forward through the cloth
            (end_x, 0, TABLE_TOP_Z + 0.08),
        ]

    # mutable progress shared between calls (a dict because step_fn can't reassign locals)
    state = {"index": 0, "deadline": None,
             "start_centroid": None, "finished_time": None, "checked": False}

    def step_fn(model, data):
        if data.time > 0.5:   # first 0.5s: hands off, let the cloth settle
            if state["start_centroid"] is None:
                state["start_centroid"] = cloth_centroid(model, data)
                close_gripper(model, data)

            if state["index"] < len(waypoints):
                if state["deadline"] is None:
                    state["deadline"] = data.time + 5.0
                reached = ik_step(model, data, waypoints[state["index"]])
                # move on after 5s even if not reached, so one unreachable
                # waypoint can't stall the whole trajectory forever
                if reached or data.time > state["deadline"]:
                    if not reached:
                        print(f"waypoint {state['index']} not reached, moving on")
                    state["index"] = state["index"] + 1
                    state["deadline"] = None
                    if state["index"] == len(waypoints):
                        state["finished_time"] = data.time
            elif not state["checked"] and data.time > state["finished_time"] + 1.0:
                state["checked"] = True
                dx = cloth_centroid(model, data)[0] - state["start_centroid"][0]
                print(f"centroid moved {dx * 100:.1f} cm in +x")
                if dx > 0.02:
                    print("PASS")
                else:
                    print("FAIL: cloth barely moved")

        tracked_step(model, data)

    def reset_fn(model, data):
        # called by the viewer's Reset button: rewind the physics AND the script,
        # so pressing Reset then Play replays the whole trajectory
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        state["index"] = 0
        state["deadline"] = None
        state["start_centroid"] = None
        state["finished_time"] = None
        state["checked"] = False

    return step_fn, reset_fn

def make_render_fn(model, data):
    # mjviser only draws geoms and skips flex objects entirely, so the cloth
    # would be invisible -- push it to the browser ourselves as a triangle mesh.
    # faces never change, so build them once; vertex positions update every frame.
    faces = np.array(model.flex_elem).reshape(-1, 3)

    def render_fn(scene):
        scene.update_from_mjdata(data)   # everything mjviser normally draws
        vertices = np.array(data.flexvert_xpos)

        # mjviser shifts its whole scene by -tracked_body_pos when "Track camera"
        # is on (it is by default), so shift the cloth the same way or it floats
        if scene.camera_tracking_enabled and scene._tracked_body_id is not None:
            vertices = vertices - data.xpos[scene._tracked_body_id]
        scene.server.scene.add_mesh_simple(
            "/cloth",
            vertices=vertices,
            faces=faces,
            color=(204, 51, 51),
            side="double",   # cloth is visible from both sides
        )

    return render_fn

def run_drop():
    model = compile_model(0.001, task="drop")
    data = mujoco.MjData(model)
    mjviser.Viewer(model, data, step_fn=tracked_step,
                   render_fn=make_render_fn(model, data)).run()

def run_push():
    model = compile_model(ARM_TIMESTEP, task="push")
    data = mujoco.MjData(model)
    step_fn, reset_fn = make_step_fn(model, data, "push")
    mjviser.Viewer(model, data, step_fn=step_fn, reset_fn=reset_fn,
                   render_fn=make_render_fn(model, data)).run()

def run_task_loop(model, data, viewer, task):
    if task == "drag":
        z = DRAG_Z
        start_x = -CLOTH_HALF + 0.02   # drag starts ON the cloth
    else:
        z = PUSH_Z
        start_x = -CLOTH_HALF - 0.05   # push starts outside the cloth edge
    end_x = 0.02

    settle(model, data, viewer, 0.5)
    move_gripper_to(model, data, viewer, (end_x, 0, z))
    close_gripper(model, data)
    move_gripper_to(model, data, viewer, (end_x+0.002, 0, z))
    start_centroid = cloth_centroid(model, data)

    settle(model, data, viewer, 1.0)

    end_centroid = cloth_centroid(model, data)
    dx = end_centroid[0] - start_centroid[0]
    print(f"centroid moved {dx * 100:.1f} cm in +x")
    assert dx > 0.02, "FAIL: cloth barely moved"
    print("PASS")

def run_drag():
    model = compile_model(ARM_TIMESTEP, task="drag")
    data = mujoco.MjData(model)
    step_fn, reset_fn = make_step_fn(model, data, "drag")
    mjviser.Viewer(model, data, step_fn=step_fn, reset_fn=reset_fn,
                   render_fn=make_render_fn(model, data)).run()

def main():
    if TASK == "drop":
        run_drop()
    elif TASK == "push":
        run_push()
    elif TASK == "drag":
        run_drag()
    else:
        print(f"unknown task: {TASK} (use drop|push|drag)")
        return

    print(f"max |qacc|: {_max_qacc:.1f}  (healthy: <1e3-ish; 1e5+ is no good)")

if __name__ == "__main__":
    main()
