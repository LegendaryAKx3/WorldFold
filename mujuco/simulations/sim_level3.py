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
import gymnasium as gym
import so101_nexus
import so101_nexus.mujoco
import mujoco.viewer
from so101_nexus import PickConfig, YCBObject
import numpy as np
import mjviser

TASK                  = "drag" # the options are: drop, push, drag
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
GRIPPER_CLOSED   = 0.2                   # mostly closed; -0.1 jams the jaw into its own finger pad

PUSH_Z           = TABLE_TOP_Z + 0.022  
DRAG_Z           = TABLE_TOP_Z + 0.010  
ARM_TIMESTEP     = 0.0005              
MAX_STEP_MOVE    = 0.0002  
MAX_JOINT_STEP   = 0.005
CLOTH_DAMPING    = 0.3

def build_xml_file(timestep, task):

    # spawns the cloth high and tilted
    if task == "drop":
        spawn = f'pos="0 0 {CLOTH_SPAWN_Z}" euler="25 15 0"'
    else:
        spawn = f'pos="0 0 {TABLE_TOP_Z + CLOTH_RADIUS + 0.001}"'

    xml = f"""

    <mujoco model="cloth_{task}">
        <option timestep="{timestep}" integrator="implicitfast"/>
        <visual><global offwidth="1280" offheight="720"/></visual>
        <worldbody>
        <light pos="0 0 2" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
        <light pos="1 -1 1.5" dir="-0.5 0.5 -1" diffuse="0.4 0.4 0.4"/>
        <geom name="floor" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.35 1"/>
        <geom name="table" type="box" size="0.30 0.30 {TABLE_TOP_Z / 2}"
                pos="0 0 {TABLE_TOP_Z / 2}" friction="0.4 0.005 0.0001"
                rgba="0.55 0.4 0.25 1"/>
        <camera name="main" pos="0.75 -0.75 0.75" xyaxes="0.707 0.707 0 -0.19 0.19 0.96"/>

        <flexcomp name="cloth" type="grid" count="{CLOTH_COUNT} {CLOTH_COUNT} 1"
                    spacing="{CLOTH_SPACING} {CLOTH_SPACING} {CLOTH_SPACING}"
                    {spawn} radius="{CLOTH_RADIUS}" mass="{CLOTH_MASS}"
                    dim="2" rgba="0.8 0.2 0.2 1">
            <contact condim="3" solref="0.01 1" solimp="0.95 0.99 0.001"
                    friction="0.4 0.005 0.0001" selfcollide="none" internal="false"/>
            <edge equality="true" damping="0.2"/>
        </flexcomp>
        </worldbody>
    </mujoco>

    """

    return xml

def compile_model(timestep, task):
    xml = build_xml_file(timestep, task=task)
    spec = mujoco.MjSpec.from_string(xml)

    if task == "push" or task == "drag":
        arm_spec = mujoco.MjSpec.from_file(ARM_XML_PATH)
        frame = spec.worldbody.add_frame(pos=ARM_BASE_POS)
        frame.attach_body(arm_spec.body("base"), "", "")

    model = spec.compile()

    if task == "drop":
        damping = 0.001
    else:
        damping = CLOTH_DAMPING

    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name is None:
            model.dof_damping[model.jnt_dofadr[i]] = damping

    # the arm's finger pads ship as 1.25mm boxes with very stiff contact params, which explode the 1-gram cloth vertices on touch. soften and enlarge them.
    if task == "push" or task == "drag":
        for pad in ["static_finger_pad", "moving_finger_pad"]:
            g = model.geom(pad).id
            model.geom_condim[g] = 3
            model.geom_solref[g] = [0.02, 1]
            model.geom_solimp[g] = [0.8, 0.9, 0.01, 0.5, 2]
            model.geom_size[g] = [0.004, 0.004, 0.004]

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

def compute_vertex_normals(vertices, faces):
    # one upward-pointing unit normal per cloth vertex: sum the normals of every triangle touching the vertex, then normalize
    normals = np.zeros((len(vertices), 3))
    for tri in faces:
        v0 = vertices[tri[0]]
        v1 = vertices[tri[1]]
        v2 = vertices[tri[2]]
        face_normal = np.cross(v1 - v0, v2 - v0)
        normals[tri[0]] = normals[tri[0]] + face_normal
        normals[tri[1]] = normals[tri[1]] + face_normal
        normals[tri[2]] = normals[tri[2]] + face_normal

    for i in range(len(normals)):
        length = float(np.linalg.norm(normals[i]))
        if length > 1e-12:
            normals[i] = normals[i] / length
        # "pointing up towards the sky": flip any normal facing down
        if normals[i][2] < 0:
            normals[i] = -normals[i]

    return normals

# angle in degrees between each vertex normal and the reference normal
def compute_relative_angles(normals, ref_index):
    ref = normals[ref_index]
    angles = np.zeros(len(normals))
    for i in range(len(normals)):
        dot = float(np.dot(normals[i], ref))
        if dot > 1.0:
            dot = 1.0
        if dot < -1.0:
            dot = -1.0
        angles[i] = np.degrees(np.arccos(dot))
    return angles

def ik_step(model, data, target, tol=0.008):
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

def close_gripper(model, data):
    gripper_act = model.actuator("gripper").id
    data.ctrl[gripper_act] = GRIPPER_CLOSED

def make_step_fn(model, data, task):
    end_x = 0.02
    if task == "drag":
        z = DRAG_Z
        start_x = -CLOTH_HALF + 0.02   
        waypoints = [
            (start_x, 0, HOVER_Z),
            (start_x, 0, z), # descend onto the cloth
            (end_x, 0, z), # drag in +x
            (end_x, 0, HOVER_Z), # lift
        ]
    else:
        z = PUSH_Z
        waypoints = [
            (-CLOTH_HALF - 0.01, 0, TABLE_TOP_Z + 0.08),
            (-CLOTH_HALF - 0.03, 0, z),  # descend behind the cloth edge
            (end_x, 0, z), # push forward through the cloth
            (end_x, 0, TABLE_TOP_Z + 0.08),
        ]

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

                # move on after 5s even if not reached
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

        mujoco.mj_step(model, data)

    def reset_fn(model, data):
        # called by the viewer's Reset button
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        state["index"] = 0
        state["deadline"] = None
        state["start_centroid"] = None
        state["finished_time"] = None
        state["checked"] = False

    return step_fn, reset_fn

def make_render_fn(model, data):

    faces = np.array(model.flex_elem).reshape(-1, 3)
    nvert = int(faces.max()) + 1

    top_faces = faces
    bottom_faces = faces[:, ::-1] + nvert   # reversed winding, bottom vertex set
    edge_count = {}
    for tri in faces:
        for k in range(3):
            a = int(tri[k])
            b = int(tri[(k + 1) % 3])
            if a > b:
                a, b = b, a
            edge_count[(a, b)] = edge_count.get((a, b), 0) + 1
    side_faces = []
    for (a, b), count in edge_count.items():
        if count == 1:   # edge belongs to only one triangle = cloth boundary
            side_faces.append([a, b, a + nvert])
            side_faces.append([b, b + nvert, a + nvert])
    all_faces = np.concatenate([top_faces, bottom_faces, np.array(side_faces)])

    last = {"drawn": None, "ref_index": None, "stats_html": None}

    def render_fn(scene):
        scene.update_from_mjdata(data)   # everything mjviser normally draws
        centers = np.array(data.flexvert_xpos)

        if last["drawn"] is None:
            drawn = centers
        else:
            drawn = last["drawn"].copy()
            moved = np.linalg.norm(centers - drawn, axis=1) > 0.0015
            drawn[moved] = centers[moved]
        last["drawn"] = drawn

        top = drawn.copy()
        top[:, 2] = top[:, 2] + CLOTH_RADIUS
        bottom = drawn.copy()
        bottom[:, 2] = bottom[:, 2] - CLOTH_RADIUS
        vertices = np.concatenate([top, bottom])

        # mjviser shifts its whole scene by -tracked_body_pos when camera
        # tracking is on; everything we draw needs the same shift
        offset = np.zeros(3)
        if scene.camera_tracking_enabled and scene._tracked_body_id is not None:
            offset = data.xpos[scene._tracked_body_id]

        scene.server.scene.add_mesh_simple(
            "/cloth",
            vertices=vertices - offset,
            faces=all_faces,
            color=(204, 51, 51),
            side="double",   # cloth is visible from both sides
        )

        # reference "0 degree" vertex 
        if last["ref_index"] is None:
            center_xy = drawn[:, 0:2].mean(axis=0)
            distances = np.linalg.norm(drawn[:, 0:2] - center_xy, axis=1)
            last["ref_index"] = int(np.argmin(distances))
        ref_index = last["ref_index"]

        normals = compute_vertex_normals(drawn, faces)
        angles = compute_relative_angles(normals, ref_index)

        # colored green (0 deg) -> red (90+ deg), ref arrow is blue
        starts = top
        ends = starts + normals * 0.025
        points = np.stack([starts, ends], axis=1)
        colors = np.zeros((len(starts), 2, 3), dtype=np.uint8)
        for i in range(len(starts)):
            t = angles[i] / 90.0
            if t > 1.0:
                t = 1.0
            red = int(255 * t)
            green = int(255 * (1.0 - t))
            colors[i, 0] = (red, green, 40)
            colors[i, 1] = (red, green, 40)
        colors[ref_index, 0] = (50, 100, 255)
        colors[ref_index, 1] = (50, 100, 255)

        scene.server.scene.add_line_segments(
            "/normals", points=points - offset, colors=colors, line_width=3)
        scene.server.scene.add_label(
            "/zero_ref", "0° ref", position=ends[ref_index] + np.array([0, 0, 0.01]) - offset)

        # live min/mean/max readout in the side panel
        if last["stats_html"] is None:
            last["stats_html"] = scene.server.gui.add_html("")
        last["stats_html"].content = (
            f"<div style='padding: 0 1em 0.5em 1em; font-size: 0.85em;'>"
            f"<strong>Normal angles:</strong> min {angles.min():.1f}° / "
            f"mean {angles.mean():.1f}° / max {angles.max():.1f}°</div>")

    return render_fn

def run_drop():
    model = compile_model(0.001, task="drop")
    data = mujoco.MjData(model)
    mjviser.Viewer(model, data, render_fn=make_render_fn(model, data)).run()

def run_arm_task(task):
    model = compile_model(ARM_TIMESTEP, task=task)
    data = mujoco.MjData(model)
    step_fn, reset_fn = make_step_fn(model, data, task)
    mjviser.Viewer(model, data, step_fn=step_fn, reset_fn=reset_fn, render_fn=make_render_fn(model, data)).run()

def main():
    if TASK == "drop":
        run_drop()
    elif TASK == "push" or TASK == "drag":
        run_arm_task(TASK)
    else:
        print(f"unknown task: {TASK} (use drop|push|drag)")
        return

    print(f"max |qacc|: {_max_qacc:.1f}  (healthy: <1e3-ish; 1e5+ is no good)")

if __name__ == "__main__":
    main()
