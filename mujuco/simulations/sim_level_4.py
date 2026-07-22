'''
Level 4: TWO SO101 arms grab opposite front corners of the cloth and pull them
apart. Built on sim.py -- the one real change is that every arm helper now takes
a `prefix` ("left_" / "right_") so the two identical arms don't collide on
joint / actuator / site names when both are attached to the same model.
'''

import os
import time
import gymnasium as gym
import so101_nexus
import so101_nexus.mujoco
import mujoco.viewer
import numpy as np
import mjviser

# hi

TABLE_TOP_Z           = 0.42
CLOTH_COUNT           = 11      # grid resolution; higher = finer/drapier mesh
CLOTH_SPACING         = 0.03    # vertex gap; chosen with COUNT to keep span = (COUNT-1)*spacing = 0.30m
CLOTH_RADIUS          = 0.01    # collision thickness (physics only); MUST be >0 or cloth sits IN the table
VISUAL_THICKNESS      = 0.003   # how thick the cloth LOOKS (render only), decoupled from collision radius
CLOTH_MASS            = 0.05    # level3 value; stability comes from soft finger pads + heavy damping
CLOTH_HALF            = (CLOTH_COUNT-1) * CLOTH_SPACING / 2   # cloth spans +-0.09m
HOVER_Z               = TABLE_TOP_Z + 0.10
GRAB_Z                = TABLE_TOP_Z + 0.006   # fingertip presses onto the cloth; safe now that finger pads are soft. tune +-2mm

# SO101 arm model
ARM_XML_PATH     = os.path.join(os.path.dirname(so101_nexus.__file__), "assets", "SO101", "so101_new_calib.xml")
# each base sits directly BEHIND its target corner (same y), so reaching the corner
# is a straight-ahead +x/-x reach -- the arm's strong direction. reaching sideways
# (to an off-axis corner) is its weak direction and falls ~3cm short.
ARM_BASE_LEFT    = (-0.30,  CLOTH_HALF, TABLE_TOP_Z)   # behind left corner (-h,+h); faces +x
ARM_BASE_RIGHT   = ( 0.30, -CLOTH_HALF, TABLE_TOP_Z)   # behind right corner (+h,-h); faces -x (rotated 180)
ARM_JOINTS       = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_CLOSED   = -0.1

ARM_TIMESTEP     = 0.0005     # contact-heavy grasp needs 0.5ms; 1ms explodes on contact
MAX_STEP_MOVE    = 0.00015    # m per step = 0.3 m/s at 0.5ms steps; gentler approach = softer contact
MAX_JOINT_STEP   = 0.005
CLOTH_DAMPING    = 0.3        # viscous damping per cloth vertex DOF; calms jitter/explosions (level3 value)
PULL_DIST        = 0.13       # how far each arm drags its corner outward (toward its base)

def build_cloth_xml(timestep):
    # flat cloth spawned already at rest on the table (no drop -> no bounce/jitter)
    spawn = f'pos="0 0 {TABLE_TOP_Z + CLOTH_RADIUS + 0.001}"'
    edge = '<edge equality="true" damping="0.2"/>'

    xml = f"""
    <mujoco model="cloth_level4">
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
            {edge}
        </flexcomp>
        </worldbody>
    </mujoco>
    """
    return xml

def compile_model(timestep):
    spec = mujoco.MjSpec.from_string(build_cloth_xml(timestep))

    # attach two independent copies of the SO101 arm, each with its own name prefix
    left_spec = mujoco.MjSpec.from_file(ARM_XML_PATH)
    lf = spec.worldbody.add_frame(pos=ARM_BASE_LEFT)
    lf.attach_body(left_spec.body("base"), "left_", "")

    right_spec = mujoco.MjSpec.from_file(ARM_XML_PATH)
    rf = spec.worldbody.add_frame(pos=ARM_BASE_RIGHT)
    rf.quat = [0.0, 0.0, 0.0, 1.0]   # 180 deg about z, so this arm faces -x toward the cloth
    rf.attach_body(right_spec.body("base"), "right_", "")

    # SIM-ONLY GRASP CHEAT: a real gripper can't reliably pinch flat cloth, so we fake
    # the grab with a WELD equality tying the corner vertex to the gripper hand. inactive
    # until the arm 'grabs' (close_gripper sets the relpose to the current grab geometry
    # and flips data.eq_active). WELD is used over CONNECT because its relative pose is
    # settable at runtime -- CONNECT bakes its anchor at compile (home pose), which would
    # hold the cloth ~9cm from the claw. the cloth grid is row-major (index = ix*COUNT+iy),
    # so the two diagonal corners are:
    left_corner_body  = f"cloth_{CLOTH_COUNT - 1}"                    # (-h, +h)
    right_corner_body = f"cloth_{(CLOTH_COUNT - 1) * CLOTH_COUNT}"    # (+h, -h)
    for prefix, body in [("left_", left_corner_body), ("right_", right_corner_body)]:
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name1 = f"{prefix}gripper"     # body1: the gripper hand
        eq.name2 = body                   # body2: the corner cloth vertex
        eq.name = f"{prefix}weld"         # so we can look it up by id at runtime
        eq.active = False                 # off until the arm grabs
        # data = [anchor(3), relpose_pos(3), relpose_quat(4), torquescale(1)].
        # torquescale=0 -> position-only weld (a point vertex has no meaningful orientation);
        # relpose_pos is overwritten at grab time in close_gripper.
        eq.data[:] = [0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0,  0.0]

    model = spec.compile()

    # calm the cloth: light damping on every cloth vertex DOF. the cloth's joints
    # are the unnamed ones; both arms' joints now have names (left_/right_ prefixed).
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name is None:
            model.dof_damping[model.jnt_dofadr[i]] = CLOTH_DAMPING

    # each arm's finger pads ship as 1.25mm boxes with very stiff contact params,
    # which explode the ~1-gram cloth vertices on touch. soften and enlarge them
    # for BOTH arms (names are left_/right_ prefixed). this is the real fix for
    # the "arm resets on contact" divergence -- not keeping the finger away.
    for prefix in ["left_", "right_"]:
        for pad in ["static_finger_pad", "moving_finger_pad"]:
            g = model.geom(f"{prefix}{pad}").id
            model.geom_condim[g] = 3
            model.geom_solref[g] = [0.02, 1]
            model.geom_solimp[g] = [0.8, 0.9, 0.01, 0.5, 2]
            model.geom_size[g] = [0.004, 0.004, 0.004]

    return model

def cloth_x_extent(model, data):
    # width of the cloth along x -- grows as the two arms pull the corners apart
    xs = np.array(data.flexvert_xpos)[:, 0]
    return float(xs.max() - xs.min())

_max_qacc = 0.0

def tracked_step(model, data):
    global _max_qacc
    mujoco.mj_step(model, data)
    step_qacc = float(np.max(np.abs(data.qacc)))
    if step_qacc > _max_qacc:
        _max_qacc = step_qacc

def step(model, data, viewer):
    step_start = time.perf_counter()
    tracked_step(model, data)
    if viewer is not None:
        viewer.sync()
        elapsed = time.perf_counter() - step_start
        sleep_time = model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

def ik_step(model, data, target, prefix, tol=0.008):
    # ONE nudge of ONE arm's joint targets toward a cartesian fingertip target.
    # `prefix` selects which arm ("left_" / "right_"). returns True once within tol.
    site_id = model.site(f"{prefix}gripperframe").id
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
        dof_ids.append(model.joint(f"{prefix}{name}").dofadr[0])
    J = jacp[:, dof_ids]

    # solve J * dq = step_vec for the smallest joint motion dq
    dq = np.linalg.lstsq(J, step_vec, rcond=None)[0]
    biggest = float(np.max(np.abs(dq)))
    if biggest > MAX_JOINT_STEP:
        dq = dq * (MAX_JOINT_STEP / biggest)

    for k in range(len(ARM_JOINTS)):
        jname = f"{prefix}{ARM_JOINTS[k]}"
        qpos_addr = model.joint(jname).qposadr[0]
        act_id = model.actuator(jname).id
        new_target = data.qpos[qpos_addr] + dq[k]
        low = model.actuator_ctrlrange[act_id][0]
        high = model.actuator_ctrlrange[act_id][1]
        if new_target < low:
            new_target = low
        if new_target > high:
            new_target = high
        data.ctrl[act_id] = new_target

    return False

def close_gripper(model, data, prefix):
    gripper_act = model.actuator(f"{prefix}gripper").id
    data.ctrl[gripper_act] = GRIPPER_CLOSED
    # SIM-ONLY: weld the corner vertex to the gripper (real friction grip is too weak).
    # set the weld's relpose_pos to the corner's position in the gripper's CURRENT frame,
    # so the cloth is pinned right where the fingers are at grab (not the home-pose offset).
    eqid = model.equality(f"{prefix}weld").id
    b1 = model.eq_obj1id[eqid]           # gripper hand
    b2 = model.eq_obj2id[eqid]           # corner cloth vertex
    R1 = data.xmat[b1].reshape(3, 3)
    model.eq_data[eqid, 3:6] = R1.T @ (data.xpos[b2] - data.xpos[b1])
    data.eq_active[eqid] = 1

def make_arm(prefix, corner_xy, pull_dx):
    # corner_xy = (x, y) of the front corner this arm grabs
    # pull_dx   = how far in x to drag it once grabbed (- for left, + for right)
    cx, cy = corner_xy
    # with the corner welded to the gripper (see close_gripper), drag it horizontally
    # outward toward the arm's base -- the arm can't lift here (it just retracts), but
    # retracting IS the outward drag, and the weld carries the corner along reliably.
    waypoints = [
        (cx, cy, HOVER_Z),               # hover above the corner
        (cx, cy, GRAB_Z),                # descend onto the corner (grab -> weld on)
        (cx + pull_dx, cy, GRAB_Z),      # drag the corner outward horizontally
        (cx + pull_dx, cy, GRAB_Z),      # hold at the stretched position
    ]
    return {"prefix": prefix, "waypoints": waypoints,
            "index": 0, "deadline": None, "grabbed": False}

def drive_arm(model, data, arm):
    # advance ONE arm by one waypoint-nudge
    wp = arm["waypoints"]
    p = arm["prefix"]
    if arm["index"] >= len(wp):
        return
    if arm["deadline"] is None:
        arm["deadline"] = data.time + 5.0

    reached = ik_step(model, data, wp[arm["index"]], p)

    # GRAB when the fingertip is genuinely NEAR the corner (wp[1]), not merely when the
    # waypoint index ticks over. welding while still far freezes a big gripper->cloth
    # offset, so the cloth trails 8cm from the claw (looks like telekinesis). gating on
    # proximity locks in a <2cm offset -> the cloth sits right at the gripper.
    if not arm["grabbed"] and arm["index"] >= 1:
        sid = model.site(f"{p}gripperframe").id
        if np.linalg.norm(data.site_xpos[sid] - np.array(wp[1])) < 0.02:
            close_gripper(model, data, p)
            arm["grabbed"] = True

    if reached or data.time > arm["deadline"]:
        if not reached:
            print(f"{p}waypoint {arm['index']} not reached, moving on")
        arm["index"] += 1
        arm["deadline"] = None

def make_step_fn(model, data):
    # front-left and front-right corners of the cloth (y = +CLOTH_HALF edge)
    arms = [
        make_arm("left_",  (-CLOTH_HALF, CLOTH_HALF), -PULL_DIST),
        make_arm("right_", ( CLOTH_HALF, -CLOTH_HALF), +PULL_DIST),
    ]
    state = {"start_extent": None, "finished": False}

    def step_fn(model, data):
        if data.time > 1.0:   # first 1.0s: hands off, let the cloth settle flat
            if state["start_extent"] is None:
                state["start_extent"] = cloth_x_extent(model, data)
            for arm in arms:
                drive_arm(model, data, arm)

            # both arms done -> report how much the cloth stretched, once
            if not state["finished"] and all(a["index"] >= len(a["waypoints"]) for a in arms):
                state["finished"] = True
                grew = cloth_x_extent(model, data) - state["start_extent"]
                print(f"cloth x-extent grew {grew * 100:.1f} cm")
                print("PASS" if grew > 0.02 else "FAIL: corners barely moved")

        tracked_step(model, data)

    def reset_fn(model, data):
        # viewer Reset button: rewind physics AND both arms' scripts
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        for i, arm in enumerate(arms):
            arm.update({"index": 0, "deadline": None, "grabbed": False})
        state["start_extent"] = None
        state["finished"] = False

    return step_fn, reset_fn

def make_render_fn(model, data):
    # mjviser skips flex objects, so push the cloth to the browser as a triangle mesh.
    # build ONE closed slab: top surface + bottom surface + side walls, so it reads
    # as a solid sheet with thickness instead of two disconnected floating layers.
    # faces/topology never change; only vertex positions update each frame.
    faces = np.array(model.flex_elem).reshape(-1, 3)
    n = int(np.array(data.flexvert_xpos).shape[0])   # vertices per layer

    # boundary edges = edges used by exactly ONE triangle (the outline of the sheet)
    edge_count = {}
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (a, b) if a < b else (b, a)
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary = [e for e, c in edge_count.items() if c == 1]

    # assemble the static face list once (layer indices: top = 0..n-1, bottom = n..2n-1)
    side_faces = []
    for (a, b) in boundary:
        # two triangles bridging the top edge (a,b) to the bottom edge (a+n,b+n)
        side_faces.append([a, b, b + n])
        side_faces.append([a, b + n, a + n])
    all_faces = np.vstack([
        faces,                       # top surface
        faces[:, ::-1] + n,          # bottom surface (reversed winding, shifted to bottom verts)
        np.array(side_faces, dtype=faces.dtype).reshape(-1, 3),  # side walls
    ])

    def render_fn(scene):
        # mjviser auto-tracks the first movable body -- which here is a cloth corner
        # vertex, so the camera chases that wobbling corner and the whole scene
        # shimmers. pin the camera to the world instead (fixed table-top view).
        scene.camera_tracking_enabled = False
        scene.update_from_mjdata(data)
        vertices = np.array(data.flexvert_xpos)
        if scene.camera_tracking_enabled and scene._tracked_body_id is not None:
            vertices = vertices - data.xpos[scene._tracked_body_id]
        # visual slab, decoupled from the physics radius: a vertex center rests
        # ~CLOTH_RADIUS above the table, so drop to the cloth's actual bottom surface
        # (center - radius, ~table level) and build a thin VISUAL_THICKNESS slab up
        # from there -- so it looks like fabric, not a mattress. (render only.)
        base = vertices - np.array([0.0, 0.0, CLOTH_RADIUS])
        # the physics vertex sinks a few mm into the table's soft contact; clamp the
        # drawn bottom so it never dips below the table top (else the cloth visually
        # clips through). vertices that are lifted stay untouched.
        base[:, 2] = np.maximum(base[:, 2], TABLE_TOP_Z + 0.001)
        combined = np.vstack([base + np.array([0.0, 0.0, VISUAL_THICKNESS]), base])
        scene.server.scene.add_mesh_simple(
            "/cloth", vertices=combined, faces=all_faces,
            color=(204, 51, 51), side="double",
        )
    return render_fn

def main():
    model = compile_model(ARM_TIMESTEP)
    data = mujoco.MjData(model)
    step_fn, reset_fn = make_step_fn(model, data)
    mjviser.Viewer(model, data, step_fn=step_fn, reset_fn=reset_fn,
                   render_fn=make_render_fn(model, data)).run()
    print(f"max |qacc|: {_max_qacc:.1f}  (healthy: <1e3-ish; 1e5+ is no good)")

if __name__ == "__main__":
    main()
