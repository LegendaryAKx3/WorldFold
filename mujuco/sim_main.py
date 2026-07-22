import os
import time
import gymnasium as gym
import so101_nexus
import so101_nexus.mujoco
import mujoco.viewer
import numpy as np
import mjviser

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
ARM_XML_PATH          = os.path.join(os.path.dirname(so101_nexus.__file__), "assets", "SO101", "so101_new_calib.xml")
ARM_BASE_LEFT         = (-0.30,  CLOTH_HALF, TABLE_TOP_Z) 
ARM_BASE_RIGHT        = ( 0.30, -CLOTH_HALF, TABLE_TOP_Z)  
ARM_JOINTS            = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_CLOSED        = -0.1
ARM_TIMESTEP         = 0.0005     # contact-heavy grasp needs 0.5ms; 1ms explodes on contact
MAX_STEP_MOVE        = 0.0003    # max amount can move per step
MAX_JOINT_STEP       = 0.005
CLOTH_DAMPING        = 0.3        # viscous damping per cloth vertex DOF; calms jitter/explosions (level3 value)
PULL_DIST            = 0.13       # how far each arm drags its corner outward (toward its base)

# Cloth Fold params
GRIPPER_OPEN         = 1.0      # gripper ctrlrange is [-0.175, 1.745]
ROT_WEIGHT           = 0.2     
MAX_STEP_ROT         = 0.002    # rad per substep
GRASP_RADIUS         = 0.03     # weld engages when gripper is this close to its corner
JOINT_DELTA_SCALE    = 0.05     # rad per control step at full action, joint_delta mode
HOLD_STEPS           = 10       # consecutive success steps (0.5s) before terminating
SETTLE_STEPS         = 2000     # 1.0s hands-off settle at reset, mirrors the demo above
WORKSPACE_XY         = 0.45
WORKSPACE_Z_LOW      = TABLE_TOP_Z + 0.003
WORKSPACE_Z_HIGH     = TABLE_TOP_Z + 0.35
SUCCESS_FOLD_SCORE   = 0.85
CORNER_PLACED_DIST   = 0.03
ANCHOR_SLACK         = 0.05
QACC_LIMIT           = 1e5

class ClothFoldEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, control_dt=0.05, max_episode_steps=200, action_scale_pos=0.03, action_scale_rot=0.1):
        self.control_dt = control_dt
        self.max_episode_steps = max_episode_steps
        self.n_substeps = int(round(control_dt / ARM_TIMESTEP))   # 100

        self.model = compile_model(ARM_TIMESTEP)
        self.data = mujoco.MjData(self.model)
        self.prefixes = ["left_", "right_"]

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)
        self._step_count = 0

        self.action_scale_pos = action_scale_pos
        self.action_scale_rot = action_scale_rot
        self._site_id = {}
        self._arm_qpos_adr = {}
        self._arm_dof_adr = {}
        self._arm_act_id = {}
        for prefix in self.prefixes:
            site = self.model.site(f"{prefix}gripperframe")
            self._site_id[prefix] = site.id
            qpos_adrs = []
            dof_adrs = []
            act_ids = []
            for name in ARM_JOINTS:
                joint = self.model.joint(f"{prefix}{name}")
                qpos_adrs.append(joint.qposadr[0])
                dof_adrs.append(joint.dofadr[0])
                actuator = self.model.actuator(f"{prefix}{name}")
                act_ids.append(actuator.id)
            self._arm_qpos_adr[prefix] = qpos_adrs
            self._arm_dof_adr[prefix] = dof_adrs
            self._arm_act_id[prefix] = act_ids
        self._target_pos = {}
        self._target_quat = {}

        self._weld_id = {}
        self._gripper_act = {}
        self._corner_body = {}
        self._gripper_closed = {"left_": False, "right_": False}
        corner_names = {"left_": f"cloth_{CLOTH_COUNT - 1}",
                        "right_": f"cloth_{(CLOTH_COUNT - 1) * CLOTH_COUNT}"}
        for prefix in self.prefixes:
            weld = self.model.equality(f"{prefix}weld")
            self._weld_id[prefix] = weld.id
            gripper_actuator = self.model.actuator(f"{prefix}gripper")
            self._gripper_act[prefix] = gripper_actuator.id
            corner = self.model.body(corner_names[prefix])
            self._corner_body[prefix] = corner.id

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        for prefix in self.prefixes:
            gripper_act = self.model.actuator(f"{prefix}gripper").id
            self.data.ctrl[gripper_act] = GRIPPER_OPEN
            self.data.eq_active[self._weld_id[prefix]] = 0
            self._gripper_closed[prefix] = False
        mujoco.mj_forward(self.model, self.data)
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)
        for prefix in self.prefixes:
            site_id = self._site_id[prefix]
            self._target_pos[prefix] = np.array(self.data.site_xpos[site_id])
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, self.data.site_xmat[site_id])
            self._target_quat[prefix] = quat
        self._step_count = 0
        return {}, {}
    
    def update_ee_target(self, prefix, pos_delta, rot_delta):
        delta = np.asarray(pos_delta, dtype=float) * self.action_scale_pos
        new_pos = self._target_pos[prefix] + delta
        if new_pos[0] < -WORKSPACE_XY:
            new_pos[0] = -WORKSPACE_XY
        if new_pos[0] > WORKSPACE_XY:
            new_pos[0] = WORKSPACE_XY
        if new_pos[1] < -WORKSPACE_XY:
            new_pos[1] = -WORKSPACE_XY
        if new_pos[1] > WORKSPACE_XY:
            new_pos[1] = WORKSPACE_XY
        if new_pos[2] < WORKSPACE_Z_LOW:
            new_pos[2] = WORKSPACE_Z_LOW
        if new_pos[2] > WORKSPACE_Z_HIGH:
            new_pos[2] = WORKSPACE_Z_HIGH
        self._target_pos[prefix] = new_pos
        rotvec = np.asarray(rot_delta, dtype=float) * self.action_scale_rot
        mujoco.mju_quatIntegrate(self._target_quat[prefix], rotvec, 1.0)

    def ik_substep(self, prefix):
        # 6D best-effort version of ik_step() above (5-DOF arm, so rotation is a down-weighted wish so bc of that position dominates)
        site_id = self._site_id[prefix]
        err_pos = self._target_pos[prefix] - self.data.site_xpos[site_id]
        dist = float(np.linalg.norm(err_pos))
        if dist > MAX_STEP_MOVE:
            err_pos = err_pos * (MAX_STEP_MOVE / dist)
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, self.data.site_xmat[site_id])
        err_rot = np.zeros(3)
        mujoco.mju_subQuat(err_rot, self._target_quat[prefix], site_quat)
        rot_norm = float(np.linalg.norm(err_rot))
        if rot_norm > MAX_STEP_ROT:
            err_rot = err_rot * (MAX_STEP_ROT / rot_norm)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        dofs = self._arm_dof_adr[prefix]
        jac_pos = jacp[:, dofs]
        jac_rot = ROT_WEIGHT * jacr[:, dofs]
        J = np.vstack([jac_pos, jac_rot])
        weighted_rot = ROT_WEIGHT * err_rot
        err = np.concatenate([err_pos, weighted_rot])
        solution = np.linalg.lstsq(J, err, rcond=None)
        dq = solution[0]
        biggest = float(np.max(np.abs(dq)))
        if biggest > MAX_JOINT_STEP:
            dq = dq * (MAX_JOINT_STEP / biggest)

        for k in range(len(ARM_JOINTS)):
            qpos_adr = self._arm_qpos_adr[prefix][k]
            act_id = self._arm_act_id[prefix][k]
            new_target = self.data.qpos[qpos_adr] + dq[k]
            low = self.model.actuator_ctrlrange[act_id][0]
            high = self.model.actuator_ctrlrange[act_id][1]
            if new_target < low:
                new_target = low
            if new_target > high:
                new_target = high
            self.data.ctrl[act_id] = new_target

    def set_gripper(self, prefix, command):
        # hysteresis: < -0.3 close, > 0.3 open, else hold current state
        if command < -0.3:
            self._gripper_closed[prefix] = True
        elif command > 0.3:
            self._gripper_closed[prefix] = False
        act_id = self._gripper_act[prefix]
        eqid = self._weld_id[prefix]
        if self._gripper_closed[prefix]:
            self.data.ctrl[act_id] = GRIPPER_CLOSED
            if self.data.eq_active[eqid] == 0:
                site = self.data.site_xpos[self._site_id[prefix]]
                corner = self.data.xpos[self._corner_body[prefix]]
                gap = float(np.linalg.norm(site - corner))
                if gap < GRASP_RADIUS:
                    b1 = self.model.eq_obj1id[eqid]
                    b2 = self.model.eq_obj2id[eqid]
                    R1 = self.data.xmat[b1].reshape(3, 3)
                    offset_world = self.data.xpos[b2] - self.data.xpos[b1]
                    offset_local = R1.T @ offset_world
                    self.model.eq_data[eqid, 3:6] = offset_local
                    self.data.eq_active[eqid] = 1
        else:
            self.data.ctrl[act_id] = GRIPPER_OPEN
            self.data.eq_active[eqid] = 0

    def step(self, action):
        raw = np.asarray(action, dtype=np.float32)
        raw = raw.reshape(14)
        clipped = np.clip(raw, -1.0, 1.0)

        self.set_gripper("left_", float(clipped[6]))
        self.set_gripper("right_", float(clipped[13]))

        self.update_ee_target("left_", clipped[0:3], clipped[3:6])
        self.update_ee_target("right_", clipped[7:10], clipped[10:13])
        for i in range(self.n_substeps):
            self.ik_substep("left_")
            self.ik_substep("right_")
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        return {}, 0.0, False, False, {}

def run_env_check():
    env = ClothFoldEnv()
    env.reset(seed=0)
    site_id = env._site_id["left_"]
    print("start z:", env.data.site_xpos[site_id][2])
    action = np.zeros(14)
    action[2] = -1.0   # push left EE down
    for i in range(10):
        env.step(action)
    print("end z:  ", env.data.site_xpos[site_id][2])
    print("milestone 2: z should have dropped ~0.1m")

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

def test_idk(env):
    action = np.zeros(14, dtype=np.float32)
    pos_slice = {"left_": 0, "right_": 7}
    grip_index = {"left_": 6, "right_": 13}
    for prefix in env.prefixes:
        site = env.data.site_xpos[env._site_id[prefix]]
        weld_active = env.data.eq_active[env._weld_id[prefix]]
        if weld_active:
            target = np.array([0.0, 0.0, TABLE_TOP_Z + 0.10])
        else:
            target = env.data.xpos[env._corner_body[prefix]]
        direction = target - site
        cmd = np.clip(direction * 20.0, -1.0, 1.0)
        start = pos_slice[prefix]
        action[start] = cmd[0]
        action[start + 1] = cmd[1]
        action[start + 2] = cmd[2]
        action[grip_index[prefix]] = -1.0
    return action

def main():
    env = ClothFoldEnv()
    env.reset(seed=0)
    state = {"i": 0}

    def step_fn(model, data):
        if state["i"] % env.n_substeps == 0:
            action = test_idk(env)
            clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            env.set_gripper("left_", float(clipped[6]))
            env.set_gripper("right_", float(clipped[13]))
            env.update_ee_target("left_", clipped[0:3], clipped[3:6])
            env.update_ee_target("right_", clipped[7:10], clipped[10:13])
        env.ik_substep("left_")
        env.ik_substep("right_")
        mujoco.mj_step(model, data)
        state["i"] += 1

    def reset_fn(model, data):
        env.reset(seed=0)
        state["i"] = 0

    mjviser.Viewer(env.model, env.data, step_fn=step_fn, reset_fn=reset_fn, render_fn=make_render_fn(env.model, env.data)).run()

if __name__ == "__main__":
    main()
