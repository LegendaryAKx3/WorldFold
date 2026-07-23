import os
import gymnasium as gym
import so101_nexus
import mujoco
import numpy as np
import mjviser

TABLE_TOP_Z           = 0.42
CLOTH_COUNT           = 11      # grid resolution; higher = finer/drapier mesh
CLOTH_SPACING         = 0.03    # vertex gap; chosen with COUNT to keep span = (COUNT-1)*spacing = 0.30m
CLOTH_RADIUS          = 0.01    # collision thickness (physics only); MUST be >0 or cloth sits IN the table
VISUAL_THICKNESS      = 0.003   # how thick the cloth LOOKS (render only), decoupled from collision radius
CLOTH_MASS            = 0.05    # level3 value; stability comes from soft finger pads + heavy damping
CLOTH_HALF            = (CLOTH_COUNT-1) * CLOTH_SPACING / 2   # cloth spans +-0.09m

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
QACC_LIMIT           = 1e5
TASK_NAMES           = ["fold", "drop", "push", "drag"]   # index = task id (one-hot slot)

class ClothFoldEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, control_dt=0.05, max_episode_steps=200, action_scale_pos=0.03, action_scale_rot=0.1,
                 action_mode="ee_delta", observation_mode="state", image_size=(84, 84),
                 camera_names=None, domain_randomization=False, n_cloth_samples=9, n_tasks=4):
        self.control_dt = control_dt
        self.max_episode_steps = max_episode_steps
        self.n_substeps = int(round(control_dt / ARM_TIMESTEP))   # 100

        self.model = compile_model(ARM_TIMESTEP)
        self.data = mujoco.MjData(self.model)
        self.prefixes = ["left_", "right_"]

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)

        # behaviour only
        self.action_mode = action_mode
        self.domain_randomization = domain_randomization

        # shape-deciding
        self.observation_mode = observation_mode
        self.image_size = image_size
        self.camera_names = ["main"] if camera_names is None else camera_names
        self.n_cloth_samples = n_cloth_samples   # K: interior mesh points summarizing cloth shape
        self.n_tasks = n_tasks                   # number of task types (one-hot identifier)

        # proprio (per arm): joint pos + joint vel + gripper + EE pose(7) + EE vel(6) + grasp(1)
        per_arm = len(ARM_JOINTS) * 2 + 1 + 7 + 6 + 1
        P = per_arm * len(self.prefixes)

        # cloth_state: 4 corner pos + 4 corner vel + K sampled points + CoM + height stats + 4 corner-to-goal
        C = (4 * 3) + (4 * 3) + (self.n_cloth_samples * 3) + 3 + 3 + (4 * 3)

        # task: one-hot id + normalized stage(1) + goal keypoints(4x3) + per-corner progress(4) + time-left(1)
        T = self.n_tasks + 1 + (4 * 3) + 4 + 1

        # image (pixels/hybrid only): stacked RGB, one camera's 3 channels after another
        H, W = self.image_size
        img_shape = (H, W, 3 * len(self.camera_names))

        self._use_image = self.observation_mode in ("pixels", "hybrid")
        self._use_cloth = self.observation_mode in ("state", "hybrid")

        spaces = {
            "proprio": gym.spaces.Box(-np.inf, np.inf, shape=(P,), dtype=np.float32),
            "task": gym.spaces.Box(-np.inf, np.inf, shape=(T,), dtype=np.float32),
        }
        if self._use_cloth:
            spaces["cloth_state"] = gym.spaces.Box(-np.inf, np.inf, shape=(C,), dtype=np.float32)
        if self._use_image:
            spaces["image"] = gym.spaces.Box(0, 255, shape=img_shape, dtype=np.uint8)
        self.observation_space = gym.spaces.Dict(spaces)
        self._renderer = None   # lazily created on first image render

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

        # cloth vertices are bodies cloth_0 .. cloth_(N*N-1) (row-major grid)
        n_vert = CLOTH_COUNT * CLOTH_COUNT
        self._cloth_body_ids = [self.model.body(f"cloth_{i}").id for i in range(n_vert)]
        corner_idx = [0, CLOTH_COUNT - 1, (CLOTH_COUNT - 1) * CLOTH_COUNT, n_vert - 1]
        self._corner_ids = [self._cloth_body_ids[i] for i in corner_idx]
        sample_idx = np.linspace(0, n_vert - 1, self.n_cloth_samples).astype(int)
        self._sample_ids = [self._cloth_body_ids[i] for i in sample_idx]

        self._base_body_mass = self.model.body_mass.copy()
        self._base_geom_friction = self.model.geom_friction.copy()
        self._base_dof_damping = self.model.dof_damping.copy()
        self._table_geom_id = self.model.geom("table").id
        self._cloth_qpos_adr = []
        self._cloth_dof_adr = []
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name is None:
                self._cloth_qpos_adr.append(self.model.jnt_qposadr[i])
                self._cloth_dof_adr.append(self.model.jnt_dofadr[i])
        self._domain_params = {}

        # task/goal state (filled by reset)
        self._task_id = 0
        self._goal_corners = np.zeros((4, 3))
        self._goal_scale = np.ones(4)
        self._success_steps = 0
        self._action_clipped = False

    # ---- helpers ----
    def _body_linvel(self, bid):
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, bid, v, 0)
        return v[3:6]   # [angular(3), linear(3)] -> keep linear

    def _corner_dists(self):
        corners = self.data.xpos[self._corner_ids]
        return np.linalg.norm(self._goal_corners - corners, axis=1)   # (4,)

    def _corner_progress(self):
        return (self._corner_dists() < CORNER_PLACED_DIST).astype(np.float32)   # (4,)

    def _fold_score(self):
        return float(np.clip(1.0 - self._corner_dists() / self._goal_scale, 0.0, 1.0).mean())

    def _ik_ok(self, prefix):
        if self.action_mode == "joint_delta":
            return True   # no IK in joint mode
        site = self.data.site_xpos[self._site_id[prefix]]
        return bool(np.linalg.norm(self._target_pos[prefix] - site) < 0.02)

    def _failed(self):
        allc = self.data.xpos[self._cloth_body_ids]
        if np.any(np.abs(allc[:, :2]) > WORKSPACE_XY):
            return True
        if np.max(np.abs(self.data.qacc)) > QACC_LIMIT:
            return True
        return False

    def _render_image(self):
        # stacked RGB: each camera's (H,W,3) concatenated along the channel axis
        if self._renderer is None:
            H, W = self.image_size
            self._renderer = mujoco.Renderer(self.model, height=H, width=W)
        frames = []
        for cam in self.camera_names:
            self._renderer.update_scene(self.data, camera=cam)
            frames.append(self._renderer.render())
        return np.concatenate(frames, axis=2).astype(np.uint8)

    def _get_obs(self):
        # proprio (per arm): joint pos(5) + joint vel(5) + gripper(1) + EE pose(7) + EE vel(6) + grasp(1)
        proprio = []
        for p in self.prefixes:
            proprio += [self.data.qpos[a] for a in self._arm_qpos_adr[p]]
            proprio += [self.data.qvel[a] for a in self._arm_dof_adr[p]]
            proprio.append(self.data.ctrl[self._gripper_act[p]])
            sid = self._site_id[p]
            proprio += list(self.data.site_xpos[sid])
            q = np.zeros(4); mujoco.mju_mat2Quat(q, self.data.site_xmat[sid])
            proprio += list(q)
            v = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, sid, v, 0)
            proprio += list(v)
            proprio.append(float(self.data.eq_active[self._weld_id[p]]))
        proprio = np.array(proprio, dtype=np.float32)

        # task: one-hot(n_tasks) + stage(1) + goal keypoints(12) + progress(4) + time-left(1)
        onehot = np.zeros(self.n_tasks, dtype=np.float32); onehot[self._task_id] = 1.0
        stage = np.array([self._corner_progress().mean()], dtype=np.float32)
        tleft = np.array([1.0 - self._step_count / self.max_episode_steps], dtype=np.float32)
        task = np.concatenate([onehot, stage, self._goal_corners.ravel(),
                               self._corner_progress(), tleft]).astype(np.float32)

        obs = {"proprio": proprio, "task": task}

        if self._use_cloth:
            # cloth_state: 4 corner pos + 4 corner vel + K samples + CoM + height(min/max/mean z) + 4 corner-to-goal
            xpos = self.data.xpos
            corners = xpos[self._corner_ids]
            cvel = np.array([self._body_linvel(b) for b in self._corner_ids])
            samples = xpos[self._sample_ids]
            allc = xpos[self._cloth_body_ids]
            com = allc.mean(axis=0)
            z = allc[:, 2]
            height = np.array([z.min(), z.max(), z.mean()])
            to_goal = self._goal_corners - corners
            obs["cloth_state"] = np.concatenate([corners.ravel(), cvel.ravel(), samples.ravel(),
                                                 com, height, to_goal.ravel()]).astype(np.float32)
        if self._use_image:
            obs["image"] = self._render_image()

        return obs

    def _step_info(self, terms, reason=None):
        return {
            "reward_terms": terms,
            "success": self._success_steps >= HOLD_STEPS,
            "fold_score": self._fold_score(),
            "action_clipped": self._action_clipped,
            "left_ik_success": self._ik_ok("left_"),
            "right_ik_success": self._ik_ok("right_"),
            "left_grasp_active": bool(self.data.eq_active[self._weld_id["left_"]]),
            "right_grasp_active": bool(self.data.eq_active[self._weld_id["right_"]]),
            "termination_reason": reason,
        }

    def _reward(self):
        dist_term = -float(self._corner_dists().mean())
        success_term = 10.0 if self._fold_score() >= SUCCESS_FOLD_SCORE else 0.0
        terms = {"corner_dist": dist_term, "success_bonus": success_term}
        return dist_term + success_term, terms

    def shift_cloth(self, dx, dy):
        for j in range(0, len(self._cloth_qpos_adr), 3):
            self.data.qpos[self._cloth_qpos_adr[j]] += dx
            self.data.qpos[self._cloth_qpos_adr[j + 1]] += dy

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        opts = options or {}

        randomize = bool(opts.get("randomization", self.domain_randomization))
        self.model.body_mass[:] = self._base_body_mass
        self.model.geom_friction[:] = self._base_geom_friction
        self.model.dof_damping[:] = self._base_dof_damping
        self._domain_params = {}
        if randomize:
            mass_scale = float(self.np_random.uniform(0.7, 1.3))
            friction_scale = float(self.np_random.uniform(0.7, 1.3))
            damping_scale = float(self.np_random.uniform(0.7, 1.3))
            for bid in self._cloth_body_ids:
                self.model.body_mass[bid] *= mass_scale
            self.model.geom_friction[self._table_geom_id] *= friction_scale
            for adr in self._cloth_dof_adr:
                self.model.dof_damping[adr] *= damping_scale
            self._domain_params = {"cloth_mass_scale": mass_scale,
                                   "table_friction_scale": friction_scale,
                                   "cloth_damping_scale": damping_scale}

        mujoco.mj_resetData(self.model, self.data)
        for prefix in self.prefixes:
            gripper_act = self.model.actuator(f"{prefix}gripper").id
            self.data.ctrl[gripper_act] = GRIPPER_OPEN
            self.data.eq_active[self._weld_id[prefix]] = 0
            self._gripper_closed[prefix] = False

        if "cloth_pose" in opts:
            pose = np.asarray(opts["cloth_pose"], dtype=float).ravel()
            self.shift_cloth(float(pose[0]), float(pose[1]))
        elif randomize:
            offset = self.np_random.uniform(-0.03, 0.03, size=2)
            self.shift_cloth(float(offset[0]), float(offset[1]))
            self._domain_params["cloth_offset_xy"] = [float(offset[0]), float(offset[1])]

        mujoco.mj_forward(self.model, self.data)
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)
        for prefix in self.prefixes:
            site_id = self._site_id[prefix]
            self._target_pos[prefix] = np.array(self.data.site_xpos[site_id])
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, self.data.site_xmat[site_id])
            self._target_quat[prefix] = quat

        # sample task, then derive its goal keypoints (goal_pose option overrides)
        self._task_id = int(opts.get("task", self.np_random.integers(self.n_tasks)))
        corners0 = self.data.xpos[self._corner_ids].copy()
        if "goal_pose" in opts:
            self._goal_corners = np.asarray(opts["goal_pose"], dtype=float).reshape(4, 3).copy()
        else:
            # goal = fold cloth onto itself (each corner -> its diagonal-opposite start pos)
            self._goal_corners = corners0[[3, 2, 1, 0]].copy()
        self._goal_scale = np.maximum(np.linalg.norm(self._goal_corners - corners0, axis=1), 1e-6)
        self._success_steps = 0
        self._action_clipped = False
        self._step_count = 0

        info = {
            "task_name": TASK_NAMES[self._task_id],
            "episode_seed": seed,
            "goal_keypoints": self._goal_corners.copy(),
            "initial_cloth_keypoints": corners0,
            "domain_parameters": dict(self._domain_params),
        }
        return self._get_obs(), info
    
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

    def apply_joint_delta(self, prefix, deltas):
        for k in range(len(ARM_JOINTS)):
            act_id = self._arm_act_id[prefix][k]
            qpos_adr = self._arm_qpos_adr[prefix][k]
            new_target = self.data.qpos[qpos_adr] + float(deltas[k]) * JOINT_DELTA_SCALE
            low = self.model.actuator_ctrlrange[act_id][0]
            high = self.model.actuator_ctrlrange[act_id][1]
            if new_target < low:
                new_target = low
            if new_target > high:
                new_target = high
            self.data.ctrl[act_id] = new_target

    def step(self, action):
        raw = np.asarray(action, dtype=np.float32).reshape(14)
        clipped = np.clip(raw, -1.0, 1.0)
        self._action_clipped = bool(np.any(raw != clipped))

        self.set_gripper("left_", float(clipped[6]))
        self.set_gripper("right_", float(clipped[13]))

        if self.action_mode == "joint_delta":
            self.apply_joint_delta("left_", clipped[0:5])
            self.apply_joint_delta("right_", clipped[7:12])
            for _ in range(self.n_substeps):
                mujoco.mj_step(self.model, self.data)
        else:
            self.update_ee_target("left_", clipped[0:3], clipped[3:6])
            self.update_ee_target("right_", clipped[7:10], clipped[10:13])
            for _ in range(self.n_substeps):
                self.ik_substep("left_")
                self.ik_substep("right_")
                mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        reward, terms = self._reward()

        # terminated = held success OR unrecoverable failure; truncated = time limit
        self._success_steps = self._success_steps + 1 if self._fold_score() >= SUCCESS_FOLD_SCORE else 0
        reason = None
        if self._success_steps >= HOLD_STEPS:
            reason = "success"
        elif self._failed():
            reason = "cloth_out_of_bounds"
        terminated = reason is not None
        truncated = (not terminated) and self._step_count >= self.max_episode_steps

        return self._get_obs(), reward, terminated, truncated, self._step_info(terms, reason)

class StateOnlyWrapper(gym.ObservationWrapper):

    KEYS = ("proprio", "cloth_state", "task") # flat state vector for sb3, fixed key order is needed

    def __init__(self, env):
        super().__init__(env)
        dim = 0
        for k in self.KEYS:
            dim += env.observation_space[k].shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)

    def observation(self, obs):
        parts = []
        for k in self.KEYS:
            parts.append(obs[k])
        return np.concatenate(parts).astype(np.float32)

def train_sb3(use_state_wrapper=False, total_timesteps=100_000):
    from stable_baselines3 import PPO
    if use_state_wrapper:
        env = StateOnlyWrapper(ClothFoldEnv(observation_mode="state"))
        model = PPO("MlpPolicy", env, verbose=1)
    else:
        env = ClothFoldEnv(observation_mode="hybrid")
        model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps)
    return model

def dreamer_transition(obs, action, reward, terminated, truncated, is_first, gamma=0.99):
    return {
        "image": obs["image"],
        "proprio": obs["proprio"], # proprioception
        "action": np.asarray(action, dtype=np.float32),
        "reward": np.float32(reward),
        "is_first": is_first,
        "is_last": terminated or truncated,
        "is_terminal": terminated,
        "discount": np.float32(0.0 if terminated else gamma),
    }

def collect_episode(env, policy_fn, gamma=0.99):
    episode = []
    obs, info = env.reset()
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    episode.append(dreamer_transition(obs, zero_action, 0.0, False, False, True, gamma))
    while True:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode.append(dreamer_transition(obs, action, reward, terminated, truncated, False, gamma))
        if terminated or truncated:
            return episode

# verification before running
def check_contract(env, n_episodes=4, n_steps=5):
    ref = None
    for ep in range(n_episodes):
        obs, info = env.reset(options={"task": ep % env.n_tasks})
        for _ in range(n_steps):
            sig = {}
            for k in sorted(obs.keys()):
                sig[k] = (obs[k].shape, str(obs[k].dtype))
            if ref is None:
                ref = sig
            assert sig == ref, f"contract drift: {sig} != {ref}"
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print("contract ok:", ref)

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

    # left = blue, right = yellow 
    # this is js so that its easier for ppl to tell which arm is which
    arm_rgba = {"left_": [0.25, 0.45, 0.95, 1.0], "right_": [0.95, 0.75, 0.15, 1.0]}
    for g in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
        if body_name is None:
            continue
        for prefix, rgba in arm_rgba.items():
            if body_name.startswith(prefix):
                model.geom_matid[g] = -1   # drop the mesh material so the flat colour shows
                model.geom_rgba[g] = rgba

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
        corner = env.data.xpos[env._corner_body[prefix]]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, env.data.site_xmat[env._site_id[prefix]])
        env._target_quat[prefix] = q
        anchors = getattr(env, "_demo_anchor", None)
        if anchors is None:
            anchors = {}
            env._demo_anchor = anchors
        if weld_active:
            if prefix not in anchors:
                anchors[prefix] = np.array([site[0], site[1]])
            target = np.array([anchors[prefix][0], anchors[prefix][1], TABLE_TOP_Z + 0.10])
            grip = -1.0
        else:
            anchors.pop(prefix, None)
            horiz = float(np.linalg.norm((site - corner)[:2]))
            if horiz > 0.02:
                target = corner + np.array([0.0, 0.0, 0.06])
                grip = 1.0
            else:
                target = corner
                grip = -1.0
        direction = target - site
        cmd = np.clip(direction * 20.0, -1.0, 1.0)
        start = pos_slice[prefix]
        action[start] = cmd[0]
        action[start + 1] = cmd[1]
        action[start + 2] = cmd[2]
        action[grip_index[prefix]] = grip
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
