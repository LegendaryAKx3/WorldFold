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


import time
import gymnasium as gym
import so101_nexus.mujoco
import mujoco.viewer
from so101_nexus import PickConfig, YCBObject
import numpy as np

TABLE_TOP_Z           = 0.42
CLOTH_COUNT           = 7
CLOTH_SPACING         = 0.03
CLOTH_RADIUS          = 0.005
CLOTH_MASS            = 0.05
CLOTH_HALF            = (CLOTH_COUNT-1) * CLOTH_SPACING / 2
CLOTH_SPAWN_Z         = TABLE_TOP_Z + 0.10
HOVER_Z               = 0.55
GRIP_Z                = 0.45

# def build_elasticity_block(mode):
#     if mode == "native":
#         top = ""
#         child = '<elasticity young="3e4" poisson="0.0" thickness="1e-3" damping="0.05"/>'
#         return top, child
#     else:
#         top = '<extension><plugin plugin="mujoco.elasticity.shell"/></extension>'
#         child = """<plugin plugin="mujoco.elasticity.shell">
#         <config key="young" value="3e4"/>
#         <config key="poisson" value="0.0"/>
#         <config key="thickness" value="1e-3"/>
#         </plugin>"""
#         return top, child
    
# def build_gripper_body(name, x, y):
#     xml = f"""
#     <body name="{name}" mocap="true" pos="{x} {y} {HOVER_Z}">
#     <geom type="box" size="0.008 0.004 0.025" pos="0 -0.006 0" rgba="0.2 0.2 0.8 1"
#             friction="1.2 0.005 0.0001"/>
#     <geom type="box" size="0.008 0.004 0.025" pos="0  0.006 0" rgba="0.2 0.2 0.8 1"
#             friction="1.2 0.005 0.0001"/>
#     </body>"""
#     return xml

def build_xml_file(timestep, mode):
    if mode == "native":
        top = ""
        edge = '<edge equality="true" damping="0.002"/>'
        block = ""
    else:
        top = '<extension><plugin plugin="mujoco.elasticity.shell"/></extension>'
        edge = '<edge equality="true" damping="0.002"/>'
        block = """<plugin plugin="mujoco.elasticity.shell">
            <config key="young" value="3e4"/>
            <config key="poisson" value="0.0"/>
            <config key="thickness" value="1e-3"/>
        </plugin>"""

    # <geom name="ball" type="sphere" size="0.04" pos="0 0 0.04" rgba="0.4 0.4 0.5 1"/>

    xml = f"""
    <mujoco model="cloth_level1">
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
                    pos="0 0 {CLOTH_SPAWN_Z}" euler="25 15 0" radius="{CLOTH_RADIUS}" mass="{CLOTH_MASS}"
                    dim="2" rgba="0.8 0.2 0.2 1">
            <contact condim="3" solref="0.01 1" solimp="0.9 0.95 0.001"
                    friction="1.0 0.005 0.0001" selfcollide="none" internal="false"/>
            {edge}
            {block}
        </flexcomp>
        </worldbody>
    </mujoco>"""

    return xml

def compile_model(timestep):
    xml = build_xml_file(timestep, mode="native")
    model = mujoco.MjModel.from_xml_string(xml)
    return model

def debug(model):
    names = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name is not None and name.startswith("cloth_"):
            names.append(name)
    print(f"cloth vertex bodies: {len(names)} "
        f"(first: {names[0]}, last: {names[-1]})")
    
def main():
    model = compile_model(timestep=0.001)
    data = mujoco.MjData(model)
    debug(model)

    max_qacc = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 20.0:
            step_start = time.perf_counter()
            mujoco.mj_step(model, data)

            step_qacc = float(np.max(np.abs(data.qacc)))
            if step_qacc > max_qacc:
                max_qacc = step_qacc

            viewer.sync()

            # makes it so the sim doesnt finish instantly
            elapsed = time.perf_counter() - step_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print(f"max |qacc|: {max_qacc:.1f}  (healthy: <1e3-ish; 1e5+ is no good)")

if __name__ == "__main__":
    main()













