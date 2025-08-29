import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import casadi as cs
from collections import namedtuple
import jax.numpy as jnp
import jax

StateOrder = namedtuple("StateOrder", ["pos", "vel", "act"])

nominal_order = StateOrder(
    pos=["x", "y", "z", "q_x", "q_y", "q_z",
         "L_thigh_joint", "L_calf_joint", "L_wheel_joint",
         "R_thigh_joint", "R_calf_joint", "R_wheel_joint"],
    vel=["lin_v_x", "lin_v_y", "lin_v_z", "ang_v_x", "ang_v_y", "ang_v_z",
         "L_thigh_joint", "L_calf_joint", "L_wheel_joint",
         "R_thigh_joint", "R_calf_joint", "R_wheel_joint"],
    act=["L_thigh_joint", "L_calf_joint", "L_wheel_joint",
         "R_thigh_joint", "R_calf_joint", "R_wheel_joint"]
)

mujoco_order = StateOrder(
    pos=["x", "y", "z", "q_x", "q_y", "q_z",
         "R_thigh_joint", "R_calf_joint", "R_wheel_joint",
         "L_thigh_joint", "L_calf_joint", "L_wheel_joint"],
    vel=["lin_v_x", "lin_v_y", "lin_v_z", "ang_v_x", "ang_v_y", "ang_v_z",
         "R_thigh_joint", "R_calf_joint", "R_wheel_joint",
         "L_thigh_joint", "L_calf_joint", "L_wheel_joint"],
    act=["R_thigh_joint", "R_calf_joint", "R_wheel_joint",
         "L_thigh_joint", "L_calf_joint", "L_wheel_joint"]
)


def q_idx_to_jnt_idx(model, q_idx):
    for jnt_idx in range(model.njoints):
        joint = model.joints[jnt_idx]
        if joint.idx_q <= q_idx < joint.idx_q + joint.nq:
            return jnt_idx
            
def v_idx_to_jnt_idx(model, v_idx):
    for jnt_idx in range(model.njoints):
        joint = model.joints[jnt_idx]
        if joint.idx_v <= v_idx < joint.idx_v + joint.nv:
            return jnt_idx

def define_state_order(model):
    # Define state order
    state_order = [model.names[q_idx_to_jnt_idx(model, i)] for i in range(model.nq)] + \
                        [model.names[v_idx_to_jnt_idx(model, i)] for i in range(model.nv)]
    
    # Fix quaternions, detect unsupported joints
    for jnt_idx in range(model.njoints):
        joint = model.joints[jnt_idx]
        if joint.nq == 1: 
            continue
        elif joint.nq == 7:  # Modify symbolic order to match with our convention by swapping q_w and q_x
            state_order[joint.idx_q:joint.idx_q + joint.nq] = ["x", "y", "z", "q_w", "q_x", "q_y", "q_z"]
            state_order[model.nq + joint.idx_v:model.nq + joint.idx_v + joint.nv] = ["lin_v_x", "lin_v_y", "lin_v_z", "ang_v_x", "ang_v_y", "ang_v_z"]
    return state_order

LIB_PATH = '/home/crazydog/test/pineapple_ctrl/PinnZoo/build/libpineapple_6dof.so'
# LIB_PATH = '/Users/juanalvarez/Documents/CMU/PhD/REXLab/pineapple_ctrl/models/pineapple/compiled_code/libpineapple_6dof.dylib'
kinematics = cs.external('kinematics', LIB_PATH)
M_func = cs.external('M_func', LIB_PATH)
C_func = cs.external('C_func', LIB_PATH)
velocity_kinematics = cs.external('velocity_kinematics', LIB_PATH)
kinematics_velocity_jacobian = cs.external('kinematics_velocity_jacobian', LIB_PATH)
kinematics_jacobian = cs.external('kinematics_jacobian', LIB_PATH)
kinematics_rotation = cs.external('kinematics_rotation', LIB_PATH)
wheels_kinematics = cs.external('wheels_kinematics', LIB_PATH)
wheels_kinematics_jacobian = cs.external('wheels_kinematics_jacobian', LIB_PATH)

def residuals(x_val, wheel_radius=0.069):
    foot_zs = kinematics(x_val)[[2,5]] 
    return foot_zs - wheel_radius

def residuals_jacobian(x_val):
    J = kinematics_jacobian(x_val)  # shape: (num_kinematics_outputs, nx)
    return J[[2, 5], 2]

def newton_update(x_val, wheel_radius=0.069):
    r = residuals(x_val, wheel_radius)
    J = residuals_jacobian(x_val)
    dz_base = np.mean(r / J)  # average Newton update across both feet
    x_val[2] -= dz_base       # update base z
    return x_val, dz_base

def build_x(model, z_body, ori_body, 
            left_joints, right_joints, left_joints_vel, right_joints_vel):
    x_val = np.zeros(model.nq + model.nv)
    x_val[2] = z_body
    x_val[3:7] =  ori_body
    x_val[7:10] = left_joints
    x_val[10:13] = right_joints
    x_val[model.nq+6:model.nq+6+3] = left_joints_vel
    x_val[model.nq+6+3:model.nq+6+6] = right_joints_vel
    return x_val

def estimate_z(x_val, wheel_radius=0.069):
    """
    Estimate the base z position using Newton's method.
    """
    for i in range(10):
        x_val, delta_z = newton_update(x_val, wheel_radius)
        if abs(delta_z) < 1e-6:
            break
    return x_val[2]

def estimate_body_xy_velocity(model, x_v):
    x_val = x_v.copy()
    x_val[3:7] = [1, 0, 0, 0]  # Reset orientation to identity quaternion]
    JJ_wheel_contact = wheels_kinematics_jacobian(x_val)
    J_wheel_contact = JJ_wheel_contact[0:6, model.nq:]

    JJ_wheel_center = kinematics_velocity_jacobian(x_val)
    J_wheel_center  = JJ_wheel_center[:, model.nq:]

    wheels_speed = -wheels_kinematics(x_val)[:6]
    #body_vel = 0.5 * (wheels_speed[[0,1]] + wheels_speed[[3,4]])
    body_vel = np.linalg.pinv(J_wheel_center)[:3] @ wheels_speed
    return body_vel


def reorder_actuation(u: np.ndarray, from_order: StateOrder, to_order: StateOrder) -> np.ndarray:
    act_idx_map = {name: i for i, name in enumerate(from_order.act)}
    return np.array([u[act_idx_map[name]] for name in to_order.act])

def reorder_state(x: np.ndarray, from_order: StateOrder, to_order: StateOrder) -> np.ndarray:
    n_pos = len(from_order.pos)
    n_vel = len(from_order.vel)
    assert x.shape[0] == n_pos + n_vel, "State vector length mismatch"

    x_pos = x[:n_pos]
    x_vel = x[n_pos:]

    pos_idx_map = {name: i for i, name in enumerate(from_order.pos)}
    vel_idx_map = {name: i for i, name in enumerate(from_order.vel)}

    x_pos_reordered = np.array([x_pos[pos_idx_map[name]] for name in to_order.pos])
    x_vel_reordered = np.array([x_vel[vel_idx_map[name]] for name in to_order.vel])

    return np.concatenate([x_pos_reordered, x_vel_reordered])

def skew(v):
    """Skew-symmetric matrix for vector v."""
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ]
    ], dtype=float)

def quat_to_rot(q):
    """Convert quaternion [qw, qx, qy, qz] → 3×3 rotation matrix."""
    qw, qx, qy, qz = q
    qv = np.array([qx, qy, qz])
    K = skew(qv)
    I = np.eye(3)
    # I + 2*qw*K + 2*K@K corresponds to standard quaternion-to-rotation
    return I + 2*qw*K + 2*K @ K

def L_mult(q):
    qw, qx, qy, qz = q
    qv = np.array([qx, qy, qz])
    I = np.eye(3)
    top = np.hstack([[qw], -qv])                    # shape (4,)
    bottom = np.hstack([qv.reshape(3,1), qw*I + skew(qv)])  # shape (3,4)
    return np.vstack([top, bottom])                # shape (4,4)


def quat_to_axis_angle(q, tol=1e-12):
    """
    Convert quaternion [qw, qx, qy, qz] → axis-angle vector.
    Uses qs<0 flip and small-angle approximation.
    """
    qw, qv = q[0], q[1:].astype(float)
    norm_qv = np.linalg.norm(qv)
    if qw < 0:
        qw, qv = -qw, -qv
    if norm_qv >= tol:
        theta = 2 * np.arctan2(norm_qv, qw)
        axis = qv / norm_qv
        return theta * axis
    else:
        return 2 * qv  # first-order approx

def state_error(x, x0):
    """
    Python version of state_error(...) from Julia,
    using only NumPy.
    """
    # position error in body frame
    R0 = quat_to_rot(x0[3:7])
    dp = x[0:3] - x0[0:3]
    err_pos = R0.T @ dp

    # quaternion error → axis-angle
    quat_err = L_mult(x0[3:7]).T @ x[3:7]
    err_q = quat_to_axis_angle(quat_err)

    # remaining states
    err_rest = x[7:] - x0[7:]

    return np.concatenate([err_pos, err_q, err_rest])

@jax.jit
def wrap_to_pi_jit(angle):
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

@jax.jit
def skew_jit(v):
    """JAX-compatible skew-symmetric matrix for vector v."""
    return jnp.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0]
    ])

@jax.jit
def L_mult_jit(q):
    """Left quaternion multiplication matrix (JAX version)."""
    qw, qx, qy, qz = q
    qv = jnp.array([qx, qy, qz])
    I = jnp.eye(3)
    top = jnp.hstack([qw, -qv])
    bottom = jnp.hstack([qv.reshape((3,1)), qw * I + skew_jit(qv)])
    return jnp.vstack([top, bottom])

@jax.jit
def quat_to_axis_angle_jit(q, tol=1e-12):
    qw = q[0].squeeze()  # Ensure scalar
    qv = q[1:].reshape(-1)[:3]  # Ensure 3-element vector
    qv_norm = jnp.linalg.norm(qv)
    # Avoid tuple output in jnp.where
    sign = jnp.where(qw < 0, -1.0, 1.0)
    qw = qw * sign
    qv = qv * sign
    theta = 2 * jnp.arctan2(qv_norm, qw)
    axis = jnp.where(qv_norm > tol, qv / qv_norm, jnp.zeros_like(qv))
    angle_axis = jnp.where(qv_norm > tol, theta * axis, 2 * qv)
    return angle_axis

@jax.jit
def measurement_error(y, y0):
    """
    Python version of state_error(...) from Julia,
    using only NumPy.
    """
    # position error in body frame
    #print(f"y0: {y0}, y: {y}")
    err_z = y[0] - y0[0]

    #jax.debug.print("quaternion: {}, {}", y0[1:5], y[1:5])
    # # quaternion error → axis-angle
    quat_err = L_mult_jit(y0[1:5]).T @ y[1:5]
    #jax.debug.print("quat_err: {}", quat_err)

    err_q = quat_to_axis_angle_jit(quat_err)

    #jax.debug.print("err_q: {}", err_q)

    # # remaining states
    err_rest = y[5:] - y0[5:]

    #jax.debug.print("err_rest: {}", err_rest)

    err_rest = err_rest.at[2].set(wrap_to_pi_jit(err_rest[2]))
    err_rest = err_rest.at[5].set(wrap_to_pi_jit(err_rest[5]))

    return jnp.concatenate([jnp.atleast_1d(err_z), err_q, err_rest])
    #return jnp.concatenate([jnp.atleast_1d(err_z)])