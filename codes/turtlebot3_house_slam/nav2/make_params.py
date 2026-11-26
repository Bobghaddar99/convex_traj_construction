#!/usr/bin/python3
"""Nav2 parameter files for pure DWA (DWB): stock nav2_params.yaml plus the overrides below.

    make_params.py            writes params_paper.yaml
    make_params.py optimized  also applies optimized.json (existing parameters only)

Table 3 -> DWB
    search time step 0.1 s           sim_period 0.1, and one control step per 0.1 s of sim time
    max v 2 m/s, max w 0.35 rad/s    max_vel_x, max_speed_xy, max_vel_theta
    0.5 m/s^2, 0.87 rad/s^2          acc_lim / decel_lim
    v step 0.02 in a 0.10 window     vx_samples 6
    w step 0.017 in a 0.174 window   vtheta_samples 11
    constant (v, w) prediction       LimitedAccelGenerator (DWA rather than trajectory rollout)
Nav2 Humble paces its control loop on wall time, so the simulator is held at RTF = 0.5 and
controller_frequency is 10 * RTF = 5 Hz: exactly one step per 0.1 s of simulated time.

Not given in the paper: prediction 3 s (as in the MATLAB sample); critics and weights are stock Nav2.
"""
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.realpath(__file__))
RTF = 0.5
name = sys.argv[1] if len(sys.argv) > 1 else 'paper'
with open('/opt/ros/humble/share/nav2_bringup/params/nav2_params.yaml') as f:
    stock = yaml.safe_load(f)
p = {k: stock[k] for k in ('amcl', 'map_server', 'planner_server', 'controller_server',
                           'local_costmap', 'global_costmap')}

p['map_server']['ros__parameters']['yaml_filename'] = os.path.join(
    os.path.dirname(HERE), 'maps', 'turtlebot3_house.yaml')
p['amcl']['ros__parameters'].update(
    set_initial_pose=True, initial_pose=dict(x=0.0, y=0.0, z=0.0, yaw=0.0),
    update_min_d=0.1, update_min_a=0.1, scan_topic='scan')
p['planner_server']['ros__parameters']['GridBased'].update(use_astar=True, allow_unknown=False)

c = p['controller_server']['ros__parameters']
c['controller_frequency'] = 10.0 * RTF
c['general_goal_checker']['yaw_goal_tolerance'] = 6.3  # position goals, as in the paper
c['FollowPath'].update(
    trajectory_generator_name='dwb_plugins::LimitedAccelGenerator', sim_period=0.1,
    debug_trajectory_details=False,
    max_vel_x=2.0, max_speed_xy=2.0, max_vel_theta=0.35,
    acc_lim_x=0.5, decel_lim_x=-0.5, acc_lim_theta=0.87, decel_lim_theta=-0.87,
    vx_samples=6, vy_samples=1, vtheta_samples=11, sim_time=3.0)

for k in ('local_costmap', 'global_costmap'):
    p[k][k]['ros__parameters']['robot_radius'] = 0.12  # Burger (stock value is for a Waffle)
# 2 m/s for 3 s is 6 m ahead: the stock 3 m window would silently cap the speed
p['local_costmap']['local_costmap']['ros__parameters'].update(width=14, height=14)

if name != 'paper':
    with open(os.path.join(HERE, name + '.json')) as f:
        changed = json.load(f)
    unknown = set(changed) - set(c['FollowPath'])
    if unknown:
        sys.exit(f'not existing DWB parameters: {unknown}')
    c['FollowPath'].update(changed)
with open(os.path.join(HERE, f'params_{name}.yaml'), 'w') as f:
    yaml.safe_dump(p, f, sort_keys=False)
print('wrote', f'params_{name}.yaml')
