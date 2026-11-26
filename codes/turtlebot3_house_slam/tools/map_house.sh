#!/usr/bin/env bash
# Automatic mapping run: ./map_house.sh [run-name]
# Everything is written to runs/<run-name>/ while it runs.
P=$(dirname "$(dirname "$(realpath "$0")")")
R=$P/runs/${1:-$(date +%Y%m%d_%H%M%S)}
mkdir "$R" || exit 1
source /opt/ros/humble/setup.bash
unset GTK_PATH SNAP_COMMON
export PYTHONDONTWRITEBYTECODE=1 GZ_PARTITION=house_slam ROS_LOG_DIR=$R/roslog
since=$(date '+%Y-%m-%d %H:%M:%S')

# Run-only copies: true-pose publisher for grading, private velocity topic.
sed 's#^  </model>#    <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher"><odom_topic>/truth</odom_topic><tf_topic>/truth_tf</tf_topic><dimensions>2</dimensions><odom_publish_frequency>30</odom_publish_frequency></plugin>\n  </model>#' \
  "$P/models/turtlebot3_burger/model.sdf" > "$R/robot.sdf"
sed 's#ros_topic_name: "cmd_vel"#ros_topic_name: "house_sim/cmd_vel"#' "$P/config/bridge.yaml" > "$R/bridge.yaml"

groups=()
start() { local log=$1; shift; setsid "$@" > "$R/$log" 2>&1 < /dev/null & groups+=($!); }
cleanup() {
  for g in "${groups[@]}"; do kill -INT -- -"$g" 2>/dev/null; done
  sleep 10
  for g in "${groups[@]}"; do kill -KILL -- -"$g" 2>/dev/null; done
  echo STOPPED
}
trap cleanup EXIT
trap 'exit 130' INT TERM

start sim.log ros2 launch "$P/launch/slam_house.launch.py" gui:=false rviz:=true robot:="$R/robot.sdf" bridge:="$R/bridge.yaml"
start truth.log ros2 run ros_gz_bridge parameter_bridge '/truth@nav_msgs/msg/Odometry[gz.msgs.Odometry'
start bag.log ros2 bag record -o "$R/bag" --max-cache-size 0 /clock /scan /imu /wheel_odom /odom /tf /tf_static /truth
sleep 12

gz_pids=$(pgrep -u "$USER" -f '^gz sim')
[ -n "$gz_pids" ] || { echo "ABORT gazebo is not running"; exit 1; }
for p in $gz_pids; do
  if ls -l /proc/$p/fd 2>/dev/null | grep -qE '/dev/(nvidia|dri)' || grep -qE 'nvidia' /proc/$p/maps 2>/dev/null; then
    echo "ABORT gazebo opened the GPU"; exit 1
  fi
done
echo "gazebo is isolated from the GPU"

setsid timeout 2700 /usr/bin/python3 "$P/tools/drive_route.py" "$R" > "$R/drive.log" 2>&1 & driver=$!
groups+=("$driver")
while kill -0 "$driver" 2>/dev/null; do
  if journalctl -k --no-pager --since "$since" 2>/dev/null | grep -qE 'NVRM: Xid|fallen off the bus'; then
    echo "ABORT nvidia error in kernel log"; exit 1
  fi
  sleep 10
done
wait "$driver"; echo "driver rc=$?"
sleep 4
timeout 40 ros2 run nav2_map_server map_saver_cli -f "$R/map_005" --ros-args -p save_map_timeout:=20.0 > "$R/save.log" 2>&1
echo "map save rc=$?"
timeout 30 ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: '$R/posegraph'}" >> "$R/save.log" 2>&1
echo "posegraph rc=$?"
sync
echo FINISHED
