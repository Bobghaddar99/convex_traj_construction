#!/usr/bin/env bash
R=/home/ibrahim/Desktop/convex/turtlebot3_house_slam/runs/live_142803
source /opt/ros/humble/setup.bash
export PYTHONDONTWRITEBYTECODE=1 ROS_LOG_DIR=/tmp/explore_log
setsid ros2 bag record -o $R/bag --max-cache-size 0 /clock /scan /imu /wheel_odom /odom /tf /tf_static > $R/bag.log 2>&1 &
BAG=$!
bwrap --dev-bind / / --tmpfs /home --tmpfs /tmp --ro-bind /home/ibrahim/Desktop/convex/turtlebot3_house_slam/tools/explore /tmp/explore --chdir /tmp sh -c 'echo "house files visible to explorer: $(find /home /tmp -iname "*.sdf" -o -iname "*house*" 2>/dev/null | wc -l)"; exec timeout 2100 /usr/bin/python3 /tmp/explore/explore.py' &
EXP=$!
since=$(date '+%Y-%m-%d %H:%M:%S'); n=0
while kill -0 $EXP 2>/dev/null; do
  sleep 10; n=$((n+1))
  if journalctl -k --no-pager --since "$since" | grep -qE 'NVRM: Xid|fallen off the bus'; then
    echo "ABORT nvidia error"; kill -TERM $EXP; kill -INT -- -$(cat $R/sim.pgid); kill -INT -- -$BAG; echo SESSION_END; exit 1
  fi
  [ $((n % 6)) -eq 0 ] && timeout 20 ros2 run nav2_map_server map_saver_cli -f $R/checkpoint --free 0.196 > $R/save.log 2>&1
done
sleep 3
timeout 40 ros2 run nav2_map_server map_saver_cli -f $R/map_005 --free 0.196 --ros-args -p save_map_timeout:=20.0 > $R/save.log 2>&1; echo "save rc=$?"
timeout 30 ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: '$R/posegraph'}" >> $R/save.log 2>&1; echo "posegraph rc=$?"
kill -INT -- -$BAG
echo SESSION_END
