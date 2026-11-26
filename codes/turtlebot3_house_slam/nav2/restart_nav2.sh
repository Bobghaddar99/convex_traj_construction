#!/usr/bin/env bash
# restart_nav2.sh <paper|optimized>   restarts Nav2 only; Gazebo is never restarted
D=$(dirname "$(realpath "$0")")
R=$D/results; mkdir -p "$R"
source /opt/ros/humble/setup.bash
export ROS_LOG_DIR=$R/roslog PYTHONDONTWRITEBYTECODE=1
unset GTK_PATH SNAP_COMMON
if [ -f "$R/nav2.pgid" ]; then
  g=$(cat "$R/nav2.pgid")
  if ps -o args= -p "$g" | grep -q 'nav2/nav2.launch.py'; then
    kill -INT -- -"$g"
    for i in $(seq 1 15); do kill -0 -- -"$g" 2>/dev/null || break; sleep 1; done
    kill -KILL -- -"$g" 2>/dev/null
  fi
fi
L=$R/nav2_$1.log
setsid nohup bash -c "echo \$\$ > $R/nav2.pgid; exec ros2 launch $D/nav2.launch.py params:=$D/params_$1.yaml" > "$L" 2>&1 < /dev/null &
for i in $(seq 1 50); do
  grep -q 'Managed nodes are active' "$L" && { echo "nav2 active ($1)"; exit 0; }
  sleep 1
done
echo "nav2 did NOT activate ($1)"; tail -5 "$L"; exit 1
