#!/usr/bin/env bash
# Diagonal A* + Nav2 DWA, same goals, settings, repeats and timing as experiment.py.
#
#   experiment_diagonal.sh [RViz window id for screenshots]
#
# Gazebo (sim.launch.py) must already be running. Only the global path differs from experiment.py.
D=$(dirname "$(realpath "$0")")
O=$D/results/diagonal
mkdir -p "$O"
source /opt/ros/humble/setup.bash
export PYTHONDONTWRITEBYTECODE=1
export GZ_PARTITION=house_slam
export PLAN_DIR=$D/diagonal_astar
export RVIZ_WINDOW=$1
since=$(date '+%Y-%m-%d %H:%M:%S')

for run in 1 2 3; do
  for setting in paper optimized; do
    if journalctl -k --no-pager --since "$since" | grep -qE 'NVRM: Xid|fallen off the bus'; then
      echo "ABORT nvidia error"
      exit 1
    fi
    if ! bash "$D/restart_nav2.sh" "$setting"; then
      echo "retrying nav2 ($setting)"
      bash "$D/restart_nav2.sh" "$setting" || { echo "SKIPPED $setting run $run"; continue; }
    fi
    /usr/bin/python3 -u "$D/run_goals.py" "diagonal/${setting}_run$run" test 150
  done
done

/usr/bin/python3 "$D/report.py" "$O" "Diagonal A*"

shots=()
for setting in paper optimized; do
  for goal in A_south_west_room B_middle_room C_south_east_room; do
    image=$O/${setting}_run1/screenshots/${goal}_end.png
    [ -f "$image" ] && shots+=(-label "$setting: $goal" "$image")
  done
done
if [ ${#shots[@]} -gt 0 ]; then
  montage "${shots[@]}" -tile 3x2 -geometry 900x610+6+6 -pointsize 28 "$O/screenshots_overview.png"
fi
echo DIAGONAL_DONE
