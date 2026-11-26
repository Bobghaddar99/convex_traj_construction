# SLAM results

Open this file with the preview (Ctrl+Shift+V) to see all figures.

| Goals reached (3 runs each) | NavFn, paper | NavFn, alternative | Diagonal A*, paper | Diagonal A*, alternative |
|---|---|---|---|---|
| A, south-west room | 0/3 | 0/3 | 0/3 | 0/3 |
| B, middle room | 1/3 | 0/3 | 1/3 | 0/3 |
| C, south-east room | 2/3 | 2/3 | 2/3 | 0/3 |
| **Total** | **3/9** | **2/9** | **3/9** | **0/9** |
| Max velocity (m/s) | 0.71 | 1.96 | 0.71 | 1.30 |
| Max acceleration, with sudden stops (m/s²) | 6.2 | 16.1 | 6.5 | 0.50 |

State these next to the figures:

- Everything is simulation: TurtleBot3 house, front door closed. Table 3's 2 m/s is far above a real Burger's 0.22 m/s.
- The map was driven by an explorer that saw only the SLAM map and lidar. Four scan-matcher settings were
  tuned against true position on an earlier run in the same world.
- The east side of the map has 20-25 cm of drift. Eight low furniture bases are invisible to the lidar and show as free.
- "Paper" is Table 3. The paper gives no weights, so those are stock Nav2, with a 3 s prediction.
- "Alternative weights" (called optimized in file names) were tuned on two goals only and are not optimal.
- Nav2 ran without recovery behaviors or velocity smoother, with a 0.12 m radius and a 14 m local costmap.
- Velocity figures show run 1. The photos and tables include all three runs.
- Photos 01-10 are recorded trials replayed in RViz, showing all runs, including those that stopped short of the goal; 00, 11, 12 and 13 were captured live.

## map

### accuracy map vs truth

![accuracy_map_vs_truth](1_map/accuracy_map_vs_truth.png)

### slam map 020

![slam_map_020](1_map/slam_map_020.png)

### slam map

![slam_map](1_map/slam_map.png)

## paths

### 01 slam mapping path

![01_slam_mapping_path](2_paths/01_slam_mapping_path.png)

### 02 diagonal astar waypoints

![02_diagonal_astar_waypoints](2_paths/02_diagonal_astar_waypoints.png)

### 03 navfn dwa trajectories

![03_navfn_dwa_trajectories](2_paths/03_navfn_dwa_trajectories.png)

### 04 diagonal dwa trajectories

![04_diagonal_dwa_trajectories](2_paths/04_diagonal_dwa_trajectories.png)

## rviz photos

### 00 live window

![00_live_window](3_rviz_photos/00_live_window.png)

### 01 slam map

![01_slam_map](3_rviz_photos/01_slam_map.png)

### 02 diagonal astar waypoints

![02_diagonal_astar_waypoints](3_rviz_photos/02_diagonal_astar_waypoints.png)

### 03 navfn goal A

![03_navfn_goal_A](3_rviz_photos/03_navfn_goal_A.png)

### 04 navfn goal B

![04_navfn_goal_B](3_rviz_photos/04_navfn_goal_B.png)

### 05 navfn goal C

![05_navfn_goal_C](3_rviz_photos/05_navfn_goal_C.png)

### 06 navfn all goals

![06_navfn_all_goals](3_rviz_photos/06_navfn_all_goals.png)

### 07 diagonal goal A

![07_diagonal_goal_A](3_rviz_photos/07_diagonal_goal_A.png)

### 08 diagonal goal B

![08_diagonal_goal_B](3_rviz_photos/08_diagonal_goal_B.png)

### 09 diagonal goal C

![09_diagonal_goal_C](3_rviz_photos/09_diagonal_goal_C.png)

### 10 diagonal all goals

![10_diagonal_all_goals](3_rviz_photos/10_diagonal_all_goals.png)

### 11 slam mapping live

![11_slam_mapping_live](3_rviz_photos/11_slam_mapping_live.png)

### 12 rqt node graph

![12_rqt_node_graph](3_rviz_photos/12_rqt_node_graph.png)

### 13 during the trials

![13_during_the_trials](3_rviz_photos/13_during_the_trials.png)

## velocity acceleration

### 01 slam mapping drive

![01_slam_mapping_drive](4_velocity_acceleration/01_slam_mapping_drive.png)

### 02 goal A

![02_goal_A](4_velocity_acceleration/02_goal_A.png)

### 02 goal B

![02_goal_B](4_velocity_acceleration/02_goal_B.png)

### 02 goal C

![02_goal_C](4_velocity_acceleration/02_goal_C.png)

### 03 navfn paper

![03_navfn_paper](4_velocity_acceleration/03_navfn_paper.png)

### 04 navfn alternative weights

![04_navfn_alternative_weights](4_velocity_acceleration/04_navfn_alternative_weights.png)

### 05 diagonal paper

![05_diagonal_paper](4_velocity_acceleration/05_diagonal_paper.png)

### 06 diagonal alternative weights

![06_diagonal_alternative_weights](4_velocity_acceleration/06_diagonal_alternative_weights.png)

## tables

- [all_36_trials.csv](5_tables/all_36_trials.csv)
- [astar_search.csv](5_tables/astar_search.csv)
- [diagonal_velocity_acceleration.csv](5_tables/diagonal_velocity_acceleration.csv)
- [navfn_velocity_acceleration.csv](5_tables/navfn_velocity_acceleration.csv)
- [slam_mapping_drive.csv](5_tables/slam_mapping_drive.csv)
