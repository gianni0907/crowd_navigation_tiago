# Safe robot navigation in a crowd: Application to the TIAGo mobile manipulator
The project addresses the problem of safe mobile robot navigation in a crowded environment. The adopted sensor-based scheme consists of two main modules:
-   Crowd prediction module: acquires information via an on-board laser rangefinder and produces predictions of the surrounding agents' motion.  
-   Motion generation module: given a desired target position and the crowd motion prediction, runs a Nonlinear Model Predictive Control (NMPC) algorithm to generate a collision-free robot motion. Collision avoidance constraints are formulated via discrete-time Control Barrier Functions (CBFs).

For further details, the interested reader is referred to the thesis and/or the slides in the `media` folder, where also videos of some simulations and experiments are reported.

## Installation
The project is realized using Ubuntu 20.04 with ROS Noetic (incompatibilities may arise considering different versions).

- Create a catkin workspace and clone this repository in the `src` folder:
```
mkdir -p <ws_name>/src
cd <ws_name>/src
git clone https://github.com/gianni0907/crowd_navigation_tiago
```
- In the `<ws_name>` directory, set the *Release* mode and build by running:
```
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin build
```
## Usage
To run the Gazebo simulation (make sure `simulation=True` in `crowd_navigation_core/src/crowd_navigation_core/Hparams.py`):
```bash
roslaunch labrob_tiago_gazebo tiago_gazebo.launch public_sim:=true end_effector:=pal-gripper world:=WORLD
```
where `WORLD` is one of the worlds in the `labrob_gazebo_worlds/worlds` directory. Depending on the chosen `WORLD`, modify accordingly the `n_actors` variable specified in `crowd_navigation_core/src/crowd_navigation_core/Hparams.py`.

If the real robot is considered, thus `simulation=False`, make sure it is inside the admissible region defined by `vertexes` in `crowd_navigation_core/src/crowd_navigation_core/Hparams.py`.

To run the crowd prediction module:
```bash
roslaunch crowd_navigation_core crowd_prediction.launch
```

To run the motion generation module:
```bash
roslaunch crowd_navigation_core motion_generation.launch
```

It is also possible to run the two modules simultaneously:
```bash
roslaunch crowd_navigation_core crowd_navigation.launch
```

To set a desired TIAGo target position:
```bash
roslaunch crowd_navigation_core send_desired_target_position.launch x_des:=X y_des:=Y
```
where `X` and `Y` are the coordinates of the desired goal position.

Note: the desired target position can be set multiple times, running the corresponding launch file with different coordinates.

When the nodes are shutdown, relevant data are logged in `.json` files saved in the folder `/tmp/crowd_navigation_tiago/data`, with names specified by the variable `filename` in `crowd_navigation_core/src/crowd_navigation_core/Hparams.py`. To plot results run:
```bash
roslaunch crowd_navigation_core plotter.launch filename:=FILENAME
```
where `FILENAME` is the name of the saved `.json` file (WITHOUT extension). Plots and animations shown are also saved in `/tmp/crowd_navigation_tiago/plots` and `/tmp/crowd_navigation_tiago/simulations` folders, respectively.
