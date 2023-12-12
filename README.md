# Crowd Navigation with TIAGo
## Project on-track
Implementation of NMPC to navigate the TIAGo robot to a desired target position in an environment with 5 humans standing still. The humans avoidance is realized introducing nonlinear constraints defined by CBFs in the NMPC model of the Acados solver. 
## Next goal
Implement a TIAGo navigation within a dynamic environment:

-   Humans moving along pre-defined linear trajectories

-   TIAGo is provided with the current humans position and estimates their future trajectories (KFs)

-   Based on the trajectory estimations, adapt the NMPC to avoid moving obstacles

## Usage
To run the Gazebo simulation:
```bash
roslaunch my_tiago_gazebo tiago_gazebo.launch public_sim:=true end_effector:=pal-gripper world:=WORLD
```
where `WORLD` is one of the worlds in `my_tiago_gazebo/worlds` or `pal_gazebo_worlds` package.


To run the NMPC controller module:
```bash
roslaunch my_tiago_controller nmpc_controller.launch
```

Note that the position of the obstacles for this module is independent from the humans position specified in the file `my_tiago_controller/worlds/5_humans.world`. Thus, if you modify the humans position, you need to properly update also the corresponding variable in `my_tiago_controller/src/my_tiago_controller/Hparams.py`.
When the node is shutdown, relevant data are logged in a `.json` file located in `/tmp/crowd_navigation_tiago/data` folder, with the name specified in `my_tiago_controller/src/my_tiago_controller/Hparams.py`

To set a desired TIAGo target position:
```bash
roslaunch my_tiago_controller send_desired_target_position.launch x_des:=X y_des:=Y
```
where `X` and `Y` are the coordinates of the desired goal position.

Once the simulation is completed (nodes are shutdown), you can plot animations of the results as follows:
```bash
roslaunch my_tiago_controller plotter.launch filename:=FILENAME
```
where `FILENAME` is the name of the saved `.json` file (WITHOUT extension). The two created animations are saved respectively in `/tmp/crowd_navigation_tiago/simulations` and `/tmp/crowd_navigation_tiago/plots` folders.

