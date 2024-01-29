# Crowd Navigation with TIAGo
## Project on-track
Accomplished task: TIAGo robot navigation to a desired target position in an environment populated by `n_actors` moving actors (along linear paths).
2 modules implemented:
-   crowd prediction module:
    read the laser scan measurement and estimate their state (position and velocity) using `n_clusters` Kalman Filters (KFs), managed via as many Finite State Machines (FSMs). Then, propagate the estimated state over the whole control horizon assuming a constant velocity motion. Finally, send a message containing the predicted actors trajectory.
    Note: if `fake_sensing=True` the module does not consider the laser scan measurement but a custom ground truth trajectory of the actors. 

-   controller module:
    given a desired target position, at each iteration, read the crowd motion prediction message and the current robot configuration and run the NMPC. The extended kinematic model of the TIAGo robot is considered, with wheels angular accelerations as control inputs. The moving humans avoidance is realized introducing nonlinear constraints defined by discrete-time Control Barrier Functions (CBFs) in the NMPC model of the Acados solver. CBFs constraints are also adopted to ensure the permanence of the robot within a predefined region.

## Usage
To run the Gazebo simulation (if `simulation=True`):
```bash
roslaunch labrob_tiago_gazebo tiago_gazebo.launch public_sim:=true end_effector:=pal-gripper world:=WORLD
```
Note that as `WORLD` it is suggested to use the ones in the `worlds` directory of the repository, since they have been modified in order to create more challenging scenario (please copy them in the `labrob_gazebo_worlds/worlds` directory). Of course, depending on the chosen `WORLD`, modify also the  `n_actors` variable specified in `my_tiago_controller/src/my_tiago_controller/Hparams.py`.
Note if `fake_sensing=True` the selected `WORLD` is not relevant, hence running Gazebo with an empty world is suggested (do not specify any `world`).

If `simulation=False`, before running the controller module, make sure the real robot is within the admissible region defined by `vertexes` in `my_tiago_controller/src/my_tiago_controller/Hparams.py`.

To run the NMPC controller module:
```bash
roslaunch my_tiago_controller nmpc_controller.launch
```
When the node is shutdown, relevant data are logged in a `.json` file located in `/tmp/crowd_navigation_tiago/data` folder, with the name specified in `my_tiago_controller/src/my_tiago_controller/Hparams.py` for the variable `controller_file`.

To run the Crowd Prediction module:
```bash
roslaunch my_tiago_controller crowd_prediction.launch
```

It is also possible to run the two modules simultaneously:
```bash
roslaunch my_tiago_controller crowd_navigation.launch
```

If `fake_sensing=True`, to generate and send the actors trajectory:
```bash
roslaunch my_tiago_controller send_actors_trajectory.launch
```

To set a desired TIAGo target position:
```bash
roslaunch my_tiago_controller send_desired_target_position.launch x_des:=X y_des:=Y
```
where `X` and `Y` are the coordinates of the desired goal position.

Note: the desired target position can be changed or set multiple times, running the corresponding launch file with different coordinates.

Once the simulation is completed (nodes are shutdown), you can plot animations of the results as follows:
```bash
roslaunch my_tiago_controller plotter.launch filename:=FILENAME
```
where `FILENAME` is the name of the saved `.json` file (WITHOUT extension). A plot of the quantities profile and an animation are saved respectively in `/tmp/crowd_navigation_tiago/plots` and `/tmp/crowd_navigation_tiago/simulations` folders.