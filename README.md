# Crowd Navigation with TIAGo
## Project on-track
Accomplished ask: TIAGo robot navigation to a desired target position in an environment populated by `n_actors` moving actors (along linear paths). 
2 modules implemented:
-   crowd prediction module:
    assuming to know the whole actors trajectory and the data association, consider iteratively the current actors position and estimate their next state (position and velocity) using `n_actors` Kalman Filters (KFs), managed via as many Finite State Machines (FSMs). Then, propagate the estimated state over the whole control horizon assuming a constant velocity motion. Finally, send a message containing the predicted actors trajectory.

-   controller module:
    given a desired target position, at each iteration, read the crowd motion prediction message and the current robot configuration and run the NMPC. The kinematic model of the TIAGo robot is considered, with wheels angular velcoities as control inputs. The moving humans avoidance is realized introducing nonlinear constraints defined by CBFs in the NMPC model of the Acados solver. CBFs constraints are also adopted to ensure the permanence of the robot within a predefined region.
## Next goal
Implemenation of the sensing module:

-   Read the data from the Laser Scan and implement a data association

## Usage
To run the Gazebo simulation:
```bash
roslaunch labrob_tiago_gazebo tiago_gazebo.launch public_sim:=true end_effector:=pal-gripper
```
Note: the TIAGo robot is spawn in an empty world. Hence, you cannot see the moving actors in the Gazebo simulation since it would not be straightforward to generate a motion corresponding to the synthetically generated trajectory. In order to read from the laser scan (i.e., implementing the next goal), the first step is to delete synthetic trajectories and consider moving actors in Gazebo simulation. 

To run the NMPC controller module:
```bash
roslaunch my_tiago_controller nmpc_controller.launch
```
When the node is shutdown, relevant data are logged in a `.json` file located in `/tmp/crowd_navigation_tiago/data` folder, with the name specified in `my_tiago_controller/src/my_tiago_controller/Hparams.py` for the variable `controller_file`

To run the Crowd Prediction module:
```bash
roslaunch my_tiago_controller crowd_prediction.launch
```

To generate and send the actors trajectory:
```bash
roslaunch my_tiago_controller send_actors_trajectory.launch
```

To set a desired TIAGo target position:
```bash
roslaunch my_tiago_controller send_desired_target_position.launch x_des:=X y_des:=Y
```
where `X` and `Y` are the coordinates of the desired goal position.

Note: the desired target position can be changed or set multiple times, running the corresponding launch file with different coordinates. Similarly, the actor motion can also be triggered multiple times by running the corresponding launch file. However, if actors are already moving because of previously sent trajectories, the newest ones are not accepted.

Once the simulation is completed (nodes are shutdown), you can plot animations of the results as follows:
```bash
roslaunch my_tiago_controller plotter.launch filename:=FILENAME
```
where `FILENAME` is the name of the saved `.json` file (WITHOUT extension). A plot of the quantities profile and an animation are saved respectively in `/tmp/crowd_navigation_tiago/plots` and `/tmp/crowd_navigation_tiago/simulations` folders.

