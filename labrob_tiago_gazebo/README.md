# labrob_tiago_gazebo
## Usage
To run the Gazebo simulation:
```bash
roslaunch labrob_tiago_gazebo tiago_gazebo.launch public_sim:=true end_effector:=pal-gripper world:=WORLD
```
where `WORLD` is one of the worlds in one of the packages `labrob_gazebo_worlds`
or `pal_gazebo_worlds`.
