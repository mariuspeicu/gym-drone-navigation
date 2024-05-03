from gym.envs.registration import register

register(
    id="task_rl/DroneNavigation-v0",
    entry_point="task_rl.envs:DroneNavigation",
)
