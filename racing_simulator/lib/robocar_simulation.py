from mlagents_envs.environment import UnityEnvironment, ActionTuple
from simulation_parameters import RAY_COUNT
from .my_keyboard import is_key_pressed
from .robocar_env import env
from dataclasses import dataclass
from itertools import count
import numpy as np



@dataclass
class Observation:
    rays: list
    reward: float
    speed: float
    steering: float
    x: float
    y: float
    z: float


@dataclass
class Action:
    speed: float = 0
    steering: float = 0


class Simulation:
    def __init__(self, env: UnityEnvironment):
        self.env = env
        self.action_tuple = ActionTuple(np.array([[0., 0.]], dtype=np.float32))
        self.frame_index = 1
    
    def run(self, step_func, step_count = -1):
        self.env.reset()
        
        # infinite loop if step_count is -1
        for _ in count() if step_count == -1 else range(step_count):
            decision, _ = self.env.get_steps("Agent0?team=0")
            observation = Observation(
                rays = decision.obs[0][0][:RAY_COUNT].tolist(),
                reward = decision.reward[0],
                speed = decision.obs[0][0][RAY_COUNT],
                steering = decision.obs[0][0][RAY_COUNT + 1],
                x = decision.obs[0][0][RAY_COUNT + 2],
                y = decision.obs[0][0][RAY_COUNT + 3],
                z = decision.obs[0][0][RAY_COUNT + 4]
            )
            action = step_func(self.frame_index, observation)
            if is_key_pressed('esc'):
                self.env.close()
                exit(0)
            self.action_tuple.continuous[0][0] = action.speed
            self.action_tuple.continuous[0][1] = action.steering
            self.env.set_actions("Agent0?team=0", self.action_tuple)
            self.env.step()
            self.frame_index += 1
        self.env.close()
        exit(0)
    
    def exit(self):
        self.env.close()


simulation = Simulation(env)
