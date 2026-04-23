from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.environment import UnityEnvironment
import os



SCREEN_WIDTH = 700 #1920
SCREEN_HEGHT = 500 #1080
TIME_SCALE = 1
GRAPHIC_MODE = True



if not os.path.exists("RacingSimulatorLinux/BuildLinux/RacingSimulator.x86_64"):
    print(f"{os.path.abspath('.')}/RacingSimulatorLinux/BuildLinux/RacingSimulator.x86_64 does not exist")
    print("Maybe you don't have the simulator")
    print(f"Put it in {os.path.abspath('.')}")
    exit(1)

channel = EngineConfigurationChannel()

env = UnityEnvironment(
    file_name="RacingSimulatorLinux/BuildLinux/RacingSimulator.x86_64",
    base_port=5004,
    additional_args = ["-batch-mode", "--config-path", "config.json"],
    no_graphics_monitor= not GRAPHIC_MODE,
    no_graphics= not GRAPHIC_MODE,
    log_folder=".",
    timeout_wait=10,
    side_channels=[channel]
)

channel.set_configuration_parameters(width=SCREEN_WIDTH, height=SCREEN_HEGHT, time_scale=TIME_SCALE)
