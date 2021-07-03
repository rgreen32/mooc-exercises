#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import numpy as np
from aido_schemas import (
    Context,
    DB20Commands,
    DB20Observations,
    EpisodeStart,
    GetCommands,
    JPGImage,
    LEDSCommands,
    protocol_agent_DB20,
    PWMCommands,
    RGB,
    wrap_direct,
)

import duckietown_code_utils as dcu
from connections import get_motor_left_matrix, get_motor_right_matrix
from preprocessing import preprocess


@dataclass
class BraitenbergAgentConfig:
    gain: float = 0.1
    const: float = 0.1


class BraitenbergAgent:
    config = BraitenbergAgentConfig()

    left: Optional[np.ndarray]
    right: Optional[np.ndarray]
    rgb: Optional[np.ndarray]
    l_max: float
    r_max: float
    l_min: float
    r_min: float

    def init(self, context: Context):
        context.info("init()")
        self.rgb = None
        self.l_max = -math.inf
        self.r_max = -math.inf
        self.l_min = math.inf
        self.r_min = math.inf
        self.left = None
        self.right = None

    def on_received_seed(self, data: int):
        np.random.seed(data)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        context.info(f'Starting episode "{data.episode_name}".')

    def on_received_observations(self, context: Context, data: DB20Observations):
        camera: JPGImage = data.camera
        if self.rgb is None:
            context.info("received first observations")
        self.rgb = dcu.bgr_from_rgb(dcu.bgr_from_jpg(camera.jpg_data))

    def compute_commands(self, context: Context) -> Tuple[float, float]:
        """ Returns the commands (pwm_left, pwm_right) """
        # If we have not received any image, we don't move
        if self.rgb is None:
            return 0.0, 0.0

        if self.left is None:
            # if it is the first time, we initialize the structures
            shape = self.rgb.shape[0], self.rgb.shape[1]
            self.left = get_motor_left_matrix(shape)
            self.right = get_motor_right_matrix(shape)

        # let's take only the intensity of RGB
        P = preprocess(self.rgb)
        # now we just compute the activation of our sensors
        observed_left_intensity = float(np.sum(P * self.left))
        observed_right_intensity = float(np.sum(P * self.right))

        # These are big numbers -- we want to normalize them.
        # We normalize them using the history

        # first, we remember the high/low of these raw signals
        self.l_max = max(observed_left_intensity, self.l_max)
        self.r_max = max(observed_right_intensity, self.r_max)
        self.l_min = min(observed_left_intensity, self.l_min)
        self.r_min = min(observed_right_intensity, self.r_min)

        # now rescale from 0 to 1
        left_scale = rescale(observed_left_intensity, self.l_min, self.l_max)
        right_scale = rescale(observed_right_intensity, self.r_min, self.r_max)

        gain = self.config.gain
        const = self.config.const
        pwm_left = const + left_scale * gain
        pwm_right = const + right_scale * gain
        
        context.info(f"==const== {const}")
        context.info(f"==gain== {gain}")
        
        context.info("=====================Left=======================")
        context.info(f"==observed_left_intensity {observed_left_intensity}==")
        context.info(f"==left_scale== {left_scale}")
        context.info(f"==pwm_left== {pwm_left}")
        context.info("=====================Left=======================")
        
        context.info("=====================Right=======================")
        context.info(f"==observed_right_intensity== {observed_right_intensity}")
        context.info(f"==right_scale== {right_scale}")
        context.info(f"==pwm_right== {pwm_right}")
        context.info("=====================Right=======================")
        

        
        return pwm_left, pwm_right

    def on_received_get_commands(self, context: Context, data: GetCommands):
        pwm_left, pwm_right = self.compute_commands(context)

        col = RGB(0.0, 0.0, 1.0)
        col_left = RGB(pwm_left, pwm_left, 0.0)
        col_right = RGB(pwm_right, pwm_right, 0.0)
        led_commands = LEDSCommands(col, col_left, col_right, col_left, col_right)
        pwm_commands = PWMCommands(motor_left=pwm_left, motor_right=pwm_right)
        commands = DB20Commands(pwm_commands, led_commands)
        context.write("commands", commands)

    def finish(self, context: Context):
        context.info("finish()")


def rescale(observed_intensity: float, min_intensity: float, max_intensity: float):
    if np.allclose(min_intensity, max_intensity):
        return 0.0
    return (observed_intensity - min_intensity) / (max_intensity - min_intensity)


def main():
    node = BraitenbergAgent()
    protocol = protocol_agent_DB20
    wrap_direct(node=node, protocol=protocol)


if __name__ == "__main__":
    main()
