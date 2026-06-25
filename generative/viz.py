import torch
import conditional_diffusion

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.nn.modules.base import AnalogLayerBase
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.simulator.configs import InferenceRPUConfig

rpu_config = InferenceRPUConfig()
rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
rpu_config.drift_compensation = GlobalDriftCompensation()
rpu_config.pre_post.input_range.enable = True  # alpha
rpu_config.pre_post.input_range.learn_input_range = True
rpu_config.forward.inp_res = 1 / (2**8 - 2)
rpu_config.forward.out_res = 1 / (2**8 - 2)
rpu_config.forward.out_bound = 10  # ADC saturation
rpu_config.forward.out_noise = 0.04  # N(0, 0.04^2)
rpu_config.forward.w_noise = 0.0175  # N(0, 0.0175^2)
rpu_config.forward.ir_drop = 1.0
rpu_config.forward.ir_drop_g_ratio = 571428.57
rpu_config.mapping.digital_bias = True
rpu_config.mapping.out_scale_columnwise = True  # gamma_i
rpu_config.mapping.max_input_size = 512
rpu_config.mapping.max_output_size = 512

digital_model = conditional_diffusion.Diffusion()
obj = torch.load("hwa_diffusion_model.pth", map_location="cpu", weights_only=False)
state_dict = torch.load("diffusion_model.pth", map_location="cpu")
digital_model.load_state_dict(state_dict)
analog_model = convert_to_analog(
    digital_model,
    rpu_config=rpu_config
)

print(analog_model)
