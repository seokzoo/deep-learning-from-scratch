from tqdm import tqdm
import torch
import torchvision
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.nn.modules.base import AnalogLayerBase
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel

import conditional_diffusion

# for training
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.inference import PCMLikeNoiseModel
from aihwkit.optim import AnalogSGD

# for inference
from aihwkit.simulator.presets import PCMPreset

#rpu_config = PCMPreset()
rpu_config = InferenceRPUConfig()
#rpu_config = FloatingPointRPUConfig()

t_inference = 3600.0 # 1h
rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
rpu_config.drift_compensation = GlobalDriftCompensation()
rpu_config.pre_post.input_range.enable = True # alpha
rpu_config.pre_post.input_range.learn_input_range = True
rpu_config.forward.inp_res = 1 / (2**8 - 2)
rpu_config.forward.out_res = 1 / (2**8 - 2)
rpu_config.forward.out_bound = 10 # ADC saturation
rpu_config.forward.out_noise = 0.04 # N(0, 0.04^2)
rpu_config.forward.w_noise = 0.0175 # N(0, 0.0175^2)
rpu_config.forward.ir_drop = 1.0
rpu_config.forward.ir_drop_g_ratio = 571428.57
rpu_config.mapping.digital_bias = True
rpu_config.mapping.out_scale_columnwise = True # gamma_i
rpu_config.mapping.max_input_size = 512
rpu_config.mapping.max_output_size = 512

load_digital = False
training = False
sample = True

if load_digital:
    digital_model = conditional_diffusion.Diffusion()
    state_dict = torch.load("diffusion_model.pth", map_location="cpu")
    digital_model.load_state_dict(state_dict)

    analog_model = convert_to_analog(
        digital_model,
        rpu_config=rpu_config
    )
else:
    digital_model = conditional_diffusion.Diffusion()
    state_dict = torch.load("hwa_diffusion_model.pth", map_location="cpu", weights_only=False)
    analog_model = convert_to_analog(
        digital_model,
        rpu_config=rpu_config
    )
    analog_model.load_state_dict(state_dict)

if training == True:
    analog_model.train()
    analog_model = analog_model

    analog_model.bar_alpha_schedule = (
        analog_model.bar_alpha_schedule
    )

    analog_model.alpha_schedule = (
        analog_model.alpha_schedule
    )

    optimizer = torch.optim.SGD(
            analog_model.parameters(),
            lr=10**-5,
            momentum=0.9,
            weight_decay=10**-5
            )
    transform = torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(5):
        print(f'Epoch: {epoch}')
        sum_of_loss = 0
        cnt = 0
        for x, y in tqdm(dataloader):
            x = x
            y = y
            t = torch.randint(1, analog_model.T+1, (x.shape[0],))
            optimizer.zero_grad()
            eps = torch.randn_like(x)
            xt = torch.sqrt(analog_model.bar_alpha_schedule[t-1])*x + torch.sqrt(1-analog_model.bar_alpha_schedule[t-1])*eps
            if torch.rand(1).item() < 0.1:
                y = None
            hat_eps, hat_mu = analog_model(xt, t, y)

            loss = torch.nn.functional.mse_loss(hat_eps, eps)
            loss.backward()
            optimizer.step()
            sum_of_loss += loss.item()
            cnt += 1

        print(f'Total Loss: {sum_of_loss/cnt}')

    torch.save(analog_model.state_dict(), 'hwa_diffusion_model.pth')

if sample == True:
    analog_model.eval()
    analog_model.program_analog_weights()
    analog_model.drift_analog_weights(t_inference)
    transform = torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    '''
    for name, module in analog_model.named_modules():
        if isinstance(module, AnalogLayerBase):
            print(name, len(list(module.analog_tiles())))
    '''

    T = analog_model.T // 2
    size = dataset[0][0].shape[1]
    xt = torch.randn((20, 1, size, size))
    y = torch.arange(0, 10).repeat(2)
    with torch.no_grad():
        for t in tqdm(range(T)):
            t = T - t
            t_tensor = torch.full((20,), t, dtype=torch.int)
            cond_hat_eps, cond_hat_mu = analog_model(xt, t_tensor, y)
            uncond_hat_eps, uncond_hat_mu = analog_model(xt, t_tensor, None)

            cond_hat_score = -cond_hat_eps / torch.sqrt(1-analog_model.bar_alpha_schedule[t-1])
            uncond_hat_score = -uncond_hat_eps / torch.sqrt(1-analog_model.bar_alpha_schedule[t-1])
            cond_score = analog_model.gamma * (cond_hat_score - uncond_hat_score) + uncond_hat_score
            cond_eps = -cond_score * torch.sqrt(1-analog_model.bar_alpha_schedule[t-1])
            frac_term1 = (1-analog_model.alpha_schedule[t-1])/(torch.sqrt(1-analog_model.bar_alpha_schedule[t-1]))
            frac_term2 = 1/torch.sqrt(analog_model.alpha_schedule[t-1])
            cond_mu = (xt - frac_term1 * cond_eps)*frac_term2
            eps = torch.randn_like(xt)
            if t == 1:
                xt = cond_mu
            else:
                q_variance = (1-analog_model.alpha_schedule[t-1])*(1-analog_model.bar_alpha_schedule[t-2])/(1-analog_model.bar_alpha_schedule[t-1])
                xt = cond_mu + torch.sqrt(q_variance)*eps

    grid = torchvision.utils.make_grid(xt.reshape(-1, 1, 28, 28), nrow=4)
    torchvision.utils.save_image(grid, 'aimc_images_no_hwa.png')
