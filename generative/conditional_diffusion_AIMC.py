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
from aihwkit.optim import AnalogSGD

# for inference
from aihwkit.simulator.presets import PCMPreset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# rpu_config = PCMPreset()
rpu_config = InferenceRPUConfig()
# rpu_config = FloatingPointRPUConfig()

t_inference = 3600.0  # 1h
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
state_dict = torch.load("diffusion_model.pth", map_location="cpu")
digital_model.load_state_dict(state_dict)

digital_model = digital_model.to(device)
analog_model = convert_to_analog(
    digital_model,
    rpu_config=rpu_config
).to(device)

training = True
sample = True

if training:
    analog_model.train()

    optimizer = AnalogSGD(
        analog_model.parameters(),
        lr=5e-5,
        momentum=0.9,
        weight_decay=1e-5
    )
    optimizer.regroup_param_groups(analog_model)

    transform = torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=2 if device.type == "cuda" else 0,
    )

    for epoch in range(10):
        print(f"Epoch: {epoch}")
        sum_of_loss = 0
        cnt = 0

        for x, y in tqdm(dataloader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            t = torch.randint(
                1, analog_model.T + 1, (x.shape[0],),
                device=device
            )

            optimizer.zero_grad()

            eps = torch.randn_like(x, device=device)
            bar_alpha_t = analog_model.bar_alpha_schedule[t - 1].view(-1, 1, 1, 1)
            xt = torch.sqrt(bar_alpha_t) * x + torch.sqrt(1 - bar_alpha_t) * eps

            if torch.rand(1, device=device).item() < 0.1:
                y = None

            hat_eps, hat_mu = analog_model(xt, t, y)

            loss = torch.nn.functional.mse_loss(hat_eps, eps)
            loss.backward()
            optimizer.step()

            sum_of_loss += loss.item()
            cnt += 1

        print(f"Total Loss: {sum_of_loss / cnt}")

    torch.save(analog_model.state_dict(), "hwa_diffusion_model.pth")

if sample:
    analog_model.eval()
    analog_model.program_analog_weights()
    analog_model.drift_analog_weights(t_inference)

    transform = torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    T = analog_model.T // 2
    size = dataset[0][0].shape[1]

    xt = torch.randn((20, 1, size, size), device=device)
    y = torch.arange(0, 10, device=device).repeat(2)

    with torch.no_grad():
        for step in tqdm(range(T)):
            t = T - step
            t_tensor = torch.full((20,), t, dtype=torch.long, device=device)

            cond_hat_eps, cond_hat_mu = analog_model(xt, t_tensor, y)
            uncond_hat_eps, uncond_hat_mu = analog_model(xt, t_tensor, None)

            bar_alpha_t = analog_model.bar_alpha_schedule[t - 1]
            alpha_t = analog_model.alpha_schedule[t - 1]

            cond_hat_score = -cond_hat_eps / torch.sqrt(1 - bar_alpha_t)
            uncond_hat_score = -uncond_hat_eps / torch.sqrt(1 - bar_alpha_t)
            cond_score = analog_model.gamma * (cond_hat_score - uncond_hat_score) + uncond_hat_score

            cond_eps = -cond_score * torch.sqrt(1 - bar_alpha_t)
            frac_term1 = (1 - alpha_t) / torch.sqrt(bar_alpha_t)
            frac_term2 = 1 / torch.sqrt(alpha_t)
            cond_mu = (xt - frac_term1 * cond_eps) * frac_term2

            eps = torch.randn_like(xt)

            if t == 1:
                xt = cond_mu
            else:
                q_variance = (1 - alpha_t) * (1 - analog_model.bar_alpha_schedule[t - 2]) / (1 - bar_alpha_t)
                xt = cond_mu + torch.sqrt(q_variance) * eps

    grid = torchvision.utils.make_grid(xt.reshape(-1, 1, 28, 28), nrow=4)
    torchvision.utils.save_image(grid, "aimc_images_no_hwa.png")
