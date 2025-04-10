import torch
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm

class conv_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_size=100):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.batchnorm1 = torch.nn.BatchNorm2d(out_channels)
        self.batchnorm2 = torch.nn.BatchNorm2d(out_channels)
        self.in_channels = in_channels

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(time_embedding_size, in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(in_channels, in_channels)
        )

    def forward(self, x, t):
        t = self.mlp(t).view(-1, self.in_channels, 1, 1)
        x = self.conv1(x + t)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)

        return x

def positional_encoding(ts, embedding_size):
    i = torch.arange(embedding_size)
    div = 10000.0 ** (i / embedding_size)

    vs = torch.zeros(ts.shape[0], embedding_size)

    for i, t in enumerate(ts):
        v = torch.zeros(embedding_size)
        v[0::2] = torch.sin(t/div[0::2] )
        v[1::2] = torch.cos(t/div[1::2] )
        vs[i] = v

    return vs

class UNet(torch.nn.Module):
    def __init__(self, channels, time_embedding_size=100):
        super().__init__()
        self.channels = channels
        self.time_embedding_size = time_embedding_size

        self.conv1 = conv_block(channels, 64, time_embedding_size)
        self.conv2 = conv_block(64, 128, time_embedding_size)
        self.conv3 = conv_block(128, 256, time_embedding_size)

        self.conv4 = conv_block(128+256, 128, time_embedding_size)
        self.conv5 = conv_block(128+64, 64, time_embedding_size)

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv6 = torch.nn.Conv2d(64, 1, kernel_size=1, stride=1)


    def forward(self, xt, t):
        positional_embedding = positional_encoding(t, self.time_embedding_size)

        x1 = self.conv1(xt, positional_embedding)
        x = self.maxpool(x1)
        x2 = self.conv2(x, positional_embedding)
        x = self.maxpool(x2)
        x = self.conv3(x, positional_embedding)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv4(x, positional_embedding)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv5(x, positional_embedding)
        x = self.conv6(x)

        return x

class Diffusion(torch.nn.Module):
    def __init__(self, beta_start=0.0001, beta_end=0.02, T=1000):
        super().__init__()
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.T = T
        beta_schedule = torch.linspace(beta_start, beta_end, T)
        alpha_schedule = 1 - beta_schedule
        self.alpha_schedule = alpha_schedule.view(-1, 1, 1, 1)
        self.bar_alpha_schedule = torch.cumprod(alpha_schedule, 0).view(-1, 1, 1, 1)
        self.model = UNet(1)

    def forward(self, xt, t):
        hat_eps = self.model(xt, t)
        frac_term1 = (1-self.alpha_schedule[t-1])/(torch.sqrt(1-self.bar_alpha_schedule[t-1]))
        frac_term2 = 1/torch.sqrt(self.alpha_schedule[t-1])
        hat_mu = (xt - frac_term1 * hat_eps)*frac_term2

        return hat_eps, hat_mu

def tensor_to_images(tensor):
    tensor *= 255
    tensor = tensor.clamp(0, 255)
    tensor = tensor.to(torch.uint8)
    return torchvision.transforms.ToPILImage()(tensor)

if __name__ == '__main__':

    transform = torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    #img = tensor_to_images(dataset[0][0]) # 1, 28, 28

    diffusion = Diffusion()

    training = False
    sample = True

    if training == True:
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=0.001)

        for epoch in range(10):
            print(f'Epoch: {epoch}')
            sum_of_loss = 0
            cnt = 0
            for x, _ in tqdm(dataloader):
                t = torch.randint(1, diffusion.T+1, (x.shape[0],))
                optimizer.zero_grad()
                eps = torch.randn_like(x)
                xt = torch.sqrt(diffusion.bar_alpha_schedule[t-1])*x + torch.sqrt(1-diffusion.bar_alpha_schedule[t-1])*eps
                hat_eps, hat_mu = diffusion(xt, t)

                loss = torch.nn.functional.mse_loss(hat_eps, eps)
                loss.backward()
                optimizer.step()
                sum_of_loss += loss.item()
                cnt += 1

            print(f'Total Loss: {sum_of_loss/cnt}')

        torch.save(diffusion.state_dict(), 'diffusion_model.pth')

    if sample == True:
        diffusion.load_state_dict(torch.load('diffusion_model.pth'))
        diffusion.eval()
        size = dataset[0][0].shape[1]
        xt = torch.randn((20, 1, size, size))
        with torch.no_grad():
            for t in tqdm(range(diffusion.T)):
                t = diffusion.T - t
                hat_eps, hat_mu = diffusion(xt, (torch.ones((20))*t).int())
                eps = torch.randn_like(xt)
                if t == 1:
                    xt = hat_mu
                else:
                    q_variance = (1-diffusion.alpha_schedule[t-1])*(1-diffusion.bar_alpha_schedule[t-2])/(1-diffusion.bar_alpha_schedule[t-1])
                    xt = hat_mu + torch.sqrt(q_variance)*eps

        grid = torchvision.utils.make_grid(xt.reshape(-1, 1, 28, 28), nrow=4)
        torchvision.utils.save_image(grid, 'images.png')
