import torch
import torchvision
from tqdm import tqdm

class Encoder(torch.nn.Module):
    def __init__(self, input_dim=784, hidden_dim=100, latent_dim=20, embbed_dim=100):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, latent_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, latent_dim)
        self.fc = torch.nn.Linear(embbed_dim, input_dim)

    def forward(self, x, y):
        y = torch.relu(self.fc(y))
        h = torch.relu(self.fc1(x + y))
        mu = self.fc2(h)
        sigma = torch.exp(0.5 * self.fc3(h))
        return mu, sigma

class Decoder(torch.nn.Module):
    def __init__(self, input_dim=784, hidden_dim=100, latent_dim=20, embbed_dim=100):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, latent_dim)
        self.fc = torch.nn.Linear(embbed_dim, input_dim)

    def forward(self, x, y):
        y = torch.relu(self.fc(y))
        h = torch.relu(self.fc1(x + y))
        x_hat = torch.sigmoid(self.fc2(h))
        return x_hat

class VAE(torch.nn.Module):
    def __init__(self, input_dim=784, hidden_dim=100, latent_dim=20, embbed_dim=100):
        super().__init__()
        self.label_encoder = torch.nn.Embedding(10, embbed_dim)
        self.x_to_z1 = Encoder(input_dim, hidden_dim, latent_dim)
        self.z1_to_z2 = Encoder(latent_dim, hidden_dim, latent_dim)
        self.z2_to_z1 = Decoder(latent_dim, hidden_dim, latent_dim)
        self.z1_to_x = Decoder(latent_dim, hidden_dim, input_dim)

    def encoder(self, x, y=None):
        mu1, sigma1 = self.x_to_z1(x, y)
        z1 = mu1 + sigma1 * torch.randn_like(mu1)
        mu2, sigma2 = self.z1_to_z2(z1, y)
        z2 = mu2 + sigma2 * torch.randn_like(mu2)

        return z1, mu1, sigma1, z2, mu2, sigma2

    def decoder(self, z2, y=None):
        z1_hat = self.z2_to_z1(z2, y)
        z1 = z1_hat + torch.randn_like(z1_hat)
        x = self.z1_to_x(z1, y)

        return x, z1, z1_hat

    def forward(self, x, y=None):
        if y is not None:
            y = self.label_encoder(y)

        z1, mu1, sigma1, z2, mu2, sigma2 = self.encoder(x, y)
        z1_hat = self.z2_to_z1(z2, y)
        x_hat = self.z1_to_x(z1, y)

        L1 = ((x-x_hat)**2).sum(dim=1)
        L2 = -(1+torch.log(sigma2 ** 2) - mu2 ** 2 - sigma2 ** 2).sum(dim=1)
        L3 = -(1+torch.log(sigma1 ** 2) - (mu1-z1_hat) ** 2 - sigma1 ** 2).sum(dim=1)

        return (L1 + L2 + L3).mean()


if __name__ == "__main__":
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.view(-1))
    ])
    mnist = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transforms)
    dataloader = torch.utils.data.DataLoader(mnist, batch_size=64, shuffle=True)
    model = VAE(input_dim=784, hidden_dim=100, latent_dim=20)
    optim = torch.optim.Adam(model.parameters(), lr=0.003)

    train = True
    eval = True

    if train:
        for epoch in tqdm(range(40)):
            total_loss = 0
            cnt = 0
            for x, y in dataloader: # 128, 1, 28, 28
                optim.zero_grad()
                loss = model(x, y)
                total_loss += loss.item()
                cnt += 1
                loss.backward()
                optim.step()
            print(f"Epoch {epoch+1}: Loss = {total_loss/cnt}")

        torch.save(model.state_dict(), 'vae_model.pth')

    if eval:
        model.load_state_dict(torch.load('vae_model.pth'))
        model.eval()
        with torch.no_grad():
            z2 = torch.randn(20, 20)
            y = torch.arange(10).repeat(2)
            y = model.label_encoder(y)
            x, z1, z1_hat = model.decoder(z2, y)

        grid = torchvision.utils.make_grid(x.reshape(-1, 1, 28, 28), nrow=4)
        torchvision.utils.save_image(grid, 'images.png')
