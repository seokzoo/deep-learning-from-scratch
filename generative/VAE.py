import torch
import torchvision

class Encoder(torch.nn.Module):
    def __init__(self, input_dim=784, hidden_dim=200, output_dim=20):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.to_log_squared_sigma = torch.nn.Linear(hidden_dim, output_dim)
        self.to_mu = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.sigmoid(self.fc1(x))
        log_squared_sigma = self.to_log_squared_sigma(h)
        mu = self.to_mu(h)
        sigma = torch.exp(0.5 * log_squared_sigma)

        return mu, sigma

class Decoder(torch.nn.Module):
    def __init__(self, input_dim=20, hidden_dim=200, output_dim=784):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        x_hat = torch.sigmoid(self.fc3(z))

        return x_hat

class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim=784, hidden_dim=200, output_dim=20):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, output_dim)
        self.decoder = Decoder(output_dim, hidden_dim, input_dim)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = mu + sigma * torch.randn_like(sigma)
        x_hat = self.decoder(z)

        return x_hat, mu, sigma

if __name__ == '__main__':
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.reshape(-1))
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(100):
        loss_sum = 0
        count = 0
        for i, batch in enumerate(torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)):
            images, labels = batch
            x_hat, mu, sigma = model(images)
            loss = ((torch.norm(images - x_hat, p=2, dim=1)**2) - (torch.log(sigma**2) - mu**2 - sigma**2).sum(dim=1)).mean(dim=0)
            loss_sum += loss.item()
            count += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'loss : {loss_sum / count}')

    with torch.no_grad():
        z = torch.randn(64, 20)
        x_hat = model.decoder(z)
        grid = torchvision.utils.make_grid(x_hat.reshape(-1, 1, 28, 28), nrow=4)
        torchvision.utils.save_image(grid, 'images.png')
