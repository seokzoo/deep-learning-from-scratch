import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class RMSNorm(torch.nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(latent_dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return norm * x * self.scale

class VisionTransformer(torch.nn.Module):
    def __init__(self, embed=True, num_layers=2, num_heads=3, patch_length=4, context_length=7, input_dim=128, latent_dim=128, output_dim=128, base=1000.):
        super().__init__()
        self.embed = embed
        self.num_layers = num_layers
        self.patch_size = patch_length ** 2
        self.layers = torch.nn.ModuleList([TransformerBlock(num_heads, context_length, input_dim, latent_dim, output_dim, base) for _ in range(num_layers)])
        self.embed_layer = torch.nn.Linear(self.patch_size, latent_dim)

    def forward(self, x, indices=None):
        if self.embed:
            x = self.embed_layer(x)
        for i in range(self.num_layers):
            x = self.layers[i](x, indices)

        return x

class TransformerBlock(torch.nn.Module):
    def __init__(self, num_heads=3, context_length=7, input_dim=128, latent_dim=128, output_dim=128, base=1000.):
        super().__init__()
        self.num_heads = num_heads
        self.context_length = context_length
        self.latent_dim = latent_dim
        self.q = torch.nn.Linear(input_dim, latent_dim*num_heads)
        self.k = torch.nn.Linear(input_dim, latent_dim*num_heads)
        self.v = torch.nn.Linear(input_dim, latent_dim*num_heads)
        self.register_buffer('r', self.get_2d_rotary_matrix(base))
        self.rms_norm1 = RMSNorm(latent_dim)
        self.rms_norm2 = RMSNorm(latent_dim)
        self.heads_to_latent = torch.nn.Linear(latent_dim*num_heads, latent_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.GELU(),
            torch.nn.Linear(latent_dim, output_dim)
        )

    def get_2d_rotary_matrix(self, base=1000.):
        half_latent_dim = self.latent_dim // 2
        quarter_latent_dim = self.latent_dim // 4
        rotary_mat = torch.zeros(self.context_length, self.context_length, self.latent_dim, self.latent_dim)
        latent_mat = torch.zeros(self.context_length, half_latent_dim, half_latent_dim)
        theta = torch.einsum(
                'i,j->ij',
                torch.arange(self.context_length), 
                ((base) ** (-2 * torch.arange(quarter_latent_dim) / half_latent_dim))
        )
        
        pos = torch.arange(half_latent_dim)
        latent_mat[:,pos,pos] = torch.cos(theta).repeat_interleave(2, dim=-1)
        latent_mat[:,pos[::2],pos[1::2]] = -torch.sin(theta)
        latent_mat[:,pos[1::2],pos[::2]] = torch.sin(theta)

        rotary_mat[:,:,half_latent_dim:,half_latent_dim:] = latent_mat.unsqueeze(1)
        rotary_mat[:,:,:half_latent_dim,:half_latent_dim] = latent_mat.unsqueeze(0)

        return rotary_mat.reshape(-1, self.latent_dim, self.latent_dim)

    def forward(self, x, indices=None):
        batch_size = x.shape[0]
        context_length = x.shape[1]
        norm_x = self.rms_norm1(x)

        q = self.q(norm_x).reshape(batch_size, context_length, self.num_heads, self.latent_dim).transpose(1, 2)
        k = self.k(norm_x).reshape(batch_size, context_length, self.num_heads, self.latent_dim).transpose(1, 2)
        v = self.v(norm_x).reshape(batch_size, context_length, self.num_heads, self.latent_dim).transpose(1, 2)

        if indices is not None: # indicies = j * context_length + i
            rotated_q = torch.einsum('bhcl,clk->bhck', q, self.r[indices,:,:])
            rotated_k = torch.einsum('bhcl,clk->bhck', k, self.r[indices,:,:])
        else:
            rotated_q = torch.einsum('bhcl,clk->bhck', q, self.r)
            rotated_k = torch.einsum('bhcl,clk->bhck', k, self.r)

        attention_score = torch.einsum('bhql,bhlk->bhqk', rotated_q, rotated_k.transpose(-1,-2)) * (self.latent_dim**-0.5)
        attention_prob = torch.nn.functional.softmax(attention_score, dim=-1)
        attention = torch.einsum('bhdc,bhcl->bhdl', attention_prob, v).transpose(1, 2).reshape(batch_size, context_length, -1)
        attention = self.heads_to_latent(attention)

        attention = x + attention

        output = attention + self.ffn(self.rms_norm2(attention))

        return output

class JEPA(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context_length = config['img_size'] // config['patch_length']
        self.context_encoder = VisionTransformer(
                embed=True,
                num_layers=config['num_vit_layers'],
                num_heads=config['num_vit_heads'],
                patch_length=config['patch_length'],
                context_length=self.context_length,
                input_dim=config['latent_dim'],
                latent_dim=config['latent_dim'],
                output_dim=config['latent_dim'],
                base=1000.
                )
        self.target_encoder = VisionTransformer(
                embed=True,
                num_layers=config['num_vit_layers'],
                num_heads=config['num_vit_heads'],
                patch_length=config['patch_length'],
                context_length=self.context_length,
                input_dim=config['latent_dim'],
                latent_dim=config['latent_dim'],
                output_dim=config['latent_dim'],
                base=1000.
                )
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        self.target_encoder.eval()
        self.target_encoder.requires_grad_(requires_grad=False)
        self.predictor = VisionTransformer(
                embed=False,
                num_layers=config['num_vit_layers'],
                num_heads=config['num_vit_heads'],
                patch_length=config['patch_length'],
                context_length=self.context_length,
                input_dim=config['latent_dim'],
                latent_dim=config['latent_dim'],
                output_dim=config['latent_dim'],
                base=1000.
                )
        self.mask_token = torch.nn.Parameter(torch.empty(config['latent_dim']))
        torch.nn.init.normal_(self.mask_token, std=0.2)

    def forward(self, x):
        pass

def main(config):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor()
    ])
    training_data = torchvision.datasets.CIFAR100(root="data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR100(root="data", train=False, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=config['train_batch_size'], shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=config['test_batch_size'], shuffle=True)

    def get_targets(num_targets, context_length):
        r = np.random.uniform(0.75, 1.5, num_targets)
        s = np.random.uniform(0.15, 0.2, num_targets)

        h = np.round(np.sqrt(s * r) * context_length)
        w = np.round(np.sqrt(s / r) * context_length)

        h_left = np.random.randint(0, context_length - h + 1)
        w_left = np.random.randint(0, context_length - w + 1)

        h_list = np.array([])
        w_list = np.array([])

        for i in range(num_targets):
            hh = np.arange(h[i]) + h_left[i]
            ww = np.arange(w[i]) + w_left[i]
            hv, wv = np.meshgrid(hh, ww)
            h_list = np.concat((h_list, hv.flatten()), axis=-1)
            w_list = np.concat((w_list, wv.flatten()), axis=-1)

        coords = np.unique(np.vstack([h_list, w_list]), axis=1)

        return np.vsplit(coords, 2)

    def get_context(context_length):
        s = np.random.uniform(0.75, 1.0)
        x = np.round(np.sqrt(s) * context_length)

        hh = np.random.randint(0, context_length - x + 1) + np.arange(x)
        ww = np.random.randint(0, context_length - x + 1) + np.arange(x)
        hv, wv = np.meshgrid(hh, ww)
        hv = hv.flatten()
        wv = wv.flatten()
        
        return hv, wv

    jepa = JEPA(config)

    def evaluate():
        test_losses = []
        jepa.eval()
        step = 0
        for imgs, labels in tqdm(test_dataloader):
            y_list, x_list = get_targets(config['num_targets'], jepa.context_length)
            patches = torch.nn.functional.unfold(imgs, kernel_size=config['patch_length'], stride=config['patch_length']).transpose(-1,-2)
            targets = torch.from_numpy((y_list * jepa.context_length + x_list).reshape(-1)).long()
            with torch.no_grad():
                encoded_patches = jepa.target_encoder(patches)
                y = encoded_patches[:,targets,...]

            y_list, x_list = get_context(jepa.context_length)
            context = (y_list * jepa.context_length + x_list).reshape(-1)

            context = torch.from_numpy(np.setdiff1d(context, targets)).long()
            encoded_context = jepa.context_encoder(patches[:,context,...], context)

            batch_size = imgs.shape[0]
            merged_patches = torch.cat((jepa.mask_token.repeat(batch_size, len(targets), 1), encoded_context), dim=1)
            merged_indices = torch.concat((targets, context), dim=-1)
            predicted = jepa.predictor(merged_patches, merged_indices)
            y_hat = predicted[:,:len(targets),:]

            loss = torch.nn.functional.mse_loss(y, y_hat)
            if step % 20 == 0:
                test_losses.append(loss.item())
            step += 1
        jepa.train()
        return np.mean(test_losses)

    def train():
        optim = torch.optim.Adam(jepa.parameters(), lr=config['learning_rate'])
        train_losses = []
        test_losses = []

        step = 0
        for epoch in range(config['epoch']):
            for imgs, labels in tqdm(train_dataloader):
                optim.zero_grad()
                y_list, x_list = get_targets(config['num_targets'], jepa.context_length)
                patches = torch.nn.functional.unfold(imgs, kernel_size=config['patch_length'], stride=config['patch_length']).transpose(-1,-2)
                targets = torch.from_numpy((y_list * jepa.context_length + x_list).reshape(-1)).long()
                with torch.no_grad():
                    encoded_patches = jepa.target_encoder(patches)
                    y = encoded_patches[:,targets,...]

                y_list, x_list = get_context(jepa.context_length)
                context = (y_list * jepa.context_length + x_list).reshape(-1)

                context = torch.from_numpy(np.setdiff1d(context, targets)).long()
                encoded_context = jepa.context_encoder(patches[:,context,...], context)

                batch_size = imgs.shape[0]
                merged_patches = torch.cat((jepa.mask_token.repeat(batch_size, len(targets), 1), encoded_context), dim=1)
                merged_indices = torch.concat((targets, context), dim=-1)
                predicted = jepa.predictor(merged_patches, merged_indices)
                y_hat = predicted[:,:len(targets),:]

                loss = torch.nn.functional.smooth_l1_loss(y, y_hat)
                    
                loss.backward()
                optim.step()

                if step % 20 == 0:
                    train_losses.append((step, loss.item()))

                if step % 100 == 0:
                    test_losses.append((step, evaluate()))
                step += 1

                with torch.no_grad():
                    alpha = config['ema_alpha']
                    for t, s in zip(jepa.target_encoder.parameters(), jepa.context_encoder.parameters()):
                        t.data.mul_(alpha).add_((1 - alpha) * s.data)

        torch.save(jepa.state_dict(), 'jepa.pth')

        train_steps, train_losses = zip(*train_losses)
        test_steps, test_losses = zip(*test_losses)

        plt.plot(train_steps, train_losses, label='Train Loss', color='blue', marker='o')
        plt.plot(test_steps, test_losses, label='Test Loss', color='red', marker='s')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which='both', ls='--', alpha=.3)
        plt.show()


    def test():
        jepa.load_state_dict(torch.load('jepa.pth'))
        jepa.eval()
        imgs, labels = test_dataloader.__iter__().__next__()

        y_list, x_list = get_targets(config['num_targets'], jepa.context_length)
        patches = torch.nn.functional.unfold(imgs, kernel_size=config['patch_length'], stride=config['patch_length']).transpose(-1,-2)
        targets = torch.from_numpy((y_list * jepa.context_length + x_list).reshape(-1)).long()
        encoded_patches = jepa.target_encoder(patches)
        y = encoded_patches[:,targets,...]

        y_list, x_list = get_context(jepa.context_length)
        context = (y_list * jepa.context_length + x_list).reshape(-1)

        context = torch.from_numpy(np.setdiff1d(context, targets)).long()
        encoded_context = jepa.context_encoder(patches[:,context,...], context)

        batch_size = imgs.shape[0]
        merged_patches = torch.cat((jepa.mask_token.repeat(batch_size, len(targets), 1), encoded_context), dim=1)
        merged_indices = torch.concat((targets, context), dim=-1)
        predicted = jepa.predictor(merged_patches, merged_indices)
        y_hat = predicted[:,:len(targets),:]

        invisible = np.setdiff1d(np.arange(64), context.numpy())
        cloned = patches.clone() 
        cloned[:,invisible,:] = 0
        folded = torch.nn.functional.fold(cloned.transpose(-1,-2), kernel_size=config['patch_length'], stride=config['patch_length'], output_size=(32, 32)).squeeze()

        _, ax = plt.subplots(6, 4)
        for i in range(6):
            ax[i, 0].imshow(imgs[i].squeeze())
            ax[i, 1].imshow(folded[i].squeeze())
            ax[i, 2].imshow(y[i].detach().numpy().squeeze())
            ax[i, 3].imshow(y_hat[i].detach().numpy().squeeze())
        column_titles = ['original image', 'visible area', 'target patches', 'predicted patches']
        for a, col in zip(ax[0], column_titles):
            a.set_title(col, fontsize=14, pad=20)
        plt.show()

    #train()
    test()

if __name__ == "__main__":
    config = {
            "epoch": 1,
            "learning_rate": 1e-4,
            "train_batch_size": 64,
            "test_batch_size": 64,
            "ema_alpha": 0.994,
            "num_targets": 3,
            "num_vit_layers": 2,
            "num_vit_heads": 3,
            "img_size": 32,
            "patch_length": 4,
            "latent_dim": 128
    }
    main(config) 
