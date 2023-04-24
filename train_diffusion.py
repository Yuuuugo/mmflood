import torch
from floods.datasets.flood import RGBFloodDataset
from floods.models.consistency_model import ConsistencyModel
from floods.utils.keras import kerras_boundaries
from floods.datasets.flood import FloodDataset
from torch.utils.data import DataLoader
from accelerate import Accelerator

# Implementation of Consistency Model
# https://arxiv.org/pdf/2303.01469.pdf


from typing import List
from tqdm import tqdm
import math

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid



def train(
    model,
    accelerator : Accelerator,
    n_epoch: int = 20,
    dataloader=DataLoader(RGBFloodDataset("processed_data/")),
    n_channels=2,
):
    model = ConsistencyModel(n_channels, D=256)
    device = accelerator.device
    #device = "gpu:0"
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Define \theta_{-}, which is EMA of the params
    ema_model = ConsistencyModel(n_channels, D=256)
    ema_model.to(device)
    ema_model.load_state_dict(model.state_dict())

    model, optim, dataloader = accelerator.prepare(model, optim, dataloader)
    for epoch in range(1, n_epoch):
        N = math.ceil(math.sqrt((epoch * (150**2 - 4) / n_epoch) + 4) - 1) + 1
        boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(device)

        pbar = tqdm(dataloader)
        loss_ema = None
        model.train()
        for x, _ in pbar:
            print(x.shape)
            optim.zero_grad()

            z = torch.randn_like(x)
            t = torch.randint(0, N - 1, (x.shape[0], 1), device=device)
            t_0 = boundaries[t]
            t_1 = boundaries[t + 1]

            loss = model.loss(x, z, t_0, t_1, ema_model=ema_model)

            accelerator.backward(loss)
            #loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()

            optim.step()
            with torch.no_grad():
                mu = math.exp(2 * math.log(0.95) / N)
                # update \theta_{-}
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.mul_(mu).add_(p, alpha=1 - mu)

            pbar.set_description(f"loss: {loss_ema:.10f}, mu: {mu:.10f}")

        #model.eval()
        """with torch.no_grad():
            # Sample 5 Steps
            xh = model.sample(
                torch.randn_like(x).to(device=device) * 80.0,
                list(reversed([5.0, 10.0, 20.0, 40.0, 80.0])),
            )
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"./contents/ct_{name}_sample_5step_{epoch}.png")

            # Sample 2 Steps
            xh = model.sample(
                torch.randn_like(x).to(device=device) * 80.0,
                list(reversed([2.0, 80.0])),
            )
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"./contents/ct_{name}_sample_2step_{epoch}.png")

        """
        # save model
        torch.save(model.state_dict(), f"./saved_model/diffusion_model_{epoch}.pth")


if __name__ == "__main__":
    accelerator = Accelerator()

    ds = RGBFloodDataset("processed_data/", subset="train")
    train_loader = DataLoader(ds)
    train(
        accelerator= accelerator,
        dataloader=train_loader, 
        n_channels=2, 
        )