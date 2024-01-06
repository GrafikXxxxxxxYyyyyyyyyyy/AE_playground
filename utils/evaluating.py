import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import make_grid


def show_images(batch, nrow=3):
    batch = batch.detach().numpy()

    # Создаём подграфик
    fig, ax = plt.subplots(figsize=(10, 10))
    # Убираем кооринаты
    ax.set_xticks([]); ax.set_yticks([])
    # Рисуем картинки
    ax.imshow(make_grid(torch.tensor(batch[ :min(len(batch), int(nrow**2))]), nrow).permute(1,2,0))

    return 
