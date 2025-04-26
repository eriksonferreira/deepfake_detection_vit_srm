import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelConv2d(nn.Module):
    def __init__(self, inc=3, learnable=False):
        super(SobelConv2d, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        self.sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        '''
        x: img tensor (Batch, Channels, Height, Width)
        '''
        batch_size, channels, height, width = x.shape

        # Move os kernels para o mesmo dispositivo que x, se necessário
        device = x.device
        sobel_kernel_x = self.sobel_kernel_x.to(device)
        sobel_kernel_y = self.sobel_kernel_y.to(device)

        # Repete os kernels para cada canal da imagem de entrada
        sobel_x = F.conv2d(x, sobel_kernel_x.repeat(channels, 1, 1, 1), padding=1, groups=channels)
        sobel_y = F.conv2d(x, sobel_kernel_y.repeat(channels, 1, 1, 1), padding=1, groups=channels)
        
        # Calcula a magnitude da borda para cada canal
        sobel = torch.sqrt(sobel_x ** 2 + sobel_y ** 2)
        
        # Limita os valores entre -3 e 3 usando Hardtanh
        sobel = self.truc(sobel)

        return sobel
