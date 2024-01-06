import torch
import torch.nn as nn
import torch.nn.functional as F



###########################################################################################################################################
# Base convolutional block
###########################################################################################################################################
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, conv_num=1, batch_norm=True, activation="relu"):
        """
        Инициализация одного свёрточного блока ConvBlock

        Параметры:
            - in_c:         Количество входных каналов.
            - out_c:        Количество выходных каналов.
            - conv_num:     Количество сверточных слоев.
            - batch_norm:   Аргумент включения/отключения батч-нормализации.
            - activation:   Выбранная функция активации
        """
        super(ConvBlock, self).__init__()

        layers = []
        for i in range(1, conv_num + 1):
            if i == 1:
                layers.append(nn.Conv2d(in_c, out_c, kernel_size=(1 + 2 * i), stride=1, padding=i))
            else:
                layers.append(nn.Conv2d(out_c, out_c, kernel_size=(1 + 2 * i), stride=1, padding=i))

        # Опциональный батчнорм
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_c))
            
        # Опциональная активация
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "selu":
            layers.append(nn.SELU())
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU())
        else:
            raise ValueError("Неподдерживаемая функция активации. Используйте 'relu', 'selu' или 'leakyrelu'")

        self.conv_block = nn.Sequential(*layers)

        return

    
    def forward(self, img):
        return self.conv_block(img)
###########################################################################################################################################
    


###########################################################################################################################################
# Base convolutional encoder block for ConvAE
###########################################################################################################################################
class ConvEncoderBlock (nn.Module):
    def __init__ (self, in_c, out_c, conv_blocks=1, downsample='conv', down_k_size=2):
        super(ConvEncoderBlock, self).__init__()
        
        blocks = []
        for i in range(1, conv_blocks + 1):
            if i == 1:
                blocks.append(ConvBlock(in_c, out_c))
            else:
                blocks.append(ConvBlock(out_c, out_c))

        # Опциональный downsample
        if downsample == 'conv':
            blocks.append(nn.Conv2d(out_c, out_c, kernel_size=(1+down_k_size), stride=down_k_size, padding=1))
        elif downsample == 'avgpool':
            blocks.append(nn.AvgPool2d(kernel_size=down_k_size, stride=down_k_size))
        elif downsample == 'maxpool':
            blocks.append(nn.MaxPool2d(kernel_size=down_k_size, stride=down_k_size))
        else:
            raise ValueError("Неподдерживаемый метод downsample. Используйте 'conv', 'maxpool' или 'avgpool'")


        self.encoder_block = nn.Sequential(*blocks)
        
        return
    
    
    def forward (self, img):        
        return self.encoder_block(img)
###########################################################################################################################################
    


###########################################################################################################################################
# Base convolutional decoder block for ConvAE
###########################################################################################################################################
class ConvDecoderBlock (nn.Module):
    def __init__ (self, in_c, out_c, conv_blocks=1, upsample='conv', up_k_size=2):
        super(ConvDecoderBlock, self).__init__()
        
        blocks = []

        # Опциональный upsample
        if upsample == 'conv':
            blocks.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=(1+up_k_size), stride=up_k_size, padding=1, output_padding=1))
        elif upsample == 'dummy':
            blocks.append(nn.Upsample(scale_factor=up_k_size))
        else:
            raise ValueError("Неподдерживаемый метод downsample. Используйте 'conv', 'maxpool' или 'avgpool'")
        
        for i in range(1, conv_blocks+1):
            blocks.append(ConvBlock(out_c, out_c))

        self.decoder_block = nn.Sequential(*blocks)
        
        return
    
    
    def forward (self, img):
        return self.decoder_block(img)
###########################################################################################################################################
    


###########################################################################################################################################
# Base encoder block
###########################################################################################################################################
class ConvEncoder (nn.Module):
    def __init__ (self, input_dim, latent_dim, num_blocks, hidden_c, img_size, k_size):
        super(ConvEncoder, self).__init__()
        
        # Проверка что кол-во блоков соотв-т размерам изображения
        hidden_size = int(img_size / (k_size**num_blocks)) 
        if hidden_size < 1:
            raise ValueError(f"Too many blocks for image of size {img_size}x{img_size}")
        
        blocks = []
        current_hidden = hidden_c
        for i in range(1, num_blocks+1):
            if i == 1:
                blocks.append(ConvEncoderBlock(input_dim, current_hidden, conv_blocks=1, downsample="conv", down_k_size=k_size))
            else:
                blocks.append(ConvEncoderBlock(current_hidden, current_hidden*2, conv_blocks=1, downsample="conv", down_k_size=k_size))
                current_hidden *= 2
        
        # Оборачиваем весь энкодер в Sequential
        self.encoder = nn.Sequential(*blocks)
        
        # Переводим скрытое изображение в латентный вектор
        self.to_latent = nn.Sequential(
            nn.Conv2d(current_hidden, latent_dim, kernel_size=1),
            nn.Flatten(),
            nn.Linear(latent_dim * hidden_size * hidden_size, latent_dim)
        )
        
        return
        

    def forward (self, x):
        return self.to_latent(self.encoder(x))
###########################################################################################################################################



###########################################################################################################################################
# Base decoder block
###########################################################################################################################################
class ConvDecoder (nn.Module):
    def __init__ (self, latent_dim, out_c, num_blocks, hidden_c, img_size, k_size):
        super(ConvDecoder, self).__init__()
        
        # Проверка что кол-во блоков соотв-т размерам изображения
        hidden_size = int(img_size / (k_size**num_blocks))
        if hidden_size < 1:
            raise ValueError(f"Too many blocks for image of size {img_size}x{img_size}")
        
        # Извлекаем из латентного вектора скрытое изображение
        self.from_latent = nn.Sequential (
            nn.Linear(latent_dim, latent_dim * hidden_size * hidden_size),
            nn.Unflatten(1, (latent_dim, hidden_size, hidden_size)),
            nn.Conv2d(latent_dim, int(hidden_c * (2**(num_blocks-1))), kernel_size=1)
        )    
        
        blocks = []
        current_hidden = int(hidden_c * (2**(num_blocks-1)))
        for i in range(1, num_blocks+1):
            if i != num_blocks:
                blocks.append(ConvDecoderBlock(current_hidden, int(current_hidden / 2), conv_blocks=1, upsample="conv", up_k_size=k_size))
                current_hidden = int(current_hidden / 2)
            else:
                blocks.extend([ConvDecoderBlock(current_hidden, out_c, conv_blocks=1, upsample="conv", up_k_size=k_size)])
        
        # Выходной свёрточный слой с нелинейностью
        blocks.extend   ([
                            nn.Conv2d(out_c, out_c, kernel_size=1, stride=1, padding=0),
                            nn.Sigmoid()
                        ])


        self.decoder = nn.Sequential(*blocks)

        return 
    

    def forward (self, z):
        return self.decoder(self.from_latent(z))
###########################################################################################################################################
    


                                #####################################################################
                                #     ________  _______       ____  ____        ____  ___    __  __ #
                                #    / ____/ / / / ___/      / __ \/ __ \      / __ \/   |  / / / / #
                                #   / /_  / / / /\__ \______/ /_/ / / / /_____/ / / / /| | / /_/ /  #
                                #  / __/ / /_/ /___/ /_____/ _, _/ /_/ /_____/ /_/ / ___ |/ __  /   #
                                # /_/    \____//____/     /_/ |_|\____/     /_____/_/  |_/_/ /_/    #
                                #####################################################################



###########################################################################################################################################
# Convolutional autoencoder architecture
###########################################################################################################################################
class ConvAutoencoder (nn.Module):
    def __init__(self, channels, latent_dim=256, num_blocks=4, hidden=32, img_size=128, k_size=2):
        super(ConvAutoencoder, self).__init__()

        self.encoder = ConvEncoder(channels, latent_dim, num_blocks, hidden, img_size, k_size)
        self.decoder = ConvDecoder(latent_dim, channels, num_blocks, hidden, img_size, k_size)

        return
    

    def forward (self, x):
        return self.decode(self.encode(x))
    

    def encode (self, x):
        return self.encoder(x)
    

    def decode (self, z):
        return self.decoder(z)
###########################################################################################################################################
