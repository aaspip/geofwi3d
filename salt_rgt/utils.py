import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SK(nn.Module):
    def __init__(self, channels, m=2, r=2, L=32, kernel=None):
        super(SK, self).__init__()

        if kernel is None:
            kernel = channels

        d = max(int(kernel * r), L)

        self.out = nn.Sequential(
            nn.Conv3d(channels, kernel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(kernel),
            nn.ReLU(inplace=True)
        )

        self.x1 = nn.Sequential(
            nn.Conv3d(kernel, kernel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(kernel),
            nn.ReLU(inplace=True)
        )

        self.x2 = nn.Sequential(
            nn.Conv3d(kernel, kernel, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm3d(kernel),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Linear(kernel, d)
        self.fc2 = nn.Linear(d, kernel * m)

        self.kernel = kernel
        self.m = m

    def forward(self, inputs):
        out = self.out(inputs)

        x1 = self.x1(out)
        x2 = self.x2(out)

        _x1 = torch.mean(x1, dim=(2, 3, 4))
        _x2 = torch.mean(x2, dim=(2, 3, 4))

        U = _x1 + _x2

        z = self.fc1(U)
        z = F.relu(z)
        z = self.fc2(z)

        z = z.view(z.shape[0], self.m, self.kernel, 1, 1, 1)
        scale = F.softmax(z, dim=1)

        x = torch.stack([x1, x2], dim=1)
        r = torch.sum(scale * x, dim=1)

        return r

class DenseNet(nn.Module):
    def __init__(self, in_channels, layers, filters):
        super(DenseNet, self).__init__()

        self.layers = layers
        self.blocks = nn.ModuleList()

        current_channels = in_channels

        for i in range(layers):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv3d(current_channels, filters, kernel_size=3, padding=1, stride=1),
                    nn.BatchNorm3d(filters),
                    nn.ReLU(inplace=True)
                )
            )
            current_channels += filters

        self.out_channels = current_channels

    def forward(self, inp):
        x4 = inp
        x5 = None

        for i, block in enumerate(self.blocks):
            x3 = block(x4)

            if i == 0:
                x4 = torch.cat([x3, inp], dim=1)
                x5 = x4
            else:
                x4 = torch.cat([x3, x4], dim=1)

            if (i > 0) and (i < self.layers - 1):
                x5 = torch.cat([x5, x4], dim=1)

        return x5 if x5 is not None else x4


class TD(nn.Module):
    def __init__(self, in_channels, filters, U):
        super(TD, self).__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, filters, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=U, stride=U)
        )

    def forward(self, inp):
        return self.block(inp)


class TU(nn.Module):
    def __init__(self, in_channels, filters, U):
        super(TU, self).__init__()

        if isinstance(U, int):
            output_padding = U - 1
        else:
            output_padding = tuple(u - 1 for u in U)

        self.block = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                filters,
                kernel_size=3,
                padding=1,
                stride=U,
                output_padding=output_padding
            ),
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, inp):
        return self.block(inp)


def TicTocGenerator():
    ti = 0
    tf = time.time()

    while True:
        ti = tf
        tf = time.time()
        yield tf - ti


TicToc = TicTocGenerator()


def toc(tempBool=True):
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds.\n" % tempTimeInterval)


def tic():
    toc(False)


def patch3d_optimized(A, l1=4, l2=4, l3=4, s1=2, s2=2, s3=2):
    pad1 = (l1 - A.shape[0] % s1) % s1
    pad2 = (l2 - A.shape[1] % s2) % s2
    pad3 = (l3 - A.shape[2] % s3) % s3

    A_padded = np.pad(A, ((0, pad1), (0, pad2), (0, pad3)), mode='constant')

    n1, n2, n3 = A_padded.shape

    n1_patches = (n1 - l1) // s1 + 1
    n2_patches = (n2 - l2) // s2 + 1
    n3_patches = (n3 - l3) // s3 + 1

    shape = (n1_patches, n2_patches, n3_patches, l1, l2, l3)

    strides = (
        s1 * A_padded.strides[0],
        s2 * A_padded.strides[1],
        s3 * A_padded.strides[2]
    ) + A_padded.strides

    patches = np.lib.stride_tricks.as_strided(
        A_padded,
        shape=shape,
        strides=strides
    )

    patches = patches.reshape(-1, l1 * l2 * l3)

    return patches


def patch3d_inv_optimized(X, n1, n2, n3, l1=4, l2=4, l3=4, s1=2, s2=2, s3=2):
    pad1 = (l1 - n1 % s1) % s1
    pad2 = (l2 - n2 % s2) % s2
    pad3 = (l3 - n3 % s3) % s3

    A = np.zeros((n1 + pad1, n2 + pad2, n3 + pad3))
    mask = np.zeros_like(A)

    X = X.reshape(-1, l1, l2, l3)

    n1_patches = (A.shape[0] - l1) // s1 + 1
    n2_patches = (A.shape[1] - l2) // s2 + 1
    n3_patches = (A.shape[2] - l3) // s3 + 1

    idx = 0

    for i in range(n1_patches):
        for j in range(n2_patches):
            for k in range(n3_patches):
                A[
                    i * s1:i * s1 + l1,
                    j * s2:j * s2 + l2,
                    k * s3:k * s3 + l3
                ] += X[idx]

                mask[
                    i * s1:i * s1 + l1,
                    j * s2:j * s2 + l2,
                    k * s3:k * s3 + l3
                ] += 1

                idx += 1

    mask[mask == 0] = 1
    A /= mask

    return A[:n1, :n2, :n3]


class conv_bn_relu_3d(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=3):
        super(conv_bn_relu_3d, self).__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, filters, kernel_size=kernel_size, padding=kernel_size // 2, stride=1),
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class encoder_block(nn.Module):
    def __init__(self, in_channels, filters, layer=1):
        super(encoder_block, self).__init__()

        self.dense = DenseNet(in_channels, layer, filters)
        self.sk = SK(self.dense.out_channels, m=2, r=2, L=8, kernel=filters)
        self.down = TD(filters, filters, 2)

    def forward(self, x):
        x = self.dense(x)
        x = self.sk(x)
        down = self.down(x)
        return x, down


class decoder_block(nn.Module):
    def __init__(self, in_channels, filters, layer=1, up_size=2):
        super(decoder_block, self).__init__()

        self.up = TU(in_channels, filters, up_size)
        self.dense = DenseNet(filters, layer, filters)
        self.sk = SK(self.dense.out_channels, m=2, r=2, L=8, kernel=filters)

    def forward(self, x):
        x = self.up(x)
        x = self.dense(x)
        x = self.sk(x)
        return x


class SaltRGT3DNet(nn.Module):
    def __init__(
        self,
        input_channels=1,
        D=2,
        layer=1,
        layers=1
    ):
        super(SaltRGT3DNet, self).__init__()

        self.stem = conv_bn_relu_3d(input_channels, D)

        self.encoder_1 = encoder_block(D, D * 2, layer=layer)
        self.encoder_2 = encoder_block(D * 2, D * 4, layer=layer)
        self.encoder_3 = encoder_block(D * 4, D * 8, layer=layer)

        self.bottleneck_dense = DenseNet(D * 8, layer, D * 16)
        self.bottleneck_sk = SK(
            self.bottleneck_dense.out_channels,
            m=2,
            r=2,
            L=8,
            kernel=D * 16
        )

        self.rgt_decoder_1 = decoder_block(D * 16, D * 8, layer=layer, up_size=(2, 2, 2))
        self.rgt_decoder_2 = decoder_block(D * 8, D * 4, layer=layer, up_size=2)
        self.rgt_decoder_3 = decoder_block(D * 4, D * 2, layer=layer, up_size=2)
        self.rgt_final = conv_bn_relu_3d(D * 2, D)
        self.out_rgt = nn.Conv3d(D, 1, kernel_size=3, padding=1)

        self.lab_decoder_1 = decoder_block(D * 16, D * 8, layer=layers, up_size=(2, 2, 2))
        self.lab_decoder_2 = decoder_block(D * 8, D * 4, layer=layers, up_size=2)
        self.lab_decoder_3 = decoder_block(D * 4, D * 2, layer=layers, up_size=2)
        self.outlab = nn.Conv3d(D * 2, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.stem(x)

        enc1, down1 = self.encoder_1(x)
        enc2, down2 = self.encoder_2(down1)
        enc3, down3 = self.encoder_3(down2)

        bottleneck = self.bottleneck_dense(down3)
        bottleneck = self.bottleneck_sk(bottleneck)

        rgt = self.rgt_decoder_1(bottleneck)
        rgt = self.rgt_decoder_2(rgt)
        rgt = self.rgt_decoder_3(rgt)
        rgt = self.rgt_final(rgt)
        out_rgt = self.out_rgt(rgt)

        lab = self.lab_decoder_1(bottleneck)
        lab = self.lab_decoder_2(lab)
        lab = self.lab_decoder_3(lab)
        out_lab = torch.sigmoid(self.outlab(lab))

        return out_rgt, out_lab



    #### For RGT prediction


import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScale3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )

        self.fuse = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x3 = self.conv3(x)
        x1 = self.conv1(x)

        x = torch.cat([x3, x1], dim=1)
        x = self.fuse(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=2, ff_dim=128, dropout=0.05):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)

        return x


class SeismicRGTNet(nn.Module):
    def __init__(
            self,
            input_channels=1,
            base_channels=8,
            embed_dim=32,
            num_heads=2,
            ff_dim=64,
            num_transformer_blocks=1,
            dropout=0.05
    ):
        super().__init__()

        # 64
        self.enc1 = nn.Sequential(
            MultiScale3DBlock(input_channels, base_channels)
        )

        self.down1 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.GELU()
        )

        # 32
        self.enc2 = nn.Sequential(
            MultiScale3DBlock(base_channels * 2, base_channels * 2)
        )

        self.down2 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.GELU()
        )

        # 16
        self.enc3 = nn.Sequential(
            MultiScale3DBlock(base_channels * 4, base_channels * 4)
        )

        self.down3 = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 8),
            nn.GELU()
        )

        # 8
        self.bottleneck = nn.Sequential(
            MultiScale3DBlock(base_channels * 8, base_channels * 8)
        )

        self.token_proj = nn.Linear(base_channels * 8, embed_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for _ in range(num_transformer_blocks)
        ])

        self.token_back = nn.Linear(embed_dim, base_channels * 8)

        # 8 -> 16
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.GELU()
        )

        self.dec1 = nn.Sequential(
            MultiScale3DBlock(base_channels * 8, base_channels * 4)
        )

        # 16 -> 32
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.GELU()
        )

        self.dec2 = nn.Sequential(
            MultiScale3DBlock(base_channels * 4, base_channels * 2)
        )

        # 32 -> 64
        self.up3 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.GELU()
        )

        self.dec3 = nn.Sequential(
            MultiScale3DBlock(base_channels * 2, base_channels)
        )

        self.out_rgt = nn.Conv3d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)          # (B, C, 64, 64, 64)
        x = self.down1(e1)

        e2 = self.enc2(x)          # (B, 2C, 32, 32, 32)
        x = self.down2(e2)

        e3 = self.enc3(x)          # (B, 4C, 16, 16, 16)
        x = self.down3(e3)

        # bottleneck
        x = self.bottleneck(x)     # (B, 8C, 8, 8, 8)

        # transformer
        B, C, D, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.token_proj(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.token_back(x)
        x = x.transpose(1, 2).reshape(B, C, D, H, W)

        # decoder + skip connections
        x = self.up1(x)            # (B, 4C, 16, 16, 16)
        x = torch.cat([x, e3], dim=1)
        x = self.dec1(x)

        x = self.up2(x)            # (B, 2C, 32, 32, 32)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)

        x = self.up3(x)            # (B, C, 64, 64, 64)
        x = torch.cat([x, e1], dim=1)
        x = self.dec3(x)

        out = self.out_rgt(x)

        return out