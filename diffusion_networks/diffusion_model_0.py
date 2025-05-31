import torch
from torch import nn
import torch.utils.checkpoint as checkpt
import math



class ResidualBlock(nn.Module):
    def __init__(self, channels, time_emb_dim=256):
        super(ResidualBlock, self).__init__()
        padding = (3 - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(channels),
            nn.GroupNorm(num_groups=8, num_channels=channels, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(channels),
            nn.GroupNorm(num_groups=8, num_channels=channels, affine=True),
        )
        self.silu = nn.SiLU(inplace=True)
        self.time_emb_proj = nn.Linear(time_emb_dim, channels)

    def forward(self, x, t_emb):
        # with torch.autocast(device_type="cuda", enabled=False):
        #     x = x.float()
        #     t_emb = t_emb.float()
        # Add time embedding to features (broadcast to spatial)
        t_added = self.time_emb_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        out = self.block[0](x + t_added)
        out = self.block[1](out)
        out = self.block[2](out)
        out = self.block[3](out)
        out = self.block[4](out)
        x = self.silu(x + out)
        # x = x + self.block(x)
        # x = self.silu(x)
        return x

class ResidualBlockEnd(nn.Module):
    def __init__(self, channels, time_emb_dim=256):
        super(ResidualBlockEnd, self).__init__()
        padding = (3 - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(channels),
            nn.InstanceNorm2d(channels, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(channels),
            nn.InstanceNorm2d(channels, affine=True),
        )
        self.silu = nn.SiLU(inplace=True)
        self.time_emb_proj = nn.Linear(time_emb_dim, channels)

    def forward(self, x, t_emb):
        # with torch.autocast(device_type="cuda", enabled=False):
        #     x = x.float()
        #     t_emb = t_emb.float()
        # Add time embedding to features (broadcast to spatial)
        t_added = self.time_emb_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        out = self.block[0](x + t_added)
        out = self.block[1](out)
        out = self.block[2](out)
        out = self.block[3](out)
        out = self.block[4](out)
        x = self.silu(x + out)
        # x = x + self.block(x)
        # x = self.silu(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_dim, time_emb_dim=None):
        super().__init__()
        self.in_dim = in_dim
        self.time_emb_dim = time_emb_dim

        mid_dim = max(1, in_dim // 8)
        self.query_conv = nn.Conv2d(in_dim, mid_dim, 1)
        self.key_conv   = nn.Conv2d(in_dim, mid_dim, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, in_dim)

    def forward(self, x, t=None):
        B, C, H, W = x.size()

        if self.time_emb_dim is not None and t is not None:
            x = x + self.time_mlp(t)[:, :, None, None]

        q = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key_conv(x).view(B, -1, H * W)
        v = self.value_conv(x).view(B, -1, H * W)

        attn = self.softmax(torch.bmm(q, k))
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)

        return self.gamma * out + x

class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim, context_dim, heads=4, dim_head=32):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        # Projections
        self.to_q = nn.Conv2d(query_dim, inner_dim, 1)
        self.to_k = nn.Conv2d(context_dim, inner_dim, 1)
        self.to_v = nn.Conv2d(context_dim, inner_dim, 1)
        self.to_out = nn.Conv2d(inner_dim, query_dim, 1)

    def forward(self, x, context):
        B, C, H, W = x.shape
        _, Cc, Hc, Wc = context.shape
        h = self.heads

        # Project to queries, keys, and values
        q = self.to_q(x).reshape(B, h, -1, H * W)          # [B, h, d, HW]
        k = self.to_k(context).reshape(B, h, -1, Hc * Wc)   # [B, h, d, Hc*Wc]
        v = self.to_v(context).reshape(B, h, -1, Hc * Wc)   # [B, h, d, Hc*Wc]

        # Compute attention
        sim = torch.einsum("bhdi,bhdj->bhij", q, k) * self.scale  # [B, h, HW, Hc*Wc]
        attn = sim.softmax(dim=-1)

        # Apply to values
        out = torch.einsum("bhij,bhdj->bhdi", attn, v)      # [B, h, d, HW]
        out = out.reshape(B, -1, H, W)                      # [B, C, H, W]

        return self.to_out(out)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        # Create sinusoidal embeddings
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(t.device)
        emb = t[:, None] * emb[None, :]  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)
        return self.mlp(emb)  # (B, dim)


class EncodeBlock(nn.Module):
    def __init__(self, input_channels, output_channels, time_emb_dim=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            #nn.BatchNorm2d(output_channels),
            nn.GroupNorm(num_groups=8, num_channels=output_channels, affine=True),
            nn.SiLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),  # downsampling here
            nn.GroupNorm(num_groups=8, num_channels=output_channels, affine=True),
            nn.SiLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),  # downsampling here
            nn.GroupNorm(num_groups=8, num_channels=output_channels, affine=True),
            nn.SiLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),  # downsampling here
            nn.GroupNorm(num_groups=8, num_channels=output_channels, affine=True),
            nn.SiLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),  # downsampling here
            nn.GroupNorm(num_groups=8, num_channels=output_channels, affine=True),
            nn.SiLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),  # downsampling here
            nn.GroupNorm(num_groups=8, num_channels=output_channels, affine=True),
            nn.SiLU(inplace=True),
        )
        self.residual = ResidualBlock(output_channels, time_emb_dim)
        self.attention = SelfAttention(output_channels, time_emb_dim)
        self.cross_attention1 = CrossAttentionBlock(input_channels, input_channels)
        self.cross_attention2 = CrossAttentionBlock(output_channels, output_channels)

    def forward(self, x, t_emb, x_cond):
        x = x + self.cross_attention1(x, x_cond)
        x = self.block(x)
        x_cond = self.block(x_cond)
        x = self.residual(x, t_emb)
        x = self.attention(x, t_emb)
        x = x + self.cross_attention2(x, x_cond)

        x = self.block2(x)
        x = self.residual(x, t_emb)
        x = self.attention(x, t_emb)
        x = x + self.cross_attention2(x, x_cond)
        return x, x_cond


class DecodeBlock(nn.Module):
    def __init__(self, input_channels, output_channels, time_emb_dim=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(output_channels),
            nn.GroupNorm(num_groups=8, num_channels=output_channels, affine=True),
            nn.SiLU(True),

            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),  # downsampling here
            nn.GroupNorm(num_groups=8, num_channels=output_channels, affine=True),
            nn.SiLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),  # downsampling here
            nn.GroupNorm(num_groups=8, num_channels=output_channels, affine=True),
            nn.SiLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),  # downsampling here
            nn.GroupNorm(num_groups=8, num_channels=output_channels, affine=True),
            nn.SiLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),  # downsampling here
            nn.GroupNorm(num_groups=8, num_channels=output_channels, affine=True),
            nn.SiLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),  # downsampling here
            nn.GroupNorm(num_groups=8, num_channels=output_channels, affine=True),
            nn.SiLU(inplace=True),
        )
        self.residual = ResidualBlock(output_channels, time_emb_dim)
        self.attention = SelfAttention(output_channels, time_emb_dim)
        self.skipper = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, padding=1),
        )
        self.cross_attention1 = CrossAttentionBlock(input_channels, input_channels)
        self.cross_attention2 = CrossAttentionBlock(output_channels, output_channels)


    def forward(self, x, skip, t_emb, x_cond):
        x = torch.cat([x, skip], dim=1)
        x = self.skipper(x)
        # x_cond = torch.cat([x, x_cond], dim=1)
        # x_cond = self.skipper(x_cond)
        x = x + self.cross_attention1(x, x_cond)

        x = self.block(x)
        x_cond = self.block(x_cond)
        x = self.residual(x, t_emb)
        x = self.attention(x, t_emb)
        x = x + self.cross_attention2(x, x_cond)

        x = self.block2(x)
        x = self.residual(x, t_emb)
        x = self.attention(x, t_emb)
        x = x + self.cross_attention2(x, x_cond)
        return x, x_cond


class DecodeBlockEnd(nn.Module):
    def __init__(self, input_channels, output_channels, time_emb_dim=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(output_channels),
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.SiLU(True),

            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),  # downsampling here
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.SiLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),  # downsampling here
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.SiLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),  # downsampling here
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.SiLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),  # downsampling here
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.SiLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),  # downsampling here
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.SiLU(inplace=True),
        )
        self.residual = ResidualBlockEnd(output_channels, time_emb_dim)
        self.attention = SelfAttention(output_channels, time_emb_dim)
        self.skipper = nn.Sequential(
            nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, padding=1),
        )
        self.cross_attention1 = CrossAttentionBlock(input_channels, input_channels)
        self.cross_attention2 = CrossAttentionBlock(output_channels, output_channels)

    def forward(self, x, skip, t_emb, x_cond):
        x = torch.cat([x, skip], dim=1)
        x = self.skipper(x)
        # x_cond = torch.cat([x, x_cond], dim=1)
        # x_cond = self.skipper(x_cond)
        x = x + self.cross_attention1(x, x_cond)

        x = self.block(x)
        x_cond = self.block(x_cond)

        x = self.residual(x, t_emb)
        x = self.attention(x, t_emb)
        x = x + self.cross_attention2(x, x_cond)

        x = self.block2(x)
        x = self.residual(x, t_emb)
        x = self.attention(x, t_emb)
        x = x + self.cross_attention2(x, x_cond)
        return x, x_cond




class BottleneckLayer(nn.Module):
    def __init__(self, channel, time_emb_dim=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            #nn.BatchNorm2d(channel),
            nn.GroupNorm(num_groups=8, num_channels=channel, affine=True),
            nn.SiLU(True),

            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            # nn.BatchNorm2d(channel),
            nn.GroupNorm(num_groups=8, num_channels=channel, affine=True),
            nn.SiLU(True),

            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            # nn.BatchNorm2d(channel),
            nn.GroupNorm(num_groups=8, num_channels=channel, affine=True),
            nn.SiLU(True),
        )
        self.residual = ResidualBlock(channel, time_emb_dim)
        self.attention = SelfAttention(channel, time_emb_dim)

        self.cross_attention1 = CrossAttentionBlock(channel, channel)
        self.cross_attention2 = CrossAttentionBlock(channel, channel)

    def forward(self, x, t_emb, x_cond):
        x = x + self.cross_attention1(x, x_cond)
        x = self.block(x)
        x_cond = self.block(x_cond)
        x = self.residual(x, t_emb)
        x = self.attention(x, t_emb)
        x = x + self.cross_attention2(x, x_cond)

        x = self.block(x)
        x_cond = self.block(x_cond)
        x = self.residual(x, t_emb)
        x = self.attention(x, t_emb)
        x = x + self.cross_attention2(x, x_cond)
        return x, x_cond



def compute_downsample_layers(h, w, target=8):
    ds_h = int(math.log2(h // target))
    ds_w = int(math.log2(w // target))
    assert ds_h == ds_w, "Width and height must downsample to 8x8 symmetrically"
    #print(f'ds_h:{ds_h}')
    #print(f'ds_w:{ds_w}')
    return ds_h



class Diffusion_Model_0(nn.Module):
    def __init__(self, input_resolution, time_emb_dim=256):
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.input_resolution = input_resolution
        self.num_downsamples = compute_downsample_layers(*input_resolution)

        channels = [3] + [128 * (2 ** i) for i in range(self.num_downsamples)]
        #print(f'channels: {channels}')
        # Encoder
        self.encode_blocks = nn.ModuleList([
            EncodeBlock(channels[i], channels[i + 1], time_emb_dim)
            for i in range(self.num_downsamples)
        ])

        # Bottleneck
        self.bottleneck_block = BottleneckLayer(channels[-1], time_emb_dim)

        channel_first = channels.pop(0)

        self.decode_blocks = nn.ModuleList([
            DecodeBlock(channels[i + 1], channels[i], time_emb_dim)
            for i in reversed(range(self.num_downsamples - 1))
        ])
        self.decode_blocks.append(DecodeBlockEnd(channels[0], channel_first, time_emb_dim))


        # self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        #
        # # Encoder layers
        # self.encode_blocks1 = EncodeBlock(3, 32, time_emb_dim)
        # self.encode_blocks2 = EncodeBlock(32, 128, time_emb_dim)
        #
        # # Bottleneck Layers
        # self.bottleneck_block = BottleneckLayer(128, time_emb_dim)
        #
        # # Decoder layers
        # self.decode_blocks3 = DecodeBlock(128, 32, time_emb_dim)
        # self.decode_blocks4 = DecodeBlockEnd(32, 3, time_emb_dim)


    def forward(self, x, t_emb, x_cond):
        #t_emb = self.time_embedding(t_emb)
        t_emb = checkpt.checkpoint(self.time_embedding, t_emb, use_reentrant=True)
        enc_feats = []
        #print(f'x.shape: {x.shape}')
        for encode in self.encode_blocks:
            x, x_cond = checkpt.checkpoint(encode, x, t_emb, x_cond, use_reentrant=True)
            #print(f'x.shape: {x.shape}')
            enc_feats.append((x, x_cond))

        for _ in range(3):
            x, x_cond = checkpt.checkpoint(self.bottleneck_block, x, t_emb, x_cond, use_reentrant=True)

        #print(f'x.shape: {x.shape}')
        #print(f'x_cond.shape: {x_cond.shape}')
        # After bottleneck input

        assert x.shape[-1] == 8 and x.shape[-2] == 8, f"Expected 8x8 bottleneck, got {x.shape[-2:]}"
        #print('---------dssssssssss-----------')
        #print(f'x.shape: {x.shape}')
        total_decodes = len(self.decode_blocks)
        for decode, (skip_x, skip_cond) in zip(self.decode_blocks, reversed(enc_feats)):
            #print(f'x.shape before: {x.shape}')
            #print(f'skip_x.shape: {skip_x.shape}')
            total_decodes = total_decodes - 1
            # if total_decodes > 1:
            #     x, x_cond = checkpt.checkpoint(decode, x, skip_x, t_emb, x_cond, use_reentrant=True)
            # else:
            #     x, x_cond = decode(x, skip_x, t_emb, x_cond)
            #x, x_cond = decode(x, skip_x, t_emb, x_cond)
            x, x_cond = checkpt.checkpoint(decode, x, skip_x, t_emb, x_cond, use_reentrant=True)

            #print(f'x.shape after: {x.shape}')


        return x

        #
        # t_emb = self.time_embedding(t)  # (B, D)
        #
        # encode_1, x_cond1 = self.encode_blocks1(x, t_emb, x_cond)
        # encode_2, x_cond2 = self.encode_blocks2(encode_1, t_emb, x_cond1)
        #
        # bottleneck_1, x_cond2 = self.bottleneck_block(encode_2, t_emb, x_cond2)
        #
        # decode_3, de_x_cond3 = self.decode_blocks3(bottleneck_1, encode_2, t_emb, x_cond2)
        # decode_4, de_x_cond4 = self.decode_blocks4(decode_3, encode_1, t_emb, de_x_cond3)

        # return decode_4
