'''
Efficient Concertormer for Image Deblurring and Beyond

@article{kuo2024efficient,
  title={Efficient Concertormer for Image Deblurring and Beyond},
  author={Kuo, Pin-Hung and Pan, Jinshan and Chien, Shao-Yi and Yang, Ming-Hsuan},
  journal={arXiv preprint arXiv:2404.06135},
  year={2024}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from einops import rearrange, repeat
import math


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    x = rearrange(x, 'b (h kh) (w kw) c -> (b h w c) (kh kw)', kh=window_size, kw=window_size)

    return x


class CSAttention(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, shift):
        super(CSAttention, self).__init__()
        h_dim = int(ffn_expansion_factor*dim)
        self.dim = dim
        self.h_dim = h_dim
        self.shift = shift
        self.num_heads = num_heads
        self.ffn_expansion_factor = ffn_expansion_factor
        self.ks = 8

        self.bias = bias
        self.temperature = nn.Parameter(torch.ones(4, num_heads, 1, 1, 1))
        self.r_talking = nn.Parameter(torch.empty(num_heads, h_dim // num_heads // 4, h_dim // num_heads // 4, num_heads))
        self.g_talking = nn.Parameter(torch.empty(num_heads, self.ks ** 2, self.ks ** 2, num_heads))
        nn.init.kaiming_normal_(self.r_talking, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.g_talking, a=math.sqrt(5))
        self.b_talking = nn.Conv3d(num_heads, num_heads, (3, 3, 1), padding='same', bias=bias)
        self.l_talking = nn.Conv3d(num_heads, num_heads, (3, 3, 1), padding='same', bias=bias)

        self.shift_size = self.ks // 2

        self.to_quz = nn.Conv2d(dim, h_dim*3, kernel_size=1, bias=bias)
        self.to_kv = nn.Conv2d(dim, h_dim*2, kernel_size=1, bias=bias)

        self.dw_u = nn.Conv2d(h_dim, h_dim, 3, padding=1, groups=h_dim, bias=bias)

        self.project = nn.Conv2d(h_dim, h_dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(h_dim, dim, kernel_size=1, bias=bias)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=h_dim, out_channels=h_dim, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True)
        )

    def forward(self, x, y=None):
        b, c, h, w = x.shape

        k, v = self.to_kv(y).chunk(2, dim=1) if y is not None else self.to_kv(x).chunk(2, dim=1)
        q, u, z = self.to_quz(x).chunk(3, dim=1)

        if self.shift:
            q, k, v = map(lambda t: torch.roll(t, shifts=(-self.shift_size, -self.shift_size), dims=(-1, -2)), (q, k, v))
        q, k, v = map(lambda t: rearrange(t, 'b (head c) (h kh) (w kw) -> b head (h w) (kh kw) c', head=self.num_heads, kh=self.ks, kw=self.ks), (q, k, v))
        # b:cs / l:cc / g:rc / r:rs
        qb, ql, qg, qr = q.chunk(4, dim=-1)
        kb, kl, kg, kr = k.chunk(4, dim=-1)
        vb, vl, vg, vr = v.chunk(4, dim=-1)

        qb, kb = map(lambda t: torch.nn.functional.normalize(t, dim=-1), (qb, kb))
        ql, kl = map(lambda t: torch.nn.functional.normalize(t, dim=-2), (ql, kl))
        qr, qg, kr, kg = map(lambda t: torch.nn.functional.normalize(t, dim=-3), (qr, qg, kr, kg))

        attn_b = torch.matmul(qb, kb.permute(0, 1, 2, 4, 3)) * self.temperature[0]    # b head hw k k
        attn_l = torch.matmul(ql.permute(0, 1, 2, 4, 3), kl) * self.temperature[1]    # b head hw c c
        attn_g = torch.matmul(qg.permute(0, 1, 3, 4, 2), kg.permute(0, 1, 3, 2, 4)) * self.temperature[2]  # b head k c c
        attn_r = torch.matmul(qr.permute(0, 1, 4, 3, 2), kr.permute(0, 1, 4, 2, 3)) * self.temperature[3]  # b head c k k

        attn_b = rearrange(self.b_talking(rearrange(attn_b, 'b head (h w) k1 k2 -> b head h w (k1 k2)', h=h//self.ks)),
                           'b head h w (k1 k2) -> b head (h w) k1 k2', k1=self.ks**2)
        attn_l = rearrange(self.l_talking(rearrange(attn_l, 'b head (h w) c1 c2 -> b head h w (c1 c2)', h=h//self.ks)),
                           'b head h w (c1 c2) -> b head (h w) c1 c2', c1=self.h_dim // self.num_heads // 4)
        attn_g = torch.einsum('hklt, bhk... -> btl...', self.g_talking, attn_g)
        attn_r = torch.einsum('hcdt, bhc... -> btd...', self.r_talking, attn_r)

        if self.shift:
            if not hasattr(self, 'attn_mask'):
                attn_mask = self.calculate_mask(h, w).to(q.device)
                self.register_buffer("attn_mask", attn_mask, persistent=False)
            elif self.attn_mask.shape[0] != attn_b.shape[2]:
                self.attn_mask = self.calculate_mask(h, w).to(q.device)
            attn_b = attn_b + self.attn_mask[None, None, ...]

        attn_b, attn_l, attn_g, attn_r = map(lambda t: torch.softmax(t, dim=-1), (attn_b, attn_l, attn_g, attn_r))

        out_b = rearrange(torch.matmul(attn_b, vb), 'b head (h w) (kh kw) c -> b (head c) (h kh) (w kw)', b=b, h=h//self.ks, kh=self.ks)
        out_l = rearrange(torch.matmul(attn_l, vl.transpose(-1, -2)), 'b head (h w) c (kh kw) -> b (head c) (h kh) (w kw)', b=b, h=h//self.ks, kh=self.ks)
        out_g = rearrange(torch.matmul(attn_g, vg.permute(0, 1, 3, 4, 2)), 'b head (kh kw) c (h w) -> b (head c) (h kh) (w kw)', b=b, h=h//self.ks, kh=self.ks)
        out_r = rearrange(torch.matmul(attn_r, vr.transpose(-1, -3)), 'b head c (kh kw) (h w) -> b (head c) (h kh) (w kw)', b=b, h=h//self.ks, kh=self.ks)

        out = torch.cat([out_b, out_l, out_g, out_r], dim=1)

        if self.shift:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(-1, -2))

        out = self.project(out * self.sca(out)) + self.dw_u(u)
        out = self.project_out(out * z)

        return out

    def calculate_mask(self, H, W):
        # calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.ks),
                    slice(-self.ks, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.ks),
                    slice(-self.ks, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.ks)  # nW, window_size, window_size, 1
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask


class LocalCSAttention(CSAttention):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, shift, base_size=None, kernel_size=None, fast_imp=False, train_size=None):
        super().__init__(dim, num_heads, ffn_expansion_factor, bias, shift)
        self.base_size = base_size
        self.kernel_size = kernel_size
        self.fast_imp = fast_imp
        self.train_size = train_size

    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, self.h_dim, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt


    def _forward(self, q, kv):
        b, c, h, w = q.shape
        if self.shift:
            q, kv = map(lambda t: torch.roll(t, shifts=(-self.shift_size, -self.shift_size), dims=(-1, -2)), (q, kv))
        k, v = kv.chunk(2, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b (head c) (h kh) (w kw) -> b head (h w) (kh kw) c', head=self.num_heads, kh=self.ks, kw=self.ks), (q, k, v))

        qb, ql, qg, qr = q.chunk(4, dim=-1)
        kb, kl, kg, kr = k.chunk(4, dim=-1)
        vb, vl, vg, vr = v.chunk(4, dim=-1)

        qb, kb = map(lambda t: torch.nn.functional.normalize(t, dim=-1), (qb, kb))
        ql, kl = map(lambda t: torch.nn.functional.normalize(t, dim=-2), (ql, kl))
        qr, qg, kr, kg = map(lambda t: torch.nn.functional.normalize(t, dim=-3), (qr, qg, kr, kg))

        attn_b = torch.matmul(qb, kb.permute(0, 1, 2, 4, 3)) * self.temperature[0]    # b head hw k k
        attn_l = torch.matmul(ql.permute(0, 1, 2, 4, 3), kl) * self.temperature[1]    # * temperature[1]  # b head hw c c
        attn_g = torch.matmul(qg.permute(0, 1, 3, 4, 2), kg.permute(0, 1, 3, 2, 4)) * self.temperature[2]  # b head k c c
        attn_r = torch.matmul(qr.permute(0, 1, 4, 3, 2), kr.permute(0, 1, 4, 2, 3)) * self.temperature[3]  # b head c k k

        attn_b = rearrange(self.b_talking(rearrange(attn_b, 'b head (h w) k1 k2 -> b head h w (k1 k2)', h=h//self.ks)),
                           'b head h w (k1 k2) -> b head (h w) k1 k2', k1=self.ks**2)
        attn_l = rearrange(self.l_talking(rearrange(attn_l, 'b head (h w) c1 c2 -> b head h w (c1 c2)', h=h//self.ks)),
                           'b head h w (c1 c2) -> b head (h w) c1 c2', c1=self.h_dim // self.num_heads // 4)
        attn_g = torch.einsum('hklt, bhk... -> btl...', self.g_talking, attn_g)
        attn_r = torch.einsum('hcdt, bhc... -> btd...', self.r_talking, attn_r)

        if self.shift:
            attn_mask = self.calculate_mask(h, w).to(q.device)
            attn_b = attn_b + attn_mask[None, None, ...]

        attn_b, attn_l, attn_g, attn_r = map(lambda t: torch.softmax(t, dim=-1), (attn_b, attn_l, attn_g, attn_r))
        out_b = rearrange(torch.matmul(attn_b, vb), 'b head (h w) (kh kw) c -> b (head c) (h kh) (w kw)', b=b, h=h//self.ks, kh=self.ks)
        out_l = rearrange(torch.matmul(attn_l, vl.transpose(-1, -2)), 'b head (h w) c (kh kw) -> b (head c) (h kh) (w kw)', b=b, h=h//self.ks, kh=self.ks)
        out_g = rearrange(torch.matmul(attn_g, vg.permute(0, 1, 3, 4, 2)), 'b head (kh kw) c (h w) -> b (head c) (h kh) (w kw)', b=b, h=h//self.ks, kh=self.ks)
        out_r = rearrange(torch.matmul(attn_r, vr.transpose(-1, -3)), 'b head c (kh kw) (h w) -> b (head c) (h kh) (w kw)', b=b, h=h//self.ks, kh=self.ks)
        out = torch.cat([out_b, out_l, out_g, out_r], dim=1)

        if self.shift:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(-1, -2))

        return out * self.sca(out)

    def _pad(self, x):
        b, c, h, w = x.shape
        k1, k2 = self.kernel_size
        mod_pad_h = (k1 - h % k1) % k1
        mod_pad_w = (k2 - w % k2) % k2
        pad = (mod_pad_w//2, mod_pad_w-mod_pad_w//2, mod_pad_h//2, mod_pad_h-mod_pad_h//2)
        x = F.pad(x, pad, 'reflect')
        return x, pad

    def forward(self, x, y=None):

        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

        b, c, h, w = x.shape
        quz = self.to_quz(x)
        kv = self.to_kv(x) if y is None else self.to_kv(y)
        q, u, z = quz.chunk(3, dim=1)

        if self.fast_imp:
            raise NotImplementedError
            # qkv, pad = self._pad(qkv)
            # b,C,H,W = qkv.shape
            # k1, k2 = self.kernel_size
            # qkv = qkv.reshape(b,C,H//k1, k1, W//k2, k2).permute(0,2,4,1,3,5).reshape(-1,C,k1,k2)
            # out = self._forward(qkv)
            # out = out.reshape(b,H//k1,W//k2,c,k1,k2).permute(0,3,1,4,2,5).reshape(b,c,H,W)
            # out = out[:,:,pad[-2]:pad[-2]+h, pad[0]:pad[0]+w]
        else:
            q = self.grids(q)
            kv = self.grids(kv)
            out = self._forward(q, kv)
            out = self.grids_inverse(out)

        out = self.project(out) + self.dw_u(u)
        return self.project_out(z * out)


class CSABlock(nn.Module):
    def __init__(self, dim, head, ffn_expansion_factor, idx, cross=False) -> None:
        super().__init__()
        self.cross = cross
        shift = idx % 2

        if self.cross:
            self.norm1 = LayerNorm2d(dim)
            self.norm2 = LayerNorm2d(dim)
        else:
            self.norm = LayerNorm2d(dim)

        self.m = CSAttention(dim, num_heads=head, ffn_expansion_factor=ffn_expansion_factor, bias=False, shift=shift)

    def forward(self, x):
        if self.cross:
            x, y = x.chunk(2, dim=1)
            x = x + self.m(self.norm1(x), self.norm2(y))
        else:
            x = x + self.m(self.norm(x))

        return x


class Concertormer(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, ffn_expansion_factor=1, enc_blk_nums=[], dec_blk_nums=[], enc_heads=[], middle_heads=1, dec_heads=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)

        self.intros = nn.ModuleList()
        self.endings = nn.ModuleList()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.n_layer = len(enc_blk_nums)
        for i in range(self.n_layer):
            self.intros.append(
                nn.Conv2d(in_channels=img_channel, out_channels=width*(2**(i+1)), kernel_size=3, padding=1, stride=1, groups=1,
                          bias=True)
            )
            self.endings.append(
                nn.Conv2d(in_channels=width*(2**(self.n_layer-i)), out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                          bias=True)
            )

        chan = width
        for num, head in zip(enc_blk_nums, enc_heads):
            self.encoders.append(
                nn.Sequential(
                    *[CSABlock(chan, head, ffn_expansion_factor, i, cross=(i == 0 and len(self.encoders))) for i in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(*[CSABlock(chan, middle_heads, ffn_expansion_factor, i, cross=(i == 0)) for i in range(middle_blk_num)]) if middle_blk_num else nn.Identity()

        for num, head in zip(dec_blk_nums, dec_heads):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[CSABlock(chan, head, ffn_expansion_factor, i, cross=(i == 0)) for i in range(num)]
                )
            )

        self.padder_size = 2 ** (len(self.encoders) + 3)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        inps = []

        for i, intro in enumerate(self.intros):
            inps.append(F.interpolate(inp, scale_factor=0.5**(i+1), mode='bilinear'))

        encs = []

        for encoder, down, intro, _inp in zip(self.encoders, self.downs, self.intros, inps):
            x = encoder(x)
            encs.append(x)
            x = down(x)
            x = torch.cat([x, intro(_inp)], dim=1)

        x = self.middle_blks(x)

        decs = []

        for decoder, up, enc_skip, _end, _inp in zip(self.decoders, self.ups, encs[::-1], self.endings, inps[::-1]):
            decs.append(_end(x) + _inp)
            x = up(x)
            x = torch.cat([x, enc_skip], dim=1)
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        decs.append(x[:, :, :H, :W])
        return decs

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, CSAttention):
            attn = LocalCSAttention(dim=m.dim, num_heads=m.num_heads, ffn_expansion_factor=m.ffn_expansion_factor, bias=m.bias, shift=m.shift, base_size=base_size, fast_imp=False, train_size=train_size)
            setattr(model, n, attn)


class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)


class ConcertormerLocal(Local_Base, Concertormer):
    def __init__(self, *args, train_size=(1, 3, 256, 256), grid_factor=2., fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        Concertormer.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * grid_factor), int(W * grid_factor))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)