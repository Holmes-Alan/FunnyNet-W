import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
        )
        self.layer2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x) * self.layer2(x)

        return x
    

class projection_net(nn.Module):
    def __init__(self, N, n_embedding_dim):
        super(projection_net, self).__init__()
        self.D = n_embedding_dim
        self.n = 16
        scale = 1.0
        self.audio_feat = nn.Sequential(
            linear(2048, 1024),
            linear(1024, 512),
            nn.LayerNorm(512)
        )
        self.vision_feat = nn.Sequential(
            linear(2048, 512), # large 2048, base 1536
            linear(512, 512),
            nn.LayerNorm(512)
        )
        self.sub_feat1= nn.LSTM(self.D, 256, batch_first=True)
        self.sub_feat2 = nn.Sequential(
            nn.Dropout(0.5),
            linear(256*64, 512),
            linear(512, 512),
            nn.LayerNorm(512)
        )
        self.ZA = CA_SA(dim=32)
        self.ZV = CA_SA(dim=32)
        self.ZT = CA_SA(dim=32)
        self.SA = CA_SA(dim=32)
        self.pre = nn.Sequential(
            linear(512, 2)
        )

    def forward(self, audio, vision, sub):
        audio_feat = self.audio_feat(audio)
        x = torch.mean(vision[:, 1:], dim=1)
        vision = torch.cat((x, vision[:, 0]), dim=1)
        vis_feat = self.vision_feat(vision)
        # sub_feat = sub.last_hidden_state.to(torch.float32)
        sub_feat, h = self.sub_feat1(sub)
        sub_feat = sub_feat.reshape(audio.shape[0], -1)
        sub_feat = self.sub_feat2(sub_feat)
        # CA
        b = audio_feat.shape[0]
        z_feat = torch.cat((F.normalize(audio_feat, dim=1), F.normalize(vis_feat, dim=1), F.normalize(sub_feat, dim=1)), dim=1)
        z_feat = z_feat.view(b, 3 * self.n, -1)
        feat_ZA = self.ZA(z_feat, audio_feat.view(b, self.n, -1))
        feat_ZV = self.ZV(z_feat, vis_feat.view(b, self.n, -1))
        feat_ZT = self.ZT(z_feat, sub_feat.view(b, self.n, -1))
        # SA
        feat = feat_ZA + feat_ZV + feat_ZT
        feat = self.SA(feat, feat) + feat
        # feat = feat.mean(dim=1)
        feat1, feat2, feat3 = feat.chunk(3, dim=1)
        feat = feat1.view(b,-1) + feat2.view(b,-1) + feat3.view(b,-1)
        prob = self.pre(feat)

        return audio_feat, vis_feat, sub_feat, prob





from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# from vit_module import Transformer


class CA_SA(nn.Module):
    def __init__(self, dim=32):
        super(CA_SA, self).__init__()
        self.dim = dim
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.Q = nn.Linear(dim, dim)
        self.attend = nn.Softmax(dim = -1)
    def forward(self, feat1, feat2):
        K = self.K(feat2)
        V = self.V(feat2)
        Q = self.Q(feat1)
        dots = torch.bmm(Q, K.permute(0, 2, 1))
        attn = self.attend(dots)
        out = torch.bmm(attn, V)

        # out = out.mean(dim=1)

        return out



class ContrastiveLossELI5(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        # if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")

        def l_ij(i, j):
            # z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            # if self.verbose: print("sim({i}, {j})={sim_i_j}")

            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * self.batch_size,)).to(emb_i.device).scatter_(0, torch.tensor([i]).to(emb_i.device), 0.0)
            # if self.verbose: print("1{{k!={i}}}", one_for_not_i)

            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )
            # if self.verbose: print("Denominator", denominator)

            loss_ij = -torch.log(numerator / denominator)
            # if self.verbose: print("loss({i},{j})={loss_ij}\n")

            return loss_ij.squeeze(0)

        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2 * N) * loss



