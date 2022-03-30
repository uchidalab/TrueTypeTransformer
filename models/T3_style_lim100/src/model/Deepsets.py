import torch
import torch.nn as nn


class PermutationEquivariant(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermutationEquivariant, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        xm, _ = x.max(1, keepdim=True)
        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm
        return x


class DeepSets(nn.Module):
    def __init__(self, x_dim, d_dim):
        super(DeepSets, self).__init__()
        self.x_dim = x_dim
        self.d_dim = d_dim

        self.phi = nn.Sequential(
            PermutationEquivariant(self.x_dim, self.d_dim),
            nn.ELU(inplace=True),
        )

        self.rho = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.d_dim, self.d_dim),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.d_dim, 300),
        )
        print(self)

    def forward(self, x):
        phi_output = self.phi(x)
        sum_output = phi_output.mean(1)
        rho_output = self.rho(sum_output)
        return rho_output


if __name__ == '__main__':
    model = DeepSets(300, 300)
    test = torch.ones(23, 43, 300)
    print(model(test).size())
