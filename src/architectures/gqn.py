import torch
from torch import nn
from torch.nn import functional as F


class GQN_Pool(nn.Module):
    def __init__(self, representation_dim, covariate_dim, net=None, **kwargs):
        assert representation_dim == 256
        assert (isinstance(net, nn.Identity) or (net is None))
        super(GQN_Pool, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256 + covariate_dim, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256 + covariate_dim, 128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.pool = nn.AvgPool2d(16)

    def forward(self, x, xi=None):
        # Residual connection
        skip_in = F.relu(self.conv1(x))
        skip_out = F.relu(self.conv2(skip_in))

        r = F.relu(self.conv3(skip_in))
        r = F.relu(self.conv4(r)) + skip_out

        if xi is not None:
            # You should set covariate_dim=0 to exercise this branch
            # Broadcast / upsample
            xi = xi.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 16, 16)

            # Residual connection
            # Concatenate
            skip_in = torch.cat((r, xi), dim=1)
        else:
            skip_in = r
        skip_out = F.relu(self.conv5(skip_in))

        r = F.relu(self.conv6(skip_in))
        r = F.relu(self.conv7(r)) + skip_out
        r = F.relu(self.conv8(r))

        # Pool
        r = self.pool(r)

        r = r.squeeze(-1).squeeze(-1)
        return r
