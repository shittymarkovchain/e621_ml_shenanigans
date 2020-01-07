import torch

class Network(torch.nn.Module):
    def forward(self, favs, to_identify, fingerprint=None):
        if not fingerprint:
            favs = favs.float()
            to_identify = to_identify.float()
            x = favs
            for l in self.conv:
                x = l(x)
            x_mean = x.mean(dim=2)
            x_var = x.var(dim=2)
            inputs_mean = favs.mean(dim=2)
            x = torch.cat([x_mean, x_var, inputs_mean], dim=1)
            for l in self.features_extract:
                x = l(x)
            
            fingerprint = x
        
        user_extended = x.view(-1, 64, 1).repeat(1, 1, to_identify.shape[-1])
        x = torch.cat([to_identify, user_extended], dim=1)
        
        for l in self.is_fav:
            x = l(x)
        return x.reshape(x.shape[0], -1), fingerprint