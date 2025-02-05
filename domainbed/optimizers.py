import torch
from adabelief_pytorch import AdaBelief
from domainbed.opt.GENIE import GENIE


def get_optimizer(name, params, **kwargs):
    name = name.lower()
    optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW, 
                  "adabelief":AdaBelief, "genie": GENIE
                  }
    
    optim_cls = optimizers[name]
    print("--------------",optim_cls)

    return optim_cls(params, **kwargs)
