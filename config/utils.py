import json

def read_cfg(cfg_file):
    with open(cfg_file, 'r') as f:
        cfg = json.loads(f.read())
        cfg = update_cfg(cfg)
    return cfg

def update_cfg(cfg):
    # this function should be extended everytime we increase the config version
    if cfg["config_version"] == 0:
        cfg = makeVer1(cfg)
    if cfg["config_version"] == 1:
        return cfg
    else:
        raise ValueError # unreachable state

def makeVer1(old):
    cfg = {}
    cfg["config_version"] = 1    
    
    model = {}
    model["hidden_size"] = old["hidden_size"]
    model["layer_size"] = old["layer_size"]
    model["dropout"] = old["dropout"]
    model["bidirectional"] = old["bidirectional"]
    model["use_attention"] = old["use_attention"]
    model["max_len"] = old["max_len"]
    cfg["model"] = model
    
    cfg["batch_size"] = old["batch_size"]
    cfg["workers"] = old["workers"]
    cfg["max_epochs"] = old["max_epochs"]
    cfg["lr"] = old["lr"]
    cfg["teacher_forcing"] = old["teacher_forcing"]
    return cfg
