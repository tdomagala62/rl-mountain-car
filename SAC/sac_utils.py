import json
import numpy as np
import dacite
from dataclasses import dataclass, field

@dataclass
class SAC_Database:
    #observation state 1 min,max == [-1.2 0.6]
    #observation state 2 min,max == [-0.07 0.07]
    NOISE_STD:np.ndarray = field(default_factory=lambda: np.array([0.15, 0.03]))
    PROCESS_NOISE:np.ndarray = field(default_factory=lambda: np.array([1e-3, 1e-5]))
    FRAME_STACK_N:int = 4


@dataclass
class SAC_TRAIN_ENV:
    verbose:int = 0
    learning_rate:float = 1e-3
    buffer_size:int = 100_000
    batch_size:int = 256
    ent_coef:str = "auto"
    gamma:float = 0.99
    tau:float =  0.005
 
 
DACITE_CONFIG = dacite.Config(
    type_hooks={np.ndarray: np.array},
    strict=True,  
)
 
 
def load_config(path: str, cfg_class: type) -> object:
    with open(path, "r") as f:
        data = json.load(f)
    return dacite.from_dict(data_class=cfg_class, data=data, config=DACITE_CONFIG)


def main():
    data_path = "./SAC/sac_json/sac_data_1.json"
    print(SAC_DATABASE())
    modified = load_config(data_path, SAC_DATABASE)
    print(modified)

if __name__ == "__main__":
    main()