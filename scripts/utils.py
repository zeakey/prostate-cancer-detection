import numpy as np
import pickle, torch, io

def rescale(x, alpha):
    return (np.exp((1 - x) * alpha) - 1) / np.e**alpha

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)