import torch
import os
import config

class ModelCheckpoint():
    def __init__(self, model, filepath, mode: str='min', verbose: bool=False):
        """
        model: model that needs saving
        filepath: path for saving (ex. path.pt)
        mode: 'min' - save when loss decreasing
              'max' - save when loss increasing
        verbose: show saving info
        """
        self.model = model
        self.filepath = filepath
        self.mode = mode
        self.verbose = verbose
        self.best_loss = float('inf') if mode == 'min' else -float('inf')
    
        os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    def __call__(self, current_loss):
        if (self.mode == 'min' and current_loss < self.best_loss) or (self.mode == 'max' and current_loss > self.best_loss):

            self.best_loss = current_loss
            torch.save(self.model.state_dict(), self.filepath)

            if self.verbose:
                print(f"Model was saved. Best loss: {self.best_loss:.2f}")