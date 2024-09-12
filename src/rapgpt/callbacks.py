import numpy as np
from pathlib import Path
import torch
from rapgpt.config import Config
from rapgpt.model import TransformerModel
from loguru import logger

class Callback:
    pass

class Checkpoint(Callback):
    def __init__(self, config: Config, exp_name: str) -> None:
        self.save_path: Path = Path(config.checkpoint.save_path) / exp_name
        self.save_path.mkdir(exist_ok=True, parents=True)
        self.save_path_model = self.save_path / "model.pt"
        self.best_metric = np.infty

    def update(self, metric: float, model: TransformerModel) -> None:
        if metric >= self.best_metric:
            return
        self.best_metric = metric
        torch.save(model.state_dict(), self.save_path_model)
        logger.info("Best model updated")
