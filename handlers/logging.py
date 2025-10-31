from typing import Optional, Dict

class Logger:
    def __init__(self, project: str, name: Optional[str] = None) -> None:
        global wandb
        import wandb
        wandb.init(project=project, name=name)

    def log(self, data: Dict[str, float], n_steps: int) -> None:
        wandb.log(data, n_steps)