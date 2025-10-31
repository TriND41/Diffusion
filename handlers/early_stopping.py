from typing import Literal

class EarlyStopping:
    def __init__(self, n_patiences: int = 3, score_type: Literal['up', 'down'] = 'up') -> None:
        self.n_patiences = n_patiences
        self.best_score = None
        self.current_patiences = 0

        self.score_type = score_type

    def decrease_patiences(self) -> None:
        if self.current_patiences > 0:
            self.current_patiences -= 1

    def step(self, score: float) -> None:
        if self.best_score is None:
            self.best_score = score
        else:
            if self.best_score == score:
                self.current_patiences += 0.5
            elif self.score_type == 'up':
                if score > self.best_score:
                    self.best_score = score
                    self.decrease_patiences()
                else:
                    self.current_patiences += 1
            else:
                if score > self.best_score:
                    self.current_patiences += 1
                else:
                    self.best_score = score
                    self.decrease_patiences()

    def early_stop(self) -> bool:
        return self.current_patiences >= self.n_patiences