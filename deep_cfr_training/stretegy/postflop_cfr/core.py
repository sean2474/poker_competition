from interface.model import PostflopModel


class Postflop(PostflopModel):
    def __init__(self):
        pass

    def action(self, hand: list, board: list, history: str,
               hero_range, opp_range) -> tuple[str, dict]:
        return 'c', {'c': 1.0}

    def train(self, **kwargs):
        pass