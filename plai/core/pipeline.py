import typing


class Pass:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def __call__(self, x):
        return self.fn(x)

    def __repr__(self):
        return f"Pass({self.name})"


class Pipeline(Pass):
    def __init__(self, name: str = None, steps: typing.List[Pass] = None, metadata: dict = None):
        super().__init__(name or f'pipeline', self.call_steps)
        self.steps = steps or []
        self.metadata = metadata or {}

    def call_steps(self, x):
        for step in self.steps:
            x = step(x)
        return x
