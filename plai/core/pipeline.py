import typing


class Pass:
    def __init__(self, name):
        self.name = name

    def __call__(self, graph) -> bool:
        """
        :param graph:
        :return: True when changed.
        """
        return False

    def __repr__(self):
        return f"Pass({self.name})"


class FnPass(Pass):
    def __init__(self, fn):
        super().__init__(fn.__name__)
        self.fn = fn

    def __call__(self, graph) -> bool:
        """
        :param graph:
        :return: True when changed.
        """
        return self.fn(graph)


class UntilStablePass(Pass):
    def __init__(self, name: str = None, step: Pass = None):
        super().__init__(name or f'until_stable')
        self.step = step

    def __call__(self, graph) -> bool:
        changed = True
        while changed:
            changed = self.step(graph)
        return changed

    def __repr__(self):
        return f"UntilStable({repr(self.step)})"


class Pipeline(Pass):
    def __init__(self, name: str = None, steps: typing.List[Pass] = None, metadata: dict = None):
        super().__init__(name or f'pipeline')
        self.steps = steps or []
        self.metadata = metadata or {}

    def __call__(self, graph) -> bool:
        changed = False
        for step in self.steps:
            step_changed = step(graph)
            changed = changed or step_changed
        return changed

    def __repr__(self):
        return f"Pipeline({repr(self.steps)})"

    def add_step(self, step: Pass):
        self.steps.append(step)



