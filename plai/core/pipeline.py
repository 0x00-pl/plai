import typing


class Pass:
    def __init__(self, name=None):
        name = name or self.__class__.__name__
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
    def __init__(self, name: str = None, passes: typing.Sequence[Pass] = None, metadata: dict = None):
        super().__init__(name or f'pipeline')
        self.passes = list(passes) or []
        self.metadata = metadata or {}

    def __call__(self, graph) -> bool:
        changed = False
        for cur_pass in self.passes:
            # print(f'=== before pass {cur_pass.name} ===')
            # print(graph)

            step_changed = cur_pass(graph)
            changed = changed or step_changed
        return changed

    def __repr__(self):
        return f"Pipeline({repr(self.passes)})"

    def add_pass(self, cur_pass: Pass):
        self.passes.append(cur_pass)
