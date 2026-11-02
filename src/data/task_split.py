"""Deterministic class-incremental splits; labels are positions in class_order."""

import numpy as np


class TaskSplit:
    def __init__(self, total_classes=100, initial_classes=50, increment=10,
                 class_order_seed=1993):
        if (not 1 <= initial_classes <= total_classes <= 100 or increment <= 0
                or (total_classes - initial_classes) % increment):
            raise ValueError('Require 1 <= initial <= total <= 100 and equal increments')
        # PyCIL uses the legacy MT19937 permutation, not default_rng.
        self.class_order = np.random.RandomState(class_order_seed).permutation(100).tolist()
        self.class_order = self.class_order[:total_classes]
        self.label_map = {original: mapped for mapped, original in enumerate(self.class_order)}
        self.tasks = [list(range(initial_classes))]
        self.tasks += [list(range(start, start + increment))
                       for start in range(initial_classes, total_classes, increment)]

    def remap(self, labels):
        try:
            return [self.label_map[int(label)] for label in labels]
        except KeyError as exc:
            raise ValueError(f'Class not in this protocol: {exc.args[0]}') from exc

    def classes(self, task, seen=False):
        if not 0 <= task < len(self.tasks):
            raise ValueError(f'Invalid task {task}; expected 0..{len(self.tasks) - 1}')
        return list(range(self.tasks[task][-1] + 1)) if seen else self.tasks[task]

    def indices(self, labels, task, seen=False):
        allowed = {self.class_order[i] for i in self.classes(task, seen)}
        return [i for i, label in enumerate(labels) if int(label) in allowed]