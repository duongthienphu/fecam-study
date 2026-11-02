import unittest

import numpy as np

from src.data.task_split import TaskSplit


class TaskSplitTests(unittest.TestCase):
    def test_paper_schedule_and_legacy_class_order(self):
        split = TaskSplit()
        expected = np.random.RandomState(1993).permutation(100).tolist()
        self.assertEqual(split.class_order, expected)
        self.assertEqual([len(t) for t in split.tasks], [50, 10, 10, 10, 10, 10])
        self.assertEqual(sorted(sum(split.tasks, [])), list(range(100)))
        self.assertEqual(split.remap(expected), list(range(100)))

    def test_current_train_and_seen_test_have_no_future_classes(self):
        split = TaskSplit(total_classes=10, initial_classes=4, increment=3)
        labels = split.class_order * 2
        current = split.indices(labels, task=1, seen=False)
        seen = split.indices(labels, task=1, seen=True)
        mapped = split.remap(labels)
        self.assertEqual({mapped[i] for i in current}, {4, 5, 6})
        self.assertEqual({mapped[i] for i in seen}, set(range(7)))
        self.assertTrue(set(current).isdisjoint(split.indices(labels, 0)))

    def test_invalid_schedules_labels_and_task_ids_fail(self):
        for args in ({'increment': 0}, {'total_classes': 10, 'initial_classes': 11},
                     {'total_classes': 10, 'initial_classes': 4, 'increment': 4}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                TaskSplit(**args)
        split = TaskSplit()
        for task in (-1, 6):
            with self.assertRaises(ValueError):
                split.indices([0], task)
        with self.assertRaises(ValueError):
            split.remap([100])


if __name__ == '__main__':
    unittest.main()