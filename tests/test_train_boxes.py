import unittest
import os, sys
import torch as th

print(os.getcwd())
# Allow for import directly from './code/prediction_models'
sys.path.append("./code/prediction_models")

from train_boxes import (
    box_loss_inclusion,
    box_loss_distance,
)
from box_embeddings.parameterizations import MinDeltaBoxTensor, BoxTensor
from box_embeddings.modules.volume import (
    BesselApproxVolume,
    Volume,
    HardVolume,
    SoftVolume,
)
from box_embeddings.modules.intersection import HardIntersection, GumbelIntersection


class TestExample(unittest.TestCase):
    def setUp(self):
        pass

    def test_example(self):
        self.assertTrue(True)


class TestBoxLoss(unittest.TestCase):
    def setUp(self) -> None:
        self._working_dir = os.getcwd()
        # print(f"Changing working directory to {self._working_dir}")
        os.chdir("./code/prediction_models")

        self.test_null_box = MinDeltaBoxTensor(
            th.tensor([(0.0, 0.0), (0.0, 0.0)]), beta=1000.0
        )
        self.test_flip_box = MinDeltaBoxTensor(
            th.tensor([(1.0, 1.0), (-1.0, -1.0)]), beta=1000.0
        )
        self.test_unit_box = MinDeltaBoxTensor(
            th.tensor([(0.0, 0.0), (1.0, 1.0)]), beta=1000.0
        )
        self.test_unit_box_2 = MinDeltaBoxTensor(
            th.tensor([(-0.3, 0.7), (1.0, 1.0)]), beta=1000.0
        )
        self.test_unit_box_3 = MinDeltaBoxTensor(
            th.tensor([(0.0, 0.0), (2.0, 0.5)]), beta=1000.0
        )

        self.volume = HardVolume(log_scale=False)
        self.hard_intersect = HardIntersection()

        pass

    def tearDown(self) -> None:
        # print(f"Changing back to {self._working_dir}")
        os.chdir(self._working_dir)

    def test_null_box_volume(self):
        self.assertAlmostEqual(
            self.volume(self.test_null_box).detach().item(), 0.0, places=4
        )

    def test_flip_box_volume(self):
        self.assertEqual(self.volume(self.test_flip_box), 0.0)

    def test_unit_box_volume(self):
        self.assertAlmostEqual(
            self.volume(self.test_unit_box).detach().item(), 1.0, places=4
        )

    def test_unit_box_volume_eq(self):
        self.assertEqual(
            self.volume(self.test_unit_box), self.volume(self.test_unit_box_2)
        )

    def test_unit_box_volume_stretched(self):
        self.assertAlmostEqual(
            self.volume(self.test_unit_box_3).detach().item(), 1.0, places=4
        )

    def test_hard_intersection(self):
        self.assertEqual(
            self.hard_intersect(self.test_unit_box, self.test_unit_box_3),
            MinDeltaBoxTensor(th.tensor([(0.0, 0.0), (1.0, 0.5)]), beta=1000.0),
        )

    def test_hard_intersection_volume(self):
        self.assertAlmostEqual(
            self.volume(self.hard_intersect(self.test_unit_box, self.test_unit_box_3)),
            0.5,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
