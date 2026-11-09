"""Offline ResNet18 contract checks, not recognition-performance evaluation.

Run from the repository root: python -m unittest tests.test_resnet18
All real networks use pretrained=False; mocked checks never load real weights.
"""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from torchvision.models import ResNet18_Weights

from src.models.resnet18 import build_resnet18


class ResNet18Tests(unittest.TestCase):
    def test_training_modes_gradients_and_batchnorm(self):
        # Use a small synthetic batch at the intended RGB resolution.
        torch.manual_seed(42)
        images = torch.randn(2, 3, 128, 128)
        labels = torch.tensor([0, 2], dtype=torch.long)
        for mode in ("none", "layer4", "all"):
            with self.subTest(mode=mode):
                model = build_resnet18(3, pretrained=False, trainable_backbone=mode)
                model.eval()
                self.assertTrue(all(not child.training for child in model.modules()))
                # Exercise repeated eval/train transitions, not just construction.
                self.assertIs(model.train(), model)
                model.train()

                for name, parameter in model.network.named_parameters():
                    expected = (
                        name.startswith("fc.")
                        or mode == "all"
                        or (mode == "layer4" and name.startswith("layer4."))
                    )
                    self.assertEqual(parameter.requires_grad, expected, name)

                snapshots = {}
                for name, module in model.network.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        expected_training = mode == "all" or (
                            mode == "layer4" and name.startswith("layer4.")
                        )
                        self.assertEqual(module.training, expected_training, name)
                        snapshots[name] = (
                            module.running_mean.clone(),
                            module.running_var.clone(),
                            module.num_batches_tracked.clone(),
                        )

                logits = model(images)
                self.assertEqual(tuple(logits.shape), (2, 3))
                self.assertTrue(torch.isfinite(logits).all().item())
                # Direct network output: the wrapper must not apply softmax.
                with patch.object(model.network, "forward", return_value=logits):
                    self.assertIs(model(images), logits)
                loss = nn.CrossEntropyLoss()(logits, labels)
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                    else:
                        self.assertIsNone(parameter.grad, name)

                for name, module in model.network.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        mean, variance, batches = snapshots[name]
                        if module.training:
                            self.assertEqual(module.num_batches_tracked.item(), batches.item() + 1)
                        else:
                            self.assertTrue(torch.equal(module.running_mean, mean), name)
                            self.assertTrue(torch.equal(module.running_var, variance), name)
                            self.assertTrue(torch.equal(module.num_batches_tracked, batches), name)
                model.eval()
                self.assertTrue(all(not child.training for child in model.modules()))

    def test_policy_can_change_without_rebuilding_weights(self):
        model = build_resnet18(3, pretrained=False)
        head = model.network.fc
        model.set_trainable_backbone("layer4")
        self.assertTrue(model.network.layer4[0].conv1.weight.requires_grad)
        self.assertTrue(model.network.layer4[0].bn1.training)
        self.assertFalse(model.network.layer3[0].bn1.training)
        parameter = model.network.layer4[0].conv1.weight
        parameter.grad = torch.ones_like(parameter)
        model.set_trainable_backbone("none")
        self.assertIsNone(parameter.grad)
        self.assertFalse(parameter.requires_grad)
        self.assertFalse(model.network.layer4[0].bn1.training)
        self.assertIs(model.network.fc, head)
        model.eval()
        model.set_trainable_backbone("all")
        self.assertFalse(model.network.bn1.training)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        with self.assertRaises(ValueError):
            model.set_trainable_backbone("invalid")
        self.assertEqual(model.trainable_backbone, "all")

    def test_invalid_arguments_fail_before_weight_loading(self):
        with patch("src.models.resnet18.resnet18") as factory:
            for value in (-1, 0, 1):
                with self.subTest(num_classes=value), self.assertRaises(ValueError):
                    build_resnet18(value)
            for value in (True, 2.5, "3", None):
                with self.subTest(num_classes=value), self.assertRaises(TypeError):
                    build_resnet18(value)
            for mode in ("invalid", "Layer4", "", None):
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    build_resnet18(3, trainable_backbone=mode)
            with self.assertRaises(TypeError):
                build_resnet18(3, pretrained="true")
            factory.assert_not_called()

    def test_offline_builder_passes_no_weights(self):
        with patch("src.models.resnet18.resnet18") as factory:
            factory.return_value = nn.Module()
            factory.return_value.fc = nn.Linear(7, 1000)
            model = build_resnet18(3, pretrained=False)
            factory.assert_called_once_with(weights=None)
            self.assertEqual(model.network.fc.in_features, 7)
            self.assertEqual(model.network.fc.out_features, 3)

    def test_pretrained_failure_propagates_without_random_fallback(self):
        # Verify routing/failure handling without any network request.
        with patch("src.models.resnet18.resnet18", side_effect=OSError("offline")) as factory:
            with self.assertRaisesRegex(OSError, "offline"):
                build_resnet18(3, pretrained=True)
            factory.assert_called_once_with(weights=ResNet18_Weights.DEFAULT)


if __name__ == "__main__":
    unittest.main()
