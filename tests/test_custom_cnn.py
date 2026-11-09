"""Synthetic offline CNN checks, not face-identification performance results.

Run from the repository root: python -m unittest tests.test_custom_cnn
Requires only PyTorch and the standard library; no datasets or weights.
"""

import io
import unittest

import torch
from torch import nn

from src.models.custom_cnn import CustomCNN, build_custom_cnn


class CustomCNNTests(unittest.TestCase):
    def test_output_shape_with_configurable_class_count(self):
        images = torch.randn(2, 3, 128, 128)
        for num_classes in (2, 5):
            with self.subTest(num_classes=num_classes):
                model = build_custom_cnn(num_classes).eval()
                with torch.no_grad():
                    logits = model(images)
                self.assertEqual(tuple(logits.shape), (2, num_classes))
                self.assertTrue(torch.isfinite(logits).all().item())

    def test_training_step_updates_weights_and_batchnorm(self):
        torch.manual_seed(42)
        model = build_custom_cnn(3).train()
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        batchnorms = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        self.assertEqual(len(batchnorms), 4)
        counters = [m.num_batches_tracked.clone() for m in batchnorms]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.classifier[-1].weight.detach().clone()
        images = torch.randn(2, 3, 128, 128)
        labels = torch.tensor([0, 2], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.CrossEntropyLoss()(model(images), labels)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
        self.assertGreater(model.features[0][0].weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.classifier[-1].weight))
        for module, counter in zip(batchnorms, counters):
            self.assertEqual(module.num_batches_tracked.item(), counter.item() + 1)

    def test_eval_is_deterministic_and_preserves_batchnorm_buffers(self):
        model = build_custom_cnn(3, dropout=0.5).train()
        dropout = model.classifier[1]
        self.assertTrue(dropout.training)
        self.assertEqual(dropout.p, 0.5)
        model.eval()
        self.assertFalse(dropout.training)
        snapshots = {
            name: buffer.clone()
            for name, buffer in model.named_buffers()
        }
        with torch.no_grad():
            images = torch.randn(1, 3, 128, 128)
            first = model(images)
            second = model(images)
        self.assertTrue(torch.equal(first, second))
        for name, buffer in model.named_buffers():
            self.assertTrue(torch.equal(buffer, snapshots[name]), name)

    def test_final_layer_returns_logits_without_softmax(self):
        model = build_custom_cnn(3).eval()
        # Make the head produce scores that cannot be probability values.
        expected = torch.tensor([-2.0, 0.0, 3.0])
        with torch.no_grad():
            model.classifier[-1].weight.zero_()
            model.classifier[-1].bias.copy_(expected)
            logits = model(torch.randn(1, 3, 128, 128))
        torch.testing.assert_close(logits, expected.unsqueeze(0))

    def test_seed_and_state_dict_reproduce_predictions(self):
        torch.manual_seed(42)
        model = build_custom_cnn(3).eval()
        torch.manual_seed(42)
        restored = build_custom_cnn(3).eval()
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, dict(restored.named_parameters())[name]), name)
        # Check serialization in memory; do not create a checkpoint artifact.
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        with torch.no_grad():
            restored.classifier[-1].weight.zero_()
        buffer.seek(0)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            images = torch.randn(1, 3, 128, 128)
            torch.testing.assert_close(model(images), restored(images))

    def test_invalid_arguments_are_rejected(self):
        for constructor in (CustomCNN, build_custom_cnn):
            for num_classes in (-1, 0, 1):
                with self.subTest(constructor=constructor, num_classes=num_classes):
                    with self.assertRaises(ValueError):
                        constructor(num_classes)
            for num_classes in (True, 2.5, "3", None):
                with self.subTest(constructor=constructor, num_classes=num_classes):
                    with self.assertRaises(TypeError):
                        constructor(num_classes)
            for dropout in (-0.1, 1.0, float("nan"), float("inf")):
                with self.subTest(constructor=constructor, dropout=dropout):
                    with self.assertRaises(ValueError):
                        constructor(3, dropout=dropout)
            for dropout in (True, "0.3", None):
                with self.subTest(constructor=constructor, dropout=dropout):
                    with self.assertRaises(TypeError):
                        constructor(3, dropout=dropout)


if __name__ == "__main__":
    unittest.main()
