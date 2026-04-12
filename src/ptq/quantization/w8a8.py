"""Simulation-only W8A8 modules used in tests and local validation."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptq.config import (
    ArtifactQuantizationSettings,
    PTQRunConfig,
    QuantizationMethod,
)
from ptq.quantization.primitives import (
    activation_reduction_axes,
    simulate_quantization,
    weight_reduction_axes,
)


class StaticActivationObserver(nn.Module):
    """Track a simple calibration statistic for static int8 activations."""

    def __init__(self, settings: ArtifactQuantizationSettings) -> None:
        super().__init__()
        self.settings = settings
        self.register_buffer("amax", torch.tensor(0.0), persistent=False)

    def observe(self, activations: torch.Tensor) -> None:
        """Update the running activation max with a new calibration batch."""
        current = activations.detach().abs().amax()
        self.amax = torch.maximum(self.amax, current)

    def freeze(self) -> torch.Tensor:
        """Convert the observed range into the serialized int8 scale."""
        if self.settings.dtype.value != "int8":
            return torch.tensor(0.0, device=self.amax.device)
        return torch.clamp(self.amax / 127.0, min=1e-12)


class SimulatedW8A8Linear(nn.Module):
    """Wrap ``nn.Linear`` with fake weight and activation quantization."""

    def __init__(
        self,
        linear: nn.Linear,
        weight_settings: ArtifactQuantizationSettings,
        activation_settings: ArtifactQuantizationSettings,
        method: QuantizationMethod | str,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(
                linear.bias.detach().clone(),
                requires_grad=False,
            )

        self.weight_settings = weight_settings
        self.activation_settings = activation_settings
        self.method = QuantizationMethod(method)
        self.observer = (
            StaticActivationObserver(activation_settings)
            if self.method is QuantizationMethod.STATIC and activation_settings.enabled
            else None
        )
        self._static_activation_scale: torch.Tensor | None = None
        self._quantized_weight, self._weight_scale = self._quantize_weight()

    def _quantize_weight(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        axis = None
        if self.weight_settings.dtype.value == "int8":
            axis = weight_reduction_axes(
                self.weight,
                self.weight_settings.granularity,
            )
        return simulate_quantization(
            self.weight,
            self.weight_settings.dtype,
            self.weight_settings.granularity,
            axis,
        )

    def calibrate(self, activations: torch.Tensor) -> None:
        """Feed activations into the static observer when calibration is enabled."""
        if self.observer is not None:
            self.observer.observe(activations)

    def freeze(self) -> None:
        """Finalize the static activation scale after calibration finishes."""
        if self.observer is not None:
            self._static_activation_scale = self.observer.freeze()

    def _quantize_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """Apply the configured activation quantization path."""
        if not self.activation_settings.enabled:
            return activations

        if self.method is QuantizationMethod.DYNAMIC:
            axis = None
            if self.activation_settings.dtype.value == "int8":
                axis = activation_reduction_axes(
                    activations,
                    self.activation_settings.granularity,
                )
            quantized, _scale = simulate_quantization(
                activations,
                self.activation_settings.dtype,
                self.activation_settings.granularity,
                axis,
            )
            return quantized

        if self.method is QuantizationMethod.STATIC:
            if self.activation_settings.dtype.value == "fp8":
                quantized, _scale = simulate_quantization(
                    activations,
                    self.activation_settings.dtype,
                    self.activation_settings.granularity,
                )
                return quantized

            if self._static_activation_scale is None:
                raise RuntimeError("static activation scale has not been frozen yet")

            scale = self._static_activation_scale
            q = torch.clamp(torch.round(activations / scale), -127, 127)
            return q * scale

        return activations

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the linear layer on quantized inputs and quantized weights."""
        quantized_inputs = self._quantize_activations(inputs)
        return F.linear(quantized_inputs, self._quantized_weight, self.bias)


def _replace_named_module(
    model: nn.Module,
    module_name: str,
    new_module: nn.Module,
) -> None:
    """Replace a submodule given its dotted module path."""
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_simulated_w8a8_to_linear(
    model: nn.Module,
    config: PTQRunConfig,
) -> nn.Module:
    """Clone a model and replace each linear layer with a simulated W8A8 layer."""
    cloned = copy.deepcopy(model)
    for name, module in list(cloned.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        replacement = SimulatedW8A8Linear(
            linear=module,
            weight_settings=config.artifacts.weights,
            activation_settings=config.artifacts.activations,
            method=config.method.name,
        )
        _replace_named_module(cloned, name, replacement)
    return cloned


def calibrate_simulated_model(
    model: nn.Module,
    sample_inputs: list[torch.Tensor],
) -> None:
    """Run calibration inputs through all simulated W8A8 layers and freeze them."""
    for module in model.modules():
        if isinstance(module, SimulatedW8A8Linear):
            for sample in sample_inputs:
                module.calibrate(sample)
            module.freeze()
