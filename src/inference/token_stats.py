import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from features.extractor import InternalModelSignal, TokenUncertaintyStat


class TokenStatProvider(Protocol):
    def collect(self, prompt: str, response: str) -> list[TokenUncertaintyStat]:
        ...

    def collect_signals(self, prompt: str, response: str) -> "CollectedModelSignals":
        ...


@dataclass(frozen=True)
class CollectedModelSignals:
    token_stats: list[TokenUncertaintyStat]
    internal_signal: InternalModelSignal | None = None


@dataclass(frozen=True)
class TransformersProviderConfig:
    model_id: str
    checkpoint_path: str | None = None
    device: str = "auto"
    torch_dtype: str = "auto"
    response_delimiter: str = "\n\n### Response:\n"
    max_memory: dict[int, str] | None = None
    enable_internal_features: bool = False
    selected_hidden_layers: tuple[int, ...] = (-1, -2)

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_memory", self._normalize_max_memory(self.max_memory))

    @classmethod
    def from_json(cls, path: str | Path) -> "TransformersProviderConfig":
        payload = json.loads(Path(path).read_text())
        return cls(
            model_id=payload["model_id"],
            checkpoint_path=payload.get("checkpoint_path"),
            device=payload.get("device", "auto"),
            torch_dtype=payload.get("torch_dtype", "auto"),
            response_delimiter=payload.get(
                "response_delimiter",
                "\n\n### Response:\n",
            ),
            max_memory=payload.get("max_memory"),
            enable_internal_features=payload.get("enable_internal_features", False),
            selected_hidden_layers=tuple(payload.get("selected_hidden_layers", [-1, -2])),
        )

    @property
    def model_source(self) -> str:
        return self.checkpoint_path or self.model_id

    @staticmethod
    def _normalize_max_memory(
        max_memory: dict[Any, str] | None,
    ) -> dict[int, str] | None:
        if max_memory is None:
            return None

        normalized: dict[int, str] = {}
        for key, value in max_memory.items():
            if isinstance(key, int):
                normalized[key] = value
                continue
            if isinstance(key, str) and key.isdigit():
                normalized[int(key)] = value
                continue
            raise ValueError(
                "max_memory keys must be integer GPU ids like 0 or 1, not cuda:N"
            )
        return normalized


@dataclass(frozen=True)
class GigaChatProviderConfig(TransformersProviderConfig):
    @classmethod
    def from_json(cls, path: str | Path) -> "GigaChatProviderConfig":
        payload = json.loads(Path(path).read_text())
        return cls(
            model_id=payload["model_id"],
            checkpoint_path=payload.get("checkpoint_path"),
            device=payload.get("device", "auto"),
            torch_dtype=payload.get("torch_dtype", "auto"),
            response_delimiter=payload.get(
                "response_delimiter",
                "\n\n### Response:\n",
            ),
            max_memory=payload.get("max_memory"),
            enable_internal_features=payload.get("enable_internal_features", False),
            selected_hidden_layers=tuple(payload.get("selected_hidden_layers", [-1, -2])),
        )


class TransformersTokenStatProvider:
    def __init__(
        self,
        *,
        config: TransformersProviderConfig,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config
        self._tokenizer = tokenizer
        self._model = model

    def collect(self, prompt: str, response: str) -> list[TokenUncertaintyStat]:
        return self.collect_signals(prompt=prompt, response=response).token_stats

    def collect_signals(self, prompt: str, response: str) -> CollectedModelSignals:
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise TypeError("prompt and response must be strings")
        if response == "":
            return CollectedModelSignals(token_stats=[], internal_signal=None)

        tokenizer = self._get_tokenizer()
        model = self._get_model()

        full_ids, response_token_indices = self._prepare_model_inputs(
            tokenizer=tokenizer,
            prompt=prompt,
            response=response,
        )
        if not response_token_indices:
            return CollectedModelSignals(token_stats=[], internal_signal=None)

        input_ids = torch.tensor(
            [full_ids],
            dtype=torch.long,
            device=self._input_device_for_model(model),
        )

        model_kwargs = {"input_ids": input_ids}
        if self.config.enable_internal_features:
            model_kwargs["output_hidden_states"] = True

        with torch.no_grad():
            outputs = model(**model_kwargs)

        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise ValueError("model output must contain logits")
        if logits.ndim != 3 or logits.shape[1] != len(full_ids):
            raise ValueError("logits shape does not align with input ids")

        stats: list[TokenUncertaintyStat] = []
        for token_index in response_token_indices:
            logit_index = token_index - 1
            token_id = full_ids[token_index]
            token_logits = logits[0, logit_index]
            log_probs = torch.log_softmax(token_logits, dim=-1)
            probs = torch.exp(log_probs)
            top2_probs = torch.topk(probs, k=min(2, probs.shape[0])).values
            top1_top2_margin = (
                float(top2_probs[0] - top2_probs[1]) if len(top2_probs) > 1 else 0.0
            )

            stats.append(
                TokenUncertaintyStat(
                    token=tokenizer.convert_ids_to_tokens([token_id])[0],
                    logprob=float(log_probs[token_id].item()),
                    entropy=float((-(probs * log_probs).sum()).item()),
                    top1_top2_margin=top1_top2_margin,
                )
            )
        internal_signal = None
        if self.config.enable_internal_features:
            internal_signal = self._extract_internal_signal(
                outputs=outputs,
                response_token_indices=response_token_indices,
            )

        return CollectedModelSignals(token_stats=stats, internal_signal=internal_signal)

    def _extract_internal_signal(
        self,
        *,
        outputs: Any,
        response_token_indices: list[int],
    ) -> InternalModelSignal:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("model output must contain hidden states")
        if not hidden_states:
            raise ValueError("hidden states must not be empty")

        selected_layers = self._select_hidden_layers(hidden_states)
        response_vectors = selected_layers[-1][0, response_token_indices, :]
        pooled_last_layer = response_vectors.mean(dim=0)

        selected_layer_norms = [
            float(layer[0, response_token_indices, :].mean(dim=0).norm().item())
            for layer in selected_layers
        ]
        layer_disagreements = []
        for previous_layer, current_layer in zip(selected_layers, selected_layers[1:]):
            previous_vector = previous_layer[0, response_token_indices, :].mean(dim=0)
            current_vector = current_layer[0, response_token_indices, :].mean(dim=0)
            layer_disagreements.append(
                float((current_vector - previous_vector).norm().item())
            )

        return InternalModelSignal(
            last_layer_pooled_l2=float(pooled_last_layer.norm().item()),
            last_layer_pooled_mean_abs=float(
                pooled_last_layer.abs().mean().item()
            ),
            selected_layer_norm_variance=float(
                self._variance(selected_layer_norms)
            ),
            layer_disagreement_mean=float(
                sum(layer_disagreements) / len(layer_disagreements)
                if layer_disagreements
                else 0.0
            ),
        )

    def _select_hidden_layers(
        self,
        hidden_states: tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor]:
        resolved_indices: list[int] = []
        for layer_index in self.config.selected_hidden_layers:
            resolved_index = layer_index
            if layer_index < 0:
                resolved_index = len(hidden_states) + layer_index
            if resolved_index < 0 or resolved_index >= len(hidden_states):
                continue
            if resolved_index not in resolved_indices:
                resolved_indices.append(resolved_index)

        if not resolved_indices:
            raise ValueError("selected hidden layers are out of range")
        return [hidden_states[index] for index in resolved_indices]

    def _prepare_model_inputs(
        self,
        *,
        tokenizer: Any,
        prompt: str,
        response: str,
    ) -> tuple[list[int], list[int]]:
        joined_text = self._join_prompt_and_response(prompt=prompt, response=response)
        prompt_char_end = len(prompt + self.config.response_delimiter)
        full_payload = self._tokenize_with_offsets(tokenizer, joined_text)
        full_ids = list(full_payload.get("input_ids", []))
        offsets = full_payload.get("offset_mapping")

        if full_ids and offsets:
            response_token_indices = [
                index
                for index, (_, end) in enumerate(offsets)
                if end > prompt_char_end
            ]
            if response_token_indices and response_token_indices[0] > 0:
                return full_ids, response_token_indices

        response_ids = self._encode_text(tokenizer, response)
        if not response_ids:
            return [], []

        prompt_ids = self._encode_text(tokenizer, prompt)
        prefix_ids = self._prefix_ids(tokenizer, prompt_ids)
        combined_ids = prefix_ids + response_ids
        response_start_index = len(prefix_ids)
        response_token_indices = list(range(response_start_index, len(combined_ids)))
        return combined_ids, response_token_indices

    def _join_prompt_and_response(self, *, prompt: str, response: str) -> str:
        return f"{prompt}{self.config.response_delimiter}{response}"

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_source,
                    trust_remote_code=True,
                )
            except Exception as error:
                raise RuntimeError(
                    f"failed to load token-stat provider from {self.config.model_source}"
                ) from error
        return self._tokenizer

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                model_load_kwargs: dict[str, Any] = {
                    "trust_remote_code": True,
                    "torch_dtype": self.config.torch_dtype,
                }
                if self._should_use_device_map_auto():
                    model_load_kwargs["device_map"] = "auto"
                    if self.config.max_memory is not None:
                        model_load_kwargs["max_memory"] = self.config.max_memory

                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_source,
                    **model_load_kwargs,
                )
                if not self._uses_dispatched_model(self._model):
                    self._model.to(self._resolved_device())
                self._model.eval()
            except Exception as error:
                raise RuntimeError(
                    f"failed to load token-stat provider from {self.config.model_source}"
                ) from error
        return self._model

    @staticmethod
    def _encode_text(tokenizer: Any, text: str) -> list[int]:
        encoded = tokenizer(text, add_special_tokens=False)
        return list(encoded["input_ids"])

    @staticmethod
    def _tokenize_with_offsets(tokenizer: Any, text: str) -> dict[str, Any]:
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
        except Exception:
            return {}
        return dict(encoded)

    @staticmethod
    def _prefix_ids(tokenizer: Any, prompt_ids: list[int]) -> list[int]:
        if prompt_ids:
            return prompt_ids

        if getattr(tokenizer, "bos_token_id", None) is not None:
            return [int(tokenizer.bos_token_id)]
        if getattr(tokenizer, "eos_token_id", None) is not None:
            return [int(tokenizer.eos_token_id)]

        raise ValueError("tokenizer must provide a BOS/EOS token for empty prompts")

    def _resolved_device(self) -> str:
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def _should_use_device_map_auto(self) -> bool:
        return self.config.device == "auto"

    def _input_device_for_model(self, model: Any) -> str:
        if self._uses_dispatched_model(model):
            return "cpu"
        return self._resolved_device()

    @staticmethod
    def _uses_dispatched_model(model: Any) -> bool:
        return bool(getattr(model, "hf_device_map", None))

    @staticmethod
    def _variance(values: list[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((value - mean) ** 2 for value in values) / len(values)


class GigaChatTokenStatProvider(TransformersTokenStatProvider):
    def __init__(
        self,
        *,
        config: GigaChatProviderConfig,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        super().__init__(config=config, tokenizer=tokenizer, model=model)
