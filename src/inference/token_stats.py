import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from features.extractor import TokenUncertaintyStat


class TokenStatProvider(Protocol):
    def collect(self, prompt: str, response: str) -> list[TokenUncertaintyStat]:
        ...


@dataclass(frozen=True)
class TransformersProviderConfig:
    model_id: str
    checkpoint_path: str | None = None
    device: str = "auto"
    torch_dtype: str = "auto"
    response_delimiter: str = "\n\n### Response:\n"

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
        )

    @property
    def model_source(self) -> str:
        return self.checkpoint_path or self.model_id


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
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise TypeError("prompt and response must be strings")
        if response == "":
            return []

        tokenizer = self._get_tokenizer()
        model = self._get_model()

        full_ids, response_token_indices = self._prepare_model_inputs(
            tokenizer=tokenizer,
            prompt=prompt,
            response=response,
        )
        if not response_token_indices:
            return []

        input_ids = torch.tensor(
            [full_ids],
            dtype=torch.long,
            device=self._resolved_device(),
        )

        with torch.no_grad():
            outputs = model(input_ids=input_ids)

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
        return stats

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
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_source,
                    trust_remote_code=True,
                    torch_dtype=self.config.torch_dtype,
                )
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


class GigaChatTokenStatProvider(TransformersTokenStatProvider):
    def __init__(
        self,
        *,
        config: GigaChatProviderConfig,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        super().__init__(config=config, tokenizer=tokenizer, model=model)
