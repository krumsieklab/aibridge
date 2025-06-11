"""
Wrapper for the OpenAI API.
Includes web search support and cost accounting.

This one is not compliant with the rest of the aibridge codebase.
"""
import os
import datetime
from dotenv import load_dotenv
from openai import OpenAI
import openai
from typing import Dict, Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

SYSTEM_PROMPT_DEFAULT = "You are a helpful AI assistant."


# ---------------------------------------------------------------------------
# Hard-coded pricing table (USD)
# ---------------------------------------------------------------------------
# All token prices are **per-1 000 tokens**.
# Web-search prices are **per individual tool call**.
MODEL_PRICING: Dict[str, Dict] = {
    # ── GPT-4o family ─────────────────────────────────────────────────────
    "gpt-4o-mini": {
        "input_per_1k": 0.0006,      # $0.60 / 1M
        "output_per_1k": 0.0024,     # $2.40 / 1M
        "search_per_call": {
            "low":    0.0250,
            "medium": 0.0275,        # default
            "high":   0.0300,
        },
    },
    "gpt-4o": {
        "input_per_1k": 0.0050,      # $5.00 / 1M
        "output_per_1k": 0.0200,     # $20.00 / 1M
        "search_per_call": {
            "low":    0.0300,
            "medium": 0.0350,        # default
            "high":   0.0500,
        },
    },

    # ── GPT-4.1 family ────────────────────────────────────────────────────
    "gpt-4.1": {
        "input_per_1k": 0.0020,      # $2.00 / 1M
        "output_per_1k": 0.0080,     # $8.00 / 1M
        "search_per_call": {
            "low":    0.0300,
            "medium": 0.0350,
            "high":   0.0500,
        },
    },
    "gpt-4.1-mini": {
        "input_per_1k": 0.0004,      # $0.40 / 1M
        "output_per_1k": 0.0016,     # $1.60 / 1M
        "search_per_call": {
            "low":    0.0250,
            "medium": 0.0275,
            "high":   0.0300,
        },
    },
    "gpt-4.1-nano": {
        "input_per_1k": 0.0001,      # $0.10 / 1M
        "output_per_1k": 0.0004,     # $0.40 / 1M
        "search_per_call": {
            "low":    0.0250,
            "medium": 0.0275,
            "high":   0.0300,
        },
    },

    # ── OpenAI "o" reasoning models ───────────────────────────────────────
    "o4-mini": {
        "input_per_1k": 0.0011,      # $1.10 / 1M
        "output_per_1k": 0.0044,     # $4.40 / 1M
        "search_per_call": {
            "low":    0.0250,
            "medium": 0.0275,
            "high":   0.0300,
        },
    },
    "o3": {
        "input_per_1k": 0.0100,      # $10.00 / 1M
        "output_per_1k": 0.0400,     # $40.00 / 1M
        "search_per_call": {
            "low":    0.0300,
            "medium": 0.0350,
            "high":   0.0500,
        },
    },
    "o3-mini": {
        "input_per_1k": 0.0011,      # $1.10 / 1M
        "output_per_1k": 0.0044,     # $4.40 / 1M
        "search_per_call": {
            "low":    0.0250,
            "medium": 0.0275,
            "high":   0.0300,
        },
    },
    "o1": {
        "input_per_1k": 0.0150,      # $15.00 / 1M
        "output_per_1k": 0.0600,     # $60.00 / 1M
        "search_per_call": {
            "low":    0.0300,
            "medium": 0.0350,
            "high":   0.0500,
        },
    },
    "o1-mini": {
        "input_per_1k": 0.0011,      # $1.10 / 1M
        "output_per_1k": 0.0044,     # $4.40 / 1M
        "search_per_call": {
            "low":    0.0250,
            "medium": 0.0275,
            "high":   0.0300,
        },
    },
    "o1-pro": {
        "input_per_1k": 0.1500,      # $150.00 / 1M
        "output_per_1k": 0.6000,     # $600.00 / 1M
        "search_per_call": {
            "low":    0.0300,
            "medium": 0.0350,
            "high":   0.0500,
        },
    },
}

REASONING_MODELS = {"o4-mini", "o3", "o3-mini", "o1"}
VALID_EFFORT      = {"low", "medium", "high"}

# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
class OpenAIWrapper:
    """Convenience wrapper around the Responses API.

    Parameters
    ----------
    model_name : {"gpt-4o-mini", "gpt-4o", "gpt-4.1"}, optional
        Defaults to "gpt-4o-mini".
    reasoning_effort : {"low","medium","high"} | None, optional
        Required **only** for models that support it (o4-mini, o3, o3-mini, o1)
        and **forbidden** for all other models.
    temperature : float | None, optional
        Controls randomness in the output. Higher values make output more random,
        lower values make it more deterministic. Must be between 0 and 2.
        Defaults to None (model default).
    logging_dir : str | None, optional
        If provided, every prompt/completion pair is written to this directory
        using the following scheme::

            YYYY_MM_DD__HH_MM_SS__a_prompt(.txt)
            YYYY_MM_DD__HH_MM_SS__a_prompt_websearch(.txt)  # web_search
            YYYY_MM_DD__HH_MM_SS__b_completion.txt

        The directory is created automatically if it does not exist.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        system_prompt: str = SYSTEM_PROMPT_DEFAULT,
        *,
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        logging_dir: Optional[str] = None,
    ) -> None:
        # ----------------------- model check -----------------------
        if model_name not in MODEL_PRICING:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Supported models: {', '.join(MODEL_PRICING)}"
            )

        # ----------------- reasoning-effort validation -------------
        if model_name in REASONING_MODELS:
            if reasoning_effort is None:
                raise ValueError(
                    f"Model '{model_name}' requires a reasoning_effort "
                    f"argument: one of {', '.join(sorted(VALID_EFFORT))}."
                )
            if reasoning_effort not in VALID_EFFORT:
                raise ValueError(
                    f"reasoning_effort must be one of "
                    f"{', '.join(sorted(VALID_EFFORT))}."
                )
        else:
            if reasoning_effort is not None:
                raise ValueError(
                    f"Model '{model_name}' does NOT support reasoning_effort."
                )

        # ----------------- temperature validation -------------
        if model_name in REASONING_MODELS and temperature is not None:
            raise ValueError(
                f"Model '{model_name}' does not support the temperature parameter."
            )

        # ----------------- temperature validation -------------
        if temperature is not None and not (0 <= temperature <= 2):
            raise ValueError("temperature must be between 0 and 2")

        # ---------------------------- init ---------------------------
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self._pricing = MODEL_PRICING[model_name]
        self.system_prompt = system_prompt

        # -------------------------- logging ---------------------------
        self.logging_dir: Optional[str] = logging_dir
        if self.logging_dir:
            os.makedirs(self.logging_dir, exist_ok=True)

        # ---------------------- cost accounting -----------------------
        self._token_tally: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._search_calls = 0
        self._search_cost_total = 0.0

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _log_pair(self, prompt: str, completion: str, *, is_websearch: bool) -> None:

        """Write prompt + completion to disk if ``logging_dir`` is set."""
        if not self.logging_dir:
            return  # no‑op when logging disabled

        ts = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S__")
        prompt_suffix = "a_prompt_websearch.txt" if is_websearch else "a_prompt.txt"
        comp_suffix = "b_completion.txt"
        prompt_path = os.path.join(self.logging_dir, f"{ts}{prompt_suffix}")
        comp_path = os.path.join(self.logging_dir, f"{ts}{comp_suffix}")

        # best-effort logging; unwrap any exceptions so user is not interrupted
        try:
            with open(prompt_path, "w", encoding="utf-8") as fp:
                fp.write(prompt)
            with open(comp_path, "w", encoding="utf-8") as fp:
                header = f"==== [{self.model_name}]"
                if self.reasoning_effort:
                    header += f" effort={self.reasoning_effort}"
                header += " ====\n\n"
                fp.write(header + completion)
        except Exception as exc:  # pragma: no cover – logging must never crash caller
            print(f"[OpenAIWrapper] Logging error: {exc}")

    # -----------------------------------------------------------------
    # Public helpers
    # -----------------------------------------------------------------
    def run_prompt_with_web_search(
        self,
        prompt: str,
        *,
        context_size: str = "medium",
        as_dict: bool = False,
    ):
        """One-shot query with the ``web_search_preview`` tool enabled."""
        if context_size not in {"low", "medium", "high"}:
            raise ValueError("context_size must be 'low', 'medium', or 'high'.")

        tools = [
            {
                "type": "web_search_preview",
                "search_context_size": context_size,
            }
        ]

        response = self.client.responses.create(
            model=self.model_name,
            instructions=self.system_prompt,
            input=prompt,
            tools=tools,
            **(
                {"reasoning_effort": self.reasoning_effort}
                if self.reasoning_effort
                else {}
            ),
            **({"temperature": self.temperature} if self.temperature is not None else {}),
        )

        completion_text = response.output_text
        # ------------------- meter token usage ---------------------
        if response.usage is not None:
            usage = (
                response.usage.model_dump(exclude_none=True)
                if hasattr(response.usage, "model_dump")
                else response.usage.dict(exclude_none=True)
            )
            p = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            c = usage.get("completion_tokens", usage.get("output_tokens", 0))
            self._token_tally["prompt_tokens"] += p
            self._token_tally["completion_tokens"] += c
            self._token_tally["total_tokens"] += usage.get("total_tokens", p + c)

        # ------------------- meter search cost ---------------------
        self._search_calls += 1
        self._search_cost_total += self._pricing["search_per_call"][context_size]

        # ----------------------- logging ---------------------------
        self._log_pair(prompt, completion_text, is_websearch=True)

        return {"text": completion_text} if as_dict else completion_text

    # ----------------------- update usage -----------------------
    def _update_usage(self, usage: Dict[str, int]) -> None:
        """Merge a usage block into the running tally."""
        p = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        c = usage.get("completion_tokens", usage.get("output_tokens", 0))
        self._token_tally["prompt_tokens"] += p
        self._token_tally["completion_tokens"] += c
        self._token_tally["total_tokens"] += usage.get("total_tokens", p + c)

    # prompt with no tools
    def run_prompt(
        self,
        prompt: str,
        response_schema: Optional[BaseModel] = None,
    ):
        """
        Unified helper that *always* calls the chat-completions endpoint.

        Parameters
        ----------
        prompt : str
            User input.
        response_schema : pydantic.BaseModel | None
            • If provided  → request JSON output and validate it with Pydantic.  
            • If None      → return the model's raw text.
        """

        # ------------- shared kwargs (apply to every call) -------------
        chat_args = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.reasoning_effort:
            chat_args["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            chat_args["temperature"] = self.temperature

        # ------------- choose branch -------------
        if response_schema is None:                       # ── plain text ────────────
            chat_resp = self.client.chat.completions.create(**chat_args)
            result              = chat_resp.choices[0].message.content
            completion_for_log  = result

        
        else:                                             # ── structured JSON ───────
            
            # One-shot call that returns a fully-parsed model instance
            chat_resp = self.client.beta.chat.completions.parse(
                **chat_args,                       # model + messages + opts
                response_format=response_schema,   # just the class
            )

            msg     = chat_resp.choices[0].message
            result  = msg.parsed                  # already typed (BaseModel)

            # keep a JSON string for the log
            completion_for_log = (
                result.model_dump_json(indent=2)
                if hasattr(result, "model_dump_json")  # Pydantic v2
                else result.json(indent=2)             # Pydantic v1
            )

        # ------------- cost accounting -------------
        if hasattr(chat_resp, "usage") and chat_resp.usage:
            self._update_usage(chat_resp.usage.dict(exclude_none=True))

        # ------------- prompt / completion logging -------------
        self._log_pair(prompt, completion_for_log, is_websearch=False)

        # ------------- single exit point -------------
        return result

    # ------------------------------------------------------------------
    # Accounting helpers
    # ------------------------------------------------------------------
    @property
    def token_usage(self) -> Dict[str, int]:
        """Return cumulative token counts (defensive copy)."""
        return dict(self._token_tally)

    @property
    def search_calls(self) -> int:
        """Return the number of search calls."""
        return self._search_calls

    def get_cost(self) -> float:
        """Return running estimated **USD** cost."""
        p_cost = (
            self._token_tally["prompt_tokens"] / 1000 * self._pricing["input_per_1k"]
        )
        c_cost = (
            self._token_tally["completion_tokens"]
            / 1000
            * self._pricing["output_per_1k"]
        )
        return p_cost + c_cost + self._search_cost_total

    # ------------------------------------------------------------------
    # Nicety for quick introspection
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        model_info = self.model_name
        if self.reasoning_effort:
            model_info += f" || reasoning={self.reasoning_effort}"
        return (
            f"<OpenAIWrapper model='{model_info}' "
            f"prompts={self._token_tally['prompt_tokens']} "
            f"completions={self._token_tally['completion_tokens']} "
            f"search_calls={self._search_calls} "
            f"cost=${self.get_cost():.4f}>"
        )
