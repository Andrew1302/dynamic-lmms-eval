import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.vllm import VLLM as VLLMSimple
from lmms_eval.protocol import ChatMessages

LLM, _ = optional_import("vllm", "LLM")
SamplingParams, _ = optional_import("vllm", "SamplingParams")

WORKERS = int(os.getenv("WORKERS", "32"))


def _append_reasoning_prompt(messages: list, directive: str) -> None:
    """Append `directive` to the trailing text segment of the last user message.

    OpenAI-style content is either a plain string or a list of {type, text|image_url}
    parts. We mutate in place — the directive (e.g. "/no_think") needs to land at
    the very end of the user turn for Qwen3's chat template to pick it up.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = content.rstrip() + directive
            return
        if isinstance(content, list):
            for part in reversed(content):
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = part.get("text", "").rstrip() + directive
                    return
            content.append({"type": "text", "text": directive.lstrip()})
            return
        return


# InternVL3.5's documented thinking trigger is an R1-style *system* prompt
# (not an enable_thinking chat-template flag). When InternVL runs through this
# vllm wrapper (for batching/speed), thinking is enabled by prepending this as a
# system message. Passed via model_args as the preset token `system_prompt=internvl_r1`
# (the literal string can't survive comma/newline-delimited model_args parsing).
# Kept in sync with lmms_eval/models/simple/internvl3.py::R1_SYSTEM_PROMPT.
_INTERNVL_R1_SYSTEM_PROMPT = """You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section."""

# Prompt-tuning variants for reducing InternVL over-deliberation truncation.
# Selected per job via INTERNVL_R1_VARIANT (run_eval.sh) -> system_prompt=<preset>.
# V1 = the faithful R1 with a gentle commit rule swapped into the last paragraph.
_INTERNVL_R1_V1_COMMIT = """You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Work efficiently: reason only as much as the problem requires. As soon as you have determined the answer and checked it once, immediately close the </think> tag and give the final answer — do not re-examine the graph again or second-guess an answer you have already verified. The final answer should be standalone and not reference the thinking section."""

_INTERNVL_R1_V2_CONCISE = """You are a careful problem-solver. Reason through the problem step by step inside <think> and </think>, but keep it concise: do only the reasoning needed to determine the answer, verify it once, then immediately close </think> and give the final answer on its own line. Do not explore alternative approaches or re-derive an answer you have already reached."""

_INTERNVL_R1_V3_DECISIVE = """Reason step by step inside <think> and </think>, then give the answer. Be decisive: reach a conclusion, verify it once, and commit. Do not write "wait", "actually", "alternatively", or otherwise second-guess an answer you have already found — as soon as you have it, close </think> and state the final answer on its own line."""

_INTERNVL_R1_V4_STEPBUDGET = """Reason inside <think> and </think> using at most a few short steps — no more than necessary to reach the answer. Once you have the answer, verify it a single time, then close </think> and give the final answer on its own line. Never revisit earlier steps or restart your analysis."""

_SYSTEM_PROMPT_PRESETS = {
    "internvl_r1": _INTERNVL_R1_SYSTEM_PROMPT,
    "internvl_r1_v1": _INTERNVL_R1_V1_COMMIT,
    "internvl_r1_v2": _INTERNVL_R1_V2_CONCISE,
    "internvl_r1_v3": _INTERNVL_R1_V3_DECISIVE,
    "internvl_r1_v4": _INTERNVL_R1_V4_STEPBUDGET,
}


def _prepend_system_prompt(messages: list, text: str) -> None:
    """Insert a system message at the front of the conversation (in place)."""
    messages.insert(0, {"role": "system", "content": [{"type": "text", "text": text}]})


@register_model("vllm_chat")
class VLLM(VLLMSimple):
    is_simple = False

    def __init__(
        self,
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        tensor_parallel_size=1,
        data_parallel_size=1,
        gpu_memory_utilization=0.8,
        batch_size=1,
        max_frame_num=768,
        trust_remote_code=True,
        chat_template=None,
        max_pixels: int = 1605632,
        min_image_pixels=28,
        fps: Optional[int] = None,
        nframes: Optional[int] = 32,
        reasoning_prompt: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
        skip_special_tokens: bool = True,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model,
            tensor_parallel_size,
            data_parallel_size,
            gpu_memory_utilization,
            batch_size,
            max_frame_num,
            trust_remote_code,
            chat_template,
            min_image_pixels,
            **kwargs,
        )
        self.fps = fps
        self.max_pixels = max_pixels
        self.nframes = nframes
        # Two complementary ways to control thinking for reasoning models:
        # - reasoning_prompt: appends a text directive like "/no_think" to the
        #   user message (Qwen3 trained-behavior fallback). Fragile — depends
        #   on the model recognizing the literal token.
        # - chat_template_kwargs: forwarded to vllm's chat() so kwargs like
        #   {"enable_thinking": false} reach jinja's apply_chat_template; this
        #   is the official Qwen3 mechanism and pre-injects an empty
        #   <think></think> block at the prompt level.
        self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n") if reasoning_prompt else None
        if isinstance(chat_template_kwargs, str):
            try:
                chat_template_kwargs = json.loads(chat_template_kwargs)
            except json.JSONDecodeError as e:
                raise ValueError(f"chat_template_kwargs must be valid JSON: {chat_template_kwargs!r}") from e
        self.chat_template_kwargs = chat_template_kwargs or None
        # skip_special_tokens=False keeps model special tokens in the decoded
        # text. Needed for Gemma-4, whose reasoning is delimited by the special
        # tokens <|channel>thought ... <channel|> — with the default True those
        # delimiters are stripped and the answer can't be isolated from the
        # reasoning. The task yaml's reasoning_tags then strip that channel block.
        self.skip_special_tokens = str(skip_special_tokens).lower() not in ("false", "0", "no")
        # Optional system prompt, injected as a leading system message. Accepts a
        # preset token (e.g. "internvl_r1") or a literal string. Used to enable
        # InternVL3.5 thinking via its R1 system prompt when running through vllm.
        self.system_prompt = _SYSTEM_PROMPT_PRESETS.get(system_prompt, system_prompt) if system_prompt else None

    def make_one_request(self, request: Instance) -> Tuple[list[dict], dict]:
        """
        Build OpenAI-style messages and per-request sampling params from an Instance.
        Returns (messages, params_dict). Does not mutate input.
        """
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
        raw_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        chat_messages = ChatMessages(messages=raw_messages)
        # Copy to avoid side-effects across threads
        _gen = dict(gen_kwargs or {})
        _gen.setdefault("max_new_tokens", 4096)
        _gen.setdefault("temperature", 0)
        _gen.setdefault("top_p", 0.95)

        params = {
            "temperature": _gen["temperature"],
            "max_tokens": _gen["max_new_tokens"],
            "top_p": _gen["top_p"],
            "skip_special_tokens": self.skip_special_tokens,
        }
        # Native thinking-token budget (vllm SamplingParams): caps the reasoning
        # separately from the total output — when the budget is hit, vllm forces
        # the reasoning_end token (</think>) so the model must emit its answer,
        # which is never lost to truncation. Requires the engine to be built with
        # a reasoning_parser + reasoning_config (see simple/vllm.py). max_tokens
        # stays the total cap (budget + answer allowance).
        if _gen.get("thinking_token_budget") is not None:
            params["thinking_token_budget"] = int(_gen["thinking_token_budget"])

        video_kwargs = {
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_image_pixels,
            "max_frames": self.max_frame_num,
        }
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
        else:
            video_kwargs["nframes"] = self.nframes
        messages = chat_messages.to_openai_messages(video_kwargs=video_kwargs)
        if self.system_prompt:
            _prepend_system_prompt(messages, self.system_prompt)
        if self.reasoning_prompt:
            _append_reasoning_prompt(messages, self.reasoning_prompt)
        return messages, params

    def generate_until(self, requests) -> List[GenerationResult]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        total_elapsed_time = 0
        sample_token_counts: Optional[TokenCounts] = None
        for batch_requests in batched_requests:
            batched_messages = []
            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                futures = [executor.submit(self.make_one_request, request) for request in batch_requests]
                for future in futures:
                    messages, sampling_params = future.result()
                    batched_messages.append(messages)

            sampling_params = SamplingParams(**sampling_params)
            start_time = time.time()
            chat_kwargs = dict(
                sampling_params=sampling_params,
                messages=batched_messages,
                chat_template=self.chat_template,
            )
            if self.chat_template_kwargs is not None:
                chat_kwargs["chat_template_kwargs"] = self.chat_template_kwargs
            response = self.client.chat(**chat_kwargs)
            end_time = time.time()

            response_text = [o.outputs[0].text for o in response]
            # Record the exact generated (output) token count per response — vllm
            # gives us the token ids, so the saved jsonl carries real per-sample
            # output_tokens (used to size think budgets / measure reasoning length).
            response_tc = [TokenCounts(output_tokens=len(o.outputs[0].token_ids)) for o in response]

            # Calculate timing metrics for batch
            total_elapsed_time += end_time - start_time

            assert len(response_text) == len(batch_requests)
            res.extend([GenerationResult(text=resp_text, token_counts=tc) for resp_text, tc in zip(response_text, response_tc)])
            pbar.update(len(batch_requests))

        if not self.disable_log_stats:
            metrics = self.get_format_metrics()
            total_tokens = metrics["generation_tokens"]
            avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
            metric_dict = {
                "total_gen_tokens": total_tokens,
                "total_elapsed_time": total_elapsed_time,
                "avg_speed": avg_speed,
                "additional_metrics": {
                    "ttft": metrics["ttft"],
                    "tpot": metrics["tpot"],
                    "rank": self.rank,
                },
            }
            log_metrics(**metric_dict)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4V not support"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

    def get_format_metrics(self):
        metrics = self.client.get_metrics()
        ttft = 0
        tpot = 0
        generation_tokens = 0
        for metric in metrics:
            name = metric.name
            if "time_to_first_token" in name:
                ttft = metric.sum / metric.count
            if "time_per_output_token_seconds" in name:
                tpot = metric.sum / metric.count
            if name == "vllm:generation_tokens":
                generation_tokens = metric.value

        metrics = {
            "ttft": ttft,
            "tpot": tpot,
            "generation_tokens": generation_tokens,
        }

        return metrics
