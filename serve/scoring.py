"""SCORE v1 validation. No tokenization, sampling, normalization or generation here."""
import math
import struct

MAX_PROMPT_TOKENS = 4096
MAX_SELECTED_TOKENS = 256
MAX_PENDING = 8
MAX_WIRE_BYTES = 65536


def capabilities(engine):
    info = getattr(engine, "info", {}) or {}
    return {"available": info.get("score") == 1 and info.get("score_version") == 1,
            "version": info.get("score_version"),
            "max_prompt_tokens": min(MAX_PROMPT_TOKENS, int(info.get("score_max_prompt") or MAX_PROMPT_TOKENS)),
            "max_selected_tokens": min(MAX_SELECTED_TOKENS, int(info.get("score_max_selected") or MAX_SELECTED_TOKENS)),
            "max_pending": min(MAX_PENDING, int(info.get("score_max_pending") or MAX_PENDING))}


def prepare(request, engine):
    if type(request.get("version")) is not int or request["version"] != 1:
        raise ValueError("score version must be 1")
    unknown = request.keys() - {"version", "prompt_token_ids", "selected_token_ids", "model",
                                "strata_priority", "strata_timeout_s"}
    if unknown:
        raise ValueError("unsupported score fields: " + ", ".join(sorted(unknown)))
    caps = capabilities(engine)
    prompt, selected = request.get("prompt_token_ids"), request.get("selected_token_ids")
    for name, tokens, limit in (("prompt_token_ids", prompt, min(caps["max_prompt_tokens"], engine.max_context)),
                                ("selected_token_ids", selected, caps["max_selected_tokens"])):
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= limit:
            raise ValueError(f"{name} must contain 1..{limit} tokens; never truncated")
        if any(type(token) is not int or not 0 <= token <= 2147483647 for token in tokens):
            raise ValueError(f"{name} must contain nonnegative int32 token IDs")
    payload = "v1 prompt=" + ",".join(map(str, prompt)) + " selected=" + ",".join(map(str, selected))
    if len(payload) + 72 > MAX_WIRE_BYTES:
        raise ValueError("score wire limit exceeded")
    # Own immutable-in-practice copies through terminal validation, not a caller's mutable lists.
    return list(prompt), list(selected), payload


def validate_terminal(status, result, request_id, prompt, selected):
    if (not isinstance(result, dict) or type(result.get("version")) is not int or result["version"] != 1 or
            result.get("request_id") != request_id or type(result.get("generated_tokens")) is not int or
            result["generated_tokens"] != 0):
        raise ValueError("score terminal identity/version/generated-token boundary mismatch")
    if status not in ("OK", "REJECT", "CANCELLED", "ERROR"):
        raise ValueError("unknown score terminal status")
    if status != "OK":
        if (set(result) != {"version", "request_id", "generated_tokens", "error"} or
                not isinstance(result["error"], str) or not result["error"]):
            raise ValueError("failed score terminal contains a partial option set or no error")
        return result
    receipt = {"prompt_tokens": len(prompt), "selected_tokens": len(selected),
               "logit_passes": 1, "dtype": "float32", "scale": 1}
    if (set(result) != {"version", "request_id", "generated_tokens", "prompt_tokens", "selected_token_ids",
                       "logits", "receipt"} or type(result.get("prompt_tokens")) is not int or
            result["prompt_tokens"] != len(prompt) or result.get("selected_token_ids") != selected or
            not isinstance(result["selected_token_ids"], list) or
            any(type(token) is not int for token in result["selected_token_ids"]) or
            result.get("receipt") != receipt or
            any(type(result["receipt"].get(key)) is not int for key in
                ("prompt_tokens", "selected_tokens", "logit_passes", "scale"))):
        raise ValueError("score prompt/selected/pass receipt boundary mismatch")
    logits = result.get("logits")
    if not isinstance(logits, list) or len(logits) != len(selected):
        raise ValueError("score terminal has an incomplete option set")
    rounded = []
    for value in logits:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("score terminal has non-finite or nonnumeric logits")
        try:
            fp32 = struct.unpack("!f", struct.pack("!f", value))[0]
        except (OverflowError, struct.error):
            raise ValueError("score terminal has out-of-range FP32 logits") from None
        if not math.isfinite(fp32):
            raise ValueError("score terminal has out-of-range FP32 logits")
        rounded.append(fp32)  # Restore FP32 from the native decimal's round-trip representation.
    return result | {"logits": rounded}
