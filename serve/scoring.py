"""SCORE v1 validation. No tokenization, sampling, normalization or generation here."""
import hashlib
import json
import math
import struct
import re
import time

MAX_PROMPT_TOKENS = 4096
MAX_SELECTED_TOKENS = 256
MAX_PENDING = 8
MAX_WIRE_BYTES = 65536
MAX_GROUP_QUESTIONS = 64
MAX_GROUP_CHILDREN = 512
MAX_GROUP_BYTES = 4 << 20
MAX_GROUP_PROMPT_TOKENS = 64 * MAX_PROMPT_TOKENS
MAX_GROUP_SELECTED_TOKENS = 16384
MAX_GROUP_INPUT_BYTES = 1 << 20
GROUP_DRAIN_S = 32.0  # Native drain (30 s), plus watcher/control handoff; not extra model work.
MAX_CALLER_GROUPS = 4
MAX_CALLER_PROMPT_TOKENS = 2 * MAX_GROUP_PROMPT_TOKENS
MAX_PENDING_PROMPT_TOKENS = 4 * MAX_GROUP_PROMPT_TOKENS
MAX_CALLER_GROUP_BYTES = 2 * MAX_GROUP_BYTES
MAX_PENDING_GROUP_BYTES = 4 * MAX_GROUP_BYTES
MAX_PRE_CANCELS = 128
MAX_CALLER_PRE_CANCELS = 32
FOREGROUND_CHILD_BURST = 3
MAX_RETIRED_IDENTITIES = 128
MAX_CALLER_RETIRED_IDENTITIES = 64


def validate_operation_id(value):
    if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", value)):
        raise ValueError("X-Strata-Operation must be 1..128 ASCII identifier characters")
    return value


def group_work(request):
    children = request["children"]
    return {"groups": 1, "children": len(children),
            "prompt_tokens": sum(len(c["prompt_token_ids"]) for c in children),
            "selected_tokens": sum(len(c["selected_token_ids"]) for c in children),
            "compiled_bytes": len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def validate_controls(request, max_timeout=300.0):
    priority = request.get("strata_priority", "foreground")
    if priority not in ("foreground", "background"):
        raise ValueError("strata_priority must be foreground or background")
    timeout = request.get("strata_timeout_s", max_timeout)
    if (type(timeout) not in (int, float) or not math.isfinite(timeout) or
            not 0 < timeout <= max_timeout):
        raise ValueError("strata_timeout_s must be finite, positive and no greater than the service bound")
    return timeout, priority


def validate_group_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise ValueError("invalid score group/child ID")
    return value


def prepare_group(request, engine):
    """Validate every child and aggregate bound before any native SCORE can be issued."""
    if set(request) - {"version", "group", "children", "strata_timeout_s", "strata_priority"}:
        raise ValueError("unsupported score group fields")
    if type(request.get("version")) is not int or request["version"] != 1:
        raise ValueError("score version must be 1")
    validate_controls(request)
    group = request.get("group")
    fields = {"id", "scoring_identity_sha256", "bundle_sha256", "request_sha256", "token_plan_sha256",
              "readout_family", "temperature", "question_count", "deadline_unix_s", "manifest_sha256"}
    if not isinstance(group, dict) or set(group) != fields:
        raise ValueError("missing or unsupported score group identity")
    validate_group_id(group["id"])
    if (type(group["deadline_unix_s"]) not in (int, float) or not math.isfinite(group["deadline_unix_s"]) or
            group["deadline_unix_s"] > time.time() + 300):
        raise ValueError("score group deadline must be a finite absolute Unix time within 300 seconds")
    for key in ("scoring_identity_sha256", "bundle_sha256", "request_sha256",
                "token_plan_sha256", "manifest_sha256"):
        if not isinstance(group[key], str) or not re.fullmatch(r"[0-9a-f]{64}", group[key]):
            raise ValueError("invalid score group pinned digest")
    if group["readout_family"] not in ("direct_letter_logits", "per_option_yes_no_log_odds"):
        raise ValueError("invalid score group readout family")
    if type(group["temperature"]) not in (int, float) or not math.isfinite(group["temperature"]) or group["temperature"] <= 0:
        raise ValueError("invalid score group temperature")
    count = group["question_count"]
    if type(count) is not int or not 1 <= count <= MAX_GROUP_QUESTIONS:
        raise ValueError("score group question bound exceeded")
    children = request.get("children")
    if not isinstance(children, list) or not 1 <= len(children) <= MAX_GROUP_CHILDREN:
        raise ValueError("score group child bound exceeded")
    if len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > MAX_GROUP_BYTES:
        raise ValueError("score group compiled byte bound exceeded")
    plans, seen, token_plans = [], set(), [[] for _ in range(count)]
    prompt_work = selected_work = 0
    previous = (0, -1)
    for child in children:
        if not isinstance(child, dict) or set(child) != {
                "id", "question_index", "branch_index", "plan_sha256", "prompt_token_ids", "selected_token_ids"}:
            raise ValueError("missing or unsupported score child fields")
        child_id = validate_group_id(child["id"])
        q, branch = child["question_index"], child["branch_index"]
        if (type(q) is not int or type(branch) is not int or not 0 <= q < count or
                (q, branch) not in ((previous[0], previous[1] + 1), (previous[0] + 1, 0)) or
                child_id in seen or not isinstance(child["plan_sha256"], str) or
                not re.fullmatch(r"[0-9a-f]{64}", child["plan_sha256"])):
            raise ValueError("duplicate, reordered or stale score child identity")
        if group["readout_family"] == "direct_letter_logits" and branch != 0:
            raise ValueError("direct readout must have exactly one child per question")
        seen.add(child_id)
        previous = q, branch
        token_plans[q].append(child["plan_sha256"])
        plan = {"version": 1, "prompt_token_ids": child["prompt_token_ids"],
                "selected_token_ids": child["selected_token_ids"]}
        prompt, selected, _ = prepare(plan, engine)
        if group["readout_family"] == "per_option_yes_no_log_odds" and len(selected) != 2:
            raise ValueError("yes/no readout must select exactly two tokens")
        prompt_work += len(prompt)
        selected_work += len(selected)
        plans.append(plan)
    if (any(not row for row in token_plans) or fingerprint(token_plans) != group["token_plan_sha256"] or
            fingerprint({k: v for k, v in group.items() if k != "manifest_sha256"} | {"children": children}) != group["manifest_sha256"]):
        raise ValueError("stale score group/child manifest")
    if prompt_work > MAX_GROUP_PROMPT_TOKENS or selected_work > MAX_GROUP_SELECTED_TOKENS:
        raise ValueError("score group total work/result bound exceeded")
    return group, children, plans


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
    validate_controls(request)
    caps = capabilities(engine)
    prompt, selected = request.get("prompt_token_ids"), request.get("selected_token_ids")
    for name, tokens, limit in (("prompt_token_ids", prompt, min(caps["max_prompt_tokens"], engine.max_context or MAX_PROMPT_TOKENS)),
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
