"""OpenAI-compatible LLM controllers for E-NPC experiments.

The module intentionally uses only the Python standard library so the
benchmark can run in a fresh environment. It targets the widely implemented
Chat Completions shape: POST {base_url}/chat/completions.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import error, request

from baselines import NPCResponse
from epistemic_state import (
    ABSTAIN,
    ANSWER_AS_HEARSAY,
    ANSWER_AS_INFERENCE,
    ANSWER_FACT,
    ANSWER_WITH_SOURCE,
    ASK_CLARIFICATION,
    ASK_NPC,
    ASSERTIVE_MODES,
    CORRECT_FALSE_PRESUPPOSITION,
    DECLINE_META,
    HEDGE,
    INFORMATION_SEEKING_MODES,
    INSPECT_LOCATION,
    INSPECT_OBJECT,
    QUERY_PUBLIC_RECORD,
    SUGGEST_SOURCE,
    SourceType,
    action_mode,
    allowed_modes_for_claim,
)
from query_generator import QueryItem
from world_generator import World


RESPONSE_MODES = [
    ANSWER_FACT,
    ANSWER_WITH_SOURCE,
    ANSWER_AS_HEARSAY,
    ANSWER_AS_INFERENCE,
    CORRECT_FALSE_PRESUPPOSITION,
    HEDGE,
    ABSTAIN,
    ASK_CLARIFICATION,
    SUGGEST_SOURCE,
    ASK_NPC,
    INSPECT_LOCATION,
    INSPECT_OBJECT,
    QUERY_PUBLIC_RECORD,
    DECLINE_META,
]

SOURCE_TYPES = [source.value for source in SourceType]

SUPPORTED_POLICIES = [
    "global_state",
    "local_state",
    "careful",
    "source_calibrated_nohint",
    "source_calibrated",
]

_SOURCE_RULES = {
    "direct_observation": ["answer_fact", "answer_with_source"],
    "role_knowledge": ["answer_fact", "answer_with_source"],
    "public_record": ["answer_fact", "answer_with_source"],
    "testimony": ["answer_with_source", "hedge"],
    "hearsay": ["answer_as_hearsay", "hedge", "suggest_source"],
    "environmental_evidence": ["answer_as_inference", "hedge", "suggest_source"],
    "no_local_claim": ["abstain", "ask_npc", "inspect_location", "query_public_record"],
}

_SOURCE_CALIBRATED_INSTRUCTION = (
    "Choose a response mode justified by local evidence. Direct observations "
    "and role knowledge may be asserted as facts; testimony must be attributed; "
    "hearsay must be hedged; unsupported claims require an in-world action."
)

# Strong generic-careful baseline: same local evidence, but no structured
# source-type -> speech-act mapping and no gold acceptable_modes hint.
_CAREFUL_INSTRUCTION = (
    "You are a situated character, not an omniscient narrator. Reason carefully about "
    "what THIS character could actually know from its own experience, then answer. Only "
    "assert what your own evidence supports; be honest when you are uncertain or only "
    "heard something secondhand; never state or imply facts you have no in-world basis "
    "for; if you lack evidence, say so in character or take an action to find out."
)


class ChatClient(Protocol):
    def complete_json(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class OpenAICompatibleChatClient:
    model: str
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 60.0
    max_retries: int = 2
    temperature: float = 0.0
    use_structured_output: bool = True
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def complete_json(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.model.startswith(("o1", "o3", "o4", "gpt-5")):  # reasoning models: no temperature; reserve room for reasoning+output
            payload["max_completion_tokens"] = int(os.environ.get("OPENAI_MAX_COMPLETION_TOKENS", "25000"))
        else:
            payload["temperature"] = self.temperature
        if self.use_structured_output and response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "npc_response",
                    "strict": True,
                    "schema": response_schema,
                },
            }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._post_json("/chat/completions", payload)
                usage = raw.get("usage", {})
                self.input_tokens += usage.get("prompt_tokens", 0)
                self.output_tokens += usage.get("completion_tokens", 0)
                self.calls += 1
                content = raw["choices"][0]["message"]["content"]
                return _parse_json_object(content)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"LLM request failed after retries: {last_error}") from last_error

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = request.Request(url, data=body, headers=headers, method="POST")
        timeout = 300.0 if self.model.startswith(("o1", "o3", "o4", "gpt-5")) else self.timeout
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from LLM API: {detail}") from exc


@dataclass
class GeminiChatClient:
    """Google Gemini GenerateContent client implementing the local ChatClient protocol.

    The rest of the repo uses a small `complete_json(messages, response_schema)` interface.
    Gemini does not use Chat Completions roles directly, so developer/system messages are
    folded into `systemInstruction` and user/assistant turns become `contents`.
    """

    model: str
    api_key: str
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout: float = 120.0
    max_retries: int = 2
    temperature: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def complete_json(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        system = "\n\n".join(
            m["content"] for m in messages if m["role"] in ("system", "developer")
        )
        contents = []
        for m in messages:
            if m["role"] not in ("user", "assistant"):
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Return {} as JSON."}]}]

        generation_config: dict[str, Any] = {
            "temperature": self.temperature,
            "responseMimeType": "application/json",
        }
        if response_schema is not None:
            generation_config["responseSchema"] = _gemini_schema(response_schema)
            system = (system + "\n\n" if system else "") + (
                "Return only a JSON object conforming to the requested schema."
            )

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._post(payload)
                usage = raw.get("usageMetadata", {})
                self.input_tokens += usage.get("promptTokenCount", 0)
                self.output_tokens += usage.get("candidatesTokenCount", 0)
                self.calls += 1
                text = _gemini_text(raw)
                return _parse_json_object(text)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"Gemini request failed after retries: {last_error}") from last_error

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        model_path = request.pathname2url(self.model)
        url = f"{self.base_url.rstrip('/')}/models/{model_path}:generateContent?key={self.api_key}"
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from Gemini API: {detail}") from exc


@dataclass
class AnthropicChatClient:
    """Native Anthropic Messages API client (stdlib only), implementing the ChatClient
    protocol. developer/system roles map to the top-level `system`; user/assistant map to
    `messages`. Non-structured: relies on Claude's instruction-following to return one JSON
    object (parsed leniently). Tracks token usage for cost reporting."""

    model: str
    api_key: str
    base_url: str = "https://api.anthropic.com"
    timeout: float = 60.0
    max_retries: int = 2
    max_tokens: int = 512
    anthropic_version: str = "2023-06-01"
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def complete_json(
        self,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        system = "\n\n".join(
            m["content"] for m in messages if m["role"] in ("system", "developer")
        )
        if response_schema is not None:
            system = (system + "\n\n" if system else "") + (
                "Return ONLY a single JSON object — no prose, no markdown fences, no array "
                "wrapper — conforming to this JSON Schema:\n" + json.dumps(response_schema)
            )
        msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] in ("user", "assistant")
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": msgs,
        }
        if system:
            payload["system"] = system

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._post(payload)
                usage = raw.get("usage", {})
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)
                self.calls += 1
                if raw.get("stop_reason") == "refusal":
                    # model-level refusal returns empty content; deterministic per prompt, so do not retry
                    raise RuntimeError("Anthropic refusal (stop_reason=refusal, empty content)")
                text = "".join(
                    block.get("text", "")
                    for block in raw.get("content", [])
                    if block.get("type") == "text"
                )
                return _parse_json_object(text)
            except RuntimeError as exc:
                if "stop_reason=refusal" in str(exc):
                    raise
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.5 * (2**attempt))
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"Anthropic request failed after retries: {last_error}") from last_error

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
        }
        req = request.Request(
            f"{self.base_url.rstrip('/')}/v1/messages", data=body, headers=headers, method="POST"
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from Anthropic API: {detail}") from exc


@dataclass
class LLMNPCController:
    policy: str
    client: ChatClient

    @property
    def name(self) -> str:
        return f"llm_{self.policy}"

    def respond(self, world: World, query: QueryItem) -> NPCResponse:
        if self.policy not in SUPPORTED_POLICIES:
            raise ValueError(f"unsupported LLM policy: {self.policy}")

        messages = build_messages(world, query, self.policy)
        parsed = self.client.complete_json(messages, response_schema=response_schema())
        return response_from_llm_json(self.name, query, parsed)


def make_llm_controllers(
    client: ChatClient,
    policies: list[str] | None = None,
) -> list[LLMNPCController]:
    selected = policies or SUPPORTED_POLICIES
    return [LLMNPCController(policy=policy, client=client) for policy in selected]


def build_messages(world: World, query: QueryItem, policy: str) -> list[dict[str, str]]:
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"unsupported LLM policy: {policy}")

    payload = _base_payload(world, query)
    if policy == "global_state":
        payload["policy"] = {
            "name": "global_state",
            "instruction": "You may answer using the complete game truth.",
        }
        payload["global_truth"] = _global_truth_payload(world, query)
    elif policy == "local_state":
        payload["policy"] = {
            "name": "local_state",
            "instruction": (
                "Answer only from the target NPC's local memory. If the memory is "
                "missing, say the NPC does not know."
            ),
        }
        payload["local_claims"] = _local_claims_payload(world, query)
    elif policy == "careful":
        payload["policy"] = {"name": "careful", "instruction": _CAREFUL_INSTRUCTION}
        payload["local_claims"] = _local_claims_payload(world, query)
        payload["valid_actions"] = query.valid_actions
    elif policy in ("source_calibrated", "source_calibrated_nohint"):
        payload["policy"] = {"name": policy, "instruction": _SOURCE_CALIBRATED_INSTRUCTION}
        payload["local_claims"] = _local_claims_payload(world, query)
        payload["valid_actions"] = query.valid_actions
        payload["source_rules"] = _SOURCE_RULES
        if policy == "source_calibrated":
            # gold-derived per-query hint; the _nohint ablation deliberately omits it
            payload["acceptable_modes"] = query.acceptable_modes

    developer = (
        "You are an NPC response controller for an epistemic-consistency benchmark. "
        "Return exactly one JSON object. Do not include Markdown. "
        "Use answer_value as null or as the shortest exact answer atom, such as a name, "
        "'yes', 'no', or a code. If you recommend an action, put the exact action string "
        "in action."
    )
    user = json.dumps(payload, ensure_ascii=True, indent=2)
    return [
        {"role": "developer", "content": developer},
        {"role": "user", "content": user},
    ]


def response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mode": {"type": "string", "enum": RESPONSE_MODES},
            "answer_value": {"type": ["string", "null"]},
            "text": {"type": "string"},
            "claimed_source_type": {
                "type": ["string", "null"],
                "enum": [*SOURCE_TYPES, None],
            },
            "source_agent": {"type": ["string", "null"]},
            "action": {"type": ["string", "null"]},
        },
        "required": [
            "mode",
            "answer_value",
            "text",
            "claimed_source_type",
            "source_agent",
            "action",
        ],
    }


def response_from_llm_json(
    controller_name: str,
    query: QueryItem,
    parsed: dict[str, Any],
) -> NPCResponse:
    mode = _as_string(parsed.get("mode")) or ABSTAIN
    action = _as_optional_string(parsed.get("action"))
    if mode not in RESPONSE_MODES:
        mode = action_mode(action) if action else ABSTAIN
    answer_value = _as_optional_string(parsed.get("answer_value"))
    text = _as_string(parsed.get("text")) or ""
    claimed_source_type = _as_optional_string(parsed.get("claimed_source_type"))
    if claimed_source_type not in SOURCE_TYPES:
        claimed_source_type = None

    return NPCResponse(
        controller=controller_name,
        query_id=query.query_id,
        target_npc_id=query.target_npc_id,
        mode=mode,
        answer_value=answer_value,
        text=text,
        source_type=claimed_source_type,
        source_agent=_as_optional_string(parsed.get("source_agent")),
        claimed_source_type=claimed_source_type,
        action=action,
        metadata={"raw_llm_json": parsed},
    )


def _base_payload(world: World, query: QueryItem) -> dict[str, Any]:
    npc = world.npc(query.target_npc_id)
    return {
        "world_id": world.world_id,
        "scenario_type": world.scenario_type,
        "target_npc": npc.to_dict(),
        "question": query.question,
        "proposition_id": query.proposition_id,
        "is_false_presupposition": query.is_false_presupposition,
        "presupposed_value": query.presupposed_value,
        "response_contract": {
            "modes": RESPONSE_MODES,
            "source_types": SOURCE_TYPES,
            "action_examples": query.valid_actions,
        },
    }


def _global_truth_payload(world: World, query: QueryItem) -> dict[str, Any]:
    fact = world.facts[query.proposition_id]
    return {
        "proposition": fact.proposition,
        "true_value": fact.true_value,
        "answer_text": fact.answer_text,
        "hidden": fact.hidden,
        "events": [event.to_dict() for event in world.events],
    }


def _local_claims_payload(world: World, query: QueryItem) -> list[dict[str, Any]]:
    state = world.epistemic_states[query.target_npc_id]
    return [
        claim.to_dict()
        for claim in state.claims_for(query.proposition_id)
    ]


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end < start:
            raise
        parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response was not a JSON object")
    return parsed


def _gemini_text(raw: dict[str, Any]) -> str:
    candidates = raw.get("candidates") or []
    if not candidates:
        raise ValueError(f"Gemini response had no candidates: {raw}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts)
    if not text:
        raise ValueError(f"Gemini response had no text: {raw}")
    return text


def _gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert the repo's JSON Schema subset to Gemini's responseSchema subset."""
    out: dict[str, Any] = {}
    typ = schema.get("type")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), typ[0] if typ else None)
    if typ:
        out["type"] = str(typ).upper()
    if "enum" in schema:
        out["enum"] = [x for x in schema["enum"] if x is not None]
    if "properties" in schema:
        out["properties"] = {
            k: _gemini_schema(v) for k, v in schema["properties"].items()
        }
    if "items" in schema:
        out["items"] = _gemini_schema(schema["items"])
    if "required" in schema:
        out["required"] = schema["required"]
    if "description" in schema:
        out["description"] = schema["description"]
    return out


def _as_string(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _as_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


# --- Proposed method: structured source-calibrated controller --------------------------


def constrained_response_schema(
    allowed_modes: list[str],
    valid_actions: list[str],
) -> dict[str, Any]:
    """Response schema whose `mode` enum is restricted to the provenance-licensed modes,
    so the model cannot emit an unlicensed speech act."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mode": {"type": "string", "enum": allowed_modes},
            "answer_value": {"type": ["string", "null"]},
            "text": {"type": "string"},
            "claimed_source_type": {"type": ["string", "null"], "enum": [*SOURCE_TYPES, None]},
            "source_agent": {"type": ["string", "null"]},
            "action": {"type": ["string", "null"], "enum": [*valid_actions, None]},
        },
        "required": ["mode", "answer_value", "text", "claimed_source_type", "source_agent", "action"],
    }


def build_structured_messages(
    world: World,
    query: QueryItem,
    best_claim,
    licensed_modes: list[str],
) -> list[dict[str, str]]:
    payload = _base_payload(world, query)
    payload["policy"] = {
        "name": "structured_source_calibrated",
        "instruction": (
            "You are a situated NPC. The licensed_modes below were derived from YOUR own "
            "evidence and are the ONLY response modes you may use (the schema enforces this). "
            "Use your local_claims to decide the answer; set answer_value to the shortest "
            "answer atom (a name, 'yes', 'no', or a code) when you assert a value, else null. "
            "If you choose an information-seeking mode, put the exact action string in action."
        ),
    }
    payload["local_claims"] = _local_claims_payload(world, query)
    payload["licensed_modes"] = licensed_modes
    payload["valid_actions"] = query.valid_actions
    payload["source_rules"] = _SOURCE_RULES
    developer = (
        "You are an NPC response controller. Return exactly one JSON object, no Markdown. "
        "Choose mode only from the provided enum."
    )
    user = json.dumps(payload, ensure_ascii=True, indent=2)
    return [
        {"role": "developer", "content": developer},
        {"role": "user", "content": user},
    ]


def _structured_response_from_json(
    name: str,
    query: QueryItem,
    best_claim,
    licensed_modes: list[str],
    parsed: dict[str, Any],
    deterministic_attribution: bool = True,
) -> NPCResponse:
    # When modes are enum-enforced this is already licensed; when not enforced we keep
    # whatever the model chose so the ablation can measure how often it deviates.
    mode = _as_string(parsed.get("mode")) or ABSTAIN
    answer_value = _as_optional_string(parsed.get("answer_value"))
    if mode not in ASSERTIVE_MODES:
        answer_value = None
    action = _as_optional_string(parsed.get("action"))
    if action not in (query.valid_actions or []):
        action = None
    if deterministic_attribution:
        # attribution set from the NPC's real claim, not guessed by the model
        if best_claim is not None and mode in ASSERTIVE_MODES:
            claimed = best_claim.source_type
            source_agent = best_claim.source_agent
        else:
            claimed = None
            source_agent = None
    else:
        claimed = _as_optional_string(parsed.get("claimed_source_type"))
        if claimed not in SOURCE_TYPES:
            claimed = None
        source_agent = _as_optional_string(parsed.get("source_agent"))
    return NPCResponse(
        controller=name,
        query_id=query.query_id,
        target_npc_id=query.target_npc_id,
        mode=mode,
        answer_value=answer_value,
        text=_as_string(parsed.get("text")) or "",
        source_type=claimed,
        source_agent=source_agent,
        claimed_source_type=claimed,
        action=action,
        metadata={"licensed_modes": licensed_modes, "raw_llm_json": parsed},
    )


@dataclass
class StructuredSourceCalibratedController:
    """The proposed method. Derives the licensed speech acts from the NPC's OWN evidence
    provenance via allowed_modes_for_claim (NOT the benchmark's gold acceptable_modes),
    enforces them through the response schema's mode enum, and sets provenance attribution
    deterministically. The LLM supplies only the answer value and the natural-language line.
    """

    client: ChatClient
    enforce_modes: bool = True
    deterministic_attribution: bool = True
    name: str = "structured_source_calibrated"

    def respond(self, world: World, query: QueryItem) -> NPCResponse:
        state = world.epistemic_states[query.target_npc_id]
        best_claim = state.best_claim(query.proposition_id)
        licensed_modes = allowed_modes_for_claim(
            best_claim,
            valid_actions=query.valid_actions,
            is_false_presupposition=query.is_false_presupposition,
        )
        messages = build_structured_messages(world, query, best_claim, licensed_modes)
        # enforce_modes=True restricts the mode enum to the licensed set; otherwise the
        # licensed modes are only given as a hint (full enum) so we can ablate enforcement.
        if self.enforce_modes:
            schema = constrained_response_schema(licensed_modes, query.valid_actions)
        else:
            schema = response_schema()
        parsed = self.client.complete_json(messages, response_schema=schema)
        return _structured_response_from_json(
            self.name, query, best_claim, licensed_modes, parsed, self.deterministic_attribution
        )
