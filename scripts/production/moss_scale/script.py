"""Draft and validate a dialogue before synthesis."""
import hashlib
import json
import os
import random
import re

from scripts.config import TOPIC_DOMAINS
from scripts.text.oneshot_agent.lexicon import load_bank
from scripts.text.quality import prior_preference_conflict, unsupported_assistant_action
from scripts.text.simultaneous_choices import (
    CHOICE_ACKNOWLEDGEMENTS, JOINT_CONFIRMATIONS, CHOICE_CONFIRMATIONS, CHOICE_RESOLUTIONS,
    CONFLICTING_CHOICES, CONFLICT_CLARIFICATIONS,
    choice_option_sets, choice_request_texts, choice_setup_texts,
)
from scripts.text.utils.llm import chat
from scripts.text.voices import gender_of, reference_voices, voice_bindings_snapshot, voices_for_dialogue

from .slots import EN_TRANSITION_PIVOTS, TRANSITION_PIVOTS, draft_from_slots, plan_slots
from .errors import safe_diagnostic
from .storage import atomic_json


SELF_RESUME_PAUSE_MS = (500, 2000)
HANDOFF_PAUSE_MS = (2500, 3200)
EN_TOPICS = ("travel plans", "home maintenance", "work coordination", "meal planning",
             "local events", "pet care", "technology", "shopping", "study plans",
             "community projects", "outdoor activities", "family logistics")
EN_BACKCHANNELS = ("Right", "Makes sense", "Got it", "Exactly", "I see", "That's fair")
SCENE_SEEDS = (
    ("当面商量", "in-person planning"),
    ("线上语音", "voice call"),
    ("边做事边聊", "chatting while working"),
    ("临出门确认安排", "last-minute coordination"),
    ("社区活动布置现场", "setting up a community event"),
    ("共享工位旁交接事项", "handing over tasks at a shared workspace"),
    ("等车时临时改行程", "replanning while waiting for a ride"),
    ("志愿活动结束后收拾器材", "packing up after volunteering"),
    ("楼下碰面取还东西", "meeting downstairs to return borrowed items"),
    ("排练或训练中途休息", "taking a break during rehearsal or practice"),
)


def dialogue_id(index, language="zh", backend=None, run_tag=None):
    backend = backend or os.environ.get("TTS_BACKEND", "moss_onnx")
    if backend == "moss_onnx":
        name = "moss"
    elif backend in ("qwen3", "kokoro"):
        name = backend
    else:
        name = "custom-" + hashlib.sha256(backend.encode("utf-8")).hexdigest()[:12]
    prefix = f"{name}-scale-en-20261115" if language == "en" else f"{name}-scale-20261115"
    run_tag = os.environ.get("CHORUS_RUN_TAG", "") if run_tag is None else run_tag
    if run_tag:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", run_tag):
            raise ValueError("Invalid production run tag")
        prefix += f"-{run_tag}"
    return f"{prefix}-{index:06d}"


def incomplete_pause_range(turn):
    return HANDOFF_PAUSE_MS if turn.get("resolution") == "assistant_handoff" else SELF_RESUME_PAUSE_MS


def incomplete_has_content(text, language):
    """Reject a hesitation consisting only of fillers, without claiming semantic validation."""
    if language == "en":
        text = re.sub(r"\b(?:uh+|um+|erm+|let me think|give me a second|one moment)\b", "", text, flags=re.I)
    else:
        text = re.sub(r"(?:让我想一想|让我想想|等我想想|我想一下|我想想|那个|这个|呃+|嗯+|啊+)", "", text)
    return bool(re.search(r"\w", text))


def distinct_voice_cast(ident, requested_users, wants_third_party):
    """Fit the cast to distinct registered voice bindings before requesting a draft."""
    probe = voices_for_dialogue(ident, 1)
    candidates = [probe["user_01"]] + [probe[f"third_party_{i:02d}"] for i in range(1, 7)]
    bindings = voice_bindings_snapshot([probe["assistant"], *candidates])
    assistant_binding = bindings[probe["assistant"]]
    assistant_key = (assistant_binding["kind"], str(assistant_binding.get("speaker", assistant_binding.get("sha256", ""))).casefold())
    unique, seen = [], {assistant_key}
    for voice in candidates:
        binding = bindings[voice]
        key = (binding["kind"], str(binding.get("speaker", binding.get("sha256", ""))).casefold())
        if key not in seen:
            unique.append(voice)
            seen.add(key)
    if not unique:
        raise ValueError("No distinct participant voice is available")
    include_third = wants_third_party and len(unique) >= 2
    count = min(requested_users, len(unique) - int(include_third))
    mapping = {"assistant": probe["assistant"]}
    mapping.update({f"user_{i+1:02d}": voice for i, voice in enumerate(unique[:count])})
    if include_third:
        mapping["third_party_01"] = unique[count]
    return count, mapping, include_third


class BackchannelTooLong(ValueError):
    """A validly referenced acknowledgement needs shorter wording."""


def shorten_backchannels(draft):
    turns = draft["turns"]
    candidates = {
        str(i): {"原文": turn["text"], "已听到的内容": turn["anchor_text"]}
        for i, turn in enumerate(turns)
        if turn["kind"] == "backchannel" and isinstance(turn["text"], str) and len(turn["text"]) > 14
    }
    raw = chat(
        "把每条附和缩为2到14字符的自然短反馈，保留原语气。不要添加观点。只返回JSON对象，键保持轮次编号，值为缩短后的文字。",
        json.dumps(candidates, ensure_ascii=False), temperature=.3, max_tokens=500, json_output=True,
    )
    corrections = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
    if not isinstance(corrections, dict) or set(corrections) != set(candidates):
        raise ValueError("附和修订的轮次编号不匹配")
    if any(not isinstance(value, str) or not 2 <= len(value.strip()) <= 14 for value in corrections.values()):
        raise ValueError("附和修订须为2至14字符")
    for index, text in corrections.items():
        turns[int(index)]["text"] = text.strip()


def apply_rule_quality(draft):
    """Evaluate inexpensive structural rules; semantic and audio quality need separate review."""
    from scripts.text.quality import audit
    from scripts.text.utils.schema import Dialogue

    report = audit(Dialogue.from_jsonl_line(json.dumps(draft, ensure_ascii=False)))
    if not report["passed"]:
        codes = sorted({issue["code"] for issue in report["issues"]})
        raise ValueError("Dialogue failed deterministic quality rules: " + ", ".join(codes))
    draft["profile"]["rule_quality"] = {
        "version": report["version"], "scores": report["rule_scores"], "scope": report["score_scope"],
        "warnings": [{"code": item["code"], "turn_id": item["turn_id"]} for item in report["warnings"]],
    }


def generate(index, root, mode="direct", language="zh"):
    backend = os.environ.get("TTS_BACKEND", "moss_onnx")
    ident = dialogue_id(index, language, backend=backend)
    path = root / "scripts" / f"{ident}.json"
    if path.exists():
        draft = json.loads(path.read_text(encoding="utf-8"))
        profile = draft.get("profile", {})
        if profile.get("tts_backend", "moss_onnx") != backend:
            raise ValueError("Cached dialogue uses a different TTS backend; use a separate output root.")
        bindings = profile.get("voice_bindings")
        if bindings is None:
            if os.environ.get("VOICE_REGISTRY"):
                raise ValueError("Cached dialogue has no voice bindings; use a separate output root with this registry.")
        elif bindings != voice_bindings_snapshot(profile["voice_assignments"].values()):
            raise ValueError("Voice bindings changed since this dialogue was drafted; use the original registry.")
        for turn in draft["turns"]:
            if turn["kind"] == "incomplete":
                low, high = incomplete_pause_range(turn)
                if type(turn.get("pause_ms")) is not int or not low <= turn["pause_ms"] <= high:
                    turn["pause_ms"] = random.Random(f"{ident}-{turn['turn_id']}-pause").randint(low, high)
        apply_rule_quality(draft)
        return draft
    rng = random.Random(index + 20261115)
    count = 3 if index == 0 else rng.choices(range(1, 7), [1, 3, 3, 3, 2, 1])[0]
    turns = rng.choice([14, 16, 18, 20])
    events = [kind for kind, probability in (
        ("incomplete", .70), ("backchannel", .95), ("interruption", .66), ("side_talk", .55)
    ) if rng.random() < probability]
    if len(events) < 3:
        remaining = [kind for kind in ("backchannel", "interruption", "incomplete", "side_talk")
                     if kind not in events]
        events.extend(rng.sample(remaining, 3 - len(events)))
    if index == 0:
        events = ["incomplete", "backchannel"]
    count, mapping, include_third = distinct_voice_cast(ident, count, "side_talk" in events)
    if "side_talk" in events and not include_third:
        events.remove("side_talk")
        remaining = [kind for kind in ("backchannel", "interruption", "incomplete") if kind not in events]
        if remaining:
            events.append(rng.choice(remaining))
    users = [f"user_{n:02d}" for n in range(1, count + 1)]
    listener_rng = random.Random(f"{ident}-listener-sequence")
    simultaneous = index == 0 or (count >= 2 and listener_rng.random() < .5)
    repeat_listener = not simultaneous and (count == 1 or listener_rng.random() < .4)
    group = "backchannel" in events and (index == 0 or rng.random() < .68)
    group_interrupt = count >= 2 and "interruption" in events and rng.random() < .45
    backchannel_to_interruption = "interruption" in events and not group_interrupt and random.Random(
        f"{ident}-backchannel-to-interruption"
    ).random() < .30
    hesitation_handoff = index != 0 and "incomplete" in events and random.Random(
        f"{ident}-hesitation-handoff"
    ).random() < .25
    simultaneous_requests = count >= 2 and random.Random(
        f"{ident}-simultaneous-requests"
    ).random() < .15
    topics = EN_TOPICS if language == "en" else TOPIC_DOMAINS
    relationships = ["neighbors", "coworkers", "old friends", "family", "temporary teammates"] if language == "en" else ["邻居", "同事", "老朋友", "家人", "临时组队"]
    scene = (rng.choice(("当面讨论", "线上语音讨论", "短暂碰面商量")) if language == "zh"
             else rng.choice(("in-person discussion", "voice-call discussion", "a brief planning conversation"))) \
            if simultaneous_requests else rng.choice(SCENE_SEEDS)[int(language == "en")]
    context = {
        "领域": topics[index % len(topics)],
        "关系": rng.choice(relationships),
        "场景": scene,
        "用户音色": {u: gender_of(mapping[u]) for u in users},
    }
    planned_events = events.copy()
    if index and not simultaneous_requests and "backchannel" in events and turns >= 16 and rng.random() < .65:
        planned_events.append("backchannel")
    if simultaneous_requests:
        planned_events.append("simultaneous_requests")
    slots = plan_slots(users, planned_events, group, repeat_listener, group_interrupt,
                       backchannel_to_interruption, hesitation_handoff, rng, turns, language,
                       topic=context["领域"])
    seeds = []
    if "backchannel" in events or backchannel_to_interruption:
        bank = EN_BACKCHANNELS if language == "en" else list(dict.fromkeys(
            row["text"] for row in load_bank()
            if row["type"] in {"ack", "receipt", "empathy"} and 2 <= len(row["text"]) <= 10
        ))
        seeds = random.Random(f"{ident}-backchannel-seeds").sample(bank, min(4, len(bank)))
    options = dict(hesitation_handoff=hesitation_handoff,
                   backchannel_to_interruption=backchannel_to_interruption,
                   sequential_backchannels=not simultaneous, repeat_listener=repeat_listener, language=language,
                   simultaneous_requests=simultaneous_requests)
    targeted_calls = 0
    for attempt in range(2):
        d = None
        targeted_review = None
        try:
            d = draft_from_slots(slots, users, context, mode, seeds, language)
            try:
                validate(d, users, events, group, group_interrupt, **options)
            except BackchannelTooLong:
                if language == "en":
                    raise
                shorten_backchannels(d)
                validate(d, users, events, group, group_interrupt, **options)
            if os.environ.get("CHORUS_TARGETED_SAFETY_REVIEW") == "1" and not targeted_calls:
                from .targeted_review import review_once
                targeted_review, trace = review_once(d, slots, language)
                targeted_calls += targeted_review["calls"]
                if trace is not None:
                    try:
                        validate(d, users, events, group, group_interrupt, **options)
                    except (ValueError, KeyError, TypeError):
                        d["turns"][targeted_review["turn_id"]]["text"] = trace["original"]
                        targeted_review["status"] = "rewrite_rejected"
                        trace["status"] = "rewrite_rejected"
                    atomic_json(root/"staging"/f"{ident}-safety-review-{attempt}.json", {
                        "turn_id": targeted_review["turn_id"], "status": targeted_review["status"],
                        "original_sha256": hashlib.sha256(trace["original"].encode("utf-8")).hexdigest(),
                        "revised_sha256": hashlib.sha256(trace["revised"].encode("utf-8")).hexdigest(),
                    })
            elif os.environ.get("CHORUS_TARGETED_SAFETY_REVIEW") == "1":
                targeted_review = {"status": "budget_exhausted", "calls": targeted_calls}
            break
        except (ValueError, KeyError, TypeError) as exc:
            diagnostic = safe_diagnostic(exc)
            from .upload_delivery import _assert_public_safe
            try:
                _assert_public_safe(d)
                safe_draft = d
            except ValueError:
                safe_draft = None
            atomic_json(root/"staging"/f"{ident}-planned-reject-{attempt}.json",
                        {"error":diagnostic["hint"], "code":diagnostic["code"], "draft":safe_draft})
            context["上次问题"] = diagnostic["hint"]
            if attempt:
                raise
    d.update(dialogue_id=ident, scenario="side_talk" if "side_talk" in events else "multi_qa")
    for i, t in enumerate(d["turns"]):
        overlapping = t["kind"] == "interruption" or t.get("overlap_with") is not None
        t.update(turn_id=i, gap_before_ms=-350 if overlapping else rng.randint(140, 320),
                 insert_ratio=None, events=[], trigger="cue" if t["speaker"]=="assistant" else None)
        if t["kind"] == "incomplete":
            t["pause_ms"] = rng.randint(*incomplete_pause_range(t))
        if i and d["turns"][i-1].get("resolution") == "assistant_handoff":
            t["gap_before_ms"] = 0
    spoken = {t["speaker"] for t in d["turns"]}
    voices = {k:v for k,v in mapping.items() if k in spoken}
    refs = reference_voices()
    d["profile"] = {"generator":"moss_scale_v1","tts_backend":backend,"voice_assignments":voices,
                    "voice_bindings":voice_bindings_snapshot(voices.values()),
                    "distinct_speaker_voices":True,
                    "strict_choice_consistency":True,
                    "voice_references":{v:{k:refs[v][k] for k in ("source_url","license","sha256")}
                                        for v in voices.values() if v in refs},
                    "topic_domain":context["领域"],"requested_events":events,
                    "draft_mode":mode,
                    "listener_spacing_ms":0 if simultaneous else rng.randint(90, 180),
                    "listener_sequence":"simultaneous" if simultaneous else "sequential",
                    "requested_backchannel_group": group,
                    "requested_repeat_listener": group and repeat_listener,
                    "requested_simultaneous_interruption": group_interrupt,
                    "requested_hesitation_handoff": hesitation_handoff,
                    "requested_backchannel_transition": backchannel_to_interruption,
                    "requested_simultaneous_requests": simultaneous_requests}
    if language == "en":
        d["profile"]["language"] = "en"
    if targeted_review and targeted_review["status"] != "not_triggered":
        d["profile"]["targeted_safety_review"] = targeted_review
    apply_rule_quality(d)
    atomic_json(path, d)
    return d


def validate(
    d, users, events, group, group_interrupt=False, *, hesitation_handoff=False,
    backchannel_to_interruption=False, sequential_backchannels=False, repeat_listener=False, language="zh",
    simultaneous_requests=False,
):
    turns = d["turns"]
    allowed = set(users) | {"assistant", "third_party_01"}
    if not 10 <= len(turns) <= 37:
        raise ValueError(f"轮数为{len(turns)}，应为10至37")
    if len(d["participants"]) != len(users) or {p["id"] for p in d["participants"]} != set(users):
        raise ValueError(f"参与人ID必须恰好是{users}，assistant有台词但不放在participants")
    d["participants"] = [{k:p[k] for k in ("id","persona","style")} | {"speech_rate":1.0,"state":"active"}
                         for p in d["participants"]]
    for key in ("setting", "topic"):
        if not isinstance(d[key], str) or not d[key].strip():
            raise ValueError("缺少场景或话题")
    if not set(users + ["assistant"]) <= {t["speaker"] for t in turns}:
        raise ValueError("所有用户及助手均须发言")
    if sum(t["speaker"] == "assistant" and t["kind"] in {"speech", "incomplete"} for t in turns) < 3:
        raise ValueError("助手须有至少三轮实际发言")
    kinds = {t["kind"] for t in turns}
    if kinds - {"speech", *events} or not set(events) <= kinds:
        raise ValueError(f"需实现{events}，实际是{sorted(kinds)}")
    host = None
    listeners = set()
    assistant_group = False
    interrupt_group = False
    handoffs = 0
    transitions = 0
    simultaneous_pairs = 0
    interrupters = set()
    for i, t in enumerate(turns):
        speaker, kind, content = t["speaker"], t["kind"], t["text"]
        refs = t["responds_to"]
        targets = t["addressee"] if isinstance(t["addressee"], list) else [t["addressee"]]
        if (speaker not in allowed or not isinstance(content,str) or not content.strip() or len(content)>(360 if language=="en" else 190)
            or not targets or any(a not in allowed | {"third_party","all","none"} or a==speaker for a in targets)
            or not isinstance(refs,list) or any(type(j) is not int or not 0<=j<i for j in refs)):
            raise ValueError(f"第{i}轮角色、文本、指向或回应索引无效")
        if language == "en" and (not re.search(r"[A-Za-z]", content) or re.search(r"[\u3400-\u9fff]", content)):
            raise ValueError(f"Turn {i} must be in English")
        if re.search(r"(?<![A-Za-z])(?:user|third_party)_\d{2}(?![A-Za-z0-9_])", content):
            raise ValueError(f"第{i}轮台词出现内部角色编号")
        if targets == ["assistant"] and re.match(
            r"^\s*(?:爸|妈|爸爸|妈妈|爹|娘|爷爷|奶奶|外公|外婆|哥哥|姐姐|弟弟|妹妹)[，,：:！!]",
            content,
        ):
            raise ValueError(f"第{i}轮把语音助手称作亲属")
        if speaker == "assistant" and re.search(r"我家(?:有|里|的)|我带(?:一|两)(?:袋|包|瓶).*过去", content):
            raise ValueError(f"第{i}轮助手把自己写成现实中的在场者")
        if speaker == "assistant" and unsupported_assistant_action(content):
            raise ValueError(f"第{i}轮助手声称会执行现实或外部操作")
        if (speaker=="third_party_01") != (kind=="side_talk"):
            raise ValueError("第三方须标作 side_talk")
        interaction = t.get("interaction")
        if interaction not in (None, "choice_request", "choice_setup", "simultaneous_requests", "conflict_clarification",
                               "choice_resolution", "choice_acknowledgement", "choice_confirmation"):
            raise ValueError("未知对话交互类型")
        if t.get("choice_outcome") is not None and interaction not in ("choice_acknowledgement", "choice_confirmation"):
            raise ValueError("选择结果只能标注在协商回应或确认轮")
        if t.get("overlap_with") is not None:
            other = t["overlap_with"]
            if (not simultaneous_requests or type(other) is not int or other != i - 1 or other < 0
                or kind != "speech" or speaker not in users or targets != ["assistant"]
                or interaction != "simultaneous_requests"):
                raise ValueError("同时请求须由两名用户同时向助手发出")
            first = turns[other]
            if (first["speaker"] not in users or first["speaker"] == speaker or first["kind"] != "speech"
                or first["addressee"] not in ("assistant", ["assistant"])
                or first.get("interaction") != "simultaneous_requests"
                or first["responds_to"] != refs or refs != [other - 1]
                or other < 1 or turns[other-1]["speaker"] != "assistant"):
                raise ValueError("同时请求须相互独立地基于共同的对话历史")
            setup = turns[other-1]
            choices = setup.get("choice_options")
            permitted_choices = [list(pair) for pair in choice_option_sets(language, d["topic"])]
            if (setup.get("interaction") != "choice_setup" or choices not in permitted_choices
                or setup["text"] not in choice_setup_texts(language, choices)):
                raise ValueError("同时请求前的助手须明确提出该话题的两个具体选项")
            request = turns[other-2] if other >= 2 else {}
            if (request.get("interaction") != "choice_request" or request.get("speaker") not in users
                or request.get("addressee") not in ("assistant", ["assistant"])
                or request.get("choice_options") != choices
                or request.get("text") not in choice_request_texts(language, choices)
                or setup["responds_to"] != [other-2]):
                raise ValueError("助手提出选项之前须先回应用户的选项请求")
            indices = (first.get("choice_index"), t.get("choice_index"))
            if any(type(index) is not int for index in indices) or set(indices) != {0, 1}:
                raise ValueError("同时请求须分别选择两个不同选项")
            pair = (first["text"], content) if indices == (0, 1) else (content, first["text"])
            if pair not in CONFLICTING_CHOICES[language]:
                raise ValueError("同时请求台词必须与对应的选项一致")
            if any(prior_preference_conflict(turns, turn["speaker"], other, choices, turn["choice_index"], language)
                   for turn in (first, t)):
                raise ValueError("同时选择与此前本人明确偏好冲突")
            following = turns[i+1] if i+1 < len(turns) else {}
            if (following.get("speaker") != "assistant" or following.get("kind") != "speech"
                or following.get("responds_to") != [other, i]
                or following.get("interaction") != "conflict_clarification"
                or following.get("text") not in CONFLICT_CLARIFICATIONS[language]):
                raise ValueError("助手须在两次同时请求之后澄清")
            resolution = turns[i+2] if i+2 < len(turns) else {}
            selected = resolution.get("choice_index")
            previous_choice = first["choice_index"] if resolution.get("speaker") == first["speaker"] else t["choice_index"]
            if (resolution.get("speaker") not in (first["speaker"], speaker)
                or resolution.get("addressee") not in ("assistant", ["assistant"])
                or resolution.get("kind") != "speech" or resolution.get("responds_to") != [i+1]
                or resolution.get("interaction") != "choice_resolution"
                or type(selected) is not int or selected != previous_choice
                or resolution.get("text") not in CHOICE_RESOLUTIONS[language][selected]):
                raise ValueError("用户须在澄清后明确选择一个已列出的方案")
            next_turn = turns[i+3] if i+3 < len(turns) else {}
            if next_turn.get("interaction") == "choice_acknowledgement":
                outcome = next_turn.get("choice_outcome")
                other_speaker = speaker if resolution["speaker"] == first["speaker"] else first["speaker"]
                if (outcome not in ("agreement", "open") or next_turn.get("speaker") != other_speaker
                    or next_turn.get("kind") != "speech" or next_turn.get("responds_to") != [i+2]
                    or next_turn.get("addressee") not in (resolution["speaker"], [resolution["speaker"]])
                    or next_turn.get("text") not in CHOICE_ACKNOWLEDGEMENTS[language][outcome]):
                    raise ValueError("另一位参与者须明确回应分歧")
                confirmation = turns[i+4] if i+4 < len(turns) else {}
                if (confirmation.get("speaker") != "assistant" or confirmation.get("kind") != "speech"
                    or confirmation.get("responds_to") != [i+2, i+3]
                    or confirmation.get("interaction") != "choice_confirmation"
                    or confirmation.get("choice_outcome") != outcome
                    or confirmation.get("addressee") not in ("all", ["all"])
                    or confirmation.get("text") not in JOINT_CONFIRMATIONS[language][outcome]):
                    raise ValueError("助手须根据两人的实际态度继续")
            else:
                confirmation = next_turn  # Older deliveries confirmed one person's preference only.
                if (confirmation.get("speaker") != "assistant" or confirmation.get("kind") != "speech"
                    or confirmation.get("responds_to") != [i+2]
                    or confirmation.get("interaction") != "choice_confirmation"
                    or confirmation.get("addressee") not in (resolution["speaker"], [resolution["speaker"]])
                    or confirmation.get("text") not in CHOICE_CONFIRMATIONS[language]):
                    raise ValueError("助手须确认用户的选择并基于此提供建议")
            simultaneous_pairs += 1
        resolution = t.get("resolution")
        if resolution is not None and (kind != "incomplete" or resolution != "assistant_handoff"):
            raise ValueError("未识别的话轮 resolution")
        if kind == "incomplete":
            handoff = resolution == "assistant_handoff"
            parts = t["parts"]
            expected_parts = 1 if handoff else 2
            if (not isinstance(parts, list) or len(parts) != expected_parts
                or any(not isinstance(part, str) or not part.strip() for part in parts)
                or "".join(parts) != content or parts[0].rstrip().endswith(("。", "？", "！", "?", "!"))):
                raise ValueError(f"incomplete 需要 {expected_parts} 个与正文一致的发声片段")
            if not incomplete_has_content(parts[0], language):
                raise ValueError("incomplete 停顿前须有实际没说完的内容，不能只有犹豫词")
            if handoff:
                next_turn = turns[i+1] if i+1 < len(turns) else {}
                if (not hesitation_handoff or speaker not in users
                    or next_turn.get("speaker") != "assistant" or next_turn.get("kind") != "speech"
                    or next_turn.get("responds_to") != [i]
                    or next_turn.get("addressee") not in (speaker, [speaker])):
                    raise ValueError("犹豫接话须由助手下一轮明确回应刚停下的用户")
                handoffs += 1
        if "backchannel_prefix" in t:
            prefix = t["backchannel_prefix"]
            if (not backchannel_to_interruption or kind != "interruption" or speaker not in users
                or not isinstance(prefix, str) or not 2 <= len(prefix) <= (40 if language=="en" else 14)
                or not content.startswith(prefix) or refs != [i-1]
                or turns[i-1]["speaker"] != "assistant"):
                raise ValueError("附和转打断需要用户在同一轮先短附和，并引用上一轮助手")
            pivot = content[len(prefix):].lstrip(" ，,。；;、：:…")
            if language == "en":
                valid_pivot = any(re.match(re.escape(word) + r"\b[ ,.!?—]*\w", pivot, re.IGNORECASE)
                                  for word in EN_TRANSITION_PIVOTS)
            else:
                valid_pivot = any(pivot.startswith(word) and pivot[len(word):].strip(" ，,。！？") for word in TRANSITION_PIVOTS)
            if not valid_pivot:
                raise ValueError("附和转打断需要在短附和后用转折词提出具体补充")
            transitions += 1
        if kind=="backchannel":
            anchor=t["anchor_text"]
            if (host is None or turns[host]["kind"]!="speech" or speaker==turns[host]["speaker"]
                or refs!=[host] or targets!=[turns[host]["speaker"]]
                or (speaker in listeners and not sequential_backchannels)
                or not isinstance(anchor,str) or not anchor.strip() or not turns[host]["text"].startswith(anchor)
                or (listeners and anchor!=turns[i-1]["anchor_text"])):
                raise ValueError(f"第{i}轮附和须引用同一主轮和原文前缀，同时附和须由不同用户发出")
            if (len(content.split()) > 6 if language == "en" else len(content) > 14):
                raise BackchannelTooLong(f"第{i}轮附和 text 过长，最多14字符。请改为简短反馈。")
            if len(turns[host]["text"]) - len(anchor) < 8:
                raise ValueError(f"第{i}轮附和锚点太晚，主轮在 anchor_text 之后必须至少还剩8字符。")
            if listeners and turns[host]["speaker"] == "assistant":
                previous_speaker = turns[i-1]["speaker"]
                assistant_group |= (
                    speaker in users and previous_speaker in users
                    and (speaker == previous_speaker) == repeat_listener
                )
            listeners.add(speaker)
        else:
            if kind == "interruption":
                previous = turns[i-1] if i else {}
                same_host = previous.get("kind")=="interruption" and refs==previous["responds_to"]
                expected = previous["responds_to"] if same_host else [i-1]
                if not same_host:
                    interrupters.clear()
                if (not i or refs!=expected or turns[refs[0]]["kind"]!="speech"
                    or targets != [turns[refs[0]]["speaker"]]
                    or speaker==turns[refs[0]]["speaker"]
                    or (same_host and (not group_interrupt or speaker in interrupters
                                       or speaker not in users or previous["speaker"] not in users))):
                    raise ValueError("抢话须引用原主轮。多人抢话由不同用户连续引用同一主轮")
                interrupters.add(speaker)
                interrupt_group |= same_host
            host=i
            listeners.clear()
    if hesitation_handoff and handoffs != 1:
        raise ValueError("剧本需要恰好一处犹豫接话")
    if group and not assistant_group:
        raise ValueError("缺少按指定人员组合附和同一助手主轮的两个短反馈")
    if backchannel_to_interruption and transitions != 1:
        raise ValueError("剧本需要恰好一处附和转打断")
    if group_interrupt and not interrupt_group:
        raise ValueError("缺少两名不同用户抢同一个话轮")
    if simultaneous_pairs != int(simultaneous_requests):
        raise ValueError("同时请求事件数量与计划不符")
