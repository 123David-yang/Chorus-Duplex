"""Draft and validate a dialogue before synthesis."""
import json
import random

from scripts.config import TOPIC_DOMAINS
from scripts.text.oneshot_agent.lexicon import load_bank
from scripts.text.utils.llm import chat
from scripts.text.voices import gender_of, reference_voices, voices_for_dialogue

from .storage import atomic_json


SELF_RESUME_PAUSE_MS = (500, 2000)
HANDOFF_PAUSE_MS = (2500, 3200)


def incomplete_pause_range(turn):
    return HANDOFF_PAUSE_MS if turn.get("resolution") == "assistant_handoff" else SELF_RESUME_PAUSE_MS


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


def generate(index, root):
    ident = f"moss-scale-20261115-{index:06d}"
    path = root / "scripts" / f"{ident}.json"
    if path.exists():
        draft = json.loads(path.read_text(encoding="utf-8"))
        for turn in draft["turns"]:
            if turn["kind"] == "incomplete":
                low, high = incomplete_pause_range(turn)
                if not low <= turn["pause_ms"] <= high:
                    turn["pause_ms"] = random.Random(f"{ident}-{turn['turn_id']}-pause").randint(low, high)
        return draft
    rng = random.Random(index + 20261115)
    count = 3 if index == 0 else rng.choices(range(1, 7), [1, 3, 3, 3, 2, 1])[0]
    users = [f"user_{n:02d}" for n in range(1, count + 1)]
    mapping = voices_for_dialogue(ident, count)
    turns = rng.choice([16, 20, 24, 28])
    events = [kind for kind, probability in (
        ("incomplete", .40), ("backchannel", .60), ("interruption", .27), ("side_talk", .25)
    ) if rng.random() < probability]
    if index == 0:
        events = ["incomplete", "backchannel"]
    listener_rng = random.Random(f"{ident}-listener-sequence")
    simultaneous = index == 0 or (count >= 2 and listener_rng.random() < .5)
    repeat_listener = not simultaneous and (count == 1 or listener_rng.random() < .4)
    group = "backchannel" in events and (index == 0 or rng.random() < .35)
    group_interrupt = count >= 2 and "interruption" in events and rng.random() < .35
    backchannel_to_interruption = "interruption" in events and not group_interrupt and random.Random(
        f"{ident}-backchannel-to-interruption"
    ).random() < .30
    hesitation_handoff = index != 0 and "incomplete" in events and random.Random(
        f"{ident}-hesitation-handoff"
    ).random() < .25
    system = (
        "生成自然中文多人对话，只返回 JSON：setting、topic、participants、turns。"
        "participants 只列正式用户，每项 id、persona、style。"
        "turns 每项 speaker、text、addressee、kind、responds_to；引用用从0开始的过去轮次编号数组，首轮用[]。示例：{\"speaker\":\"user_01\",\"text\":\"咱们几点走？\",\"addressee\":\"assistant\",\"kind\":\"speech\",\"responds_to\":[]}。"
        "对话前中后各安排至少一轮 assistant 的实质发言，不编造助手的生活经历；写口语，每轮最多两句。"
        "回应实际求助。只用请求中的事件，普通轮次用 speech。严格按请求轮数写，允许相差2轮。"
    )
    event_rules = {
        "incomplete": "未说完后同一人续说。parts 为两个非空片段，直接拼起来等于 text，第一段不能是完整句。",
        "backchannel": "紧跟 speech 主轮，只引用并面向主轮说话人。anchor_text 逐字复制已说完的事实前缀，后面至少剩8字。附和是2–14字短反馈，不加观点。",
        "interruption": "单人打断引用并面向紧邻的 speech。被打断轮只写来得及说出的半句，打断内容回应已听到的信息。",
        "side_talk": "第三方用 third_party_01，置于中段，先与正式用户交流再回主线，不列入 participants。",
    }
    if hesitation_handoff:
        event_rules["incomplete"] = (
            "犹豫接话：用户想法未说完便停住，不能用完整问句加省略号冒充。标记 kind=incomplete，resolution=assistant_handoff。"
            "下一轮是面向该用户的 assistant speech，只引用这一轮。助手温和提议，依据已听到的内容，静音由程序生成。"
        )
    if events:
        system += "\n" + "\n".join(event_rules[event] for event in events)
    if "backchannel" in events or backchannel_to_interruption:
        pool = list(dict.fromkeys(
            entry["text"] for entry in load_bank()
            if entry["type"] in {"ack", "receipt", "empathy"} and 2 <= len(entry["text"]) <= 10
        ))
        seeds = random.Random(f"{ident}-backchannel-seeds").sample(pool, min(4, len(pool)))
        system += "\n附和措辞参考（按语境变化）：" + "、".join(seeds)
    if group:
        listeners = "同一用户分两次" if repeat_listener else "两位不同用户各一次"
        system += f"\n附和组：{listeners}。两条紧随助手主轮，用同一个 anchor_text、responds_to 和 addressee=assistant。"
    if group_interrupt:
        system += "\n同时打断须写成连续三轮：assistant speech → 用户A interruption → 用户B interruption。后两轮 addressee=assistant，responds_to 都是同一个助手轮索引，B 不引用 A。"
    if backchannel_to_interruption:
        system += (
            "\n附和转打断：先写一轮 assistant speech，紧跟一轮用户 interruption。用户只引用该助手轮，addressee=assistant，先简短认同，再用不过、但是或等一下提出补充。"
            "backchannel_prefix 逐字复制该用户 text 开头的认同，只写2–14字。"
        )
    request = {
        "领域": TOPIC_DOMAINS[index % len(TOPIC_DOMAINS)],
        "关系": rng.choice(["邻居", "同事", "老朋友", "家人", "临时组队"]),
        "场景": rng.choice(["当面商量", "线上语音", "边做事边聊", "临出门确认安排"]),
        "轮数": turns, "用户及音色性别": {u: gender_of(mapping[u]) for u in users},
        "必须各自有实际台词的 speaker": users + ["assistant"], "助手实际发言次数": "3到5轮", "本条事件": events, "对助手安排一组附和": group, "安排一次两人同时抢话": group_interrupt, "安排一次附和转打断": backchannel_to_interruption,
    }
    for attempt in range(2):
        raw = ""
        try:
            raw = chat(system, json.dumps(request, ensure_ascii=False, separators=(",", ":")),
                       temperature=.85 if attempt == 0 else .65, max_tokens=7000, json_output=True)
            d = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
            for turn in d["turns"]:
                refs = turn.get("responds_to")
                if refs is None:
                    turn["responds_to"] = []
                elif type(refs) is int:
                    turn["responds_to"] = [refs]
                if turn.get("resolution") == "assistant_handoff":
                    turn["parts"] = [turn["text"]]
            try:
                validate(
                    d, users, events, group, group_interrupt, hesitation_handoff=hesitation_handoff,
                    backchannel_to_interruption=backchannel_to_interruption,
                    sequential_backchannels=not simultaneous, repeat_listener=repeat_listener,
                )
            except BackchannelTooLong:
                shorten_backchannels(d)
                validate(
                    d, users, events, group, group_interrupt, hesitation_handoff=hesitation_handoff,
                    backchannel_to_interruption=backchannel_to_interruption,
                    sequential_backchannels=not simultaneous, repeat_listener=repeat_listener,
                )
            break
        except (ValueError, KeyError, TypeError) as exc:
            (root/"staging"/f"{ident}-text-reject-{attempt}.txt").write_text(raw,encoding="utf-8")
            request["本次修正要求"] = f"重新生成一段完整剧本，解决：{exc}"
            if attempt == 1:
                raise
    d.update(dialogue_id=ident, scenario="side_talk" if "side_talk" in events else "multi_qa")
    for i, t in enumerate(d["turns"]):
        t.update(turn_id=i, gap_before_ms=-350 if t["kind"]=="interruption" else rng.randint(140, 320),
                 insert_ratio=None, events=[], trigger="cue" if t["speaker"]=="assistant" else None)
        if t["kind"] == "incomplete":
            t["pause_ms"] = rng.randint(*incomplete_pause_range(t))
        if i and d["turns"][i-1].get("resolution") == "assistant_handoff":
            t["gap_before_ms"] = 0
    spoken = {t["speaker"] for t in d["turns"]}
    voices = {k:v for k,v in mapping.items() if k in spoken}
    refs = reference_voices()
    d["profile"] = {"generator":"moss_scale_v1","tts_backend":"moss_onnx","voice_assignments":voices,
                    "voice_references":{v:{k:refs[v][k] for k in ("source_url","license","sha256")}
                                        for v in voices.values() if v in refs},
                    "topic_domain":request["领域"],"requested_events":events,
                    "listener_spacing_ms":0 if simultaneous else rng.randint(90, 180),
                    "listener_sequence":"simultaneous" if simultaneous else "sequential",
                    "requested_backchannel_group": group,
                    "requested_repeat_listener": group and repeat_listener,
                    "requested_simultaneous_interruption": group_interrupt,
                    "requested_hesitation_handoff": hesitation_handoff,
                    "requested_backchannel_transition": backchannel_to_interruption}
    atomic_json(path, d)
    return d


def validate(
    d, users, events, group, group_interrupt=False, *, hesitation_handoff=False,
    backchannel_to_interruption=False, sequential_backchannels=False, repeat_listener=False,
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
    interrupters = set()
    for i, t in enumerate(turns):
        speaker, kind, content = t["speaker"], t["kind"], t["text"]
        refs = t["responds_to"]
        targets = t["addressee"] if isinstance(t["addressee"], list) else [t["addressee"]]
        if (speaker not in allowed or not isinstance(content,str) or not content.strip() or len(content)>190
            or not targets or any(a not in allowed | {"third_party","all","none"} or a==speaker for a in targets)
            or not isinstance(refs,list) or any(type(j) is not int or not 0<=j<i for j in refs)):
            raise ValueError(f"第{i}轮角色、文本、指向或回应索引无效")
        if (speaker=="third_party_01") != (kind=="side_talk"):
            raise ValueError("第三方须标作 side_talk")
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
                or not isinstance(prefix, str) or not 2 <= len(prefix) <= 14
                or not content.startswith(prefix) or refs != [i-1]
                or turns[i-1]["speaker"] != "assistant"):
                raise ValueError("附和转打断需要用户在同一轮先短附和，并引用上一轮助手")
            pivot = content[len(prefix):].lstrip(" ，,。；;、：:…")
            if not any(pivot.startswith(word) and pivot[len(word):].strip(" ，,。！？") for word in ("不过", "但是", "等一下")):
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
            if len(content) > 14:
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
