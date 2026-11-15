"""Draft and validate a dialogue before synthesis."""
import json
import random

from scripts.config import TOPIC_DOMAINS
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
        json.dumps(candidates, ensure_ascii=False), temperature=.3, max_tokens=500,
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
    system = """写一段自然的中文多人对话，仅输出 JSON 对象，字段 setting、topic、participants、turns。
participants 中每个人含 id、persona、style。turns 每项含 speaker、text、addressee、kind、responds_to。
kind 取 speech 或本次指定的事件。responds_to 是实际回应的过去轮次索引，从0开始。
实际说出的话写在 text，别写括号舞台指令、角色名标签、轮次或省略号占位。每轮1至2句，话题细节鲜活具体，不重复同一种开头或套话。
按指定事件创作，未指定事件不要硬加。普通轮次 kind=speech。助手应自然回应实际求助。"""
    event_rules = {
        "incomplete": "incomplete 是同一人话说到一半犹豫后继续说，额外给 parts 数组（2个实际说出的片段）；第二段可按语境用“嗯”“啊”“那个”“我想想”“怎么说呢”“等一下我捋捋”，也可以直接续说。语气词要符合说话人习惯，避免每条重复。随后接续前半句的意思。text 必须等于两个片段直接拼接。第一段不能是已结束的完整陈述，也不要为了插停顿无意义地口吃。",
        "backchannel": "backchannel 紧跟被附和的主轮，addressee 指向主轮说话人，responds_to 仅包含主轮索引。额外给 anchor_text，逐字复制主轮开头直到已说出完整事实，主播仍至少有8个字可继续说。附和本身 2 到 14 个字符，用“对对”“我也觉得”“有道理”等短反馈，不加解释或新观点。例：主轮“第一趟车七点开，我们最好提前二十分钟到站。”，anchor_text 只填“第一趟车七点开”，附和 text 可填“对，我也觉得”。不要把整句主轮当作锚点。同一组附和连续写在剧本里，引用同一主轮和同一 anchor_text。不要把附和放在 incomplete 或被打断的轮次上。",
        "interruption": "interruption 紧接被打断的轮，其上一轮写只来得及说出的半句，不补全。本轮抢话要具体回应已经听到的内容。如果要求多人抢话，连续写两位不同用户各自抢话，responds_to 都只指向同一个原发言者轮次，addressee 都指向原发言者。两人围绕同一个关键点，各只说一句自然的话。两次打断若指向不同主轮，分别标注。",
        "side_talk": "side_talk 由 third_party_01 说出，插在中段，前面有正式成员开口搭话且后续自然返回主线。第三方不进入 participants。",
    }
    for event in events:
        if event == "incomplete" and hesitation_handoff:
            system += (
                "\n本条用犹豫接话实现 incomplete。让一位用户说到一半迟疑，例如“我本来想明天先去那个”，"
                "不用句号收尾。把该轮设为 kind=incomplete、resolution=assistant_handoff，"
                "parts 只含一个与 text 相同的发声片段。下一轮必须是 assistant 的 speech，"
                "addressee 只指这位用户，responds_to 只引该未完成轮。中间不要写静音轮。"
                "助手等待较长停顿后，按用户已经说出的需求温和地提出建议或给一个选项，"
                "例如“没关系，我先给你个思路，你看合不合适。”措辞按情境变化，避免催促或猜测没说出的信息。"
            )
        else:
            system += "\n" + event_rules[event]
    if "backchannel" in events:
        if simultaneous:
            system += "\n同时附和由两位不同的用户发出，合成时他们会同时开口。"
        elif repeat_listener:
            system += "\n连续附和：同一位用户先后两次简短表示认同。写成两个独立短反馈，措辞有自然变化，两句之间有短间隔。"
        else:
            system += "\n连续附和：两位不同用户先后表示认同，第二位等第一位说完再开口。"
    if backchannel_to_interruption:
        system += (
            "\n安排恰好一处附和转打断。助手先给出一个可认可的具体信息，再继续说，直到被打断。"
            "同一位用户的一条 interruption 先短暂认同，再用“不过”“但是”或“等一下”引出具体补充，"
            "例如“我觉得可以，不过集合地点得换一下”。backchannel_prefix 逐字复制本轮开头的短附和，"
            "例如“我觉得可以”。附和与后半句写在同一个 text 中。不要另写一轮 backchannel。"
        )
    request = {
        "领域": TOPIC_DOMAINS[index % len(TOPIC_DOMAINS)],
        "关系": rng.choice(["邻居", "同事", "老朋友", "家人", "临时组队"]),
        "场景": rng.choice(["当面商量", "线上语音", "边做事边聊", "临出门确认安排"]),
        "轮数": turns, "用户及音色性别": {u: gender_of(mapping[u]) for u in users},
        "必须各自有实际台词的 speaker": users + ["assistant"], "助手实际发言次数": "3到5轮", "本条事件": events, "对助手安排一组附和": group, "安排一次两人同时抢话": group_interrupt, "安排一次附和转打断": backchannel_to_interruption,
    }
    system += "\n每段必须有至少三轮 speaker 精确写为 assistant 的真实发言。assistant 是同一场语音会话中的 AI，跟正式用户一起交谈，不加入 participants 数组。其他说话人仅使用请求给出的角色编号。指定事件必须落在独立话轮中，使用指定 kind。若指定列表为空，全部用 speech。轮数可以略微浮动。不要把主持人的反应或犹豫自动标成未指定的事件。"
    for attempt in range(2):
        raw = chat(system, json.dumps(request, ensure_ascii=False), temperature=.95, max_tokens=7000)
        try:
            d = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
            if hesitation_handoff or backchannel_to_interruption:
                for turn in d["turns"]:
                    refs = turn["responds_to"]
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
                if attempt:
                    raise
                shorten_backchannels(d)
                validate(
                    d, users, events, group, group_interrupt, hesitation_handoff=hesitation_handoff,
                    backchannel_to_interruption=backchannel_to_interruption,
                    sequential_backchannels=not simultaneous, repeat_listener=repeat_listener,
                )
            break
        except (ValueError, KeyError, TypeError) as exc:
            (root/"staging"/f"{ident}-text-reject-{attempt}.txt").write_text(raw,encoding="utf-8")
            request["上次校验问题"] = str(exc)
            request["修正上次输出"] = raw
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
                    "listener_sequence":"simultaneous" if simultaneous else "sequential"}
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
