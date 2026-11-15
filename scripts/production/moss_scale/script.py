"""Draft and validate a dialogue before synthesis."""
import json
import random

from scripts.config import TOPIC_DOMAINS
from scripts.text.utils.llm import chat
from scripts.text.voices import gender_of, reference_voices, voices_for_dialogue

from .storage import atomic_json


def generate(index, root):
    ident = f"moss-scale-20261115-{index:06d}"
    path = root / "scripts" / f"{ident}.json"
    if path.exists():
        draft = json.loads(path.read_text(encoding="utf-8"))
        for turn in draft["turns"]:
            if turn["kind"] == "incomplete" and not 500 <= turn["pause_ms"] <= 2000:
                turn["pause_ms"] = random.Random(f"{ident}-{turn['turn_id']}-pause").randint(500, 2000)
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
    group = count >= 2 and "backchannel" in events and (index == 0 or rng.random() < .35)
    group_interrupt = count >= 2 and "interruption" in events and rng.random() < .35
    interruption_chain = count >= 2 and "interruption" in events and not group_interrupt and rng.random() < .30
    system = """写一段自然的中文多人对话，仅输出 JSON 对象，字段 setting、topic、participants、turns。
participants 中每个人含 id、persona、style。turns 每项含 speaker、text、addressee、kind、responds_to。
kind 取 speech 或本次指定的事件。responds_to 是实际回应的过去轮次索引，从0开始。
实际说出的话写在 text，别写括号舞台指令、角色名标签、轮次或省略号占位。每轮1至2句，话题细节鲜活具体，不重复同一种开头或套话。
按指定事件创作，未指定事件不要硬加。普通轮次 kind=speech。助手应自然回应实际求助。"""
    event_rules = {
        "incomplete": "incomplete 是同一人话说到一半犹豫后继续说，额外给 parts 数组（2个实际说出的片段）；第二段可按语境用“嗯”“啊”“那个”“我想想”“怎么说呢”“等一下我捋捋”，也可以直接续说。语气词要符合说话人习惯，避免每条重复。随后接续前半句的意思。text 必须等于两个片段直接拼接。第一段不能是已结束的完整陈述，也不要为了插停顿无意义地口吃。",
        "backchannel": "backchannel 紧跟被附和的主轮，addressee 指向主轮说话人，responds_to 仅包含主轮索引。额外给 anchor_text，逐字复制主轮开头直到已说出完整事实，主播仍至少有8个字可继续说。附和本身 2 到 14 个字符，用“对对”“我也觉得”“有道理”等短反馈，不加解释或新观点。例：主轮“第一趟车七点开，我们最好提前二十分钟到站。”，anchor_text 只填“第一趟车七点开”，附和 text 可填“对，我也觉得”。不要把整句主轮当作锚点。多人附和连续写不同用户，引用同一主轮和同一 anchor_text。不要把附和放在 incomplete 或被打断的轮次上。",
        "interruption": "interruption 紧接被打断的轮，其上一轮写只来得及说出的半句，不补全。本轮抢话要具体回应已经听到的内容。如果要求多人抢话，连续写两位不同用户各自抢话，responds_to 都只指向同一个原发言者轮次，addressee 都指向原发言者。两人围绕同一个关键点，各只说一句自然的话。如果要求连续打断，严格按四轮：assistant未说完→用户A打断→assistant回应A并补充，仍未说完→不同的用户B打断这次补充；每轮responds_to引用前一轮。",
        "side_talk": "side_talk 由 third_party_01 说出，插在中段，前面有正式成员开口搭话且后续自然返回主线。第三方不进入 participants。",
    }
    for event in events:
        system += "\n" + event_rules[event]
    request = {
        "领域": TOPIC_DOMAINS[index % len(TOPIC_DOMAINS)],
        "关系": rng.choice(["邻居", "同事", "老朋友", "家人", "临时组队"]),
        "场景": rng.choice(["当面商量", "线上语音", "边做事边聊", "临出门确认安排"]),
        "轮数": turns, "用户及音色性别": {u: gender_of(mapping[u]) for u in users},
        "必须各自有实际台词的 speaker": users + ["assistant"], "助手实际发言次数": "3到5轮", "本条事件": events, "安排一次两人附和助手": group, "安排一次两人同时抢话": group_interrupt, "安排一次连续打断": interruption_chain,
    }
    system += "\n每段必须有至少三轮 speaker 精确写为 assistant 的真实发言。assistant 是同一场语音会话中的 AI，跟正式用户一起交谈，不加入 participants 数组。其他说话人仅使用请求给出的角色编号。指定事件必须落在独立话轮中，使用指定 kind。若指定列表为空，全部用 speech。轮数可以略微浮动。不要把主持人的反应或犹豫自动标成未指定的事件。"
    for attempt in range(2):
        raw = chat(system, json.dumps(request, ensure_ascii=False), temperature=.95, max_tokens=7000)
        try:
            d = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
            validate(d, users, events, group, group_interrupt, interruption_chain)
            break
        except (ValueError, KeyError, TypeError) as exc:
            (root/"staging"/f"{ident}-text-reject-{attempt}.txt").write_text(raw,encoding="utf-8")
            request["上次校验问题"] = str(exc)
            request["修正上次输出"] = raw
            if attempt == 1:
                raise
    d.update(dialogue_id=ident, scenario="side_talk" if "side_talk" in events else "multi_qa")
    simultaneous = rng.random() < .5
    for i, t in enumerate(d["turns"]):
        t.update(turn_id=i, gap_before_ms=-350 if t["kind"]=="interruption" else rng.randint(140, 320),
                 insert_ratio=None, events=[], trigger="cue" if t["speaker"]=="assistant" else None)
        if t["kind"]=="incomplete":
            t["pause_ms"] = rng.randint(500, 2000)
    spoken = {t["speaker"] for t in d["turns"]}
    voices = {k:v for k,v in mapping.items() if k in spoken}
    refs = reference_voices()
    d["profile"] = {"generator":"moss_scale_v1","tts_backend":"moss_onnx","voice_assignments":voices,
                    "voice_references":{v:{k:refs[v][k] for k in ("source_url","license","sha256")}
                                        for v in voices.values() if v in refs},
                    "topic_domain":request["领域"],"requested_events":events,
                    "listener_spacing_ms":0 if simultaneous else rng.randint(350, 650)}
    atomic_json(path, d)
    return d


def validate(d, users, events, group, group_interrupt=False, interruption_chain=False):
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
        if kind=="incomplete":
            parts=t["parts"]
            if not isinstance(parts,list) or len(parts)!=2 or any(not isinstance(x,str) or not x.strip() for x in parts) or "".join(parts)!=content or parts[0].rstrip().endswith(("。","？","！","?","!")):
                raise ValueError("incomplete 需要两个与正文一致的发声片段")
        if kind=="backchannel":
            anchor=t["anchor_text"]
            if (host is None or turns[host]["kind"]!="speech" or speaker==turns[host]["speaker"]
                or refs!=[host] or targets!=[turns[host]["speaker"]] or speaker in listeners
                or not isinstance(anchor,str) or not anchor.strip() or not turns[host]["text"].startswith(anchor)
                or (listeners and anchor!=turns[i-1]["anchor_text"])):
                raise ValueError(f"第{i}轮附和须回应同一主轮已说出的原文前缀，且说话人不同")
            if len(content) > 14:
                raise ValueError(f"第{i}轮附和 text 过长，最多14字符。请改为简短反馈。")
            if len(turns[host]["text"]) - len(anchor) < 8:
                raise ValueError(f"第{i}轮附和锚点太晚，主轮在 anchor_text 之后必须至少还剩8字符。")
            listeners.add(speaker)
            assistant_group |= turns[host]["speaker"]=="assistant" and len(listeners)>=2
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
    if group and not assistant_group:
        raise ValueError("缺少两名不同用户附和同一助手主轮")
    if interruption_chain:
        chain = any(
            [t["kind"] for t in turns[i:i+4]] == ["speech", "interruption", "speech", "interruption"]
            and turns[i]["speaker"] == turns[i+2]["speaker"] == "assistant"
            and turns[i+1]["speaker"] != turns[i+3]["speaker"]
            and all(turns[j]["responds_to"] == [j-1] for j in range(i+1,i+4))
            for i in range(len(turns)-3)
        )
        if not chain:
            raise ValueError("连续打断需包含助手回应A后再被不同用户B打断")
    if group_interrupt and not interrupt_group:
        raise ValueError("缺少两名不同用户抢同一个话轮")
