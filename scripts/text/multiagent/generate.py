"""多角色逐轮交互 v2：导演设定 → 历史/指向调度 → 隔离私有背景的角色生成。

覆盖脚手架显式安排 barge-in / AI 附和 / 问称呼；其余轮按真实指向继续。
这不是旧版泊松自由涌现对照臂，profile 中记录新版本和调度策略，不能混做旧实验。
只生成文本；公共校验、独立语义审核及可选增广由 pipeline 统一执行。
"""
from __future__ import annotations

import random
import re
import json
import os
from dataclasses import asdict
from ..quality import rules_block, coverage_plan,strong_question
from ..oneshot_agent.validate import build_dialogue
from ..oneshot_agent.prompts import topic_for_dialogue
from ..voices import voice_table_text

from pathlib import Path
from scripts.config import LOGS_DIR

from . import prompts
from ..utils.schema import Dialogue, Participant, Turn
from ..utils import llm
from .scheduler import Scheduler

_SPEAKER_PREFIX = re.compile(r"^(user_[a-z0-9]+|assistant)[:：]\s*")

MOCK_PERSONAS = [   # mock 自测用的小人物池（真实链路用 data/personas 大池）
    {"persona": "退休中学语文老师，65 岁，喜欢追根究底，习惯引用生活经验", "style": "耐心但啰嗦"},
    {"persona": "互联网公司产品经理，29 岁，节奏快，关注效率，说话直接", "style": "简洁理性"},
    {"persona": "大一新生，19 岁，好奇爱问，有时跑题，说话带口头禅", "style": "活泼好奇"},
    {"persona": "急诊科护士，32 岁，说话干脆利落，容易不耐烦", "style": "不耐烦"},
    {"persona": "自由职业插画师，27 岁，幽默爱开玩笑，喜欢反问", "style": "幽默诙谐"},
    {"persona": "健身教练，35 岁，说话充满能量，喜欢用比喻打气", "style": "热情外放"},
]


def mock_director(rng: random.Random, dialogue_id: str, scenario: str, num_users: int) -> Dialogue:
    """无 LLM 的规则剧本（用于链路自测）。"""
    d = Dialogue(dialogue_id=dialogue_id, scenario=scenario, setting="家庭客厅", topic="周末安排")
    for i in range(num_users):
        p = rng.choice(MOCK_PERSONAS)
        d.participants.append(Participant(
            id=f"user_{i + 1:02d}", persona=p["persona"], style=p["style"],
            speech_rate=round(rng.uniform(0.5, 1.5), 2)))
    return d


def exact_prefix_anchor(host,anchor):
    """Extend one uniquely quoted host substring to a prefix without moving its factual end point."""
    if not isinstance(anchor,str) or not anchor:return None
    if host.count(anchor)!=1:return None
    end=host.find(anchor)+len(anchor)
    if end>=len(host) or len(host)-end<6:return None
    return host[:end]

def llm_director(dialogue_id: str, scenario: str, num_users: int,
                 persona_pool: list[str] | None = None) -> Dialogue | None:
    """导演 Agent（LLM）：输出剧本 JSON → 解析为 Dialogue。"""
    domain=topic_for_dialogue(dialogue_id)
    out = llm.chat(prompts.director_prompt(scenario, num_users, persona_pool) + "\n本次主话题必须属于指定领域："+domain+"。场景发生在家里不等于家庭日常领域；请围绕该领域设定具体话题，不要固定聊电影或周末出游。\n音色性别必须对齐：" + voice_table_text(dialogue_id, num_users), "请设计剧本", temperature=0.8, max_tokens=1600)
    _trace(dialogue_id, {"stage":"director","output":out})
    script = prompts.parse_script(out)
    if script and isinstance(script.get("participants"), list):
        script["participants"] = [p for p in script["participants"] if p.get("id") != "assistant"]
    if not script or len(script.get("participants", [])) != num_users:
        return None
    d = Dialogue(dialogue_id=dialogue_id, scenario=scenario,
                 setting=script.get("setting", ""), topic=script.get("topic", ""))
    for i, pp in enumerate(script["participants"][:num_users]):
        d.participants.append(Participant(
            id=f"user_{i + 1:02d}", persona=pp.get("persona", ""), style=pp.get("style", ""),
            speech_rate=float(pp.get("speech_rate", 1.0))))
    return d


def _trace(dialogue_id, record):
    folder=LOGS_DIR/"multiagent_v2"
    folder.mkdir(parents=True,exist_ok=True)
    safe=re.sub(r"[^a-zA-Z0-9_.-]","_",dialogue_id)
    with (folder/(safe+".jsonl")).open("a",encoding="utf-8") as f:
        f.write(json.dumps(record,ensure_ascii=False)+"\n");f.flush()


def _desc(d: Dialogue, pid: str = "assistant") -> str:
    """No cross-role private personas. Public IDs are metadata, not spoken names."""
    rows=[]
    for p in d.participants:
        rows.append({"id":p.id, **({"persona":p.persona,"style":p.style} if pid==p.id else {})})
    return json.dumps(rows,ensure_ascii=False)


def _history(d: Dialogue) -> str:
    # Full dialogue: preserving introductions and public identity bindings matters more than last-6 truncation.
    return json.dumps([dict(asdict(t),turn_id=i) for i,t in enumerate(d.turns)],ensure_ascii=False)


def _step(d, i, turns, rng):
    users=[p.id for p in d.participants if p.state in ("active","returned")]
    module=d.profile["required_modules"]
    # A small deterministic coverage scaffold; role agents still choose all content using actual history.
    if i==0: return users[0],"speech","assistant","用一个具体问题开启主话题，暂不自报姓名。"
    if "identity_ask" in module:
        if i==1: return "assistant","speech",users[0],"回应刚才的问题，顺便友好请教对方称呼；不能猜姓名。"
        if i==2: return users[0],"speech","assistant","自然自我介绍姓名，然后接着刚才的话题。"
        if i==3: return "assistant","speech",users[0],"使用刚刚才得知的称呼，具体回应主话题。"
    if "barge_in" in module:
        if i==1: return "assistant","speech",users[0],"回答问题，但本轮要被打断，只写尚未说完的半句，不用句号补全。"
        if i==2: return users[0],"interruption","assistant","针对 AI 刚才的半句抢话：提出具体修正或新限制。"
        if i==3: return "assistant","speech",users[0],"停止原来的展开，回应用户刚提出的修正/新限制。"
    if "ai_backchannel" in module:
        if i==1: return users[0],"speech","assistant","向 AI 补充个人经历或事实陈述；前半段就有可被轻声附和的内容，不提问、不转折。"
        if i==2: return "assistant","backchannel",users[0],"只给在听的短附和（1~6字），不引入新判断。"
    # Explicit side-talk block, followed by at least three mainline turns.
    if d.scenario=="side_talk" and turns>=10:
        start=max(4,turns//2-1)
        if i==start: return users[0],"speech","third_party","自然转向路过的服务人员/门外的人，先招呼，不提前回应尚未出现的内容。"
        if i==start+1: return "third_party_01","side_talk",users[0],"你是临时出现的第三方，只说一句具体事情，不参与主话题。"
        if i==start+2: return users[0],"speech","third_party","只回应刚才第三方的具体事情。"
        if i==start+3: return users[0],"speech","assistant","插曲结束，明确请 AI 继续插曲前的具体主话题。"
        if i==start+4: return "assistant","speech",users[0],"回到插曲前的具体主话题，承接用户请求。"
    last=d.turns[-1]
    # Addressed users get the next floor, avoiding self-answers under random sampling.
    targets=last.addressee if isinstance(last.addressee,list) else [last.addressee]
    addressed=[u for u in users if u in targets and u!=last.speaker]
    if last.kind!="backchannel" and addressed:
        pid=addressed[0]
        not_spoken=[u for u in users if u!=pid and not any(t.speaker==u for t in d.turns)]
        if last.speaker=="assistant" and not_spoken:
            return pid,"speech",not_spoken[0],"先承接 AI 刚才的具体内容，再自然邀请这位尚未发言的用户表达意见；不要猜姓名。"
        if last.speaker!="assistant" and rng.random()<.3:
            return pid,"speech","assistant","回应刚才对你说的具体内容，然后明确请 AI 就其中问题提供建议。"
        return pid,"speech",last.speaker,"回应上一轮明确对你说的话；不要替别人回答，也不要改变自己身份。"
    if last.speaker!="assistant" and (last.addressee=="assistant" or isinstance(last.addressee,list) and "assistant" in last.addressee):
        if last.kind!="backchannel": return "assistant","speech",last.speaker,"回应实际历史里指向你的问题或内容。"
    pid=rng.choice(users)
    others=[p for p in users if p!=pid]
    addr=rng.choice(["assistant"]+others)
    return pid,"speech",addr,"承接历史的具体内容，不重复前面的问答；可以提出新限制或真实观点。"


def speak(d: Dialogue, sched: Scheduler, pid: str, mock: bool, rng: random.Random,
          kind="speech", addr=None, instruction="", turns=12, planned=None) -> bool:
    addr=addr or sched.pick_addressee(pid)
    host=next((i for i in range(len(d.turns)-1,-1,-1) if d.turns[i].kind!="backchannel"),None)
    gap=-450 if kind=="interruption" else 250
    if mock:
        text="嗯嗯" if kind=="backchannel" else "我们接着讨论周末安排，先确认时间和预算。"
        obj={"text":text,"responds_to":[host] if host is not None else []}
    else:
        own=_desc(d,pid)
        system=("当前轮的 speaker/addressee 是强制元数据，你就是 speaker，不能回答别人被问的问题。assistant 是提供帮助的 AI，不编造看电影、订票等亲身生活经历；可谈作品和提供建议。每轮最多两句短口语。你只扮演当前说话人。只输出 JSON 对象，必须含 text（字符串）、responds_to（整数数组）。不要输出 schema 或解释。别人的私有人设不对你公开，禁止替别人说话。\n"
                +rules_block(d.dialogue_id,turns)+"\n你可见的人物信息："+own)
        request={"setting":d.setting,"topic":d.topic,"history":json.loads(_history(d)),
                 "current_turn_id":len(d.turns),"speaker":pid,"kind":kind,"addressee":addr,
                 "instruction":instruction,"output_schema":{"text":"只输出当前角色的台词，不带编号",
                 "responds_to":"过去真实回应轮次的索引数组", "anchor_text":"仅附和：宿主开头到完整事实小句的原文前缀，后面还剩至少6字", "insert_ratio":"仅附和，0.4~0.75"}}
        raw=llm.chat(system,json.dumps(request,ensure_ascii=False),temperature=.8,max_tokens=700)
        _trace(d.dialogue_id,{"stage":"turn","turn_id":len(d.turns),"speaker":pid,"output":raw})
        obj=prompts.parse_script(raw)
        if not isinstance(obj,dict): return False
        if planned is not None and planned.get("_allow_ai_request") is False and strong_question(str(obj.get("text",""))):
            request["previous_output"]=obj
            request["validation_error"]="下一主轮不是AI，本轮没有AI回答槽。不能只删问号：‘差多少/是不是/怎么做’仍然是在索取回答。请换成你自己的观察、已知限制或接下来的打算，补充一条自然信息，不新增问题、不要求别人分析；保持speaker和addressee。"
            raw=llm.chat(system,json.dumps(request,ensure_ascii=False),temperature=.3,max_tokens=700)
            _trace(d.dialogue_id,{"stage":"turn_retry","turn_id":len(d.turns),"speaker":pid,"output":raw})
            obj=prompts.parse_script(raw)
            if not isinstance(obj,dict) or strong_question(str(obj.get("text",""))):return False
    if planned is not None:
        obj["responds_to"]=planned["responds_to"]
    if kind=="backchannel" and host is not None:
        original=obj.get("anchor_text");normalized=exact_prefix_anchor(d.turns[host].text,original)
        if normalized is None and not mock:
            request["validation_error"]="anchor_text必须逐字复制宿主从第一个字开始的真实前缀，结束于已经完整表达的事实，宿主后面至少还剩6字。不要改写宿主，不要选未来内容，不新增提问；短附和只用对呀或明白。"
            raw=llm.chat(system,json.dumps(request,ensure_ascii=False),temperature=.2,max_tokens=700)
            _trace(d.dialogue_id,{"stage":"anchor_retry","turn_id":len(d.turns),"speaker":pid,"output":raw})
            obj=prompts.parse_script(raw)
            if not isinstance(obj,dict):return False
            original=obj.get("anchor_text");normalized=exact_prefix_anchor(d.turns[host].text,original)
        if normalized is None:return False
        if normalized!=original:
            d.profile.setdefault("anchor_normalizations",[]).append({"turn_id":len(d.turns),"original_anchor":original,"prefix":normalized,"method":"unique_substring_prefix_extension","referenced_end_unchanged":True})
        obj["anchor_text"]=normalized
    text=str(obj.get("text","")).strip()
    if not text: return False
    rt=obj.get("responds_to",[])
    if not isinstance(rt,list) or any(type(x)is not int or not 0<=x<len(d.turns) for x in rt): return False
    if kind in ("backchannel","interruption"):
        if host is None or d.turns[host].speaker==pid: return False
        # These are scheduled structural fields, not LLM-guessed content.
        rt=[host]
        if kind=="backchannel": addr=d.turns[host].speaker
    try: ratio=float(obj.get("insert_ratio",.6)) if kind=="backchannel" else None
    except (ValueError,TypeError): return False
    d.turns.append(Turn(turn_id=len(d.turns),speaker=pid,text=text,addressee=addr,
                       kind=kind,gap_before_ms=planned["gap_before_ms"] if planned else gap,insert_ratio=ratio,responds_to=rt,
                       anchor_text=obj.get("anchor_text") if kind=="backchannel" else None,
                       trigger=planned.get("trigger") if planned else ("cue" if pid=="assistant" else None)))
    sched.after_speech(pid)
    return True


def generate_dialogue_multi(rng: random.Random, dialogue_id: str, scenario: str,
                            num_users: int, turns: int, mock: bool,
                            persona_pool: list[str] | None = None) -> Dialogue | None:
    if not 1<=num_users<=6 or turns<4: raise ValueError("users=1..6, turns>=4 required")
    d=mock_director(rng,dialogue_id,scenario,num_users) if mock else llm_director(dialogue_id,scenario,num_users,persona_pool)
    if d is None: return None
    d.profile={"topic_domain":topic_for_dialogue(dialogue_id),"director_topic_policy":"shared_crc_topic_v1","required_modules":coverage_plan(dialogue_id,turns),"generator":"multi_agent_v2",
               "scheduler":"history_conditioned_coverage_scaffold", "require_semantic_anchors":os.environ.get("DUPLEX_PRODUCTION")=="1"}
    plan=None
    if os.environ.get("DUPLEX_PRODUCTION")=="1":
        from ..oneshot_agent.production_plan import production_plan
        plan=production_plan(dialogue_id,scenario,num_users,turns,d.profile)
        if plan is None:return None
        d.profile.update(generator="multi_agent_production_v3",scheduler="varied_semantic_anchor_graph")
    sched=Scheduler(d,rng)
    for i in range(turns):
        slot=plan[i] if plan else None
        if slot:
            pid,kind,addr,instruction=slot["speaker"],slot["kind"],slot["addressee"],slot["_instruction"]
            if kind=="backchannel":instruction+=" 本生产音色仅使用已校准的‘对呀’或‘明白’，必须是听完 anchor_text 事实后自然成立的回应。"
        else:pid,kind,addr,instruction=_step(d,i,turns,rng)
        if not speak(d,sched,pid,mock,rng,kind,addr,instruction,turns,planned=slot): return None
    checked,fixes=build_dialogue(json.loads(d.to_jsonl_line()),dialogue_id,scenario,turns_requested=turns)
    _trace(dialogue_id,{"stage":"validation","fixes":fixes,"passed":checked is not None,"dialogue":json.loads(d.to_jsonl_line())})
    if checked is not None: checked.profile=d.profile
    return checked
