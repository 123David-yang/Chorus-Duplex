"""One-call script writing on a predeclared interaction graph, not per-role generation."""
from __future__ import annotations
import json
from .validate import build_dialogue
from .profile import profile_block
from .prompts import topic_for_dialogue
from ..quality import coverage_plan,rules_block,audit,strong_question
from ..voices import voice_table_text
from ..utils import llm
from ..judge.review import parse_object
from scripts.config import SCENARIOS,LOGS_DIR
import re
import os


def turn_plan(dialogue_id,scenario,users,turns,profile=None):
    if os.environ.get("DUPLEX_PRODUCTION")=="1":
        from .production_plan import production_plan
        return production_plan(dialogue_id,scenario,users,turns,profile)
    if turns<8 or scenario=="side_talk" and turns<10: return None
    module=coverage_plan(dialogue_id,turns)[0]
    side_start=min(turns//2,turns-6) if scenario=="side_talk" else None
    side=set(range(side_start,side_start+3)) if side_start is not None else set()
    end_ai=turns-4 if turns>=14 else turns-2
    if end_ai in side: return None
    ai={1,end_ai}
    target=max(3,(profile or {}).get("assistant_turns",3))
    low=(profile or {}).get("intensity")=="low"
    spacing=4 if low else 2
    for pos in range(3,turns-1):
        if len(ai)>=target: break
        if pos in side or any(abs(pos-a)<spacing for a in ai): continue
        ai.add(pos)
    if profile and abs(len(ai)-profile["assistant_turns"])>1: return None
    slots=[];counter=0
    for i in range(turns):
        if i in ai:
            speaker="assistant"
        elif side_start is not None and i==side_start+1:
            speaker="third_party_01"
        elif side_start is not None and i==side_start+2:
            speaker=slots[side_start]["speaker"]
        elif i==2 and module in ("identity_ask","barge_in"):
            speaker="user_01"
        else:
            speaker=f"user_{counter%users+1:02d}"
            counter+=1
        slots.append({"turn_id":i,"speaker":speaker,"kind":"speech","text":"",
                      "addressee":"assistant","responds_to":[],"gap_before_ms":250,
                      "events":[],"_instruction":"承接已经说出的具体内容，自然口语1~2句；不猜其他人的私有背景。"})
    if side_start is not None:
        caller=slots[side_start]["speaker"]
        slots[side_start].update(addressee="third_party",_instruction="只向门外的人招呼并询问来意，例如“您好，请问有什么事？”；第三方下一轮才告知来意。本轮严禁说放门口、谢谢、辛苦、收到、知道了等处理/回应语。")
        slots[side_start+1].update(speaker="third_party_01",kind="side_talk",addressee=caller,
                                  _instruction="第三方简短交代一件具体事情，不能参与主话题。")
        slots[side_start+2].update(speaker=caller,addressee="third_party",_instruction="回应刚才第三方的具体事情，然后结束插曲。")
    for i,t in enumerate(slots):
        if i==0:continue
        prev=slots[i-1]
        if i not in side:
            if t["speaker"]==prev["speaker"]:
                refs=[j for j in range(i) if slots[j]["speaker"]!=t["speaker"] and not slots[j]["speaker"].startswith("third_party")]
                if refs: t.update(addressee=slots[refs[-1]]["speaker"],responds_to=[refs[-1]])
            else:
                t.update(addressee=prev["speaker"] if not prev["speaker"].startswith("third_party") else "all",responds_to=[i-1])
        else:t["responds_to"]=[i-1] if i!=side_start else []
        if t["speaker"]=="assistant":
            t["trigger"]="volunteer" if prev["addressee"]=="third_party" else "cue"
            if i>1:
                t["_instruction"]="根据之前已经说出的事实提供具体帮助；不编造自己的真人经历。"
                if i-1 not in side:
                    prev["addressee"]="assistant"
                    prev["_instruction"]+=" 末尾明确请 AI 提供相关帮助。"
    slots[0]["_instruction"]="向 AI 提一个主话题的具体问题，暂不自报姓名。"
    if module=="ai_backchannel":
        slots[0]["_instruction"]="向 AI 讲述一个具体经历/事实，至少15字；前半段即可支持在听的附和，不提问、不转折。"
        slots[1].update(kind="backchannel",addressee="user_01",responds_to=[0],insert_ratio=.6,gap_before_ms=0,
                        _instruction="只输出短附和如‘嗯嗯’，表示在听；不作长回答，不抢话。")
    elif module=="barge_in":
        slots[1]["_instruction"]="回答用户，但本轮要被打断：只写确实没说完的半句，末尾停在词组中间，不补全。"
        slots[2].update(kind="interruption",addressee="assistant",responds_to=[1],gap_before_ms=-450,
                        _instruction="针对 AI 刚才的半句抢话，提出一个具体修正或新限制。")
        response=min(i for i in ai if i>2)
        slots[response]["responds_to"]=list(dict.fromkeys([2]+slots[response]["responds_to"]))
        slots[response]["_instruction"]="停止原来的展开，明确回应用户刚刚提出的修正/新限制；不要继续被打断的旧方案。"
    else:
        slots[1]["_instruction"]="简短承接主话题，友好请教 user_01 怎么称呼，不得猜名字。"
        slots[2].update(addressee="assistant",responds_to=[1],_instruction="你是被请教称呼的用户：自然自报一个名字/昵称，带回主话题。")
        response=min(i for i in ai if i>2)
        slots[response]["responds_to"]=list(dict.fromkeys([2]+slots[response]["responds_to"]))
        slots[response]["_instruction"]="先用被问用户刚刚自报的名字称呼对方，再回应主话题的具体内容。"
    if side_start is not None:
        returning=min(i for i in ai if i>side_start+2)
        slots[returning]["_instruction"]+=" 插曲已结束，回到插曲前的具体话题，不能只谈外卖/服务人员。"
    if not {f"user_{i:02d}" for i in range(1,users+1)} <= {t["speaker"] for t in slots}: return None
    return slots


def generate_planned(dialogue_id,scenario,num_users,turns,profile,persona_pool):
    plan=turn_plan(dialogue_id,scenario,num_users,turns,profile)
    if plan is None:return None
    system=("你是多人双工口语剧本编剧，一次性写完整剧本。交互图已由程序确定，不能改变角色/顺序/轮数。"
            "你只填写每个槽位对应的台词；不是逐角色交互。assistant 是 AI，不能编造个人生活经历。"
            "只输出 JSON 对象，含 setting、topic、participants（只列指定数量的用户）、utterances。"
            'participants 必须是对象数组，如 [{"id":"user_01","persona":"具体人设","style":"说话风格"}]，不能是 ID 字符串数组。'
            'utterances 格式如 [{"turn_id":0,"text":"实际台词"}, ...]，必须填写全部槽位。'
            "utterances 必须按槽位顺序，长度相等；每项含 turn_id、text。"
            "每条 backchannel 还必须给 anchor_text：逐字复制宿主从开头到一个已完整陈述事实的小句的前缀（含标点）。"
            "anchor_text 至少3字，宿主后面必须还剩至少6字；该前缀说完就足以支撑附和，不能引用尚未说出的结论。"
            "宿主不能是疑问或先抑后扬的转折；附和可用‘对’‘明白’‘是的’等1~3字，避免把应答起头当附和。"
            "人物姓名/背景不算公开知识，台词里介绍后才能被 AI 使用。每位用户必须至少说一句。"
            "话题必须属于给定 topic_domain；人物背景可调整到该领域，保留性格，避免都聊科研。"
            +rules_block(dialogue_id,turns)+"\n音色："+voice_table_text(dialogue_id,num_users)
            +(profile_block(profile) if profile else ""))
    if os.environ.get("DUPLEX_PRODUCTION")=="1" and os.environ.get("TTS_BACKEND") in ("kokoro", "moss_onnx"):
        system+="\n当前引擎的短附和仅使用已标定的‘对呀’或‘明白’，按宿主已说出的语义选择；不要使用‘嗯/嗯嗯’或用长回答冒充附和。不要为了附和编造结论。"
    user={"scenario":scenario,"scenario_description":SCENARIOS.get(scenario,""),"num_users":num_users,
          "persona_pool":persona_pool,"topic_domain":topic_for_dialogue(dialogue_id),"turn_slots":plan}
    folder=LOGS_DIR/"oneshot_v3";folder.mkdir(parents=True,exist_ok=True)
    path=folder/(re.sub(r"[^A-Za-z0-9_.-]","_",dialogue_id)+".jsonl")
    feedback=[]
    previous=None
    for attempt in range(2):
        raw=llm.chat(system,json.dumps({**user,"previous_errors":feedback,"previous_output":previous},ensure_ascii=False),temperature=.8 if attempt==0 else .3,max_tokens=4096)
        try:
            obj=parse_object(raw)
            utterances=obj["utterances"]
            if not isinstance(utterances,list) or len(utterances)!=len(plan) or any(not isinstance(u,dict) or type(u.get("turn_id")) is not int or not isinstance(u.get("text"),str) or not u["text"].strip() for u in utterances):raise ValueError("utterance_count")
            if [u.get("turn_id") for u in utterances]!=list(range(len(plan))):raise ValueError("utterance_order")
            for slot,u in zip(plan,utterances):
                if slot.get("_allow_ai_request") is False and strong_question(u["text"]):
                    raise ValueError(f"第{slot['turn_id']}轮原句【{u['text']}】：下一主轮不是AI，没有回答槽。必须改为说话人自己的观察、已知限制或具体打算，不能索取回答。不能只删问号；‘谁来管/差多少/是不是/怎么做’仍然是问题。保留本轮speaker和addressee，不要把这个问题搬到另一个禁问轮次；其他合格轮次保持原意。")
            obj["turns"]=[{**{k:v for k,v in slot.items() if not k.startswith("_")},"text":u["text"],"anchor_text":u.get("anchor_text") if slot["kind"]=="backchannel" else None} for slot,u in zip(plan,utterances)]
            d,feedback=build_dialogue(obj,dialogue_id,scenario,profile,turns_requested=turns)
            if d is not None and len(d.participants)!=num_users:d=None;feedback=["participant_count"]
            if d is not None:
                d.profile.update(require_semantic_anchors=os.environ.get("DUPLEX_PRODUCTION")=="1",generator="oneshot_production_v1" if os.environ.get("DUPLEX_PRODUCTION")=="1" else "oneshot_v3_planned",required_modules=coverage_plan(dialogue_id,turns),topic_domain=topic_for_dialogue(dialogue_id))
                if os.environ.get("DUPLEX_PRODUCTION")=="1":d.profile["production_plan_version"]="semantic-graph-v3"
                checks=audit(d)
                if not checks["passed"]:d=None;feedback=checks["issues"]
        except (ValueError,KeyError,TypeError) as exc:
            d=None;feedback=[type(exc).__name__+":"+str(exc)]
        previous=raw
        with path.open("a",encoding="utf-8") as f:
            f.write(json.dumps({"attempt":attempt,"raw":raw,"validation":feedback,"passed":d is not None},ensure_ascii=False)+"\n");f.flush()
        if d is not None:return d
    return None
