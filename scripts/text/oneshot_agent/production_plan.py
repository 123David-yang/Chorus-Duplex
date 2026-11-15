"""Deterministically varied production interaction graphs; labels precede content generation."""
from __future__ import annotations
import random,zlib
from ..quality import coverage_plan

def production_plan(dialogue_id,scenario,users,turns,profile=None):
    if turns<16 or not 1<=users<=6:return None
    rng=random.Random(zlib.crc32(("production-plan:"+dialogue_id).encode()))
    module=coverage_plan(dialogue_id,turns)[0]
    last_ai=turns-rng.choice([3,4])
    side_start=None
    if scenario=="side_talk":
        choices=list(range(max(6,turns//3),min(last_ai-3,turns-6,2*turns//3)+1))
        side_start=rng.choice(choices)
    first_ai=rng.randint(1,min(4,(side_start-3) if side_start else 4))
    follow=first_ai+1
    side=set(range(side_start,side_start+3)) if side_start is not None else set()
    ai={first_ai,last_ai}
    if module in ("barge_in","identity_ask"):
        ai.add(first_ai+2)
        if module=="barge_in" and profile is not None:profile["duplex_response_pairs"]=[[first_ai,first_ai+2]]
        if module=="identity_ask" and profile is not None:profile["identity_response_pairs"]=[[first_ai,first_ai+2]]
    if profile is not None:profile["production_plan_version"]="semantic-graph-v3"
    target=(profile or {}).get("assistant_turns",max(3,round(turns*.25)) if scenario=="multi_qa" else 3)
    controlled=profile is not None and "assistant_turns" in profile
    if users==1:
        minimum=4 if controlled else max(4,turns//3)
        target=max(target,minimum)
        if controlled and profile["assistant_turns"]!=target:
            profile["unconditioned_assistant_turns"]=profile["assistant_turns"]
            profile["assistant_turns"]=target
    spacing=4 if (profile or {}).get("intensity")=="low" else 2
    available=[i for i in range(first_ai+2,turns-1) if i not in side]
    while len(ai)<target:
        eligible=[i for i in available if min(abs(i-j) for j in ai)>=spacing]
        if not eligible:break
        distances={i:min(abs(i-j) for j in ai) for i in eligible};best=max(distances.values())
        ai.add(rng.choice([i for i,d in distances.items() if d==best]))
    if profile and abs(len(ai)-target)>1:return None
    slots=[];counter=0
    for i in range(turns):
        if i in ai:speaker="assistant"
        elif side_start is not None and i==side_start+1:speaker="third_party_01"
        elif side_start is not None and i==side_start+2:speaker=slots[side_start]["speaker"]
        elif i==follow and module in ("identity_ask","barge_in"):speaker=slots[first_ai-1]["speaker"]
        else:speaker=f"user_{counter%users+1:02d}";counter+=1
        refs=[j for j,t in enumerate(slots) if t["speaker"]!=speaker and not t["speaker"].startswith("third_party")]
        host=refs[-1] if refs else None
        slots.append({"turn_id":i,"speaker":speaker,"kind":"speech","text":"","gap_before_ms":rng.choice([180,220,250,300]),
            "addressee":slots[host]["speaker"] if host is not None else "assistant","responds_to":[host] if host is not None else [],"events":[],
            "_instruction":"自然接续已说出的具体内容，1~2句口语；背景不是公开知识，不使用未介绍的姓名。"})
    if side_start is not None:
        caller=slots[side_start]["speaker"]
        slots[side_start].update(addressee="third_party",responds_to=[],_instruction="只向第三方招呼/询问来意，禁止提前说放门口、谢谢、收到等应答；对方下一轮才交代事情。场景可为服务人员/邻居/电话等。")
        slots[side_start+1].update(addressee=caller,responds_to=[side_start],kind="side_talk",_instruction="你是第三方，简短交代具体事情，不谈主话题，不擅自为正式用户起名。")
        slots[side_start+2].update(addressee="third_party",responds_to=[side_start+1],_instruction="回应第三方刚交代的具体事情，结束插曲。")
    for i in ai:
        t=slots[i];t["trigger"]="volunteer"
        if i-1 not in side:
            slots[i-1]["addressee"]="assistant"
            slots[i-1]["_instruction"]+=" 结尾明确向 AI 请教具体问题。"
            t["trigger"]="cue"
        if i==last_ai and side_start is not None:t["_instruction"]+=" 插曲后回到插曲前的具体主话题。"
    target_user=slots[first_ai-1]["speaker"]
    slots[first_ai].update(addressee=target_user,responds_to=[first_ai-1])
    slots[0]["_instruction"]="自然开启主话题，只陈述个人需求/现状，不提问、不直接请求AI解答；具体问题留在AI发言前的指定轮次。不急着介绍全部人物。"
    if module=="ai_backchannel":
        host=first_ai-1
        slots[host].update(addressee="assistant",_instruction="向 AI 讲已成立的具体经历/事实；写至少两个可连续陈述的小句，第一小句完整，后面还剩至少6字；不提问、不做否定转折，便于在第一小句结束后附和。")
        slots[first_ai].update(kind="backchannel",gap_before_ms=0,insert_ratio=.6,addressee=slots[host]["speaker"],responds_to=[host],
                              _instruction="只给1~3字的自然附和，例如‘对’‘明白’‘嗯’；anchor_text 必须是宿主从开头到已成立事实小句的精确前缀，后面还剩至少6字。")
    elif module=="identity_ask":
        slots[first_ai-1]["_instruction"]="接住主话题，表示大家可以先认识一下再慢慢聊；本轮不问AI具体解决办法，为下一轮自然询问称呼留出口。"
        slots[first_ai]["_instruction"]=f"承接主话题，友好询问刚才发言的 {target_user} 怎么称呼，绝不猜姓名。"
        slots[follow].update(addressee="assistant",responds_to=[first_ai],_instruction="你是被问称呼的用户，简单自报一个名字/昵称，带回主话题。")
        response=min(i for i in ai if i>follow)
        slots[response]["responds_to"]=list(dict.fromkeys([follow]+slots[response]["responds_to"]))
        slots[response]["_instruction"]+=f" 用被问用户刚刚自报的名字自然称呼对方，再提供具体帮助；禁止念角色 ID 或轮次。"
    else:
        slots[first_ai]["_instruction"]="回答需求，但本轮要被抢话；只写确实没说完的半句，不能是已经完整结束的一句话。"
        slots[follow].update(kind="interruption",gap_before_ms=rng.choice([-350,-450,-550]),addressee="assistant",responds_to=[first_ai],_instruction="针对 AI 刚才的半句抢话，提出具体修正/新限制。")
        response=min(i for i in ai if i>follow)
        slots[response]["responds_to"]=list(dict.fromkeys([follow]+slots[response]["responds_to"]))
        slots[response]["_instruction"]+=" 停止原方案，立即回应用户刚刚提出的修正/新限制；不能念轮次或编号。"
    # Planned human listening slots: never overwrite the core interaction or a side-talk exchange.
    candidates=[i for i in range(follow+2,turns-2) if i not in ai|side and i-1 not in side and i+1 not in ai
                and slots[i-1]["kind"]=="speech" and slots[i]["speaker"]!=slots[i-1]["speaker"]]
    rng.shuffle(candidates);chosen=[]
    for i in candidates:
        if any(abs(i-j)<4 for j in chosen):continue
        host=i-1;listener=slots[i]["speaker"]
        slots[i].update(kind="backchannel",gap_before_ms=0,insert_ratio=.6,addressee=slots[host]["speaker"],responds_to=[host],
                        _instruction="短附和1~3字，不加新内容；给出宿主已完整陈述事实的精确 anchor_text 前缀。")
        slots[host]["_instruction"]+=" 这一轮给至少两个陈述小句：前一事实说完足以支撑附和，后面还剩至少6字；不提问、不转折。"
        for t in slots[i+1:]:
            t["responds_to"]=list(dict.fromkeys(host if r==i else r for r in t["responds_to"]))
        if slots[i+1]["addressee"]==listener and slots[i+1]["speaker"]!=slots[host]["speaker"]:slots[i+1]["addressee"]=slots[host]["speaker"]
        chosen.append(i)
        if len(chosen)>=max(1,round(turns*.07)):break
    if profile is not None:
        profile["target_turns"]=sorted(ai)
        profile["triggers"]=[slots[i].get("trigger","cue") for i in sorted(ai)]
    for i,slot in enumerate(slots):
        if slot["speaker"].startswith("user_") and slot["kind"]=="speech" and slot["addressee"]=="assistant":
            next_main=next((t for t in slots[i+1:] if t["kind"]!="backchannel"),None)
            slot["_allow_ai_request"]=bool(next_main is not None and next_main["speaker"]=="assistant")
            if not slot["_allow_ai_request"]:
                slot["_instruction"]+=" 只回应/补充已有内容；下一主轮不是 AI，所以本轮不能新增向 AI 的提问或请求，不留下无人回答的问题。"
    if not {f"user_{i:02d}" for i in range(1,users+1)}<={t["speaker"] for t in slots}:return None
    return slots
