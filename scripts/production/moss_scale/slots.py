"""Plan event relationships in code and ask the model for dialogue wording."""
import json
import re

from scripts.text.utils.llm import chat


SIDE_TALK_SITUATIONS = (
    "提醒对方身边的灯、窗或水龙头需要处理",
    "临时借一件具体的小物品",
    "问附近有没有看到一件遗落的物品",
    "提醒身边的人马上有另一项安排",
    "请对方顺手帮一个与主话题无关的小忙",
    "提醒对方有人在别处等他",
    "打招呼并确认等会儿在哪里碰面",
    "告知对方有人送了东西过来",
    "提醒对方停车计时快到了",
    "确认临时借用的教室或会议室还有多久可用",
    "提醒对方门口的快递员正在等签收",
    "问对方活动结束后该把公用器材放回哪里",
)

RESUME_CUES = (
    "想不起一个准确的名称，先卡住，随后想起来或用具体描述绕过去。",
    "想不起一个时间、数字或地点，停下来回想，确认后补出它。",
    "一个词刚说出口就意识到讲错了，停顿后明确改正。",
    "说到一个暂时拿不定的选择时卡住，想清楚后由同一人说出决定。",
    "想给出一个需要心算的时间或数量，说到计算的关键处停一下，再给出结果。",
    "说到一半发现前提漏了，停顿后把条件补上，再接回原本要说的事。",
    "一句话还没说完就发现说法不合适，停顿后用更准确或更委婉的话重新表达。",
)
HESITATION_FILLERS = ("呃", "那个，那个", "嗯", "我想一下", "让我想想", "等我想想")
TRANSITION_PIVOT_GROUPS = (
    ("不过", "但是", "可是", "只是", "话是这么说"),
    ("等一下下", "等一下", "等会儿", "稍等一下", "稍等", "先等等", "等等", "等我插一句", "我插一句"),
    ("不对", "先别急", "先别动", "先暂停一下", "先打住", "打住", "停一下", "停下", "停"),
    ("闭嘴",),
)
TRANSITION_PIVOTS = tuple(sorted(
    (word for group in TRANSITION_PIVOT_GROUPS for word in group), key=len, reverse=True,
))
EN_TRANSITION_PIVOTS = ("But", "Wait", "Hold on", "Hang on", "Actually", "One second", "Let me jump in")
from scripts.text.simultaneous_choices import (
    CHOICE_CONFIRMATIONS, CHOICE_RESOLUTIONS, CONFLICTING_CHOICES, CONFLICT_CLARIFICATIONS,
    choice_option_sets, choice_request_text, choice_setup_text,
)


def shorten_backchannel_prefix(text, language):
    """Keep an already complete short acknowledgement when a draft adds excess wording."""
    if language == "en" and len(text.split()) > 6:
        first = re.split(r"[,;.!?—]", text, maxsplit=1)[0].strip()
        prefixes = r"^(?:got it|makes sense|right|yeah|yes|exactly|agreed|okay|sure|good point|true|i see|i agree|that.s fair|sounds good)\b"
        if 1 <= len(first.split()) <= 6 and re.match(prefixes, first, re.IGNORECASE):
            return first
    if language == "zh" and len(text) > 14:
        first = re.split(r"[，。！？；,!?]", text, maxsplit=1)[0]
        if 2 <= len(first) <= 10 and re.match(r"^(对|嗯|是|好|确实|明白|没错|有道理|我也是|我也觉得)", first):
            return first
    return text


def plan_slots(users, events, group, repeat, group_interrupt, transition, handoff, rng, target, language="zh", topic=""):
    slots = []

    def add(speaker, target, kind="speech", refs=(), cue="", form=None):
        slots.append(dict(speaker=speaker, addressee=target, kind=kind,
                          responds_to=list(refs), cue=cue, form=form))
        return len(slots) - 1

    first = users[0]
    opening = add(first, "assistant", cue="用具体问题引入主线。")
    add("assistant", first, refs=[opening], cue="给出有针对性的初步回应。")
    for speaker in users[1:]:
        add(speaker, "all", refs=[opening], cue="自然补充自己的实际情况。")
    order = list(events)
    rng.shuffle(order)
    for event in order:
        listener = rng.choice(users)
        if slots[-1]["speaker"] == "assistant" and event in ("backchannel", "interruption"):
            add(listener, "assistant", refs=[len(slots)-1], cue="根据刚才的建议追问一个相关的实际细节。")
        if event == "backchannel":
            host = add("assistant", "all", refs=[len(slots)-1], cue="先说一个完整的建议，再继续讲至少十个字。",
                       form="anchor")
            speakers = [listener]
            if group:
                speakers.append(listener if repeat else rng.choice([u for u in users if u != listener]))
            for speaker in speakers:
                add(speaker, "assistant", "backchannel", [host], "四到十字的短认同，只回应助手已说完的前段，不用单独的“嗯嗯”。")
        elif event == "side_talk":
            if rng.random() < .5:
                start = add(listener, "all", refs=[len(slots)-1], cue="继续主线，说一个具体安排。")
                situation = rng.choice(SIDE_TALK_SITUATIONS)
                aside = add("third_party_01", listener, "side_talk", (),
                            f"第三方突然开口。情境是{situation}，说一句具体的短话，与主线无关。")
                slots[aside]["side_talk_entry"] = "unsolicited"
            else:
                start = add(listener, "third_party_01", cue="突然有门外或身边的第三方，转向对方说一句具体的事。")
                aside = add("third_party_01", listener, "side_talk", [start], "第三方回一句，本轮独立于主话题。")
            add(listener, "third_party_01", refs=[aside], cue="短暂回应第三方。")
            resume = add(listener, "assistant", refs=[opening], cue="明确把对话接回开头的主话题。")
            add("assistant", listener, refs=[resume], cue="继续主线。")
        elif event == "incomplete":
            fillers = ("uh", "um", "let me think", "give me a second") if language == "en" else HESITATION_FILLERS
            filler = rng.choice(fillers) if rng.random() < .8 else None
            vocal_cue = (
                f"在停顿前实际说出一次‘{filler}’，放在没说完的内容之后，并用省略号收尾。"
                if filler else "用实际说到一半的内容表现卡住。"
            )
            if handoff:
                host = add(listener, "assistant", "incomplete", [len(slots)-1],
                           "在说出真正想问的细节前卡住，停在一个没讲完的分句。"+vocal_cue, "handoff")
                add("assistant", listener, refs=[host], cue="长停顿后温和接话，给出基于已知事实的具体下一步。承接实际卡点，别套用‘先别急’之类的开头，也别猜测用户还没说出的选择。")
            else:
                add(listener, "all", "incomplete", [len(slots)-1], rng.choice(RESUME_CUES)+vocal_cue, "resume")
        elif event == "simultaneous_requests":
            first, second = rng.sample(users, 2)
            options = list(rng.choice(choice_option_sets(language, topic)))
            if slots[-1]["speaker"] != "assistant" and slots[-1]["kind"] != "backchannel":
                last = len(slots)-1
                add("assistant", slots[last]["speaker"], refs=[last],
                    cue="先具体回应上一位用户刚刚的问题或陈述，再继续；不能忽略原问题。")
            request = add(first, "assistant", refs=[len(slots)-1],
                          cue="请助手帮助两人分析前文讨论过的选择。")
            slots[request].update(interaction="choice_request", choice_options=options,
                                  fixed_text=choice_request_text(language, options))
            setup = add("assistant", "all", refs=[request],
                        cue="基于当前主话题明确提出两个互斥的具体方案，请大家选一个。")
            slots[setup].update(interaction="choice_setup", choice_options=options,
                                fixed_text=choice_setup_text(language, options))
            first_text, second_text = rng.choice(CONFLICTING_CHOICES[language])
            choices = [(first_text, 0), (second_text, 1)]
            if rng.random() < .5:
                choices.reverse()
            first_turn = add(first, "assistant", refs=[setup],
                             cue="同时说出自己更喜欢的唯一选项。")
            slots[first_turn].update(interaction="simultaneous_requests", fixed_text=choices[0][0],
                                     choice_index=choices[0][1])
            second_turn = add(second, "assistant", refs=[setup],
                              cue="同时说出与另一人不同的唯一选项。")
            slots[second_turn].update(interaction="simultaneous_requests", overlap_with=first_turn,
                                      fixed_text=choices[1][0], choice_index=choices[1][1])
            reply = add("assistant", "all", refs=[first_turn, second_turn],
                        cue="等两人都说完后，自然指出两个选择不一致并请他们确认采用哪一个。不能自行决定，也不要把并发请求说成打断。")
            slots[reply].update(interaction="conflict_clarification",
                                fixed_text=rng.choice(CONFLICT_CLARIFICATIONS[language]))
            responder, selected = rng.choice([(first, choices[0][1]), (second, choices[1][1])])
            resolution = add(responder, "assistant", refs=[reply],
                             cue="回应助手的澄清，明确选择之前提出的一个方案。")
            slots[resolution].update(interaction="choice_resolution", choice_index=selected,
                                     fixed_text=rng.choice(CHOICE_RESOLUTIONS[language][selected]))
            confirmation = add("assistant", slots[resolution]["speaker"], refs=[resolution],
                               cue="确认用户选项，并继续基于该选项提供建议，不要声称自己执行现实操作。")
            slots[confirmation].update(interaction="choice_confirmation",
                                       fixed_text=rng.choice(CHOICE_CONFIRMATIONS[language]))
        elif event == "interruption":
            host = add("assistant", "all", refs=[len(slots)-1],
                       cue="先说完一项事实，再开始下一个分句，但第二个分句在半句时被用户截断。")
            if transition:
                group_choices = rng.choices(TRANSITION_PIVOT_GROUPS, weights=(42, 40, 17, 1), k=1)[0]
                pivot = rng.choice(EN_TRANSITION_PIVOTS if language == "en" else group_choices)
                transition_cue = f"先轻声附和已说出的内容，第二段以‘{pivot}’开头，并提出具体的新限制。"
                if language == "zh" and group_choices == TRANSITION_PIVOT_GROUPS[2]:
                    transition_cue += "安排一个需要立刻阻止错误操作或纠正关键误解的场景，语气符合人物关系。"
                if pivot == "闭嘴":
                    transition_cue += "安排确实会紧急压低声音或截住泄密的场景，说明缘由，避免无缘无故骂人。"
            replies = [add(listener, "assistant", "interruption", [host],
                           transition_cue if transition else
                           "根据助手已经说出的信息，抢话补充一个新限制。", "transition" if transition else None)]
            if group_interrupt:
                other = rng.choice([u for u in users if u != listener])
                replies.append(add(other, "assistant", "interruption", [host], "与另一人同时抢话，补充不同但相关的限制。"))
            add("assistant", "all", refs=replies, cue="回应抢话带来的新限制，继续主线。")
        if len(slots) <= target - 7:
            follow = add(rng.choice(users), "assistant", refs=[len(slots)-1], cue="沿主话题提出实际下一步。")
            add("assistant", "all", refs=[follow], cue="回答并推动主线。")
    while len(slots) < max(10, target - 2):
        if not order and len(slots) == target // 2:
            add("assistant", "all", refs=[len(slots)-1], cue="结合刚才的意见给一个具体建议。")
        else:
            speaker = rng.choice(users)
            add(speaker, "all", refs=[len(slots)-1], cue="按上下文补充细节、安排或不同意见。")
    finish = add(first, "assistant", refs=[len(slots)-1], cue="确认接下来的安排。")
    add("assistant", first, refs=[finish], cue="自然收束。")
    return slots


def draft_from_slots(slots, users, context, mode, seeds, language="zh"):
    instruction = (
        "为给定结构写自然的中文多人会话。只返回 JSON 对象，键为 turn 的字符串编号，"
        "普通轮值是台词字符串；标了 form=anchor、resume 或 transition 的轮，值为两个非空台词字符串组成的数组。"
        "anchor 的第一段必须是可被后续附和的完整信息，第二段至少十字。"
        "resume 第一段要听得出卡住，句尾用……或——，可按语境用‘那个’等短填充词。第二段须接回这一处，补全、找词后说出细节或改口。"
        "不要在普通逗号、已经完整的陈述后机械切开，不要在停顿后另起不相关的问题。"
        "自然例子：[‘我能赶上的最晚一班是……’,‘哦对，七点四十那班。’]。"
        "transition 第一段只写短附和，第二段用该轮指定的开场词，转折词只说一次。"
        "带 fixed_text 的轮次必须逐字使用给定台词，不能改变这些选项。"
        "不要读角色编号。关系设定只适用于人类角色，助手是语音助手。向助手说话时不要使用亲属称呼。"
        "助手只提供建议，不声称亲历、在场、已查询实时信息、替用户操作或发群消息。保持同一话题，对已经说出的信息作回应。"
        "示例：{\"0\":\"明天几点出发？\",\"1\":[\"第一班车七点开\",\"我们最好提前二十分钟到站。\"]}。"
    )
    if language == "en":
        instruction += (
            "LANGUAGE OVERRIDE: Write every spoken line in natural conversational English. "
            "The Chinese cues describe the scene and timing only. Use English punctuation. "
            "The assistant gives suggestions only: never say it will buy, book, call, send messages, "
            "or create and share documents. It may ask a person to do those things. "
            "Keep acknowledgements to one to six words. A transition starts with a short "
            "acknowledgement and puts the requested English pivot only in the second part. "
            "For a hesitation, make the first clause sound plausible if the listener stops hearing it midway. "
            "Separate the requested filler from that clause with a comma, ellipsis, or em dash, "
            "and end the unfinished part with an ellipsis or em dash. Treat a multiword filler as a spoken aside. "
            "Example resume: [\"I'll grab the… give me a second…\", \"Right, the packing list.\"]. "
            "After the pause, finish the thought once and avoid a second unexplained cutoff."
        )
    pivots = EN_TRANSITION_PIVOTS if language == "en" else TRANSITION_PIVOTS
    comma = ", " if language == "en" else "，"
    authored = {}
    # Interactive mode hands the first half to a second drafting call as shared conversation history.
    chunks = [range(len(slots))] if mode == "direct" else [range(len(slots)//2), range(len(slots)//2, len(slots))]
    for chunk in chunks:
        turns = {str(i): {k: v for k, v in slots[i].items() if v is not None and v != []} for i in chunk}
        request = {"背景": context, "待写轮次": turns}
        setup = next((s for s in slots if s.get("interaction") == "choice_setup"), None)
        if setup:
            request["全局场景线索"] = (
                "整段主对话应围绕下面两个具体方案和用户的实际需求展开，"
                "即使当前分块还未出现这句，也请提前自然引入相关主题："
                + choice_request_text(language, setup["choice_options"])
            )
            request["角色私有倾向"] = {
                s["speaker"]: setup["choice_options"][s["choice_index"]]
                for s in slots if s.get("interaction") == "simultaneous_requests"
            }
            request["倾向说明"] = (
                "这些是作者的角色设定，前后不得无故写成相反倾向。"
                "其他角色和助手不能在该用户实际说出倾向之前知道它。"
            )
        two_part_turns = [i for i, turn in turns.items() if turn.get("form") in {"anchor", "resume", "transition"}]
        if two_part_turns:
            request["必须输出两个台词字符串数组的轮次"] = two_part_turns
        if seeds:
            request["短附和措辞参考"] = seeds
        if authored:
            request["前半段已说过的台词"] = authored
        raw = chat(instruction, json.dumps(request, ensure_ascii=False, separators=(",", ":")),
                   max_tokens=4500, temperature=.75, json_output=True)
        result = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
        if not isinstance(result, dict) or set(result) != set(turns):
            raise ValueError("台词编号与事件计划不一致")
        for i in chunk:
            if "fixed_text" in slots[i]:
                result[str(i)] = slots[i]["fixed_text"]
        authored.update(result)
    output = []
    anchors = {}
    for i, slot in enumerate(slots):
        item = {k: v for k, v in slot.items() if k not in {"cue", "form", "fixed_text"}}
        wording = authored[str(i)]
        form = slot["form"]
        split = form in {"anchor", "resume", "transition"}
        if isinstance(wording, dict):
            wording = wording.get("parts" if split else "text")
        if not split and isinstance(wording, list) and all(isinstance(part, str) for part in wording):
            wording = "".join(wording)
        if split:
            if not isinstance(wording, list) or len(wording) != 2 or not all(isinstance(x, str) and x.strip() for x in wording):
                raise ValueError(f"第{i}轮缺少两段台词")
            head, tail = (x.strip() for x in wording)
            if form == "anchor":
                head = head.rstrip("，。；、！？,.;!? ")
                item["text"] = head + comma + tail
                for index in range(i+1, len(slots)):
                    if slots[index]["kind"] != "backchannel":
                        break
                    anchors[index] = head
            elif form == "resume":
                if not head.endswith(("……", "——", "...", "…", "—")):
                    raise ValueError(f"第{i}轮未完成的第一段须以省略号或破折号标出说话卡住的位置")
                if language == "en":
                    head = re.sub(
                        r"(?<=[\w])\s+(?=(?:let me think|give me a second)(?:\.{3}|…|—)$)",
                        "… ",
                        head,
                        flags=re.IGNORECASE,
                    ) + " "
                item["parts"] = [head, tail]
                item["text"] = "".join(item["parts"])
            else:
                # Some drafts place the pivot in the first array element.
                pivot_pattern = "|".join(pivots)
                prefix_limit = 40 if language == "en" else 14
                misplaced = re.fullmatch(
                    rf"(.{{2,{prefix_limit}}}?)[，,。！？\s]+((?:{pivot_pattern}).*)", head,
                    re.IGNORECASE if language == "en" else 0,
                )
                if misplaced:
                    head, pivot = misplaced.groups()
                    if pivot not in pivots or not tail.startswith(pivots):
                        if tail.startswith(pivots):
                            repeated = next(word for word in pivots if tail.startswith(word))
                            tail = tail[len(repeated):].lstrip("，, ")
                        separator = "" if pivot.endswith(("，", "。", "！", "？")) else "，"
                        tail = pivot + (comma if separator else " " if language == "en" else "") + tail
                prefix_end = " ，,。！？.!?；;：:…"
                head = head.rstrip(prefix_end)
                item.update(text=head + comma + tail, backchannel_prefix=head)
        else:
            if not isinstance(wording, str) or not wording.strip():
                raise ValueError(f"第{i}轮台词不是非空字符串")
            item["text"] = wording.strip()
            if item["kind"] == "backchannel":
                item["anchor_text"] = anchors[i]
                item["text"] = shorten_backchannel_prefix(item["text"], language)
            if form == "handoff":
                if not item["text"].endswith(("……", "——", "...", "…", "—")):
                    raise ValueError(f"第{i}轮犹豫接话须标出尚未说完的位置")
                item.update(parts=[item["text"]], resolution="assistant_handoff")
        output.append(item)
    return dict(setting=context["场景"], topic=context["领域"],
                participants=[dict(id=u, persona=context["关系"], style="自然口语") for u in users],
                turns=output)
