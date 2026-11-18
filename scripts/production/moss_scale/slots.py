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
    "想不起准确名称：第一段只描述特征并停在‘那个叫……’一类待补位置，不先说出名称；第二段先给出名称。",
    "回想时间或地点：第一段停在尚未说出的时间或地点之前；第二段先补出该时间或地点，再解释。",
    "说错后改口：第一段在错误词语尚未说完整时停住；第二段明确改用正确词语，不先完整宣布一个结论。",
    "拿不定选择：第一段说到‘我比较想选……’一类尚未给出答案的位置；第二段先给出自己的选择。",
    "需要心算数量：第一段停在‘算下来需要……’一类尚未给出数值的位置；第二段先给结果，再补说明。",
    "补充前提：第一段只开启一个尚未说完整的条件分句；第二段先补完条件，再说后续安排。",
    "寻找准确措辞：第一段停在待补的具体对象或描述之前；第二段先补出对象或描述，而不是重述一件已说完的事。",
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
    CHOICE_ACKNOWLEDGEMENTS, JOINT_CONFIRMATIONS, PERSONAL_RESOLUTIONS, CONFLICTING_CHOICES, CONFLICT_CLARIFICATIONS,
    generation_choice_option_sets, generation_choice_request_texts, choice_request_texts, choice_setup_texts,
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
    choice_planned = "simultaneous_requests" in events
    opening = add(first, "assistant", cue=("介绍相关背景或实际条件，暂不重复二选一问题。"
                                          if choice_planned else "用具体问题引入主线。"))
    add("assistant", first, refs=[opening], cue=("回应背景并了解需求或限制，不提前宣布方案。"
                                                  if choice_planned else "给出有针对性的初步回应。"))
    for speaker in users[1:]:
        add(speaker, "all", refs=[opening], cue=("自然补充实际约束，暂不表明二选一偏好。"
                                                 if choice_planned else "自然补充自己的实际情况。"))
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
            resume = add(listener, "assistant", refs=[opening], cue="直接延续插话前尚未解决的具体细节，不必宣布‘回到刚才’或复述开场问题；提到别人的限制时说清是谁。")
            add("assistant", listener, refs=[resume], cue="继续主线。")
        elif event == "incomplete":
            fillers = ("uh", "um", "let me think", "give me a second") if language == "en" else HESITATION_FILLERS
            filler = rng.choice(fillers) if rng.random() < .8 else None
            vocal_cue = (
                f"在停顿前实际说出一次‘{filler}’，放在没说完的内容之后，并用省略号收尾。"
                if filler else "用实际说到一半的内容表现卡住，并以省略号或破折号收尾。"
            )
            if handoff:
                host = add(listener, "assistant", "incomplete", [len(slots)-1],
                           "在说出真正想问的细节前卡住，停在一个没讲完的分句。"+vocal_cue, "handoff")
                add("assistant", listener, refs=[host], cue="长停顿后温和接话，给出基于已知事实的具体下一步。承接实际卡点，别套用‘先别急’之类的开头，也别猜测用户还没说出的选择。")
            else:
                add(listener, "all", "incomplete", [len(slots)-1], rng.choice(RESUME_CUES)+vocal_cue+
                    "第一段要留下一个明确待补的名称、时间、地点、数量、对象或选择；第二段先补上该项。不要用已完整的判断加犹豫，再另讲一件事。", "resume")
        elif event == "simultaneous_requests":
            first, second = rng.sample(users, 2)
            options = list(rng.choice(generation_choice_option_sets(language, topic)))
            if slots[-1]["speaker"] != "assistant" and slots[-1]["kind"] != "backchannel":
                last = len(slots)-1
                add("assistant", slots[last]["speaker"], refs=[last],
                    cue="先具体回应上一位用户刚刚的问题或陈述，再继续；不能忽略原问题。")
            request = add(first, "assistant", refs=[len(slots)-1],
                          cue="请助手帮助两人分析前文讨论过的选择。")
            slots[request].update(interaction="choice_request", choice_options=options,
                                  fixed_text=rng.choice(generation_choice_request_texts(language, options)))
            setup = add("assistant", "all", refs=[request],
                        cue="基于当前主话题明确提出两个互斥的具体方案，请大家选一个。")
            slots[setup].update(interaction="choice_setup", choice_options=options,
                                fixed_text=rng.choice(choice_setup_texts(language, options)))
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
                                     fixed_text=rng.choice(PERSONAL_RESOLUTIONS[language][selected]))
            other = second if responder == first else first
            outcome = rng.choice(("agreement", "open"))
            acknowledgment = add(other, responder, refs=[resolution],
                                 cue="回应另一位的选择；同意先尝试或保留不同意见、先比较两种方案。")
            slots[acknowledgment].update(interaction="choice_acknowledgement", choice_outcome=outcome,
                                         fixed_text=rng.choice(CHOICE_ACKNOWLEDGEMENTS[language][outcome]))
            confirmation = add("assistant", "all", refs=[resolution, acknowledgment],
                               cue="按两人的实际态度继续，未达成一致时不要宣布已共同决定。")
            slots[confirmation].update(interaction="choice_confirmation", choice_outcome=outcome,
                                       fixed_text=rng.choice(JOINT_CONFIRMATIONS[language][outcome]))
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
            candidates = [user for user in users if user != slots[-1]["speaker"]] or users
            speaker = rng.choice(candidates)
            add(speaker, "all", refs=[len(slots)-1], cue="按上下文补充细节、安排或不同意见。")
    closing_speaker = first if slots[-1]["speaker"] != first or len(users) == 1 else rng.choice([user for user in users if user != first])
    finish = add(closing_speaker, "assistant", refs=[len(slots)-1], cue="提出自己的下一步或保留意见，不代替其他人决定。")
    add("assistant", closing_speaker, refs=[finish], cue="依照已表达的态度自然收束，不宣布未经确认的全体决定。")
    for index, slot in enumerate(slots):
        if slot.get("interaction") != "choice_confirmation":
            continue
        outcome = slot.get("choice_outcome")
        if outcome == "agreement":
            state = "截至选择确认轮，两位参与者同意先试。" + ("其余人的态度根据之后发言更新。" if len(users) > 2 else "")
        else:
            state = "截至选择确认轮仍有分歧；根据之后发言更新。比较两个方案不代表都要实施。"
        for following in slots[index + 1:]:
            if following["speaker"] == "assistant" and "fixed_text" not in following:
                following["cue"] += "写作背景，仅供理解，不要照读：" + state
                following["cue"] += "先直接回答上一句的具体问题，再自然推进。"
                break
    return slots


def draft_from_slots(slots, users, context, mode, seeds, language="zh"):
    if mode not in {"direct", "interactive", "roleplay"}:
        raise ValueError("Unknown draft mode")
    instruction = (
        "为给定结构写自然的中文多人会话。只返回 JSON 对象，键为 turn 的字符串编号，"
        "普通轮值是台词字符串；标了 form=anchor、resume 或 transition 的轮，值为两个非空台词字符串组成的数组。"
        "anchor 的第一段必须是可被后续附和的完整信息，第二段至少十字。"
        "resume 第一段要听得出卡住，句尾用……或——，可按语境用‘那个’等短填充词。第二段须接回这一处，补全、找词后说出细节或改口，不要再重复第一段末尾的同一个填充短句。"
        "未完成或接话前必须有具体没说完的内容，不能只说‘我想一下’或‘嗯’。"
        "form=handoff 的台词也必须在没说完的位置以省略号或破折号收尾。"
        "不要在普通逗号、已经完整的陈述后机械切开，不要在停顿后另起不相关的问题。"
        "自然例子：[‘我能赶上的最晚一班是……’,‘哦对，七点四十那班。’]。"
        "反例：‘我去拿箱子，等我想想……’已经说完行动，不能仅加填充词就当未完成；应在‘我去拿那个……’之后再补具体物品。不要在第一段先给出第二段才想起的答案。"
        "除非是有意改口，第二段从缺失成分本身直接续接，不要重新起一句来重复同一意图。"
        "transition 第一段只写短附和，第二段用该轮指定的开场词，转折词只说一次。"
        "带 fixed_text 的轮次必须逐字使用给定台词，不能改变这些选项。"
        "写作提示是给作者的上下文，不要将提示或规则本身照读成对话。"
        "不要读角色编号。关系设定只适用于人类角色，助手是语音助手。向助手说话时不要使用亲属称呼。"
        "台词中直接称呼或询问的人必须对应本轮addressee；提到第三方时用转述，不要让助手直接向未被标为接收者的第三方提问。姓名只有在先前已说出口且对应角色明确时才能用于称呼，指代不清就省略姓名。"
        "助手只提供建议，不声称亲历、在场、已查询实时信息、替用户操作或发群消息。保持同一话题，对已经说出的信息作回应。"
        "助手没有定时提醒能力，不声称之后会主动提醒。"
        "事实性建议只依据已知条件；不确定时先提出需核实的条件，不保证收益、安全或其他结果。"
        "不要凭色温判断亮度或阅读效果，不要预设房屋接线改动容易；不要把某种食物当作缓解症状的通用办法，宠物健康建议应以兽医要求和具体情况为准。"
        "设备型号和说明书未知时，不得编造具体的容量、尺寸、转速或安全阈值。"
        "食品、过敏与健康建议要承接已知限制；缺少食材、设备或数量信息时，不要编造具体烹饪时长、批次或熟透保证。"
        "区分无添加糖和总糖、碳水；记住每项限制是谁说的，转述别人时指明对方，不要改成‘我’；自己的新情况可以自然补充。设备兼容先确认接口，别默认一定能用。"
        "卖家政策、退货期限和保修条件未知时，请先核实，不能猜具体天数；检查刀具或设备时不要建议直接用手接触刀刃。"
        "已明确出现电器接触不良需用手按住、自动断电失灵、排插开关不可靠、线缆破损或冒烟等故障时，先建议停止使用该故障设备、安全可行时断电并请专业人员检查；减少设备数或降低负荷不等于故障排除，不得建议继续试用故障设备。"
        "后续话轮推进尚未解决的事情，不要反复重新确定已明确的日期或安排。打断必须增加新的限制或纠正，而不是重复助手刚说过的内容。"
        "已经给出的建议和已确认状态只在必要时简短承接，后续回答应解决当前的新问题。被问谁来执行时，依据已表达的意愿分工；无人承诺时直接询问谁方便，不要反复只说‘找个人’或编造已有人执行。"
        "前文要求的安全条件必须在后续操作确认中成立：尚未确认安全断电时，不能直接认可拔设备或移动故障排插；无法安全接近时交由专业人员处理。多人同时补充条件后，回答要承接每个人尚未解决的要点。"
        "用户要推荐或方案且条件足够时，助手至少给出一个符合已知限制的具体例子或可执行步骤，不能整段只让用户自己选、再讨论。条件不足时先问关键条件，不编造条件。"
        "答应比较或解释时要实际推进，没空间展开就不要作该承诺；勿将需要加工的操作说成无需烹饪。"
        "逐人保持已表达的限制；如果某人的关键限制尚未解决或尚未同意，不能宣布全体已定，应请其确认。"
        "不知道设备型号、接口或配件是否存在时，不能保证具体功能；二选一也不能无故改成两个都执行。"
        "写完后检查是否真正回答提问、遵守约束和按说话顺序推进；拿不准时自然询问，而不是编造条件。"
        "示例：{\"0\":\"明天几点出发？\",\"1\":[\"第一班车七点开\",\"我们最好提前二十分钟到站。\"]}。"
    )
    if language == "en":
        instruction += (
            "LANGUAGE OVERRIDE: Write every spoken line in natural conversational English. "
            "The Chinese cues describe the scene and timing only. Use English punctuation. "
            "Never put internal identifiers such as user_01, user_02, third_party_01 or turn numbers in spoken text. When recalling another person's words, use a natural reference or a previously established name; omit the identifier rather than reading it aloud. "
            "Direct questions and vocatives must match that turn's addressee. Refer to third parties indirectly unless they are the marked recipient. Use a name as a direct address only after it has been spoken and clearly linked to that recipient; otherwise omit it. "
            "The assistant gives suggestions only: never say it will buy, book, call, send messages, "
            "remind people later, or create and share documents. It may ask a person to do those things. "
            "Make factual advice conditional on what is known; avoid unsupported guarantees or assuming facts about health, wiring or weather. "
            "Do not invent device-specific capacity, size, speed, or safety thresholds. "
            "Keep known allergies in view; don't invent cooking time or batch counts when food, equipment or quantity are unknown. "
            "Don't guess store return windows or warranties; suggest checking the seller's actual terms. Don't suggest touching blades. "
            "For an explicitly malfunctioning electrical device, including a faulty contact, unreliable power-strip switch, or damaged cord, advise stopping use of that device, isolating power when safe and seeking professional inspection. Reducing the number of connected devices or their load does not repair a fault; do not suggest testing or continuing to use the faulty device. "
            "Move the conversation forward instead of reopening settled dates, and make interruptions add a genuinely new constraint. "
            "Keep confirmed decisions and completed actions consistent. Answer the current new question instead of repeating the same advice. When asked who should act, use stated volunteers or ask who is available; don't repeatedly say 'someone' or invent that a person has already acted. "
            "Carry required safety preconditions into later action confirmations: don't approve unplugging devices or moving faulty equipment before safe power isolation has been established; if it cannot be approached safely, defer to qualified help. Address each unresolved point when several people add constraints at once. "
            "Do not declare a group decision while a person's stated constraint or agreement remains unconfirmed. "
            "Do not promise device functions before checking the necessary equipment or turn an either/or choice into doing both. "
            "Before returning JSON, check that each question is answered and each person's constraints remain consistent. "
            "Keep acknowledgements to one to six words. A transition starts with a short "
            "acknowledgement and puts the requested English pivot only in the second part. "
            "For a hesitation, make the first clause sound plausible if the listener stops hearing it midway. "
            "Include an unfinished thought before the filler; a filler alone is not an incomplete sentence. "
            "A resume must leave a specific value, object, name, location or condition unsaid; part two first supplies that missing piece. A completed action followed by 'give me a second' is not enough, and do not reveal the answer before the hesitation. "
            "Unless deliberately correcting yourself, start part two directly with the missing word or phrase rather than restarting the whole clause. "
            "Attribute each stated restriction to its speaker: refer to the other person when recalling theirs, "
            "while allowing people to introduce their own new constraints. Distinguish added sugar from total sugar or carbohydrates, "
            "and check device connectors before assuming compatibility. Don't call toasting no preparation. "
            "When a user requests recommendations or a plan and enough is known, give at least one concrete example or actionable step consistent with the stated constraints. Don't spend the whole conversation asking users to choose and discuss it themselves. Ask for essential missing information rather than inventing it. "
            "If the assistant promises an explanation, follow through instead of repeating the plan. "
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
    if mode == "roleplay":
        chunks = [range(i, i + 1) for i in range(len(slots))]
    for chunk in chunks:
        turns = {str(i): {k: v for k, v in slots[i].items() if v is not None and v != []} for i in chunk}
        request = {"背景": context, "待写轮次": turns}
        if mode == "roleplay":
            current = chunk.start
            request["当前角色"] = slots[current]["speaker"]
            request["角色边界"] = "只写当前角色这一轮；历史中的speaker是信息持有人，addressee决定你/your指谁。不要把其他人的限制改成自己的，也不能把人类任务归给助手。"
            request["后续事件计划_并非已发生事实"] = [
                {key: turn[key] for key in ("speaker", "kind", "form") if turn.get(key) is not None}
                for turn in slots[current + 1:current + 3]
            ]
        setup = next((s for s in slots if s.get("interaction") == "choice_setup"), None)
        if setup:
            request["全局场景线索"] = (
                "整段主对话应围绕下面两个具体方案和用户的实际需求展开，"
                "即使当前分块还未出现这句，也请提前自然引入相关主题："
                + choice_request_texts(language, setup["choice_options"])[0]
            )
            request["角色私有倾向"] = {
                s["speaker"]: setup["choice_options"][s["choice_index"]]
                for s in slots if s.get("interaction") == "simultaneous_requests"
                and (mode != "roleplay" or s["speaker"] == slots[chunk.start]["speaker"])
            }
            request["倾向说明"] = (
                "这些是作者的角色设定，前后不得无故写成相反倾向。"
                "其他角色和助手不能在该用户实际说出倾向之前知道它。"
                "在固定的 choice_request 之前，只铺垫背景、约束或需要比较的因素；"
                "不要提前完整重复那道二选一问题，也不要逐一问出或宣布两人的选择。"
                "参与同时选择的两位用户在此之前只陈述条件，不提前表明选哪个；若角色后来改变偏好，必须说出改变原因。"
                "choice_outcome=open 表示保留分歧，后续不要擅自说所有人已经同意；agreement 只说明参与这次讨论的两人达成暂时一致。"
            )
        two_part_turns = [i for i, turn in turns.items() if turn.get("form") in {"anchor", "resume", "transition"}]
        if two_part_turns:
            request["必须输出两个台词字符串数组的轮次"] = two_part_turns
            request["输出结构示例_请用实际台词替换占位内容"] = {
                key: ["第一段台词", "第二段台词"] if key in two_part_turns else "本轮台词"
                for key in turns
            }
        if seeds:
            request["短附和措辞参考"] = seeds
        if authored:
            # Keep the speaker with each earlier line: text alone loses ownership
            # of preferences, restrictions and commitments across draft calls.
            request["前半段已说过的台词"] = {
                key: {"speaker": slots[int(key)]["speaker"],
                      "kind": slots[int(key)]["kind"],
                      "addressee": slots[int(key)]["addressee"], "text": wording}
                for key, wording in authored.items()
            }
            if mode == "roleplay" and slots[chunk.start]["kind"] == "backchannel":
                for host in slots[chunk.start].get("responds_to", []):
                    key = str(host)
                    if slots[host].get("form") == "anchor" and isinstance(authored.get(key), list):
                        request["前半段已说过的台词"][key]["text"] = authored[key][0]
                request["可听范围"] = "附和发生在宿主前缀说完后；历史中的宿主只展示此时已说出的内容，不能依赖后半句。"
        if mode == "roleplay" and "fixed_text" in slots[chunk.start]:
            result = {str(chunk.start): slots[chunk.start]["fixed_text"]}
        else:
            raw = chat(instruction, json.dumps(request, ensure_ascii=False, separators=(",", ":")),
                       max_tokens=700 if mode == "roleplay" else 4500, temperature=.75, json_output=True)
            result = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
        if not isinstance(result, dict) or set(result) != set(turns):
            raise ValueError("台词编号与事件计划不一致")
        for i in chunk:
            if "fixed_text" in slots[i]:
                result[str(i)] = slots[i]["fixed_text"]
            if mode == "roleplay":
                wording = result[str(i)]
                split = slots[i].get("form") in {"anchor", "resume", "transition"}
                if isinstance(wording, dict):
                    wording = wording.get("parts" if split else "text")
                if not split and isinstance(wording, list) and all(isinstance(part, str) for part in wording):
                    wording = "".join(wording)
                if split:
                    if not isinstance(wording, list) or len(wording) != 2 or not all(isinstance(part, str) and part.strip() for part in wording):
                        from .errors import DraftShapeError
                        raise DraftShapeError(i, wording)
                    wording = [part.strip() for part in wording]
                elif not isinstance(wording, str) or not wording.strip():
                    raise ValueError("逐角色台词必须是非空字符串")
                result[str(i)] = wording
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
                from .errors import DraftShapeError
                raise DraftShapeError(i, wording)
            head, tail = (x.strip() for x in wording)
            if form == "anchor":
                head = head.rstrip("，。；、！？,.;!? ")
                separator = ". " if language == "en" and tail[0].isupper() else comma
                item["text"] = head + separator + tail
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
        if any(marker in item["text"] for marker in ("本轮台词", "第一段台词", "第二段台词")):
            raise ValueError("输出包含结构示例占位内容")
        output.append(item)
    return dict(setting=context["场景"], topic=context["领域"],
                participants=[dict(id=u, persona=context["关系"], style="自然口语") for u in users],
                turns=output)
