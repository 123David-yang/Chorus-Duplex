"""Verified choices and utterances for overlapping user requests."""

CHOICE_OPTIONS = {
    "zh": {
        "家庭日常": ("周六打扫", "周日打扫"),
        "饮食与生活": ("做面条", "做米饭"),
        "职场办公": ("上午开会", "下午开会"),
        "旅行出行": ("坐高铁", "坐大巴"),
        "教育学习": ("先复习词汇", "先复习语法"),
        "消费购物": ("选小号", "选大号"),
        "娱乐影视": ("看电影", "看纪录片"),
        "邻里社区": ("周六组织活动", "周日组织活动"),
        "运动健身": ("上午锻炼", "下午锻炼"),
        "科技数码": ("买笔记本电脑", "买平板电脑"),
        "理财生活": ("按月储蓄", "按季度储蓄"),
        "宠物": ("上午带宠物去检查", "下午带宠物去检查"),
        "家居装修": ("把墙刷成白色", "把墙刷成浅灰色"),
        "节庆聚会": ("在家聚会", "在餐厅聚会"),
        "兴趣爱好": ("先练习画画", "先练习摄影"),
    },
    "en": {
        "travel plans": ("take the train", "take the bus"),
        "home maintenance": ("paint the room white", "paint it light gray"),
        "work coordination": ("meet in the morning", "meet in the afternoon"),
        "meal planning": ("make noodles", "make rice"),
        "local events": ("meet on Saturday", "meet on Sunday"),
        "pet care": ("visit the vet in the morning", "visit the vet in the afternoon"),
        "technology": ("buy a laptop", "buy a tablet"),
        "shopping": ("buy the small size", "buy the large size"),
        "study plans": ("review vocabulary first", "review grammar first"),
        "community projects": ("hold the event on Saturday", "hold the event on Sunday"),
        "outdoor activities": ("go hiking on Saturday", "go hiking on Sunday"),
        "family logistics": ("clean the house on Saturday", "clean the house on Sunday"),
    },
}
DEFAULT_CHOICES = {
    "zh": ("上午安排", "下午安排"),
    "en": ("schedule it in the morning", "schedule it in the afternoon"),
}
ALTERNATE_CHOICES = {
    "zh": {
        "家庭日常": ("周六整理衣柜", "周日整理衣柜"),
        "饮食与生活": ("早餐喝牛奶", "早餐喝豆浆"),
        "职场办公": ("周一提交报告", "周二提交报告"),
        "旅行出行": ("坐飞机", "坐火车"),
        "教育学习": ("先做阅读练习", "先做听力练习"),
        "消费购物": ("在网上购买", "去实体店购买"),
        "娱乐影视": ("看喜剧片", "看科幻片"),
        "邻里社区": ("把活动安排在上午", "把活动安排在下午"),
        "运动健身": ("周六去跑步", "周日去跑步"),
        "科技数码": ("买有线耳机", "买无线耳机"),
        "理财生活": ("储蓄收入的百分之十", "储蓄收入的百分之二十"),
        "宠物": ("周六给宠物洗澡", "周日给宠物洗澡"),
        "家居装修": ("客厅装暖色灯", "客厅装冷色灯"),
        "节庆聚会": ("聚餐吃火锅", "聚餐吃烧烤"),
        "兴趣爱好": ("先学习吉他", "先学习钢琴"),
    },
    "en": {
        "travel plans": ("take a flight", "take the train"),
        "home maintenance": ("install warm lights", "install cool lights"),
        "work coordination": ("submit the report on Monday", "submit the report on Tuesday"),
        "meal planning": ("have milk for breakfast", "have soy milk for breakfast"),
        "local events": ("hold the event in the morning", "hold the event in the afternoon"),
        "pet care": ("wash the pet on Saturday", "wash the pet on Sunday"),
        "technology": ("buy wired headphones", "buy wireless headphones"),
        "shopping": ("buy it online", "buy it in a store"),
        "study plans": ("practice reading first", "practice listening first"),
        "community projects": ("hold the meeting in the morning", "hold the meeting in the afternoon"),
        "outdoor activities": ("go running on Saturday", "go running on Sunday"),
        "family logistics": ("organize the closet on Saturday", "organize the closet on Sunday"),
    },
}

EXTRA_CHOICES = {
    "zh": {
        "家庭日常": ("周六整理冰箱", "周日整理冰箱"),
        "饮食与生活": ("午餐做饺子", "午餐做面条"),
        "职场办公": ("在会议室开会", "通过视频开会"),
        "旅行出行": ("早上出发", "下午出发"),
        "教育学习": ("先做习题", "先回顾笔记"),
        "消费购物": ("选黑色款", "选白色款"),
        "娱乐影视": ("周六看那部电影", "周日看那部电影"),
        "邻里社区": ("在室内办活动", "在室外办活动"),
        "运动健身": ("先做力量训练", "先做拉伸运动"),
        "科技数码": ("买黑色键盘", "买白色键盘"),
        "理财生活": ("每月初记账", "每月末记账"),
        "宠物": ("周六带宠物去复查", "周日带宠物去复查"),
        "家居装修": ("把墙刷成米色", "把墙刷成浅蓝色"),
        "节庆聚会": ("周六举办聚会", "周日举办聚会"),
        "兴趣爱好": ("周六练习吉他", "周日练习吉他"),
    },
    "en": {
        "travel plans": ("leave in the morning", "leave in the afternoon"),
        "home maintenance": ("paint the room beige", "paint it light blue"),
        "work coordination": ("meet in person", "meet by video"),
        "meal planning": ("make dumplings for lunch", "make noodles for lunch"),
        "local events": ("hold the event indoors", "hold the event outdoors"),
        "pet care": ("visit the vet on Saturday", "visit the vet on Sunday"),
        "technology": ("buy the black keyboard", "buy the white keyboard"),
        "shopping": ("choose the black version", "choose the white version"),
        "study plans": ("do practice questions first", "review notes first"),
        "community projects": ("meet in person", "meet by video"),
        "outdoor activities": ("go hiking in the morning", "go hiking in the afternoon"),
        "family logistics": ("organize the fridge on Saturday", "organize the fridge on Sunday"),
    },
}


def choice_option_sets(language, topic):
    first = CHOICE_OPTIONS[language].get(topic, DEFAULT_CHOICES[language])
    second = ALTERNATE_CHOICES[language].get(topic, DEFAULT_CHOICES[language])
    extra = EXTRA_CHOICES[language].get(topic, DEFAULT_CHOICES[language])
    return tuple(dict.fromkeys((first, second, extra)))


def generation_choice_option_sets(language, topic):
    options = choice_option_sets(language, topic)
    # Preserve old receipts while no longer generating an ambiguous film/documentary pair.
    if language == "zh" and topic == "娱乐影视":
        return tuple(pair for pair in options if pair != ("看电影", "看纪录片"))
    return options


def choice_setup_texts(language, choices):
    if language == "en":
        return ("Which option does each of you prefer?", "What does each of you prefer?",
                "I'd like to hear each of your preferences.", "Which would each of you choose?")
    return ("你们各自更倾向哪一个？", "你们俩分别想选哪个？",
            "我先听听你们各自的想法。", "你们分别更喜欢哪个方案？")


def choice_setup_text(language, choices):
    return choice_setup_texts(language, choices)[0]


def choice_request_texts(language, choices):
    if language == "en":
        return (f"Can you help us choose whether to {choices[0]} or {choices[1]}?",
                f"Should we {choices[0]} or {choices[1]}? Can you help us decide?",
                f"We're choosing whether to {choices[0]} or {choices[1]}. What do you think?",
                f"Could you help us decide whether to {choices[0]} or {choices[1]}?")
    return (f"我们{choices[0]}还是{choices[1]}，你能帮我们选一下吗？",
            f"你帮我们看看，是{choices[0]}好，还是{choices[1]}好？",
            f"这两个方案——{choices[0]}和{choices[1]}，你有什么建议？",
            f"{choices[0]}或者{choices[1]}，你能帮我们拿个主意吗？",
            f"我们在{choices[0]}和{choices[1]}之间犹豫，你有什么建议？")


def generation_choice_request_texts(language, choices):
    texts = choice_request_texts(language, choices)
    return texts[:-1] if language == "zh" else texts


def choice_request_text(language, choices):
    return choice_request_texts(language, choices)[0]


CONFLICTING_CHOICES = {
    "zh": (
        ("我选第一个。", "我更喜欢第二个。"),
        ("我觉得第一个更合适。", "我想选第二个。"),
        ("第一个对我来说比较好。", "第二个比较适合我。"),
        ("我倾向第一个方案。", "我倾向第二个方案。"),
        ("我想选第一种。", "我倒觉得第二种好。"),
        ("第一个听起来更合适。", "我更中意第二个。"),
        ("我的话会选第一个。", "我还是想选第二个。"),
        ("我觉得第一种挺好。", "我比较喜欢第二种。"),
    ),
    "en": (
        ("I prefer the first option.", "I'd choose the second one."),
        ("Let's go with the first one.", "The second option works better for me."),
        ("I vote for option one.", "I'd rather pick option two."),
        ("The first choice suits me.", "I prefer the second choice."),
        ("I'd go with the first one.", "The second sounds better to me."),
        ("Option one makes more sense to me.", "I'd rather go with option two."),
        ("My vote is for the first choice.", "I'm leaning toward the second."),
        ("I'd pick the first.", "I like the second one better."),
    ),
}
CONFLICT_CLARIFICATIONS = {
    "zh": ("我听到了两个不同的选择，你们想选哪一个？", "你们想选的不一样，我们先确定用哪个方案？",
           "你们的偏好不同，要先确定采用哪一个吗？", "一个选第一种，一个选第二种，接下来按哪个来？"),
    "en": ("I heard two different preferences. Which option should we choose?",
           "You prefer different options. Which one would you like to go with?",
           "I heard two different choices. Which should we settle on?",
           "You each prefer a different option. What would you like to choose?"),
}
CHOICE_RESOLUTIONS = {
    "zh": (("那就按第一个方案吧。", "好，我选第一个。", "那先选第一种吧。", "我们就用第一个方案。"),
           ("那就按第二个方案吧。", "好，我选第二个。", "那先选第二种吧。", "我们就用第二个方案。")),
    "en": (("Let's go with the first option.", "I'll choose the first one.", "Let's pick option one.", "The first choice it is."),
           ("Let's go with the second option.", "I'll choose the second one.", "Let's pick option two.", "The second choice it is.")),
}
PERSONAL_RESOLUTIONS = {
    "zh": (("我还是选第一个。", "我个人更倾向第一种。", "我的选择是第一个。"),
           ("我还是选第二个。", "我个人更倾向第二种。", "我的选择是第二个。")),
    "en": (("I still prefer the first option.", "My choice is the first one.", "I'd stick with option one."),
           ("I still prefer the second option.", "My choice is the second one.", "I'd stick with option two.")),
}
CHOICE_RESOLUTIONS = {
    language: tuple(CHOICE_RESOLUTIONS[language][index] + PERSONAL_RESOLUTIONS[language][index] for index in (0, 1))
    for language in ("zh", "en")
}
CHOICE_CONFIRMATIONS = {
    "zh": ("好，我就按照你选择的方案给出建议。", "明白，我会基于你选的方案提供建议。",
           "好，那我们就按这个选择继续讨论。", "了解，接下来就围绕这个方案来看。"),
    "en": ("Got it. I'll base my suggestions on the option you chose.",
           "Okay, I'll give suggestions based on your choice.",
           "All right, let's continue with that choice.",
           "Understood. I'll focus on the option you selected."),
}
PERSONAL_CONFIRMATIONS = {
    "zh": ("好，我先按你选的方案继续分析。", "明白，我会基于你的选择给出建议。",
           "好的，我先围绕你倾向的方案说明。", "了解，我先从你的选择说起。"),
    "en": ("Got it. I'll focus on your choice for now.", "Okay, I'll base my suggestions on your choice.",
           "Understood. I'll start with the option you prefer.", "All right, let's consider your choice first."),
}
CHOICE_CONFIRMATIONS = {language: CHOICE_CONFIRMATIONS[language] + PERSONAL_CONFIRMATIONS[language]
                        for language in ("zh", "en")}

CHOICE_ACKNOWLEDGEMENTS = {
    "zh": {
        "agreement": ("好，这次我可以先按你说的。", "可以，这次我同意先这样试试。", "行，那这次先按你的选择来。"),
        "open": ("我还是倾向另一种，不过可以先比较一下。", "我暂时想法不同，先把两个方案都看看吧。",
                 "我还没决定，咱们先听听两种方案的区别。"),
    },
    "en": {
        "agreement": ("Okay, I can go with your choice this time.", "That's fine with me for now.",
                      "All right, let's try your choice this time."),
        "open": ("I still prefer the other option, but let's compare them first.",
                 "I'm not convinced yet; let's look at both choices.",
                 "I haven't decided. Let's hear how the two options compare."),
    },
}
JOINT_CONFIRMATIONS = {
    "zh": {
        "agreement": ("好，那我先按你们两位商量的方向继续。", "明白，先按你们刚才商量的选择分析。"),
        "open": ("好，我们先比较两个方案，不急着替你们决定。", "明白，我先说明两种方案的区别。"),
    },
    "en": {
        "agreement": ("Okay, I'll start with the direction you two agreed to try.",
                      "Got it. Let's continue with the option you both agreed to try."),
        "open": ("Okay, let's compare both options before deciding.",
                 "Understood. I'll explain the differences between the two choices first."),
    },
}
