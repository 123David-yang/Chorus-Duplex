"""一体直出的提示词模板（oneshot_agent 专用）—— oneshot_script_prompt + 主题领域 hash。

在用的模板：
    oneshot_script_prompt   一对多直出（主链路，见 oneshot_agent/）
    director_prompt / p1_user_roleplay_multi / assistant_prompt_multi   多 Agent 路径（见 multiagent/）
已迁出的历史模板（论文 §11.4）：
    P1~P4 标签注入式（1:1 时代）→ 历史归档（不参与当前生成）
    P5 质量评审 → scripts/text/judge/review.py
"""
from __future__ import annotations

import zlib

from scripts.config import TOPIC_DOMAINS

from .profile import profile_block
from ..utils.schema import TURN_CAP_SLACK
from ..voices import voice_table_text
from ..quality import rules_block


def topic_for_dialogue(dialogue_id: str) -> str:
    """按 dialogue_id 确定性选主题领域（crc32 稳定哈希；领域列表见 data/topics.json）。"""
    return TOPIC_DOMAINS[zlib.crc32(f"topic-{dialogue_id}".encode()) % len(TOPIC_DOMAINS)]


def oneshot_script_prompt(scenario: str, scenario_desc: str, num_users: int, turns: int,
                          persona_pool: list[str] | None = None,
                          bc_examples: list[str] | None = None,
                          profile: dict | None = None,
                          dialogue_id: str = "") -> str:
    """一体直出：单次调用产出完整对话剧本（含时序/打断/附和标注）。

    bc_examples: 附和语库抽样（oneshot_agent/lexicon.py），供模型参考以增多样性。
    profile: 参与画像（oneshot_agent/profile.py）——给出则按"介入计划"硬约束 assistant 的发言分布。
    """
    pool_block = ""
    if persona_pool:
        pool_text = "\n".join(f"- {p}" for p in persona_pool)
        pool_block = f"""
人物背景池（必须从下面直接挑选 {num_users} 位作为参与者，不要自创人物）：
{pool_text}
把选中人物的背景转换为中文口语化人设（保留职业、性格、生活背景，可加说话风格）。
"""
    n_int = max(1, round(turns * 0.08))
    n_bc = max(1, round(turns * 0.15))
    bc_block = ""
    if bc_examples:
        bc_block = "\n可参考的附和语（按说话人的人设/语体挑选，也可自行发挥更自然的）：" \
                   + "、".join(bc_examples)
    scenario_rules = ""
    if scenario == "side_talk":
        scenario_rules = """
本场景补充要求（非参与型 side-talk，核心是训练 AI"什么时候不该说话"）：
- 至少 2 轮是某人转向**第三方**说话（如对门外外卖员、对路过的邻居、打电话给家人）：
  这种轮 addressee 填 "third_party"，**kind 仍填 "speech"**（说话人还是正式成员），且内容与 AI 无关。
- **第三方也可以真的说话**（0~2 次，不是必须）：这种轮 speaker 填 "third_party_01"（多个则 02 递增）、
  kind 填 "side_talk"（**side_talk 只用于第三方自己说话的轮**）、正常衔接（gap_before_ms 同普通轮转）、
  addressee 填与之对话的用户；每次 1 句、≤15 字、与主话题无关（如"你的外卖放门口了"）。
  有时第三方回应一句、有时完全不出声（由主角单方面说话/动作完成交互）——两种都合理，自行安排。
- **第三方对话的时序（硬性）**：主角**回应**第三方的话必须排在第三方说话之后。允许的形态：
  ① 电话类：主角招呼（"喂？"/"我去接个电话"）→ 第三方说内容（"您好，您的外卖到了"）→ 主角回应（"放门口就行，谢谢"）→（可选）第三方收尾确认；
  ② 门口类：主角察觉（"门铃响了"/"我去拿个外卖"）→ 第三方说内容 → 主角回应/喊话 →（可选）第三方确认。
  **禁止**：主角先说"放门口就行/谢谢"而第三方此前一句话没说（=回应了还没出现的内容）；**电话类必须给第三方台词**（不能只有主角单方面对着电话说话）。
- **事件一致性（硬性）**：前后事件必须自洽——已发生的事不能当成没发生（如外卖已经送到，别人不能说"你继续点外卖"，
  应该说"你先去拿外卖吧"）；细节也要一致（说了放鞋柜，别变成"挂门把手上了"）。
- 主角**回应**第三方的话（如"放门口就行，谢谢"），addressee 一律填 "third_party"，**不要填 assistant**。
- **插曲位置（硬性）**：侧写安排在全剧**中段**（约 1/3~2/3 处），插曲结束后主线必须继续至少 3 轮，
  其中至少 1 轮是 assistant 的发言、且内容**锚回插曲前的话题**（如"刚才你说的剂量分层，我想到…"）——
  体现"插曲是插曲、主话题连续"；**禁止把插曲安排在最后**（外卖收尾 = 对话被掐断，训练不到回归）。
  插曲内容提不提都行：完全不提，或一句话轻轻承接（如"你们先吃，我帮你们继续梳理刚才的…"）后回到主话题。
- **场景与入口语要多样**（不要每条都是外卖）：第三方场景从谱系里变化选择——外卖/快递、电话（含推销/打错）、
  **邻里偶遇寒暄**（如"李大妈，你去买菜啊？""是啊是啊"——直接搭话、无转场语）、店员/服务人员、路人问路、
  家人隔空喊话（"饭好了，下来吃！"）等；入口语（转场）也要多样："你等一下啊""等下""门铃响了""我去开个门"
  "稍等，我接个电话"……偶遇类可无转场语直接开口。

接电话的正确写法（参考，其余场景同理）：
  t_a user_01 → third_party : "哎等下，我接个电话——喂？"
  t_b third_party_01 → user_01 : "您好，您的外卖到了，放门口还是送上来？"
  t_c user_01 → third_party : "放门口就行，谢谢啊。"
  t_d third_party_01 → user_01 : "好嘞，给您放门口了。"（可选）
错误写法：主角一轮里说完"喂？……对，是我。……行，你放门口吧，谢谢"（把整段通话写在一轮里、抢在第三方说话之前）；
或者主角先说"你放门口就行"而第三方此时一句话都还没说。"""
    # 音色表与主题领域：与音频层共用同一确定性分配（voices_for_dialogue / topic_for_dialogue）
    voice_table = voice_table_text(dialogue_id, num_users) if dialogue_id else \
        "user_01~03 由系统固定分配；人设性别须与音色一致"
    topic_domain = topic_for_dialogue(dialogue_id) if dialogue_id else "自定"

    # 单人场景适配（人数 1~6，2026-09-16）：1 人 = 用户与 AI 的对话为主线
    one_user_block = ""
    if num_users == 1:
        one_user_block = """
单人场景（只有 1 位用户 + assistant）：**没有"用户之间的对话"**——该用户的发言一律指向
assistant（或 side-talk 插曲中的第三方）；assistant 的回应对象也只有这一位用户。
"""

    # assistant 的参与分布：有画像 → 按"介入计划"硬约束；无画像 → 默认（低参与/仅点名）
    if profile:
        assistant_rules = profile_block(profile)
    else:
        assistant_rules = """
- assistant 整条对话出现 2~3 次：被**明确点名**（叫它、直接提问）时必须回应；
  此外可在话题自然的时机主动补充一次（不打断正说到一半的话）；
  对用户之间纯粹的闲聊、对第三方说的话，assistant 保持沉默（不产出该轮次）。"""
    return \
f"""\
你是多人语音对话剧本的编剧。请一次性写出完整剧本（不要逐轮生成，直接给出整条对话）。


只输出 JSON，不要任何其他内容：
{{
  "setting": "场景环境（如：家庭晚餐后的客厅）",
  "topic": "对话主题",
  "participants": [
    {{"id": "user_01", "persona": "中文人设", "style": "说话风格"}}
  ],
  "turns": [
    {{"speaker": "user_01", "text": "话语", "addressee": "assistant", "gap_before_ms": 250, "kind": "speech", "responds_to": [], "events": []}},
    {{"speaker": "assistant", "text": "助手的回答", "addressee": "user_01", "gap_before_ms": 250, "kind": "speech", "responds_to": [0], "events": [], "trigger": "cue"}}
  ]
}}


每轮必须带时序字段 gap_before_ms（与上一轮结束的间隔，毫秒，可为负）：
- 正常轮转：150~400（如 250）；紧接着说：0~100
- 打断：约 -300 ~ -600（在上一轮结束前就开口，形成重叠）
- 附和：约 -2000 ~ -3000（压在别人话语的后半段）

其余字段：
- addressee：这句话对谁说（user_01 / assistant / third_party / ...；对多人说话给列表，如 ["user_01","user_02"]）；
  **回应第三方（外卖员/电话里的人）的话，addressee 也填 "third_party"，不要填 assistant**
- responds_to：这句话回应/综合了哪几轮（轮次 id，0 起，如 [0,2]；没有则 []）


剧本要求：
- 参与者之间可以直接对话（不总指向 assistant）；assistant 只回应指向它的话。
- assistant 的回复要有**指向性**：回应某个人的具体观点时，addressee 指向这个人（以"你"或称呼开口），
  不必每轮都对着全体说；同时回应多人的轮（addressee 为列表）保留但不要占多数。
- **称呼与身份（信息只能来自对话本身）**：给每位用户起自然的中文名字/昵称（如"小林""王姐"），
  但**名字必须先被对话"介绍"出来**（当事人自称、或别人喊 ta 一次），此后其他角色才可以用名字指称 ta；
  **在名字出现于台词之前**：单指用"你/您/这位/那位"，指向一对多或子集用"你们/你们俩/两位"，
  对全体用"大家/各位/咱们"（泛称是最后手段，不要滥用）；或在音色可辨且不产生歧义时用**性别/年龄称谓**——
  性别敬称仅当场景中该性别唯一可辨时使用（如一男一女场景可用"这位女士""这位先生"）；
  年龄称谓（"阿姨""小姑娘""大哥"等）须与说话人的音色年龄感一致且不产生歧义；
  **禁止提前使用未在对话中出现过的人名或身份信息**。
  assistant 的"感知"必须由对话历史支持——它只能引用对话里已经出现过的名字、身份、背景，不得凭空知道任何人叫什么、是谁。
  所有台词的称呼一律使用名字/昵称、上述称谓或上述代词/泛称，**严禁出现 user_01、assistant、third_party 这类内部编号**。
  如果剧情里自然出现**新加入或还没被介绍的成员**，assistant 应主动而友好地接话并请教称呼
  （如"刚才那位还没请教怎么称呼？"），对方应答后大家（含 assistant）就用这个名字指称 ta；此类场景不必每条都有。
  **输出前自检（必做）**：从头逐句检查每个人名——若某个人名在此之前的所有台词中从未出现过，
  该处必须改成"你/这位/那位"或补一句自然的介绍（如让对方自我介绍）；assistant 的台词尤其严格：它只能知道对话里已经发生过的事。
  **人设（含人设里的名字）不算"已介绍"**——只有出现在台词里的名字才算；建议在对话开头几句内就让用户自然地互相称呼
  （如"哎小林，我跟你说…"），把名字锚点建立起来，方便后续使用。
- 若场景出现**非语音声音事件**（来电铃声/门铃/消息提示/手机音乐等），在对应轮的 "events" 里标注：
  {{"type": "phone_ring|doorbell|notification|music|knock|alarm", "text": "简述（如：手机来电铃声）", "position": 0.0~1.0（在该轮音频中的位置）}}。
  事件必须与剧情自洽（如：铃声响起后有人接电话）；整条对话 0~2 个，没有就留空数组 []。{scenario_rules}
- 口语化、自然（"嗯""那个"、犹豫），每轮 1~2 句，不要书面语。
- 全剧安排约 {n_int} 次打断：被打断的人只说出**半截话**（text 直接就是不完整的，如"我觉得最好还是问问"），后面不再补全；
  打断者接一句针对半截内容的抢话，kind 标为 "interruption"，两人音频会重叠。
  被打断者不限于用户：**也可以让用户打断 assistant**（assistant 的台词同样半截中断、不再补全）——
  "AI 被抢话"是本数据的重点场景，请在剧情自然的时机适度安排（不必每条都有）。
- 全剧安排约 {n_bc} 次附和：很短（≤6 字，如"嗯嗯""对对""是啊""哈哈"），kind 标为 "backchannel"，
  附和是插在别人说话中间的不抢话轮；用 insert_ratio（0.4~0.75）指明插在上一轮话语的第百分之几处
  （如 0.6 = 对方说到六成时附和）；附和轮不要填 gap_before_ms；**同一条对话内附和语不要重复**。{bc_block}

场景类型：{scenario} —— {scenario_desc}
参与者：{num_users} 位用户（id: user_01, user_02, ...，两位数字）+ 1 个 AI 助手（id: assistant）
本条对话的**主题领域：{topic_domain}**——在该领域内自然展开；人物的身份、职业要与该领域匹配且尽量多样，
不要清一色科研/学术人员（可把人物池背景调整到该领域，保留性格）。
说话人音色（每条对话固定分配，人设与称谓必须对齐）：{voice_table}
人设的性别必须与音色一致（从人物池挑选时优先选性别相符的，必要时把人设调整到相符性别）；
性别/年龄称谓的可用性以此为准（如两位用户同为女声时不能用"女士"区分她们；某性别只有一位时，"这位先生/女士"可唯一指代）。
总轮数：目标严格写 {turns} 轮（先分配轮数预算，附和、打断、AI回应均计数；12轮时不要额外塞进无关插曲）。约 {turns} 轮（**含打断、附和等所有轮**；**硬性上限 {turns + TURN_CAP_SLACK} 轮**——超出会被判不合格）
{one_user_block}{pool_block}
{assistant_rules}
{rules_block(dialogue_id, turns)}
"""
