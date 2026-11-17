"""Local SenseVoice plus text/phonetic diagnostics. No network requests during inference."""
from __future__ import annotations
import os,re,time,unicodedata
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from math import gcd
from scripts.sound.diagnostics.asr_audit import edit_distance

def prepare(text):
    from opencc import OpenCC
    import cn2an
    text=OpenCC("t2s").convert(unicodedata.normalize("NFKC",text))
    text=re.sub(r"<\|[^|]+\|>","",text)
    # Normalize digits to spoken Chinese, never erase decimal/fraction structure.
    return cn2an.transform(text,"an2cn")

def normalize(text):
    return "".join(c.lower() for c in prepare(text) if c.isalnum())

def phonemes(text):
    from pypinyin import lazy_pinyin,Style
    tokens=[]
    # Standard third-tone/yi/bu sandhi within a clause; never cross written pause boundaries.
    for clause in re.split(r"[，。！？,.;；!?、\n]",prepare(text)):
        current=[t.lower() for t in lazy_pinyin(clause,style=Style.TONE3,neutral_tone_with_five=True,
                        tone_sandhi=True,errors=lambda x:list(x)) if t.isalnum()]
        # pypinyin applies word-local rules; cover an unambiguous two-third-tone run across words.
        # Leave longer runs to lexical/prosodic handling rather than flattening every lexical tone.
        i=0
        while i<len(current):
            if not current[i].endswith("3"):i+=1;continue
            end=i
            while end<len(current) and current[end].endswith("3"):end+=1
            if end-i==2:current[i]=current[i][:-1]+"2"
            i=end
        tokens.extend(current)
    return tokens

def numeric_values(text):
    """Conservative explicit quantities. Ordinary 一般/零食/一起 are not numerical claims."""
    import cn2an
    from fractions import Fraction
    from opencc import OpenCC
    raw=OpenCC("t2s").convert(unicodedata.normalize("NFKC",text))
    # Explicit half-unit quantities are real numeric anchors, not an unrelated integer.
    def half_unit(match):
        value=match.group(1).replace("两","二")
        number=Fraction(str(cn2an.transform(value,"cn2an")))+Fraction(1,2)
        return str(float(number))+match.group(2)
    raw=re.sub(r"([0-9零〇一二两三四五六七八九十百千万亿]+)(个|小时|天|点)半",half_unit,raw)
    raw=re.sub(r"(?<![零〇一二两三四五六七八九十百千万亿0-9])半(?=小时|天|分钟|秒|个)","0.5",raw)
    numeral=r"[零〇一二两三四五六七八九十百千万亿]+"
    atom=r"(?:\d+(?:\.\d+)?|"+numeral+r"(?:点"+numeral+r")?)"
    pattern=r"百分之"+atom+r"|"+atom+r"分之"+atom+r"|\d+(?:\.\d+)?/\d+(?:\.\d+)?|"+atom+r"%|"+atom
    values=[]
    for match in re.finditer(pattern,raw):
        value=match.group();tail=raw[match.end():]
        explicit=raw[:match.start()].endswith("第") or any(c.isdigit() for c in value) or len(value)>1 or bool(re.match(r"个|人|元|块|点|年|月|天|周|小时|分钟|秒|公里|米|成|次|只|岁|份",tail))
        if not explicit:continue
        if value in ("千万","万一") and re.match(r"别|不|要",tail):continue
        if re.fullmatch(r"[二两三四五六七八九][三四五六七八九](?:十|百|千|万)?",value) and not any(c.isdigit() for c in value):
            values.append("approx:"+value.replace("两","二"));continue
        converted=cn2an.transform(value.replace("两","二"),"cn2an")
        if "分之" in converted:
            denominator,numerator=converted.split("分之",1);converted=numerator+"/"+denominator
        if converted.endswith("%"):
            number=converted[:-1];converted=str(Fraction(number)/100)
        elif "/" in converted:
            n,d=converted.split("/",1);converted=str(Fraction(n)/Fraction(d))
        else:
            try:converted=str(Fraction(converted))
            except ValueError:continue
        values.append(converted)
    return values


def strict_numeric_text(text):
    """Exact spoken-digit equivalence, without transform()'s context-dependent year/approximation rules."""
    from opencc import OpenCC
    import cn2an
    raw=OpenCC("t2s").convert(unicodedata.normalize("NFKC",text)).replace("−","-")
    raw=re.sub(r"<\|[^|]+\|>","",raw)
    raw=raw.replace("/","分隔符").replace("⁄","分隔符").replace("%","百分号").replace("+","正号")
    def spoken(match):
        try:return cn2an.an2cn(match.group(),"low")
        except ValueError:return match.group()
    raw=re.sub(r"\d+(?:\.\d+){2,}",lambda m:"版本分隔符".join(cn2an.an2cn(part,"low") for part in m.group().split(".")),raw)
    raw=re.sub(r"-?\d+(?:\.\d+)?",spoken,raw)
    return "".join(c.lower() for c in raw if c.isalnum())

def numeric_agreement(expected,actual):
    reference=numeric_values(expected);hypothesis=numeric_values(actual)
    if reference==hypothesis:return True,"explicit_values",reference,hypothesis
    # Full character equality after strict number spelling proves the alias. Never use approximate CER here.
    left,right=strict_numeric_text(expected),strict_numeric_text(actual)
    if left and left==right:return True,"strict_spoken_alias_exact_text",reference,hypothesis
    return False,"explicit_values_mismatch",reference,hypothesis


class SenseVoice:
    def __init__(self,folder=None):
        import sherpa_onnx
        folder=Path(folder or os.environ.get("SENSEVOICE_MODEL_DIR",r"D:\DuplexVoice_resources\models\sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"))
        for name in ("model.int8.onnx","tokens.txt","asset-provenance.json"):
            if not (folder/name).is_file():raise FileNotFoundError(folder/name)
        self.recognizer=sherpa_onnx.OfflineRecognizer.from_sense_voice(model=str(folder/"model.int8.onnx"),tokens=str(folder/"tokens.txt"),
            num_threads=int(os.environ.get("ASR_THREADS","2")),language="zh",use_itn=True,provider="cpu")
    def transcribe_details(self,audio,sr,use_itn=None):
        audio=np.asarray(audio,dtype=np.float32)
        if audio.ndim>1:audio=audio.mean(axis=1)
        if sr!=16000:
            common=gcd(sr,16000);audio=resample_poly(audio,16000//common,sr//common).astype(np.float32)
        stream=self.recognizer.create_stream()
        if use_itn is not None:stream.set_option("use_itn","1" if use_itn else "0")
        stream.accept_waveform(16000,audio)
        self.recognizer.decode_stream(stream)
        result=stream.result
        return {"text":result.text.strip(),"tokens":list(result.tokens),"timestamps":list(result.timestamps)}
    def transcribe(self,audio,sr):
        self.last_decode=self.transcribe_details(audio,sr)
        return self.last_decode["text"]
    def assess(self,audio,sr,expected):
        started=time.perf_counter();actual=self.transcribe(audio,sr)
        ref,hyp=normalize(expected),normalize(actual)
        ref_p,hyp_p=phonemes(expected),phonemes(actual)
        char_errors=edit_distance(ref,hyp);phonetic_errors=edit_distance(ref_p,hyp_p)
        cer=char_errors/max(1,len(ref))
        per=phonetic_errors/max(1,len(ref_p))
        # Short clips are strict. Humming/count changes never silently become a correct transcript.
        numeric_match,numeric_rule,expected_numbers,actual_numbers=numeric_agreement(expected,actual)
        passed=numeric_match and ((ref==hyp) if len(ref)<8 else (cer<=.10 or per<=.05))
        return {"recognizer":"sensevoice_int8","expected":expected,"transcript":actual,
                "normalized_expected":ref,"normalized_transcript":hyp,"cer":round(cer,5),"phonetic_er":round(per,5),
                "char_errors":char_errors,"reference_chars":len(ref),"phonetic_errors":phonetic_errors,"reference_phonemes":len(ref_p),
                "normalization":"nfkc+t2s+spoken_numbers+clause_tone_sandhi_strict_numeric_alias_v5","numeric_values_expected":expected_numbers,"numeric_values_transcript":actual_numbers,"numeric_match":numeric_match,"numeric_match_rule":numeric_rule,"passed":passed,"elapsed_s":round(time.perf_counter()-started,4),"short_exact_match_required":len(ref)<8}
