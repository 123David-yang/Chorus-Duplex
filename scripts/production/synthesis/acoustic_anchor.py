"""Causal heard-prefix boundaries from whole-turn CTC timestamps; no duration-ratio estimates."""
import math
from .asr import strict_numeric_text,phonemes,numeric_agreement

def same_spoken_prefix(expected,actual):
    if strict_numeric_text(expected)==strict_numeric_text(actual):return True
    return numeric_agreement(expected,actual)[0] and phonemes(expected)==phonemes(actual)

def verify_anchors(audio,sr,anchors,asr):
    decoded=asr.transcribe_details(audio,sr,use_itn=False);tokens=decoded["tokens"];times=decoded["timestamps"]
    result={"passed":False,"method":"sensevoice_ctc_next_token_peak_plus_80ms_v1","decode":decoded,"offsets_samples":{},"prefix_checks":[]}
    duration=len(audio)/sr
    if len(tokens)!=len(times) or not tokens or any(not math.isfinite(t) or t<0 or t>duration+.12 for t in times) or any(a>b for a,b in zip(times,times[1:])):
        return result|{"reason":"invalid_ctc_timestamps"}
    for anchor in anchors:
        acc="";boundary=None
        for i,token in enumerate(tokens):
            acc+=token.replace("▁","")
            if same_spoken_prefix(anchor,acc):
                following=next((j for j in range(i+1,len(tokens)) if strict_numeric_text(tokens[j].replace("▁",""))),None)
                if following is not None:boundary=round((times[following]+.08)*sr)
                break
        if boundary is None or not 0<boundary<len(audio)-round(.30*sr):return result|{"reason":"unmatched_prefix_or_insufficient_continuation","anchor":anchor}
        heard=asr.assess(audio[:boundary],sr,anchor)
        result["prefix_checks"].append({"anchor_text":anchor,"boundary_sec":boundary/sr,"assessment":heard})
        if not heard["passed"]:return result|{"reason":"heard_prefix_failed","anchor":anchor}
        result["offsets_samples"][anchor]=boundary
    result["passed"]=bool(anchors)
    return result
