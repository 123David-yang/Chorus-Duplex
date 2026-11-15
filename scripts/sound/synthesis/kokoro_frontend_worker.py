"""Persistent isolated Misaki1.1 Chinese G2P; never mutates the ASR process pinyin dictionary."""
from __future__ import annotations
import contextlib,hashlib,json,sys,traceback
from pathlib import Path

def main():
    with contextlib.redirect_stdout(sys.stderr):
        import jieba
        from misaki.zh import ZHG2P
        from pypinyin import load_phrases_dict
        frontend=ZHG2P(version="1.1")
        # Context-specific words; do not globally change the legitimate rare reading of 掺.
        fixes={"掺着":[["chan1"],["zhe5"]],"掺杂":[["chan1"],["za2"]],
               "查查":[["cha2"],["cha5"]],"好呀":[["hao3"],["ya5"]],
               "得扶着":[["dei3"],["fu2"],["zhe5"]],"得集体":[["dei3"],["ji2"],["ti3"]],
               "也得列":[["ye3"],["dei3"],["lie4"]],"姓查":[["xing4"],["zha1"]],"查先生":[["zha1"],["xian1"],["sheng5"]],"得悠着":[["dei3"],["you1"],["zhe5"]]}
        load_phrases_dict(fixes)
        for word in fixes:jieba.add_word(word,freq=1000000,tag="v")
        # Lexical 地 means land; generic particle neutralization must not alter it.
        frontend.frontend.tone_modifier.must_not_neural_tone_words.add("草地")
    print(json.dumps({"ready":True,"frontend":"misaki-0.9.4-zh1.1-context-v2"}),flush=True)
    for line in sys.stdin:
        try:
            request=json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):phones,_=frontend(request["text"])
            result={"ok":True,"phonemes":phones,"frontend":"misaki-0.9.4-zh1.1-context-v2","phonemes_sha256":hashlib.sha256(phones.encode()).hexdigest()}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr);result={"ok":False,"error_type":type(exc).__name__}
        print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
