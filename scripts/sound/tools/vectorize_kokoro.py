"""Exact vectorization of the known Kokoro1.1 ONNX repeat_interleave alignment loop.
Only the pinned source graph is accepted. Original weights/files are never changed.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

SUPPORTED_SHA="99db0bc8fd0c5033d249fe922db16768b71b4a6b26cfd93d206fbf66c76efb5f"

def repeat_nodes(values,repeats,output,prefix="vector_repeat"):
    import onnx
    from onnx import helper as h,numpy_helper as nh
    def name(s):return prefix+"_"+s
    initial=[nh.from_array(np.asarray(v,dtype=np.int64),name(k)) for k,v in (("flat",[-1]),("zero",0),("one",1),("axis0",[0]),("axis1",[1]))]
    nodes=[
        h.make_node("Reshape",[values,name("flat")],[name("values")]),
        h.make_node("Reshape",[repeats,name("flat")],[name("repeats")]),
        h.make_node("CumSum",[name("repeats"),name("zero")],[name("ends")]),
        h.make_node("ReduceSum",[name("repeats"),name("axis0")],[name("total")],keepdims=0),
        h.make_node("Range",[name("zero"),name("total"),name("one")],[name("steps")]),
        h.make_node("Unsqueeze",[name("ends"),name("axis1")],[name("ends2d")]),
        h.make_node("Unsqueeze",[name("steps"),name("axis0")],[name("steps2d")]),
        h.make_node("GreaterOrEqual",[name("steps2d"),name("ends2d")],[name("after_end")]),
        h.make_node("Cast",[name("after_end")],[name("after_int")],to=onnx.TensorProto.INT64),
        h.make_node("ReduceSum",[name("after_int"),name("axis0")],[name("indices")],keepdims=0),
        h.make_node("Gather",[name("values"),name("indices")],[output],axis=0)]
    return nodes,initial


def prune(graph):
    # Retain free-variable captures for any remaining subgraphs as well as explicit inputs.
    import onnx
    needed={v.name for v in graph.output};kept=[]
    for node in reversed(graph.node):
        if not any(o in needed for o in node.output):continue
        kept.append(node);needed.update(node.input)
        for a in node.attribute:
            if a.type==onnx.AttributeProto.GRAPH:
                produced={o for n in a.g.node for o in n.output}|{i.name for i in a.g.input}|{i.name for i in a.g.initializer}
                needed.update(i for n in a.g.node for i in n.input if i not in produced)
    del graph.node[:];graph.node.extend(reversed(kept))
    initial=[i for i in graph.initializer if i.name in needed];del graph.initializer[:];graph.initializer.extend(initial)
    info=[i for i in graph.value_info if i.name in needed];del graph.value_info[:];graph.value_info.extend(info)


def transform(source,destination,seed=None,reference_only=False):
    import onnx
    source=Path(source);destination=Path(destination)
    with source.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
    if sha!=SUPPORTED_SHA:raise ValueError("Refusing an unknown source graph; transform assumptions are pinned by SHA256")
    if destination.exists():raise FileExistsError("Choose a fresh derived model path")
    model=onnx.load(source);before=len(model.graph.node)
    if not reference_only:
        original=next(n for n in model.graph.node if n.name=="/ConcatFromSequence" and n.op_type=="ConcatFromSequence")
        assert list(original.input)==["/Loop_output_0"]
        replacement,initial=repeat_nodes("/Reshape_1_output_0","/Where_1_output_0",original.output[0])
        nodes=[]
        for n in model.graph.node:nodes.extend(replacement if n.name=="/ConcatFromSequence" else [n])
        del model.graph.node[:];model.graph.node.extend(nodes);model.graph.initializer.extend(initial);prune(model.graph)
        if any(n.op_type in ("Loop","SplitToSequence","SequenceEmpty","ConcatFromSequence") for n in model.graph.node):raise ValueError("Unexpected remaining control-flow/sequence operations")
    if seed is not None:
        for n in model.graph.node:
            if n.op_type.startswith("Random"):
                attrs=[a for a in n.attribute if a.name!="seed"];del n.attribute[:];n.attribute.extend(attrs);n.attribute.append(onnx.helper.make_attribute("seed",float(seed)))
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True,exist_ok=True);onnx.save(model,destination)
    with destination.open("rb") as f:derived=hashlib.file_digest(f,"sha256").hexdigest()
    report={"source_sha256":sha,"derived_sha256":derived,"transform":"reference_seed_only" if reference_only else "repeat_interleave_cumsum_search_v1","nodes_before":before,"nodes_after":len(model.graph.node),"seed_for_equivalence_test":seed,"source_model":"https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh","not_production_qualified_yet":True}
    destination.with_suffix(".provenance.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--source",default=r"D:\DuplexVoice_resources\models\kokoro-multi-lang-v1_1\model.onnx");ap.add_argument("--out",required=True);ap.add_argument("--seed",type=int);ap.add_argument("--reference-only",action="store_true")
    args=ap.parse_args();print(json.dumps(transform(args.source,args.out,args.seed,args.reference_only)),flush=True)

if __name__=="__main__":main()
