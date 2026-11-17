"""Goal credit is one exact master span, excluding outer padding and component duplication."""
import math

def credited_frames(qa,frames,sr):
    rows=qa.get("per_turn",[])
    if not rows or not isinstance(frames,int) or frames<=0 or sr<=0:return None
    starts=[];ends=[]
    for index,row in enumerate(rows):
        timing=row.get("timing",{})
        if row.get("turn_id")!=index:return None
        values=[]
        for name in ("start_sec","end_sec"):
            value=timing.get(name)
            if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value):return None
            sample=value*sr;rounded=round(sample)
            if abs(sample-rounded)>1e-5:return None
            values.append(rounded)
        start,end=values
        if not 0<=start<end<=frames:return None
        starts.append(start);ends.append(end)
    return max(ends)-min(starts)
