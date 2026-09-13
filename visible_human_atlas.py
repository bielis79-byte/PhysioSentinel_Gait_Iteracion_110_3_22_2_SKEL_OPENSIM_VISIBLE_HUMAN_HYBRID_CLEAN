from __future__ import annotations
from pathlib import Path
import hashlib, json, zipfile, re, functools, tempfile
import numpy as np
import trimesh

CACHE_ROOT=Path("/tmp/physiosentinel_visible_human_v3220")
SOURCE_PAGE="https://digitalcommons.du.edu/visiblehuman/"
ATTRIBUTION=(
    "Visible Human Male/Female lower-extremity musculoskeletal geometry, "
    "Center for Orthopaedic Biomechanics, University of Denver, CC BY 4.0; "
    "Andreassen et al., Scientific Data 10, 34 (2023)."
)

BONES=[
    "coccyx","sacrum","pelvis","femur","patella","tibia","fibula","talus",
    "calcaneus","navicular","cuboid","cuneiform","phalanges","metatars","tarsal"
]
MUSCLES=[
    "adductor brevis","adductor longus","adductor magnus","biceps femoris long",
    "biceps femoris short","extensor digitorum longus","extensor hallucis longus",
    "flexor digitorum longus","flexor hallucis longus","gastrocnemius lateral",
    "gastrocnemius medial","gluteus maximus","gluteus medius","gluteus minimus",
    "gracilis","iliacus","inferior gemellus","obturator externus","obturator internus",
    "pectineus","peroneus longus","fibularis longus","piriformis","plantaris","popliteus",
    "psoas major","quadratus femoris","rectus femoris","sartorius","semimembranosus",
    "semitendinosus","soleus","superior gemellus","tensor fasciae latae","tibialis anterior",
    "tibialis posterior","vastus intermedius","vastus lateralis","vastus medialis"
]

def _norm(s):
    s=str(s).lower().replace("lnferior","inferior").replace("lliacus","iliacus")
    s=s.replace("lnternus","internus").replace("lntermedius","intermedius")
    s=re.sub(r"[_\-.]+"," ",s)
    s=re.sub(r"\s+"," ",s).strip()
    return s

def _side(path_text):
    t=" "+_norm(path_text)+" "
    if any(x in t for x in (" right "," rt "," r side ","_r ")):
        return "r"
    if any(x in t for x in (" left "," lt "," l side ","_l ")):
        return "l"
    # folder/file suffixes
    low=str(path_text).lower()
    if re.search(r"(^|[/\\ _-])r($|[/\\ _-])",low): return "r"
    if re.search(r"(^|[/\\ _-])l($|[/\\ _-])",low): return "l"
    return None

def _structure(path_text):
    t=_norm(Path(path_text).stem)
    # bone first
    for x in BONES:
        if x in t:
            return "bone",x
    for x in MUSCLES:
        if x in t:
            return "muscle",x
    # broad aliases
    aliases={
        "gastrocnemius":"gastrocnemius medial",
        "peroneus":"peroneus longus",
        "fibularis":"fibularis longus",
        "gluteus":"gluteus maximus",
        "vastus":"vastus lateralis",
    }
    for a,b in aliases.items():
        if a in t: return "muscle",b
    return None,None

def _region(kind,name):
    n=_norm(name)
    if kind=="bone":
        if n in ("pelvis","sacrum","coccyx"): return "pelvis"
        if n in ("femur","patella"): return "thigh"
        if n in ("tibia","fibula"): return "shank"
        return "foot"
    hip=("gluteus","iliacus","psoas","piriformis","gemellus","obturator","pectineus",
         "quadratus femoris","tensor fasciae latae")
    thigh=("adductor","biceps femoris","gracilis","rectus femoris","sartorius",
           "semimembranosus","semitendinosus","vastus")
    if any(x in n for x in hip): return "pelvis"
    if any(x in n for x in thigh): return "thigh"
    return "shank"

def prepare_visible_human_zip(data: bytes, subject="Female"):
    sha=hashlib.sha256(data).hexdigest()[:16]
    root=CACHE_ROOT/f"{subject.lower()}_{sha}"
    root.mkdir(parents=True,exist_ok=True)
    marker=root/"manifest.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    zpath=root/"source.zip"
    zpath.write_bytes(data)
    extract=root/"stl"
    extract.mkdir(exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if n.lower().endswith(".stl"):
                try: z.extract(n,extract)
                except Exception: pass
    parts=[]
    for p in extract.rglob("*.stl"):
        kind,name=_structure(str(p))
        if not kind: continue
        side=_side(str(p))
        parts.append({
            "path":str(p),"kind":kind,"name":name,"side":side,
            "region":_region(kind,name)
        })
    man={
        "ready":bool(parts),"root":str(root),"subject":subject,"sha":sha,
        "parts":parts,"bones":sum(x["kind"]=="bone" for x in parts),
        "muscles":sum(x["kind"]=="muscle" for x in parts),
        "license":"CC BY 4.0","attribution":ATTRIBUTION,
        "source_page":SOURCE_PAGE
    }
    marker.write_text(json.dumps(man,ensure_ascii=False,indent=2),encoding="utf-8")
    return man

@functools.lru_cache(maxsize=256)
def _mesh(path):
    m=trimesh.load_mesh(path,process=False)
    if isinstance(m,trimesh.Scene):
        m=trimesh.util.concatenate(tuple(g for g in m.geometry.values()))
    V=np.asarray(m.vertices,np.float32)
    F=np.asarray(m.faces,np.int32)
    return V,F

def _unit(v,fb):
    v=np.asarray(v,float); n=np.linalg.norm(v)
    return np.asarray(fb,float) if n<1e-9 else v/n

def _pca(V):
    X=np.asarray(V,float)-np.mean(V,axis=0)
    _,_,vt=np.linalg.svd(X,full_matrices=False)
    B=vt.T
    if np.linalg.det(B)<0: B[:,2]*=-1
    return B

def _target_frame(J,nm,frame,side,region):
    pel=nm.get("pelvis")
    hip=nm.get(f"femur_{side}")
    knee=nm.get(f"tibia_{side}")
    ankle=nm.get(f"talus_{side}")
    cal=nm.get(f"calcn_{side}")
    toe=nm.get(f"toes_{side}")
    other=nm.get(f"femur_{'l' if side=='r' else 'r'}")
    lr=(J[frame,hip]-J[frame,other]) if hip is not None and other is not None else np.array([1.,0.,0.])
    if region=="pelvis":
        a=J[frame,pel] if pel is not None else J[frame,hip]
        b=J[frame,hip] if hip is not None else a+np.array([0,-.2,0])
    elif region=="thigh":
        a=J[frame,hip]; b=J[frame,knee]
    elif region=="shank":
        a=J[frame,knee]; b=J[frame,ankle]
    else:
        a=J[frame,cal] if cal is not None else J[frame,ankle]
        b=J[frame,toe] if toe is not None else J[frame,ankle]+np.array([0,0,.15])
    z=_unit(b-a,[0,-1,0])
    x=np.asarray(lr,float); x=x-np.dot(x,z)*z; x=_unit(x,[1,0,0])
    y=_unit(np.cross(z,x),[0,0,1])
    x=_unit(np.cross(y,z),x)
    return np.asarray(a,float),np.stack([x,y,z],axis=1),float(np.linalg.norm(b-a))

def _donor_reference(man,side,region):
    # select key bone
    preferred={
        "pelvis":["pelvis","sacrum"],
        "thigh":["femur"],
        "shank":["tibia","fibula"],
        "foot":["calcaneus","phalanges","talus"]
    }[region]
    cand=[p for p in man["parts"] if p["kind"]=="bone" and p.get("side") in (side,None) and p["name"] in preferred]
    if not cand:
        cand=[p for p in man["parts"] if p["kind"]=="bone" and p.get("side") in (side,None)]
    if not cand: return None
    # use first preferred hit
    p=sorted(cand,key=lambda q: preferred.index(q["name"]) if q["name"] in preferred else 99)[0]
    V,_=_mesh(p["path"])
    C=np.mean(V,axis=0)
    B=_pca(V)
    # choose longest PCA axis as local Z
    spans=np.ptp((V-C)@B,axis=0)
    iz=int(np.argmax(spans))
    others=[i for i in range(3) if i!=iz]
    z=B[:,iz]; x=B[:,others[0]]; y=_unit(np.cross(z,x),B[:,others[1]])
    x=_unit(np.cross(y,z),x)
    Bb=np.stack([x,y,z],axis=1)
    L=max(float(spans[iz]),1e-6)
    # origin at proximal extreme along local z
    q=(V-C)@Bb
    origin=C+Bb[:,2]*float(np.max(q[:,2]))
    # flip z so geometry extends primarily in + local longitudinal from origin
    Bb[:,2]*=-1
    Bb[:,1]=_unit(np.cross(Bb[:,2],Bb[:,0]),Bb[:,1])
    return origin,Bb,L

def _skin_radius(skin,a,b):
    V=np.asarray(skin,float); c=.5*(a+b); L=max(np.linalg.norm(b-a),1e-6)
    d=np.linalg.norm(V-c,axis=1)
    k=min(max(50,len(V)//100),len(V))
    if k==0:return .05*L
    pts=V[np.argpartition(d,k-1)[:k]]
    return float(np.percentile(np.linalg.norm(pts-c,axis=1),65))

def build_visible_human_frame(man,joints,joint_names,skin_vertices,frame=0,
                              show_bones=True,show_muscles=True,max_faces_each=220):
    J=np.asarray(joints,np.float32); skin=np.asarray(skin_vertices,np.float32)
    nm={str(n):i for i,n in enumerate(joint_names)}
    out=[]
    refs={}
    for side in ("r","l"):
        for region in ("pelvis","thigh","shank","foot"):
            refs[(side,region)]=_donor_reference(man,side,region)
    for p in man["parts"]:
        if p["kind"]=="bone" and not show_bones: continue
        if p["kind"]=="muscle" and not show_muscles: continue
        side=p.get("side")
        if side not in ("r","l"):
            # central pelvis/sacrum shown once on right-frame convention
            side="r"
        ref=refs.get((side,p["region"]))
        if ref is None: continue
        donor_origin,donor_B,donor_L=ref
        try:
            target_origin,target_B,target_L=_target_frame(J,nm,frame,side,p["region"])
        except Exception:
            continue
        V,F=_mesh(p["path"])
        Q=(V-donor_origin)@donor_B
        scale=target_L/max(donor_L,1e-6)
        if p["kind"]=="bone":
            transverse=scale
        else:
            # preserve muscle volume better; only mild transverse adaptation to skin
            try:
                if p["region"]=="thigh":
                    a=J[frame,nm[f"femur_{side}"]]; b=J[frame,nm[f"tibia_{side}"]]
                elif p["region"]=="shank":
                    a=J[frame,nm[f"tibia_{side}"]]; b=J[frame,nm[f"talus_{side}"]]
                else:
                    a=target_origin; b=target_origin+target_B[:,2]*target_L
                skin_r=_skin_radius(skin,a,b)
                native_r=max(np.percentile(np.linalg.norm(Q[:,:2],axis=1),70),1e-6)
                transverse=float(np.clip((skin_r*.72)/native_r,scale*.65,scale*1.35))
            except Exception:
                transverse=scale
        Qs=Q*np.array([transverse,transverse,scale])[None,:]
        W=target_origin+Qs@target_B.T

        # display decimation by deterministic face sampling
        step=max(1,int(np.ceil(len(F)/float(max_faces_each))))
        Fs=np.asarray(F[::step,:3],np.int32)
        used=np.unique(Fs.reshape(-1))
        remap=np.full(len(W),-1,np.int32); remap[used]=np.arange(len(used),dtype=np.int32)
        out.append({
            "id":f"{p['kind']}:{p['name']}:{side}",
            "kind":p["kind"],"name":p["name"],"side":side,
            "V":np.asarray(W[used],np.float32),"F":remap[Fs]
        })
    return out

def atlas_counts(man):
    if not man:return {"bones":0,"muscles":0}
    return {"bones":int(man.get("bones",0)),"muscles":int(man.get("muscles",0))}


def build_visible_human_muscle_sequence(man,joints,joint_names,skin_sequence,max_faces_each=18):
    """V110.3.22.2
    Builds a compact 75-frame volumetric MUSCLE sequence only.
    Bone geometry is intentionally excluded: the active skeleton is OpenSim/Hamner.
    Faces are deterministic and shared across frames.
    """
    J=np.asarray(joints,np.float32)
    S=np.asarray(skin_sequence,np.float32)
    frames=[]
    faces_ref=None
    for t in range(len(J)):
        parts=build_visible_human_frame(
            man,J,joint_names,S[t],frame=t,
            show_bones=False,show_muscles=True,max_faces_each=max_faces_each
        )
        verts=[]
        faces=[]
        off=0
        for p in parts:
            V=np.asarray(p["V"],np.float32)
            F=np.asarray(p["F"],np.int32)
            if len(V)==0 or len(F)==0:
                continue
            verts.append(V)
            faces.append(F+off)
            off+=len(V)
        if not verts:
            return None,None
        VV=np.vstack(verts).astype(np.float32)
        FF=np.vstack(faces).astype(np.int32)
        if faces_ref is None:
            faces_ref=FF
        elif FF.shape!=faces_ref.shape:
            # deterministic fallback: keep common prefix topology.
            m=min(len(FF),len(faces_ref))
            faces_ref=faces_ref[:m]
            FF=FF[:m]
        frames.append(VV)
    # Vertex count should be stable because the same atlas parts/face sampling are used.
    nmin=min(len(v) for v in frames)
    frames=np.stack([v[:nmin] for v in frames],axis=0)
    # Faces that reference truncated vertices are removed.
    faces_ref=np.asarray(faces_ref,np.int32)
    faces_ref=faces_ref[np.all(faces_ref<nmin,axis=1)]
    return frames.astype(np.float32),faces_ref.astype(np.int32)
