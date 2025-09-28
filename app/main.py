import os, json, pickle, re, numpy as np, hnswlib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

# ====== USTAWIENIA ======
INDEX_DIR   = os.getenv("INDEX_DIR", "index")
EMB_MODEL   = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-large")
RERANK_MODEL= os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# ====== NARZĘDZIA ======
TOKEN = re.compile(r"\w+", re.U)
def tok(x:str): return TOKEN.findall(x.lower())
def rrf(rank:int): return 1.0 / (60.0 + rank)
def mmss(s: float): m = int(s // 60); ss = int(s - m*60); return f"{m:02d}:{ss:02d}"
def yt_link(url: str, start: float): sep = "&" if "?" in url else "?"; return f"{url}{sep}t={int(start)}s"

SYNONYMS = {
    "ohp": ["wyciskanie nad głowę","military press","shoulder press","wyciskanie żołnierskie"],
    "bench": ["wyciskanie leżąc","flat bench","wyciskanie sztangi leżąc","ławka płaska"],
    "przysiad": ["squat","back squat","front squat","goblet squat","siady"],
    "martwy ciąg": ["deadlift","rdl","romanian deadlift","sumo deadlift","mc"],
    "wiosłowanie": ["barbell row","pendlay row","t-bar row","wiosła"],
    "podciąganie": ["pull-up","chin-up","podchwytem","nachwytem"],
    "szrugsy": ["shrugs","wzruszanie barkami","kaptury","trapez"],
    "wznosy boczne": ["lateral raise","side raise"],
    "hip thrust": ["unoszenie bioder","glute bridge"],
    "face pull": ["ściąganie liny do twarzy","tylne aktony barków"],
    "rotatory": ["rotatory barku","external rotation","internal rotation"],
    "rir": ["reps in reserve","powtórzenia w zapasie"],
    "rpe": ["rate of perceived exertion"],
    "bracing": ["usztywnienie core","napięcie brzucha","oddech przeponowy"],
    "depresja łopatek": ["depression of scapula","ściąganie łopatek w dół"],
    "retrakcja łopatek": ["retraction of scapula"],
    "protrakcja łopatek": ["protraction of scapula"]
}
def expand_query(q: str) -> str:
    base = q.lower()
    extra = []
    for k, vs in SYNONYMS.items():
        if k in base or any(v in base for v in vs):
            extra += [k] + vs
    if extra: base += " " + " ".join(sorted(set(extra)))
    return base

# ====== ŁADOWANIE INDEKSU/MODELI ======
app = FastAPI(title="Coach AI (PL)", version="0.1")

with open(f"{INDEX_DIR}/meta.jsonl", encoding="utf-8") as f:
    META = [json.loads(l) for l in f]
ID2M = {m["id"]: m for m in META}

embedder = SentenceTransformer(EMB_MODEL)
dim = embedder.get_sentence_embedding_dimension()
hnsw = hnswlib.Index(space="cosine", dim=dim); hnsw.load_index(f"{INDEX_DIR}/dense_hnsw.bin")
with open(f"{INDEX_DIR}/bm25.pkl","rb") as f:
    BM25 = pickle.load(f)

# Reranker
reranker = FlagReranker(RERANK_MODEL, use_fp16=False)

# ====== SCHEMATY ======
class SearchBody(BaseModel):
    query: str
    k: int = 8

class AnswerBody(BaseModel):
    query: str         # treść pytania od usera
    profile: str = ""  # opcjonalny profil użytkownika („pamięć”)
    k: int = 6

# ====== SEARCH ======
@app.post("/search")
def search(body: SearchBody):
    qx = expand_query(body.query)
    q_emb = embedder.encode([f"query: {qx}"], normalize_embeddings=True)[0]

    idxs, dists = hnsw.knn_query(q_emb, k=40)
    dense_hits = [(int(i), float(1 - d)) for i,d in zip(idxs[0], dists[0])]

    bm_scores = BM25.get_scores(tok(qx))
    bm_idx = np.argsort(bm_scores)[::-1][:40]
    bm_hits = [(int(i), float(bm_scores[i])) for i in bm_idx]

    ranks = {}
    for r,(i,_) in enumerate(dense_hits): ranks[i] = ranks.get(i,0)+rrf(r+1)
    for r,(i,_) in enumerate(bm_hits):    ranks[i] = ranks.get(i,0)+rrf(r+1)

    # kandydaci do reranku
    cand = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:40]
    pairs, metas = [], []
    for i,_ in cand:
        m = ID2M[i]; text = m.get("text","")
        if text:
            pairs.append((qx, text)); metas.append(m)

    if not pairs: return []
    scores = reranker.compute_score(pairs, normalize=True)
    reranked = sorted([(metas[i], float(s)) for i,s in enumerate(scores)], key=lambda x: x[1], reverse=True)[:body.k]

    out = []
    for m,sc in reranked:
        out.append({
            "title": m["title"],
            "span": f"{mmss(m['start_sec'])}-{mmss(m['end_sec'])}",
            "url": yt_link(m["url"], m["start_sec"]),
            "score": round(sc, 4),
            "text": (m["text"][:240]+"…") if len(m["text"])>240 else m["text"]
        })
    return out

# ====== ANSWER ======
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """Jesteś wymagającym, dociekliwym, rzeczowym i wyrozumiałym trenerem.
Priorytety: bezpieczeństwo, technika, progres.
Zawsze odpowiadasz po polsku, precyzyjnie i konkretnie.
Rozróżniaj dyskomfort od bólu: mocny ból → przerwij i rozważ rehabilitację/konsultację.
Jeśli pytają o plan: podaj Serie/Powtórzenia/RIR/Tempo + prostą progresję.
Na końcu zawsze wypisz ŹRÓDŁA z timestampami ▶ [MM:SS–MM:SS] Tytuł – link.
Nie zmyślaj – jeśli brak dobrych źródeł, napisz wprost i zadaj 1–2 pytania doprecyzowujące.
"""

@app.post("/answer")
def answer(body: AnswerBody):
    hits = search(SearchBody(query=body.query, k=max(body.k,6)))
    # build short context
    ctx = []
    src = []
    for h in hits[:3]:
        ctx.append(f"[{h['span']}] {h['text']}")
    for h in hits:
        src.append(f"▶ [{h['span']}] {h['title']} – {h['url']}")

    profile_str = (f"\n\n[Profil użytkownika]: {body.profile}" if body.profile else "")
    user_prompt = f"Pytanie: {body.query}{profile_str}\n\nKontekst (fragmenty):\n" + "\n".join(ctx) + "\n\nZbuduj odpowiedź zgodnie z zasadami."

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.35,
        messages=[
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"user","content": user_prompt}
        ]
    )
    ans = resp.choices[0].message.content.strip()
    ans += "\n\n—\nŹródła:\n" + "\n".join(src)
    return {"answer": ans, "hits": hits}

@app.get("/healthz")
def healthz():
    return {"ok": True, "items": len(META)}
