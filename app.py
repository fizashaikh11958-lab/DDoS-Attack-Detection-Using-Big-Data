"""
DDoS Attack Detection Using Big Data
Simple student project: Streamlit + pandas (chunk processing) + scikit-learn.

Run locally:  streamlit run app.py
Host on Render: see README.md
"""
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ---- Project credit (edit these 3 lines) ----
STUDENT_NAME = "Fiza Shaikh"
ROLL_NO = "19048"
CLASS_NAME = "TYBSc.IT"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, "traffic_flows.csv")
ATTACK_FILE = os.path.join(DATA_DIR, "ddos_attacks_detected.csv")

LABELS = {0: "Normal", 1: "SYN Flood", 2: "UDP Flood", 3: "HTTP Flood", 4: "ICMP Flood"}
FEATURES = ["protocol", "dst_port", "duration_ms", "fwd_packets", "bwd_packets",
            "fwd_bytes", "bwd_bytes", "pkts_per_sec", "bytes_per_sec",
            "avg_pkt_size", "syn_count"]

# Fixed IP pools (normal users and a botnet)
_r = np.random.default_rng(99)
NORMAL_IPS = np.array([f"172.16.{i // 250}.{i % 250 + 1}" for i in range(3000)])
BOTNET_IPS = np.array([f"{_r.integers(11, 223)}.{_r.integers(0, 255)}."
                       f"{_r.integers(0, 255)}.{_r.integers(1, 255)}" for _ in range(500)])
_w = 1.0 / (np.arange(500) + 5)
BOT_WEIGHTS = _w / _w.sum()  # some bots send much more than others


# ---------------------------------------------------------------------------
# 1. BIG DATA: create a large traffic-flow dataset (written chunk by chunk)
# ---------------------------------------------------------------------------
def gen_flows(n, attack_pct, seed):
    """Create n network flows: normal traffic + 4 kinds of DDoS attacks."""
    rng = np.random.default_rng(seed)
    label = np.where(rng.random(n) < attack_pct / 100, rng.integers(1, 5, n), 0)

    protocol = np.zeros(n, dtype=int)
    port = np.zeros(n, dtype=int)
    dur = np.zeros(n)
    fwd = np.zeros(n)
    bwd = np.zeros(n)
    fsz = np.zeros(n)
    bsz = np.zeros(n)
    syn = np.zeros(n)

    # Normal traffic
    m = label == 0
    k = m.sum()
    protocol[m] = rng.choice([6, 17, 1], k, p=[0.72, 0.24, 0.04])
    port[m] = rng.choice([80, 443, 53, 22, 8080, 25], k)
    dur[m] = rng.exponential(2500, k) + 20
    fwd[m] = rng.poisson(14, k) + 1
    bwd[m] = rng.poisson(12, k)
    fsz[m] = rng.normal(400, 150, k).clip(60, 1500)
    bsz[m] = rng.normal(600, 250, k).clip(40, 1500)
    syn[m] = rng.integers(0, 3, k)
    heavy = m & (rng.random(n) < 0.06)  # big downloads / video (look busy but are normal)
    kh = heavy.sum()
    dur[heavy] = rng.uniform(5000, 30000, kh)
    fwd[heavy] = rng.poisson(250, kh) + 1
    bwd[heavy] = rng.poisson(450, kh) + 1
    bsz[heavy] = rng.normal(1300, 100, kh).clip(500, 1500)

    # SYN flood: many tiny SYN packets, almost no replies
    m = label == 1
    k = m.sum()
    protocol[m] = 6
    port[m] = rng.choice([80, 443], k)
    dur[m] = rng.uniform(1, 80, k)
    fwd[m] = rng.integers(60, 500, k)
    bwd[m] = rng.integers(0, 3, k)
    fsz[m] = rng.normal(60, 5, k).clip(40, 80)
    bsz[m] = 60
    syn[m] = fwd[m] * rng.uniform(0.85, 1.0, k)

    # UDP flood: huge amount of big UDP packets
    m = label == 2
    k = m.sum()
    protocol[m] = 17
    port[m] = rng.integers(1024, 65535, k)
    dur[m] = rng.uniform(1, 300, k)
    fwd[m] = rng.integers(150, 1000, k)
    bwd[m] = rng.integers(0, 3, k)
    fsz[m] = rng.normal(1100, 200, k).clip(500, 1500)
    bsz[m] = 60

    # HTTP flood: many web requests to port 80/443
    m = label == 3
    k = m.sum()
    protocol[m] = 6
    port[m] = rng.choice([80, 443], k)
    dur[m] = rng.uniform(800, 6000, k)
    fwd[m] = rng.integers(80, 300, k)
    bwd[m] = rng.integers(60, 250, k)
    fsz[m] = rng.normal(500, 120, k).clip(200, 900)
    bsz[m] = rng.normal(650, 150, k).clip(300, 1000)
    syn[m] = rng.integers(1, 5, k)

    # ICMP (ping) flood
    m = label == 4
    k = m.sum()
    protocol[m] = 1
    port[m] = 0
    dur[m] = rng.uniform(1, 200, k)
    fwd[m] = rng.integers(100, 600, k)
    bwd[m] = rng.integers(0, 5, k)
    fsz[m] = rng.normal(84, 4, k)
    bsz[m] = 84

    # Make the data a bit messy (like real life) so the model is not perfect:
    # 1) flash crowds: normal web users that look like an HTTP flood
    crowd = (label == 0) & (rng.random(n) < 0.04)
    kc = crowd.sum()
    protocol[crowd] = 6
    port[crowd] = rng.choice([80, 443], kc)
    dur[crowd] = rng.uniform(800, 6000, kc)
    fwd[crowd] = rng.integers(40, 200, kc)
    bwd[crowd] = rng.integers(40, 200, kc)
    fsz[crowd] = rng.normal(450, 120, kc).clip(200, 900)
    bsz[crowd] = rng.normal(650, 150, kc).clip(300, 1000)
    syn[crowd] = rng.integers(1, 5, kc)
    # 2) weak (low-rate) attacks that hide among normal traffic
    weak = (label > 0) & (rng.random(n) < 0.08)
    kw = weak.sum()
    dur[weak] = rng.uniform(1000, 5000, kw)
    fwd[weak] = rng.poisson(25, kw) + 1
    bwd[weak] = rng.poisson(15, kw)
    syn[weak] = rng.integers(0, 4, kw)

    fbytes = fwd * fsz
    bbytes = bwd * bsz
    pk = fwd + bwd
    secs = dur / 1000 + 0.001

    # Source IPs and timestamps (attacks come in 3 bursts)
    bot = rng.choice(500, n, p=BOT_WEIGHTS)
    src_ip = np.where(label == 0, NORMAL_IPS[rng.integers(0, 3000, n)], BOTNET_IPS[bot])
    starts = np.array([720, 1800, 2880])
    lens = np.array([300, 240, 300])
    w = rng.integers(0, 3, n)
    off_attack = starts[w] + (rng.random(n) * lens[w]).astype(int)
    offset = np.where(label == 0, rng.integers(0, 3600, n), off_attack)
    ts = np.datetime64("2026-10-29T10:00:00") + offset.astype("timedelta64[s]")

    return pd.DataFrame({
        "timestamp": np.datetime_as_string(ts, unit="s"),
        "src_ip": src_ip,
        "protocol": protocol,
        "dst_port": port,
        "duration_ms": dur.round(1),
        "fwd_packets": fwd.astype(int),
        "bwd_packets": bwd.astype(int),
        "fwd_bytes": fbytes.astype(int),
        "bwd_bytes": bbytes.astype(int),
        "pkts_per_sec": (pk / secs).round(1),
        "bytes_per_sec": ((fbytes + bbytes) / secs).round(1),
        "avg_pkt_size": ((fbytes + bbytes) / np.maximum(pk, 1)).round(1),
        "syn_count": syn.astype(int),
        "label": label,
    })


def make_dataset(total, attack_pct, chunk=50_000, progress=None):
    """Write the big CSV in chunks so memory stays small."""
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    done, i = 0, 0
    while done < total:
        n = min(chunk, total - done)
        gen_flows(n, attack_pct, seed=1000 + i).to_csv(
            DATA_FILE, mode="a", header=(i == 0), index=False)
        done += n
        i += 1
        if progress:
            progress.progress(done / total, text=f"Created {done:,} of {total:,} flows")


def count_rows(path):
    with open(path, "rb") as f:
        lines = sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))
    return max(lines - 1, 0)


# ---------------------------------------------------------------------------
# 2. MACHINE LEARNING: Random Forest classifier
# ---------------------------------------------------------------------------
@st.cache_resource
def train_model():
    df = gen_flows(30_000, 35, seed=7)
    X, y = df[FEATURES], df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    model = RandomForestClassifier(n_estimators=40, max_depth=14, random_state=1, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rep = classification_report(y_te, pred, target_names=list(LABELS.values()), output_dict=True)
    rep_df = pd.DataFrame({k: v for k, v in rep.items() if isinstance(v, dict)}).T.round(3)
    cm = pd.DataFrame(confusion_matrix(y_te, pred),
                      index=[f"Actual {v}" for v in LABELS.values()],
                      columns=[f"Pred {v}" for v in LABELS.values()])
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return {"model": model, "accuracy": accuracy_score(y_te, pred),
            "report": rep_df, "cm": cm, "importance": imp,
            "train_rows": len(X_tr), "test_rows": len(X_te)}


# ---------------------------------------------------------------------------
# 3. DETECTION: read the big file in chunks (map) and combine results (reduce)
# ---------------------------------------------------------------------------
def run_detection(path, model, chunk_size, total_rows, progress=None):
    start = time.time()
    if os.path.exists(ATTACK_FILE):
        os.remove(ATTACK_FILE)

    total = correct = labelled = 0
    type_counts = Counter()
    ip_type = Counter()
    minute_counts = {}
    first = True

    for chunk in pd.read_csv(path, chunksize=chunk_size):          # MAP step
        if "src_ip" not in chunk.columns:
            chunk["src_ip"] = "unknown"
        if "timestamp" not in chunk.columns:
            chunk["timestamp"] = "2026-01-01T00:00:00"

        pred = model.predict(chunk[FEATURES])
        chunk["attack_type"] = pd.Series(pred, index=chunk.index).map(LABELS)
        chunk["is_attack"] = pred > 0
        total += len(chunk)
        type_counts.update(chunk["attack_type"].value_counts().to_dict())

        if "label" in chunk.columns:
            correct += int((pred == chunk["label"].values).sum())
            labelled += len(chunk)

        chunk["minute"] = chunk["timestamp"].astype(str).str[:16]
        for (mn, atk), cnt in chunk.groupby(["minute", "is_attack"]).size().items():
            minute_counts.setdefault(mn, [0, 0])[int(atk)] += int(cnt)

        atk_rows = chunk[chunk["is_attack"]]
        if len(atk_rows):
            for (ip, t), cnt in atk_rows.groupby(["src_ip", "attack_type"]).size().items():
                ip_type[(ip, t)] += int(cnt)
            atk_rows[["timestamp", "src_ip", "protocol", "dst_port", "fwd_packets",
                      "pkts_per_sec", "attack_type"]].to_csv(
                ATTACK_FILE, mode="a", header=first, index=False)
            first = False

        if progress:
            progress.progress(min(total / max(total_rows, 1), 1.0),
                              text=f"Processed {total:,} flows...")

    # REDUCE step: combine the small summaries
    mdf = pd.DataFrame([(m, v[0], v[1]) for m, v in sorted(minute_counts.items())],
                       columns=["minute", "Normal", "Attack"])
    mdf["Total"] = mdf["Normal"] + mdf["Attack"]
    mdf["Attack %"] = (mdf["Attack"] / mdf["Total"] * 100).round(1)

    top = pd.DataFrame()
    if ip_type:
        ip_df = pd.DataFrame([(ip, t, c) for (ip, t), c in ip_type.items()],
                             columns=["Source IP", "Attack type", "Flows"])
        tot = ip_df.groupby("Source IP")["Flows"].sum()
        main = (ip_df.sort_values("Flows", ascending=False)
                .drop_duplicates("Source IP").set_index("Source IP")["Attack type"])
        top = pd.DataFrame({"Attack flows": tot, "Main attack type": main})
        top = top.sort_values("Attack flows", ascending=False).head(10)
        top.index.name = "Source IP"
        top = top.reset_index()

    attacks = sum(v for k, v in type_counts.items() if k != "Normal")
    secs = time.time() - start
    return {"total": total, "attacks": int(attacks), "types": dict(type_counts),
            "minutes": mdf, "top_ips": top, "seconds": secs,
            "speed": int(total / max(secs, 0.001)),
            "accuracy": (correct / labelled) if labelled else None}


def attack_windows(mdf, threshold):
    """Group consecutive minutes with a high attack share into attack windows."""
    bad = mdf[(mdf["Attack %"] >= threshold) & (mdf["Total"] >= 50)].copy()
    if bad.empty:
        return pd.DataFrame()
    bad["t"] = pd.to_datetime(bad["minute"])
    wins, cur = [], None
    for _, r in bad.iterrows():
        if cur and (r["t"] - cur["end"]) == pd.Timedelta(minutes=1):
            cur["end"] = r["t"]
            cur["flows"] += int(r["Attack"])
            cur["peak"] = max(cur["peak"], r["Attack %"])
        else:
            if cur:
                wins.append(cur)
            cur = {"start": r["t"], "end": r["t"], "flows": int(r["Attack"]), "peak": r["Attack %"]}
    wins.append(cur)
    return pd.DataFrame([{
        "Start": w["start"].strftime("%H:%M"),
        "End": (w["end"] + pd.Timedelta(minutes=1)).strftime("%H:%M"),
        "Minutes": int((w["end"] - w["start"]) / pd.Timedelta(minutes=1)) + 1,
        "Attack flows": w["flows"],
        "Peak attack %": w["peak"],
    } for w in wins])


# ===== UI starts here =====================================================
st.set_page_config(page_title="DDoS Detection", page_icon="🛡️", layout="wide")

# ---- Loading screen (once) ----
if "loaded" not in st.session_state:
    splash = st.empty()
    with splash.container():
        st.markdown(
            f"""
            <div style="text-align:center; padding:70px 10px 10px 10px;">
              <div style="font-size:70px;">🛡️</div>
              <h1 style="margin-bottom:0;">DDoS Attack Detection</h1>
              <p style="font-size:20px; opacity:0.7;">Using Big Data</p>
              <hr style="width:50%; margin:25px auto;">
              <p style="font-size:18px; margin-bottom:4px;">This project is made By</p>
              <h2 style="margin:0;">{STUDENT_NAME}</h2>
              <p style="font-size:18px;">Roll No: {ROLL_NO} &nbsp;|&nbsp; {CLASS_NAME}</p>
            </div>""",
            unsafe_allow_html=True)
        bar = st.progress(0)
        msg = st.empty()
        for i, step in enumerate(["Starting Big Data engine...", "Loading machine learning model...",
                                  "Preparing dashboard...", "Ready!"]):
            msg.markdown(f"<p style='text-align:center'>{step}</p>", unsafe_allow_html=True)
            bar.progress((i + 1) * 25)
            time.sleep(0.7)
    splash.empty()
    st.session_state.loaded = True

# ---- Sidebar ----
st.sidebar.title("🛡️ DDoS Detector")
st.sidebar.caption(f"Made by {STUDENT_NAME} | Roll No {ROLL_NO} | {CLASS_NAME}")
chunk_size = st.sidebar.selectbox("Chunk size (rows read at a time)",
                                  [10_000, 50_000, 100_000], index=1)
threshold = st.sidebar.slider("Alert if attack traffic in a minute is above (%)", 10, 90, 40)

st.title("🛡️ DDoS Attack Detection Using Big Data")
st.caption("Reads a big traffic file in small chunks, detects DDoS attacks with machine learning, and raises alerts.")

tab1, tab2, tab3, tab4 = st.tabs(["📂 1. Big Data", "🤖 2. Model", "🔍 3. Detect DDoS", "ℹ️ How it works"])

# ---- Tab 1: data ----
with tab1:
    st.subheader("Step 1: Get the traffic data")
    mode = st.radio("Data source", ["Generate sample big data", "Upload my own CSV"], horizontal=True)

    if mode.startswith("Generate"):
        c1, c2 = st.columns(2)
        total_rows = c1.slider("Number of traffic flows", 50_000, 500_000, 200_000, step=50_000)
        attack_pct = c2.slider("Attack traffic (%)", 5, 50, 20)
        if st.button("⚙️ Generate big dataset", type="primary"):
            bar = st.progress(0, text="Starting...")
            make_dataset(total_rows, attack_pct, progress=bar)
            st.session_state.rows = count_rows(DATA_FILE)
            st.session_state.pop("result", None)
            bar.empty()
            st.success("Dataset created!")
    else:
        st.caption("Required columns: " + ", ".join(FEATURES) +
                   " (optional: timestamp, src_ip, label)")
        up = st.file_uploader("Upload traffic CSV", type="csv")
        if up is not None and st.button("Use this file", type="primary"):
            with open(DATA_FILE, "wb") as f:
                f.write(up.getbuffer())
            head = pd.read_csv(DATA_FILE, nrows=5)
            missing = [c for c in FEATURES if c not in head.columns]
            if missing:
                os.remove(DATA_FILE)
                st.error("Missing columns: " + ", ".join(missing))
            else:
                st.session_state.rows = count_rows(DATA_FILE)
                st.session_state.pop("result", None)
                st.success("File loaded!")

    if os.path.exists(DATA_FILE):
        size_mb = os.path.getsize(DATA_FILE) / 1_000_000
        m1, m2 = st.columns(2)
        m1.metric("Flows (rows) in dataset", f"{st.session_state.get('rows', count_rows(DATA_FILE)):,}")
        m2.metric("File size", f"{size_mb:.1f} MB")
        st.write("**Preview (first 10 rows)**")
        st.dataframe(pd.read_csv(DATA_FILE, nrows=10), hide_index=True)
    else:
        st.info("No dataset yet. Generate or upload one.")

# ---- Tab 2: model ----
with tab2:
    st.subheader("Step 2: Machine learning model")
    with st.spinner("Training Random Forest model (first time only)..."):
        info = train_model()
    a, b, c = st.columns(3)
    a.metric("Model accuracy", f"{info['accuracy'] * 100:.2f}%")
    b.metric("Training flows", f"{info['train_rows']:,}")
    c.metric("Testing flows", f"{info['test_rows']:,}")
    st.write("**Classes the model can detect:** " + ", ".join(LABELS.values()))
    st.write("**Precision / Recall / F1 for each class**")
    st.dataframe(info["report"])
    st.write("**Confusion matrix**")
    st.dataframe(info["cm"])
    st.write("**Which features matter most?**")
    st.bar_chart(info["importance"])

# ---- Tab 3: detection ----
with tab3:
    st.subheader("Step 3: Detect DDoS attacks")
    has_data = os.path.exists(DATA_FILE)
    if not has_data:
        st.info("Create or upload a dataset in the first tab.")
    if st.button("▶ Start DDoS detection", type="primary", disabled=not has_data):
        info = train_model()
        total_rows = st.session_state.get("rows") or count_rows(DATA_FILE)
        bar = st.progress(0, text="Reading data in chunks...")
        st.session_state.result = run_detection(DATA_FILE, info["model"], chunk_size, total_rows, bar)
        bar.empty()

    res = st.session_state.get("result")
    if res:
        wins = attack_windows(res["minutes"], threshold)
        pct = res["attacks"] / max(res["total"], 1) * 100
        if len(wins):
            st.error(f"🚨 DDoS ATTACK DETECTED! {len(wins)} attack window(s) found. "
                     f"Peak attack traffic: {wins['Peak attack %'].max()}% of flows in a minute.")
        elif res["attacks"]:
            st.warning("⚠️ Some suspicious flows found, but no sustained DDoS attack window.")
        else:
            st.success("✅ No DDoS attack detected. Traffic looks normal.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Flows analysed", f"{res['total']:,}")
        c2.metric("Attack flows", f"{res['attacks']:,}", f"{pct:.1f}%", delta_color="off")
        c3.metric("Time taken", f"{res['seconds']:.1f} s")
        c4.metric("Speed", f"{res['speed']:,} flows/s")
        if res["accuracy"] is not None:
            st.caption(f"Detection accuracy on this file (labels available): {res['accuracy'] * 100:.2f}%")

        st.write("**Traffic per minute (attack bursts show up as spikes)**")
        st.line_chart(res["minutes"].set_index("minute")[["Normal", "Attack"]])

        l, r = st.columns(2)
        with l:
            st.write("**Attack types detected**")
            types = pd.Series({k: v for k, v in res["types"].items() if k != "Normal"})
            if len(types):
                st.bar_chart(types)
            else:
                st.write("None")
        with r:
            st.write("**Top attacker IPs (botnet)**")
            if len(res["top_ips"]):
                st.dataframe(res["top_ips"], hide_index=True)
            else:
                st.write("None")

        st.write("**🚨 Attack windows**")
        if len(wins):
            st.dataframe(wins, hide_index=True)
        else:
            st.write("No attack windows above the alert threshold.")

        if os.path.exists(ATTACK_FILE):
            st.write("**Detected attack flows (first 200)**")
            st.dataframe(pd.read_csv(ATTACK_FILE, nrows=200), hide_index=True)
            with open(ATTACK_FILE, "rb") as f:
                st.download_button("⬇️ Download all detected attacks (CSV)", f.read(),
                                   "ddos_attacks_detected.csv", "text/csv")

# ---- Tab 4: explanation ----
with tab4:
    st.subheader("How this project works")
    st.markdown("""
**Why Big Data?**  
Real networks create millions of traffic records. A computer cannot load all of them into memory at once,
so this project reads the file in **small chunks** and processes one chunk at a time
(the same idea as Hadoop / Spark **MapReduce**).

```
Big CSV file  ->  read 50,000 rows  ->  ML model finds attacks  ->  small summary
                  read next 50,000  ->  ML model finds attacks  ->  small summary
                  ...                                              |
                                                       combine all summaries -> dashboard + alerts
```

1. **Map:** each chunk is classified as *Normal, SYN Flood, UDP Flood, HTTP Flood or ICMP Flood*.
2. **Reduce:** the small results (counts per minute, attacker IPs, attack types) are added together.
3. **Alert:** if the share of attack traffic in a minute is above the threshold, a DDoS alert is raised.

**Features used by the model:** protocol, destination port, duration, packets, bytes, packets/sec,
bytes/sec, average packet size and SYN count.

| Attack | How it looks in the data |
|---|---|
| SYN Flood | Many tiny SYN packets, almost no replies |
| UDP Flood | Huge number of large UDP packets |
| HTTP Flood | Many web requests to port 80/443 |
| ICMP Flood | Huge number of ping packets |

**Technology:** Python, Streamlit, pandas (chunk processing), scikit-learn (Random Forest). Hosted on Render.

**Future scope:** Apache Spark / Kafka for real-time streaming, real datasets such as CIC-DDoS2019,
and automatic blocking of attacker IPs.
""")

st.divider()
st.caption(f"This project is made By {STUDENT_NAME} | Roll No {ROLL_NO} | {CLASS_NAME}")
