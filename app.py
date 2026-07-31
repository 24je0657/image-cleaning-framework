import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
from PIL import Image
import shutil

# ---- PAGE CONFIG -------------------------------
st.set_page_config(
    page_title = "Image Cleaning Framework",
    page_icon  = "🧹",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ---- CSS ---------------------
st.markdown("""
<style>
.perf-box {
    background: linear-gradient(135deg,#EEEDFE 0%,#f8f9ff 100%);
    border-radius:12px; padding:18px;
    text-align:center; border-left:4px solid #534AB7;
}
.perf-num   { font-size:28px; font-weight:700; color:#3C3489; }
.perf-label { font-size:12px; color:#5F5E5A; margin-top:4px; }
.remove-tag { background:#FAECE7; color:#712B13; padding:2px 10px;
              border-radius:12px; font-size:12px; font-weight:600; }
.review-tag { background:#FFF8E1; color:#7B5800; padding:2px 10px;
              border-radius:12px; font-size:12px; font-weight:600; }
.clean-tag  { background:#EAF3DE; color:#27500A; padding:2px 10px;
              border-radius:12px; font-size:12px; font-weight:600; }
.section-header { font-size:20px; font-weight:600; color:#2C2C2A;
                  margin-bottom:16px; padding-bottom:8px;
                  border-bottom:2px solid #EEEDFE; }
.pipeline-step  { text-align:center; padding:7px 12px;
                  background:#f8f9ff; border-radius:8px;
                  margin:3px 0; font-size:13px; }
.tech-badge     { display:inline-block; background:#EEEDFE; color:#3C3489;
                  padding:4px 12px; border-radius:20px;
                  font-size:12px; font-weight:500; margin:3px; }
.flag-item      { padding:3px 0; font-size:13px; }
.inspector-card { background:#f8f9fa; border-radius:12px; padding:16px;
                  border:1px solid #E0DED8; }
</style>
""", unsafe_allow_html=True)


# ------ HELPERS ----------------------------------

@st.cache_data
def load_master(split):
    p = f"reports/{split}_master_report.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_duplicates(split):
    p = f"reports/{split}_duplicates_report.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

def load_image(path, size=(200, 200)):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail(size)
        return img
    except:
        return None

def verdict_color(val):
    if "REMOVE" in str(val): return "background-color:#FAECE7;color:#712B13"
    if "REVIEW" in str(val): return "background-color:#FFF8E1;color:#7B5800"
    if str(val) == "CLEAN":  return "background-color:#EAF3DE;color:#27500A"
    return ""

CLASSES = ["cat", "dog", "elephant", "horse", "lion"]


# -------- SIDEBAR ------------------------------
with st.sidebar:

    if os.path.exists("assets/architecture.png"):
        st.image("assets/architecture.png", width="stretch")

    st.markdown("## 🧹 Image Cleaning Framework")
    st.caption("Deep Learning Dataset Cleaning Pipeline")

    st.markdown("---")

    # IMPORTANT: Keep this
    split = st.selectbox(
        "📂 Dataset Split",
        ["train", "val"]
    )

    st.markdown("### 📂 Dataset")

    st.info("""
**Animal Dataset**

• Classes: 5

• Train: 13,474

• Validation: 1,497

• Total: 14,971
""")

    st.markdown("### 🤖 Models")

    st.markdown("""
- **ResNet50**
- **Perceptual Hash (pHash)**
- **Cosine Similarity**
- **Convolutional Autoencoder**
- **PCA + Isolation Forest**
- **K-Means + KNN + Centroid Distance**
""")

    st.markdown("### ⚙️ Framework Components")

    st.success("Embedding Extraction")
    st.success("Duplicate Detection")
    st.success("Blur Detection")
    st.success("Noise Detection")
    st.success("Outlier Detection")
    st.success("Mislabel Detection")
    st.success("Decision Engine")


# ─── LOAD DATA ────────────────────────────────────────────────────────────────
master = load_master(split)
dup_df = load_duplicates(split)

if master is None:
    st.error("Run decision_engine.py first to generate master report.")
    st.stop()

total  = len(master)
clean  = (master["verdict"] == "CLEAN").sum()
remove = master["verdict"].str.startswith("REMOVE").sum()
review = master["verdict"].str.startswith("REVIEW").sum()

def safe_sum(col):
    return int(master[col].astype(bool).sum()) if col in master.columns else 0


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.title("🧹 Intelligent Image Data Cleaning Framework")
st.markdown(
    """A complete **deep learning-based image dataset cleaning pipeline**
    that automatically detects duplicates, blur, noise, outliers,
    and mislabels before generating a final recommendation using
    a weighted **Decision Engine**."""
)
st.markdown("---")


# ─── FRAMEWORK PERFORMANCE BAR (Improvement #10) ─────────────────────────────
p1, p2, p3, p4, p5, p6 = st.columns(6)
perf_items = [
    ("✔ Images Processed", f"{total:,}"),
    ("✔ Detection Modules", "6"),
    ("✔ Reports Generated", "12"),
    ("✔ Issues Found",      f"{remove + review:,}"),
    ("✔ Clean Images",      f"{clean:,}"),
    ("✔ Export Ready",      "YES"),
]
for col, (label, val) in zip([p1,p2,p3,p4,p5,p6], perf_items):
    col.markdown(f"""
    <div class="perf-box">
        <div class="perf-num">{val}</div>
        <div class="perf-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ─── TABS ─────────────────────────────────────────────────────────────────────
(tab_home, tab_overview, tab_compare, tab_dups,
 tab_quality, tab_outliers, tab_inspector,
 tab_analytics, tab_export) = st.tabs([
    "🏠 Home",
    "📊 Overview",
    "📈 Before vs After",
    "🔁 Duplicates",
    "🌫 Quality",
    "🔍 Outliers & Mislabels",
    "🔎 Image Inspector",
    "📈 Analytics",
    "📤 Export"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HOME  (Improvement #1)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_home:
    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.markdown("### 🎯 Project Overview")
        st.markdown("""
        This framework provides an **automated, modular pipeline** to clean
        large-scale image datasets before deep learning training.

        Real-world datasets contain: duplicate images that bias model training,
        blurry images that degrade feature learning, mislabeled samples that
        corrupt class boundaries, and irrelevant outliers. This tool detects
        and removes all of them automatically using deep learning.
        """)

        st.markdown("### 🛠 Tech Stack")
        techs = [
            "PyTorch", "ResNet50", "OpenCV", "Scikit-learn",
            "Streamlit", "NumPy", "Pandas", "Isolation Forest",
            "KMeans", "Cosine Similarity", "Autoencoder", "pHash"
        ]
        st.markdown(
            " ".join(f'<span class="tech-badge">{t}</span>' for t in techs),
            unsafe_allow_html=True
        )

        st.markdown("### 📂 Dataset")
        st.dataframe(pd.DataFrame({
            "Split"  : ["Train", "Val", "Inf"],
            "Classes": ["5", "5", "5"],
            "Images" : ["13,474", "1,497", "5"],
            "Purpose": ["Main cleaning", "Validation", "Demo"]
        }), hide_index=True, width="stretch")

        if os.path.exists("assets/project_pipeline.png"):
            st.markdown("### 🏗 Full Pipeline Diagram")
            st.image("assets/project_pipeline.png",
                     width="content")

    with col_right:
        st.markdown("### 🔄 Pipeline Workflow")
        steps = [
            ("📁", "Raw images"),
            ("✅", "Format validation"),
            ("🧠", "ResNet50 embeddings"),
            ("🔁", "Duplicate detection"),
            ("🌫", "Blur detection"),
            ("📡", "Noise detection"),
            ("🎯", "Outlier detection"),
            ("🏷", "Mislabel detection"),
            ("⚖️", "Decision engine"),
            ("✅", "Clean dataset"),
        ]
        for i, (icon, label) in enumerate(steps):
            st.markdown(
                f'<div class="pipeline-step">'
                f'{icon} <b>{label}</b>'
                f'</div>',
                unsafe_allow_html=True
            )
            if i < len(steps) - 1:
                st.markdown(
                    "<div style='text-align:center;color:#B4B2A9;"
                    "font-size:16px;margin:0'>↓</div>",
                    unsafe_allow_html=True
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OVERVIEW  (Improvements #2 #6 #7)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown('<p class="section-header">Decision Engine Dashboard</p>',
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total images", f"{total:,}")
    m2.metric("✅ Clean",     f"{clean:,}",
              f"{100*clean/total:.1f}%")
    m3.metric("❌ Remove",    f"{remove:,}",
              f"-{100*remove/total:.1f}%", delta_color="inverse")
    m4.metric("👁 Review",    f"{review:,}",
              f"{100*review/total:.1f}%", delta_color="off")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Decision distribution")
        vc = master["verdict"].value_counts()
        color_map = {
            "CLEAN"                    : "#3B6D11",
            "REVIEW — low priority"    : "#F9A825",
            "REVIEW — medium priority" : "#FB8C00",
            "REMOVE — blurry"          : "#E57373",
            "REMOVE — duplicate"       : "#EF5350",
            "REMOVE — mislabeled"      : "#C62828",
            "REMOVE — high priority"   : "#B71C1C",
        }
        fig_pie = px.pie(
            values=vc.values, names=vc.index,
            hole=0.5, color=vc.index,
            color_discrete_map=color_map
        )
        fig_pie.update_layout(height=320,
                              margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_pie, width="stretch")

    with col_r:
        st.markdown("#### Issue breakdown")
        issue_df = pd.DataFrame({
            "Issue": ["Exact dupes","Near dupes","Blurry",
                      "Noisy","Outlier","Mislabeled"],
            "Count": [
                safe_sum("Exact_duplicates"),
                safe_sum("near_duplicate"),
                safe_sum("is_blurry"),
                safe_sum("is_noisy"),
                safe_sum("is_outlier"),
                safe_sum("is_mislabeled"),
            ],
        })
        fig_bar = px.bar(
            issue_df, x="Count", y="Issue",
            orientation="h",
            color="Issue",
            color_discrete_sequence=[
                "#EF5350","#E57373","#FB8C00",
                "#F9A825","#534AB7","#0F6E56"
            ],
            text="Count"
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            showlegend=False, height=320,
            margin=dict(l=0,r=30,t=10,b=0),
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_bar, width="stretch")

    st.markdown("---")

    # ── Class statistics with progress bars (Improvement #6)
    st.markdown("#### Class statistics")
    for cls in CLASSES:
        cdf       = master[master["class"] == cls]
        ctotal    = len(cdf)
        cclean    = (cdf["verdict"] == "CLEAN").sum()
        cremove   = cdf["verdict"].str.startswith("REMOVE").sum()
        creview   = cdf["verdict"].str.startswith("REVIEW").sum()

        ca, cb, cc, cd = st.columns([1, 2, 2, 2])
        ca.markdown(f"**{cls.title()}**  \n`{ctotal} total`")
        with cb:
            st.markdown(f"✅ Clean: **{cclean}**")
            st.progress(int(cclean) / ctotal)
        with cc:
            st.markdown(f"👁 Review: **{creview}**")
            st.progress(int(creview) / ctotal)
        with cd:
            st.markdown(f"❌ Remove: **{cremove}**")
            st.progress(int(cremove) / ctotal)
        st.markdown("")

    st.markdown("---")

    # ── Color-coded DataFrame (Improvement #7)
    st.markdown("#### Master report — top 100 rows (color-coded)")
    cols_show = ["file_path","class","verdict",
                 "priority_score","total_flags",
                 "is_blurry","is_noisy",
                 "is_outlier","is_mislabeled"]
    cols_show = [c for c in cols_show if c in master.columns]
    st.dataframe(
        master[cols_show].head(100)
                         .style.map(verdict_color,
                                         subset=["verdict"]),
        width="stretch",
        height=300
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BEFORE vs AFTER  (Improvement #3)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown('<p class="section-header">Before vs After Cleaning</p>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Dataset-level comparison")
        cmp_df = pd.DataFrame({
            "Metric"         : ["Total images","Clean","Removed",
                                "Review","Exact duplicates",
                                "Near duplicates","Blurry",
                                "Noisy","Outliers","Mislabeled"],
            "Original"       : [total,total,0,0,0,0,0,0,0,0],
            "After cleaning" : [
                total, int(clean), int(remove), int(review),
                safe_sum("Exact_duplicates"),
                safe_sum("near_duplicate"),
                safe_sum("is_blurry"),
                safe_sum("is_noisy"),
                safe_sum("is_outlier"),
                safe_sum("is_mislabeled"),
            ]
        })
        st.dataframe(cmp_df, hide_index=True, width="stretch")

    with c2:
        st.markdown("#### Visual comparison")
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name="Original",
            x=["Total","Clean","Remove","Review"],
            y=[total, total, 0, 0],
            marker_color="#B4B2A9"
        ))
        fig_cmp.add_trace(go.Bar(
            name="After Cleaning",
            x=["Total","Clean","Remove","Review"],
            y=[total, int(clean), int(remove), int(review)],
            marker_color=["#534AB7","#3B6D11","#C62828","#F9A825"]
        ))
        fig_cmp.update_layout(
            barmode="group", height=320,
            margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_cmp, width="stretch")

    st.markdown("---")
    st.markdown("#### Per-class before vs after")
    cls_rows = []
    for cls in CLASSES:
        cdf     = master[master["class"]==cls]
        cremove = int(cdf["verdict"].str.startswith("REMOVE").sum())
        cls_rows.append({
            "Class"      : cls.title(),
            "Original"   : len(cdf),
            "Clean"      : int((cdf["verdict"]=="CLEAN").sum()),
            "Removed"    : cremove,
            "Review"     : int(cdf["verdict"].str.startswith("REVIEW").sum()),
            "Reduction%" : f"{100*cremove/len(cdf):.1f}%"
        })
    cls_cmp = pd.DataFrame(cls_rows)
    st.dataframe(cls_cmp, hide_index=True, width="stretch")

    fig_cls = go.Figure()
    for col, color in [("Clean","#3B6D11"),
                       ("Review","#F9A825"),
                       ("Removed","#C62828")]:
        fig_cls.add_trace(go.Bar(
            name=col, x=cls_cmp["Class"],
            y=cls_cmp[col], marker_color=color
        ))
    fig_cls.update_layout(
        barmode="stack", height=300,
        margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig_cls, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DUPLICATES
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dups:
    st.markdown('<p class="section-header">Duplicate Detection</p>',
                unsafe_allow_html=True)

    if dup_df is None:
        st.warning("No duplicates report found.")
    else:
        exact_dups = dup_df[dup_df["Exact_duplicates"] == True]
        near_dups  = dup_df[dup_df["near_duplicate"]  == True]

        d1, d2, d3 = st.columns(3)
        d1.metric("Exact duplicates", len(exact_dups))
        d2.metric("Near duplicates",  len(near_dups))
        d3.metric("Total flagged",    len(exact_dups) + len(near_dups))

        st.markdown("---")
        sub1, sub2 = st.tabs(["Exact duplicates", "Near duplicates"])

        with sub1:
            st.markdown(f"**{len(exact_dups)} exact duplicates** "
                        f"via perceptual hashing")
            n_show = st.slider("Pairs to show", 1, 10, 4, key="ex")
            shown  = 0
            for _, row in exact_dups.iterrows():
                if shown >= n_show: break
                dup_of = row.get("duplicate_of", None)
                if pd.isna(dup_of) or not dup_of: continue
                img1 = load_image(row["file_path"])
                img2 = load_image(dup_of)
                if img1 and img2:
                    c1, c2, c3 = st.columns([1,1,2])
                    c1.image(img1, caption=Path(row["file_path"]).name,
                             width="content")
                    c2.image(img2, caption=Path(dup_of).name,
                             width="content")
                    c3.markdown(
                        f"**Class:** `{row.get('class','—')}`  \n"
                        f"**Status:** <span class='remove-tag'>REMOVE</span>  \n"
                        f"**Match:** identical pHash",
                        unsafe_allow_html=True
                    )
                    st.divider()
                    shown += 1

        with sub2:
            st.markdown(f"**{len(near_dups)} near-duplicates** "
                        f"(cosine similarity ≥ 0.97)")
            n_show2 = st.slider("Pairs to show", 1, 10, 4, key="nd")
            shown2  = 0
            for _, row in near_dups.iterrows():
                if shown2 >= n_show2: break
                near_of = row.get("near_duplicate_of", None)
                if pd.isna(near_of) or not near_of: continue
                img1 = load_image(row["file_path"])
                img2 = load_image(near_of)
                if img1 and img2:
                    c1, c2, c3 = st.columns([1,1,2])
                    c1.image(img1, caption=Path(row["file_path"]).name,
                             width="content")
                    c2.image(img2, caption=Path(near_of).name,
                             width="content")
                    c3.markdown(
                        f"**Class:** `{row.get('class','—')}`  \n"
                        f"**Cosine score:** `{row.get('cosine_score',0):.4f}`  \n"
                        f"**Status:** <span class='remove-tag'>REMOVE</span>",
                        unsafe_allow_html=True
                    )
                    st.divider()
                    shown2 += 1


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — QUALITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_quality:
    st.markdown('<p class="section-header">Image Quality</p>',
                unsafe_allow_html=True)

    # Filters (Improvement #4)
    fc1, fc2 = st.columns(2)
    cls_filter_q = fc1.selectbox("Filter by class",
                                  ["ALL"] + CLASSES, key="qcls")
    qual_view    = fc2.selectbox("View",
                                  ["Blurry images","Noisy images"],
                                  key="qview")
    qdf = master.copy()
    if cls_filter_q != "ALL":
        qdf = qdf[qdf["class"] == cls_filter_q]

    if qual_view == "Blurry images":
        blurry_df = qdf[qdf["is_blurry"]==True].sort_values("blur_score")
        st.markdown( f"**{len(blurry_df)} blurry images** "
                   "(Adaptive per-class threshold using 5th percentile)")

        fig_blur = px.histogram(
            qdf, x="blur_score", nbins=80,
            color_discrete_sequence=["#534AB7"],
            log_y=True, title="Blur score distribution (log scale)"
        )
        
        fig_blur.update_layout(height=250,
                               margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_blur, width="stretch")

        n_blur = st.slider("Images to show", 4, 24, 8, key="bl")
        cols   = st.columns(4)
        for i,(_, row) in enumerate(blurry_df.head(n_blur).iterrows()):
            img = load_image(row["file_path"], (180,180))
            if img:
                with cols[i%4]:
                    st.image(img, width="content")
                    st.caption(f"{row['class']} | "
                               f"score:{row['blur_score']:.1f}")

    else:
        if "is_noisy" not in master.columns:
            st.info("Noise report not found. "
                    "Run quality.py after autoencoder training.")
        else:
            noisy_df = qdf[qdf["is_noisy"]==True].sort_values(
                "reconstruction_error", ascending=False)
            st.markdown(f"**{len(noisy_df)} noisy images** "
                        f"(reconstruction error > mean + 2σ)")

            if "reconstruction_error" in master.columns:
                mean_err = master["reconstruction_error"].mean()
                std_err  = master["reconstruction_error"].std()
                fig_n    = px.histogram(
                    qdf, x="reconstruction_error", nbins=80,
                    color_discrete_sequence=["#0F6E56"],
                    title="Reconstruction error distribution"
                )
                fig_n.add_vline(x=mean_err+2*std_err,
                                line_dash="dash", line_color="red",
                                annotation_text="Threshold (mean+2σ)")
                fig_n.update_layout(height=250,
                                    margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig_n, width="stretch")

            n_noise = st.slider("Images to show", 4, 24, 8, key="ns")
            cols    = st.columns(4)
            for i,(_, row) in enumerate(noisy_df.head(n_noise).iterrows()):
                img = load_image(row["file_path"], (180,180))
                if img:
                    with cols[i%4]:
                        st.image(img, width="content")
                        st.caption(f"{row['class']} | "
                                   f"err:{row['reconstruction_error']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — OUTLIERS & MISLABELS  (Improvements #8 #9)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_outliers:
    st.markdown('<p class="section-header">Outliers & Mislabels</p>',
                unsafe_allow_html=True)

    o1, o2 = st.tabs(["🔍 Outliers", "🏷 Mislabels"])

    with o1:
        if "is_outlier" not in master.columns:
            st.info("Outlier report not found.")
        else:
            outlier_df = master[master["is_outlier"]==True]
            st.markdown(f"**{len(outlier_df)} outlier images** "
                        f"(Isolation Forest · PCA 128-dim)")

            # Scatter plot (Improvement #9)
            scatter = f"reports/{split}_outlier_scatter.png"
            if os.path.exists(scatter):
                st.markdown("#### PCA 2D projection — outliers in red")
                st.image(scatter, width="content")
            st.markdown("---")

            cls_out = outlier_df["class"].value_counts().reset_index()
            cls_out.columns = ["class","count"]
            fig_out = px.bar(
                cls_out, x="class", y="count", color="class",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title="Outliers per class"
            )
            fig_out.update_layout(showlegend=False, height=240,
                                  margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_out, width="stretch")

            n_out    = st.slider("Images to show", 4, 20, 8, key="ol")
            show_out = (outlier_df.nsmallest(n_out,"global_score")
                        if "global_score" in outlier_df.columns
                        else outlier_df.head(n_out))
            cols = st.columns(4)
            for i,(_, row) in enumerate(show_out.iterrows()):
                img = load_image(row["file_path"], (180,180))
                if img:
                    with cols[i%4]:
                        st.image(img, width="content")
                        sc = row.get("global_score","—")
                        st.caption(
                            f"{row['class']} | "
                            f"{f'{sc:.4f}' if isinstance(sc,float) else sc}"
                        )

    with o2:
        if "is_mislabeled" not in master.columns:
            st.info("Mislabel report not found.")
        else:
            mis_df = master[master["is_mislabeled"]==True].copy()
            n_high = (mis_df.get("mislabel_confidence","")=="HIGH").sum()
            n_med  = (mis_df.get("mislabel_confidence","")=="MEDIUM").sum()

            h1, h2, h3 = st.columns(3)
            h1.metric("Total mislabeled",    len(mis_df))
            h2.metric("🔴 HIGH confidence",  int(n_high))
            h3.metric("🟡 MEDIUM confidence",int(n_med))

            # Confidence gauges (Improvement #8)
            st.markdown("#### Confidence gauges")
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("🔴 **HIGH** — almost certainly mislabeled")
                st.progress(int(n_high) / max(len(mis_df),1))
                st.caption(f"{n_high} images · 3/3 methods agree")
            with g2:
                st.markdown("🟡 **MEDIUM** — likely mislabeled")
                st.progress(int(n_med) / max(len(mis_df),1))
                st.caption(f"{n_med} images · 2/3 methods agree")

            # Heatmap
            heatmap = f"reports/{split}_mislabel_heatmap.png"
            if os.path.exists(heatmap):
                st.markdown("#### Label mismatch heatmap")
                st.image(heatmap,
                         caption="Off-diagonal = potential mislabels",
                         width="content")

            conf_filter = st.selectbox(
                "Filter by confidence", ["ALL","HIGH","MEDIUM"],
                key="conf"
            )
            fdf = mis_df if conf_filter == "ALL" else mis_df[
                mis_df["mislabel_confidence"] == conf_filter
            ]

            n_mis = st.slider("Images to show", 4, 20, 8, key="ml")
            cols  = st.columns(4)
            for i,(_, row) in enumerate(fdf.head(n_mis).iterrows()):
                img = load_image(row["file_path"], (180,180))
                if img:
                    with cols[i%4]:
                        st.image(img, width="content")
                        pred = row.get("nn_predicted","?")
                        conf = row.get("mislabel_confidence","?")
                        icon = "🔴" if conf=="HIGH" else "🟡"
                        st.caption(
                            f"{row['class']} → {pred} {icon}"
                        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — IMAGE INSPECTOR  (Improvements #4 #5 #11 — standout feature)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_inspector:
    st.markdown('<p class="section-header">Image Inspector</p>',
                unsafe_allow_html=True)
    st.markdown("Search any image by filename, or browse with filters.")

    # Search bar (Improvement #5)
    search = st.text_input(
        "🔍 Search by filename",
        placeholder="e.g. lion2167.jpg",
        key="search"
    )

    # Browse filters (Improvement #4)
    fc1, fc2, fc3 = st.columns(3)
    cls_f  = fc1.selectbox("Class",   ["ALL"]+CLASSES, key="icls")
    verd_f = fc2.selectbox("Verdict", ["ALL","CLEAN","REMOVE","REVIEW"],
                            key="iverd")
    sort_f = fc3.selectbox("Sort by",
                            ["priority_score","blur_score","total_flags"],
                            key="isort")

    filtered = master.copy()
    if search:
        filtered = filtered[
            filtered["file_path"].str.contains(search, case=False, na=False)
        ]
    if cls_f != "ALL":
        filtered = filtered[filtered["class"]==cls_f]
    if verd_f != "ALL":
        filtered = filtered[filtered["verdict"].str.contains(verd_f)]
    if sort_f in filtered.columns:
        filtered = filtered.sort_values(sort_f, ascending=False)

    st.markdown(f"**{len(filtered)} images match**")
    st.markdown("---")

    if len(filtered) == 0:
        st.info("No images match.")

    elif search and len(filtered) <= 5:
        # ── Detailed inspection view (Improvement #11)
        for _, row in filtered.iterrows():
            img = load_image(row["file_path"], (300,300))
            col_img, col_info = st.columns([1, 2])

            with col_img:
                if img:
                    st.image(img,
                             caption=Path(row["file_path"]).name,
                             width="content")

            with col_info:
                verdict = row["verdict"]
                tag_cls = (
                    "remove-tag" if "REMOVE" in verdict else
                    "review-tag" if "REVIEW" in verdict else
                    "clean-tag"
                )
                st.markdown(
                    f"**Verdict:** "
                    f"<span class='{tag_cls}'>{verdict}</span>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**Class:** `{row['class']}`")
                st.markdown(
                    f"**Priority score:** `{row.get('priority_score',0)}`")
                st.markdown(
                    f"**Total flags:** `{row.get('total_flags',0)}`")

                st.markdown("#### 🔎 Issue breakdown")
                flags = {
                    "Exact duplicate" : (row.get("Exact_duplicates",False), "🔴"),
                    "Near duplicate"  : (row.get("near_duplicate", False), "🔴"),
                    "Blurry"          : (row.get("is_blurry",      False), "🟠"),
                    "Noisy"           : (row.get("is_noisy",       False), "🟡"),
                    "Outlier"         : (row.get("is_outlier",     False), "🟣"),
                    "Mislabeled"      : (row.get("is_mislabeled",  False), "🔵"),
                }
                any_flag = False
                for issue,(flagged,icon) in flags.items():
                    if flagged:
                        st.markdown(
                            f'<div class="flag-item">'
                            f'{icon} <b>{issue}</b> — detected</div>',
                            unsafe_allow_html=True
                        )
                        any_flag = True
                if not any_flag:
                    st.markdown("🟢 No issues detected")

                if row.get("blur_score"):
                    st.markdown(
                       f"**Blur score:** `{row['blur_score']:.2f}` "
                       f"(threshold: {row['threshold_used']:.2f})"
                     )
                    
                if row.get("reconstruction_error"):
                    st.markdown(
                        f"**Noise error:** `{row['reconstruction_error']:.4f}`")
                if row.get("mislabel_confidence"):
                    conf = row["mislabel_confidence"]
                    icon = ("🔴" if conf=="HIGH" else
                            "🟡" if conf=="MEDIUM" else "🟢")
                    st.markdown(
                        f"**Mislabel confidence:** {icon} `{conf}`")
                if (row.get("nn_predicted") and
                        row.get("nn_predicted") != row["class"]):
                    st.markdown(
                        f"**Predicted class:** `{row['nn_predicted']}`")

                # Reason for removal
                if "REMOVE" in verdict:
                    reasons = [k for k,(v,_) in flags.items() if v]
                    if reasons:
                        st.markdown("#### ❌ Removed because")
                        for r in reasons:
                            st.markdown(f"✔ {r}")

            st.divider()

    else:
        # ── Grid browse view
        n_browse = st.slider("Images to show", 8, 32, 12, key="ibr")
        cols = st.columns(4)
        for i,(_, row) in enumerate(filtered.head(n_browse).iterrows()):
            img = load_image(row["file_path"], (180,180))
            if img:
                verdict = row["verdict"]
                icon    = ("🔴" if "REMOVE" in verdict else
                           "🟡" if "REVIEW" in verdict else "✅")
                with cols[i%4]:
                    st.image(img, width="content")
                    st.caption(
                        f"{icon} {row['class']} | "
                        f"p={row.get('priority_score',0)}"
                    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown('<p class="section-header">Pipeline Analytics</p>',
                unsafe_allow_html=True)

    # ── Helper: safe mean
    def safe_mean(col):
        if col in master.columns:
            return round(float(master[col].dropna().mean()), 4)
        return None

    def safe_pct(col):
        if col in master.columns:
            return round(100 * master[col].astype(bool).sum() / total, 2)
        return 0.0


    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — MODULE EXECUTION SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### 🔧 Module Execution Summary")

    modules_info = [
        {
            "Module"       : "Preprocessing",
            "Method"       : "OpenCV · PIL",
            "Input"        : f"{total:,} images",
            "Flagged"      : "0",
            "Flag %"       : "0.00%",
            "Report"       : f"reports/{split}_validation_report.csv",
            "Status"       : "✅ Complete"
        },
        {
            "Module"       : "Embedding Extractor",
            "Method"       : "ResNet50 · 2048-dim",
            "Input"        : f"{total:,} images",
            "Flagged"      : f"{total:,}",
            "Flag %"       : "100.00%",
            "Report"       : f"embeddings/{split}_embeddings.npy",
            "Status"       : "✅ Complete"
        },
        {
            "Module"       : "Duplicate Detector",
            "Method"       : "pHash + cosine ≥ 0.97",
            "Input"        : f"{total:,} images",
            "Flagged"      : str(safe_sum("Exact_duplicates")
                                  + safe_sum("near_duplicate")),
            "Flag %"       : f"{safe_pct('Exact_duplicates') + safe_pct('near_duplicate'):.2f}%",
            "Report"       : f"reports/{split}_duplicates_report.csv",
            "Status"       : "✅ Complete"
        },
        {
            "Module"       : "Blur Detector",
            "Method": "Adaptive Laplacian variance (5th percentile per class)",
            "Input"        : f"{total:,} images",
            "Flagged"      : str(safe_sum("is_blurry")),
            "Flag %"       : f"{safe_pct('is_blurry'):.2f}%",
            "Report"       : f"reports/{split}_blur_report.csv",
            "Status"       : "✅ Complete"
        },
        {
            "Module"       : "Noise Detector",
            "Method"       : "Autoencoder · mean+2σ",
            "Input"        : f"{total:,} images",
            "Flagged"      : str(safe_sum("is_noisy")),
            "Flag %"       : f"{safe_pct('is_noisy'):.2f}%",
            "Report"       : f"reports/{split}_noise_report.csv",
            "Status"       : "✅ Complete"
        },
        {
            "Module"       : "Outlier Detector",
            "Method"       : "PCA 128-dim · Isolation Forest",
            "Input"        : f"{total:,} images",
            "Flagged"      : str(safe_sum("is_outlier")),
            "Flag %"       : f"{safe_pct('is_outlier'):.2f}%",
            "Report"       : f"reports/{split}_outliers_report.csv",
            "Status"       : "✅ Complete"
        },
        {
            "Module"       : "Mislabel Detector",
            "Method"       : "KMeans + KNN + Centroid (3-vote)",
            "Input"        : f"{total:,} images",
            "Flagged"      : str(safe_sum("is_mislabeled")),
            "Flag %"       : f"{safe_pct('is_mislabeled'):.2f}%",
            "Report"       : f"reports/{split}_mislabel_report.csv",
            "Status"       : "✅ Complete"
        },
        {
            "Module"       : "Decision Engine",
            "Method"       : "Weighted priority scoring",
            "Input"        : f"{total:,} images",
            "Flagged"      : str(remove),
            "Flag %"       : f"{100*remove/total:.2f}%",
            "Report"       : f"reports/{split}_master_report.csv",
            "Status"       : "✅ Complete"
        },
    ]

    mod_df = pd.DataFrame(modules_info)
    st.dataframe(
        mod_df.style.map(
            lambda v: "color:#27500A;font-weight:600"
            if v == "✅ Complete" else "",
            subset=["Status"]
        ),
        width="stretch",
        hide_index=True,
        height=320
    )


    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — CLEANING EFFICIENCY
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧹 Cleaning Efficiency")

    eff_cols = st.columns(4)
    efficiency  = round(100 * clean / total, 2)
    noise_rate  = safe_pct("is_noisy")
    outlier_pct = safe_pct("is_outlier")
    mislabel_pct= safe_pct("is_mislabeled")

    eff_cols[0].metric("Cleaning efficiency",
                       f"{efficiency}%",
                       help="% of images passing all checks")
    eff_cols[1].metric("Issues found",
                       f"{remove + review:,}",
                       f"{100*(remove+review)/total:.1f}% of dataset",
                       delta_color="inverse")
    eff_cols[2].metric("Removed",
                       f"{remove:,}",
                       f"{100*remove/total:.1f}%",
                       delta_color="inverse")
    eff_cols[3].metric("Reviewed",
                       f"{review:,}",
                       f"{100*review/total:.1f}%",
                       delta_color="off")

    # Efficiency gauge
    fig_gauge = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = efficiency,
        delta = {"reference": 80, "suffix": "%"},
        title = {"text": "Dataset cleanliness score (%)"},
        gauge = {
            "axis"  : {"range": [0, 100]},
            "bar"   : {"color": "#3C3489"},
            "steps" : [
                {"range": [0,  60], "color": "#FAECE7"},
                {"range": [60, 80], "color": "#FFF8E1"},
                {"range": [80, 100],"color": "#EAF3DE"},
            ],
            "threshold": {
                "line" : {"color": "#C62828", "width": 3},
                "thickness": 0.75,
                "value": 80
            }
        }
    ))
    fig_gauge.update_layout(height=300,
                            margin=dict(l=30,r=30,t=30,b=10))
    st.plotly_chart(fig_gauge, width="stretch")


    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — METRIC DEEP DIVE
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📐 Metric Deep Dive")

    left, right = st.columns(2)

    with left:
        # Average blur score per class
        st.markdown("#### Average blur score per class")
        if "blur_score" in master.columns:
            blur_by_cls = (
                master.groupby("class")["blur_score"]
                      .agg(["mean","min","max"])
                      .round(2)
                      .reset_index()
            )
            blur_by_cls.columns = ["Class","Mean","Min","Max"]

            fig_blur_cls = px.bar(
                blur_by_cls, x="Class", y="Mean",
                color="Class",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                error_y=None,
                title="Mean blur score by class"
            )
            fig_blur_cls.update_layout(
                showlegend=False, height=280,
                margin=dict(l=0,r=0,t=30,b=0),
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_blur_cls, width="stretch")

            global_blur_mean = safe_mean("blur_score")
            st.info(f"📊 Global mean blur score: **{global_blur_mean:.2f}**\n\n"
                  "Blur detection uses adaptive per-class thresholds "
                 "(5th percentile).")

        # Average reconstruction error per class
        st.markdown("#### Average reconstruction error per class")
        if "reconstruction_error" in master.columns:
            noise_by_cls = (
                master.groupby("class")["reconstruction_error"]
                      .agg(["mean","max"])
                      .round(6)
                      .reset_index()
            )
            noise_by_cls.columns = ["Class","Mean error","Max error"]

            mean_err   = master["reconstruction_error"].mean()
            std_err    = master["reconstruction_error"].std()
            threshold  = mean_err + 2 * std_err

            fig_noise_cls = px.bar(
                noise_by_cls, x="Class", y="Mean error",
                color="Class",
                color_discrete_sequence=px.colors.qualitative.Safe,
                title="Mean reconstruction error by class"
            )
            fig_noise_cls.add_hline(
                y=threshold, line_dash="dash", line_color="red",
                annotation_text=f"Noise threshold ({threshold:.4f})"
            )
            fig_noise_cls.update_layout(
                showlegend=False, height=280,
                margin=dict(l=0,r=0,t=30,b=0),
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_noise_cls, width="stretch")
            st.info(f"📊 Global mean error: **{mean_err:.6f}**  "
                    f"· Noise threshold: **{threshold:.6f}**")
        else:
            st.info("Noise report not yet generated.")

    with right:
        # Outlier percentage per class
        st.markdown("#### Outlier % per class")
        if "is_outlier" in master.columns:
            out_pct = (
                master.groupby("class")["is_outlier"]
                      .apply(lambda x: round(100 * x.astype(bool).sum() / len(x), 2))
                      .reset_index()
            )
            out_pct.columns = ["Class","Outlier %"]

            fig_out_pct = px.bar(
                out_pct, x="Class", y="Outlier %",
                color="Class",
                color_discrete_sequence=px.colors.qualitative.Antique,
                title="Outlier % per class"
            )
            fig_out_pct.add_hline(
                y=5, line_dash="dash", line_color="red",
                annotation_text="5% expected (contamination)"
            )
            fig_out_pct.update_layout(
                showlegend=False, height=280,
                margin=dict(l=0,r=0,t=30,b=0),
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_out_pct, width="stretch")
            st.info(f"📊 Global outlier rate: **{outlier_pct:.2f}%**  "
                    f"(expected ~5% · Isolation Forest contamination param)")

        # Mislabel percentage per class
        st.markdown("#### Mislabel % per class")
        if "is_mislabeled" in master.columns:
            mis_pct = (
                master.groupby("class")["is_mislabeled"]
                      .apply(lambda x: round(100 * x.astype(bool).sum() / len(x), 2))
                      .reset_index()
            )
            mis_pct.columns = ["Class","Mislabel %"]

            colors_mis = []
            for v in mis_pct["Mislabel %"]:
                if v > 10:   colors_mis.append("#C62828")
                elif v > 5:  colors_mis.append("#F9A825")
                else:        colors_mis.append("#3B6D11")

            fig_mis_pct = px.bar(
                mis_pct, x="Class", y="Mislabel %",
                title="Mislabel % per class",
                color="Mislabel %",
                color_continuous_scale=["#EAF3DE","#F9A825","#C62828"]
            )
            fig_mis_pct.update_layout(
                showlegend=False, height=280,
                margin=dict(l=0,r=0,t=30,b=0),
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_mis_pct, width="stretch")
            st.info(f"📊 Global mislabel rate: **{mislabel_pct:.2f}%**  "
                    f"(2/3 method agreement threshold)")


    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — FINAL RECOMMENDATION DISTRIBUTION
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Final Recommendation Distribution")

    vc     = master["verdict"].value_counts().reset_index()
    vc.columns = ["Verdict","Count"]
    vc["Percentage"] = (vc["Count"] / total * 100).round(2)
    vc["Bar"]        = vc["Percentage"]

    color_map_v = {
        "CLEAN"                    : "#3B6D11",
        "REVIEW — low priority"    : "#F9A825",
        "REVIEW — medium priority" : "#FB8C00",
        "REMOVE — blurry"          : "#E57373",
        "REMOVE — duplicate"       : "#EF5350",
        "REMOVE — mislabeled"      : "#C62828",
        "REMOVE — high priority"   : "#B71C1C",
    }

    # Horizontal bar — styled like a recommendation breakdown
    fig_rec = px.bar(
        vc, x="Percentage", y="Verdict",
        orientation="h",
        color="Verdict",
        color_discrete_map=color_map_v,
        text=vc.apply(lambda r: f"{r['Count']:,}  ({r['Percentage']}%)", axis=1),
        title="Final recommendation breakdown"
    )
    fig_rec.update_traces(textposition="outside")
    fig_rec.update_layout(
        showlegend=False,
        height=360,
        margin=dict(l=0,r=180,t=30,b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="% of dataset",
        yaxis_title="",
        xaxis_range=[0, 110]
    )
    st.plotly_chart(fig_rec, width="stretch")

    # ── Recommendation table
    st.dataframe(
        vc[["Verdict","Count","Percentage"]]
          .style.map(verdict_color, subset=["Verdict"]),
        width="stretch",
        hide_index=True
    )

    # ── Final summary callout
    st.markdown("---")
    st.markdown("### 📋 Pipeline Verdict")
    fa, fb, fc = st.columns(3)
    with fa:
        st.success(f"✅ **{clean:,} images are clean**  \n"
                   f"{efficiency:.1f}% of the dataset passed all checks.")
    with fb:
        st.warning(f"👁 **{review:,} images need review**  \n"
                   f"Flagged by 1 module — human confirmation recommended.")
    with fc:
        st.error(f"❌ **{remove:,} images should be removed**  \n"
                 f"Flagged by 2+ modules or high-confidence detection.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.markdown('<p class="section-header">Export Cleaned Dataset</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### What to remove")
        rm_exact    = st.checkbox("Exact duplicates",              True)
        rm_near     = st.checkbox("Near duplicates",               True)
        rm_blurry   = st.checkbox("Blurry images",                True)
        rm_noisy    = st.checkbox("Noisy images",                  False)
        rm_outlier  = st.checkbox("Outliers",                      False)
        rm_mislabel = st.checkbox("HIGH-confidence mislabels only",True)
    with col2:
        st.markdown("#### Priority threshold")
        min_score = st.slider("Remove if priority score ≥", 0, 20, 10)
        st.markdown("""
        **Score guide:**
        - `≥10` — remove high-priority issues
        - `≥5`  — remove moderate issues too
        - `0`   — remove everything flagged
        """)

    # Build mask
    mask = pd.Series(False, index=master.index)
    if rm_exact:   mask |= master["Exact_duplicates"].astype(bool)
    if rm_near:    mask |= master["near_duplicate"].astype(bool)
    if rm_blurry:  mask |= master["is_blurry"].astype(bool)
    if rm_noisy   and "is_noisy"       in master.columns:
        mask |= master["is_noisy"].astype(bool)
    if rm_outlier and "is_outlier"     in master.columns:
        mask |= master["is_outlier"].astype(bool)
    if rm_mislabel and "is_mislabeled" in master.columns:
        mask |= (
            master["is_mislabeled"].astype(bool) &
            (master.get("mislabel_confidence","") == "HIGH")
        )
    mask |= master["priority_score"] >= min_score

    to_keep   = (~mask).sum()
    to_remove = mask.sum()

    st.markdown("---")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Original",  f"{total:,}")
    e2.metric("Keep",      f"{to_keep:,}",
              f"{100*to_keep/total:.1f}%")
    e3.metric("Remove",    f"{to_remove:,}",
              f"-{100*to_remove/total:.1f}%", delta_color="inverse")
    e4.metric("Reduction", f"{100*to_remove/total:.1f}%")

    # Color-coded preview (Improvement #7)
    st.markdown("---")
    st.markdown("#### Preview — images to keep (color-coded, top 50)")
    preview_cols = ["file_path","class","verdict",
                    "priority_score","total_flags"]
    preview_cols = [c for c in preview_cols if c in master.columns]
    st.dataframe(
        master[~mask][preview_cols].head(50)
                                   .style.map(verdict_color,
                                                   subset=["verdict"]),
        width="stretch", height=250
    )

    st.markdown("---")
    if st.button("🚀 Export cleaned dataset", type="primary"):
        clean_root = Path("data_clean") / split
        clean_root.mkdir(parents=True, exist_ok=True)
        keep_df  = master[~mask]
        progress = st.progress(0)
        status   = st.empty()
        copied   = 0
        failed   = 0
        for i,(_, row) in enumerate(keep_df.iterrows()):
            src = Path(row["file_path"])
            dst = clean_root / row["class"] / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
                copied += 1
            except:
                failed += 1
            progress.progress((i+1)/len(keep_df))
            if (i+1) % 500 == 0:
                status.text(f"Copied {copied}/{len(keep_df)}...")
        progress.empty()
        st.success(f"✅ {copied} images → data_clean/{split}/")
        if failed:
            st.warning(f"{failed} files failed to copy.")
        keep_df.to_csv(
            f"reports/{split}_exported_clean.csv", index=False)
        st.info(f"Export log → reports/{split}_exported_clean.csv")

    st.markdown("---")
    csv_data = master.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download master report CSV",
        data=csv_data,
        file_name=f"{split}_master_report.csv",
        mime="text/csv"
    )