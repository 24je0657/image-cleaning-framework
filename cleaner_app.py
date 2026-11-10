# cleaner_app.py
"""Upload -> configure -> process -> review -> download (Step 8)."""
# Standard library
import time

# Third-party
import streamlit as st

# Local
from cleaner_client import (
    CHECKS_NOTE,
    JOB_ID_PATTERN,
    MODULES,
    ApiError,
    CleanerClient,
    build_approval_payload,
    default_decision,
    describe_reason,
    job_error_text,
    preview_counts,
    stage_for,
    class_preview,
    empty_classes,
    removal_warning,
)

PAGE_SIZE = 12
POLL_SECONDS = 2

st.set_page_config(page_title="Clean your image dataset",
                   page_icon="🧹", layout="wide")


@st.cache_resource
def get_client() -> CleanerClient:
    return CleanerClient()


client = get_client()


# ─── State ────────────────────────────────────────────────────────────────────

def init_state() -> None:
    ss = st.session_state
    if "job_id" not in ss:
        param = st.query_params.get("job")
        ok = isinstance(param, str) and JOB_ID_PATTERN.fullmatch(param)
        ss.job_id = param if ok else None
    ss.setdefault("overrides", {})


def set_job(job_id: str) -> None:
    ss = st.session_state
    for key in ("approval", "zip"):
        ss.pop(key, None)
    ss.job_id = job_id
    ss.overrides = {}
    st.query_params["job"] = job_id


def start_over() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.query_params.clear()


def show_error(exc: ApiError) -> None:
    st.error(exc.message)
    if exc.details:
        with st.expander("Details"):
            st.json(exc.details)


# ─── Screens ──────────────────────────────────────────────────────────────────

def render_upload() -> None:
    st.subheader("1 · Upload your dataset")
    st.markdown(
        "Upload a **ZIP** containing either class folders "
        "(`cats/`, `dogs/`, …) or `train/` and `val/` folders that each "
        "contain class folders. JPG and PNG images only. Your original "
        "upload is never modified.")
    file = st.file_uploader("Dataset ZIP", type=["zip"])
    if st.button("Upload and check", type="primary", disabled=file is None):
        with st.spinner("Uploading and checking your dataset…"):
            try:
                response = client.upload_zip(file.getvalue())
            except ApiError as exc:
                show_error(exc)
                return
        set_job(response["job_id"])
        st.rerun()


def render_working(job: dict) -> None:
    progress = max(0, min(100, int(job.get("progress") or 0)))
    st.progress(progress, text=job.get("message") or "Working…")
    st.caption("This page refreshes by itself. You can close it and return "
               "to this same link later.")
    time.sleep(POLL_SECONDS)
    st.rerun()


def render_run_form(job_id: str) -> None:
    st.subheader("2 · Choose the checks to run")
    st.caption("Images are only called CLEAN if they pass the checks you "
               "enable. Large datasets can take a while on a CPU server "
               "(we measured roughly 0.15–0.2 s per image).")
    with st.form(f"run::{job_id}"):
        picks = {key: st.checkbox(label, value=True, key=f"opt::{job_id}::{key}")
                 for key, label in MODULES}
        percentile = st.slider(
            "Blur: flag roughly the blurriest N% of each class", 1, 25, 5)
        submitted = st.form_submit_button("Start cleaning", type="primary")
    if not submitted:
        return
    if not any(picks.values()):
        st.error("Enable at least one check.")
        return
    try:
        client.start_run(job_id, {**picks, "blur_percentile": percentile})
    except ApiError as exc:
        show_error(exc)
        return
    st.rerun()


def render_configure(job: dict) -> None:
    info = job.get("result") or {}
    st.subheader("Your dataset")
    cols = st.columns(4)
    cols[0].metric("Classes", len(info.get("classes") or []))
    cols[1].metric("Images accepted", info.get("total_images", 0))
    cols[2].metric("Unreadable (removed)", info.get("rejected_total", 0))
    cols[3].metric("Other files ignored", info.get("ignored_total", 0))
    if info.get("classes"):
        st.text("Classes: " + ", ".join(info["classes"]))
    if info.get("rejected_total"):
        st.warning(f"{info['rejected_total']} unreadable images were left out "
                   "of the working copy. They will not be in your download.")
    render_run_form(job["job_id"])


@st.cache_data(ttl=900, max_entries=300, show_spinner=False)
def fetch_image(job_id: str, path: str) -> bytes:
    return client.image_bytes(job_id, path)


def on_toggle(path: str, group: str, default: str, key: str) -> None:
    decision = "keep" if st.session_state[key] else "remove"
    overrides = st.session_state.overrides
    if decision == default:
        overrides.pop(path, None)
    else:
        overrides[path] = (decision, group)


def set_offset(key: str, value: int) -> None:
    st.session_state[key] = value


def render_card(job_id: str, item: dict, apply_removals: bool,
                review_default: str) -> None:
    path, group = item["path"], item["group"]
    default = default_decision(group, apply_removals, review_default)
    try:
        st.image(fetch_image(job_id, path), width=220)
    except ApiError:
        st.caption("Preview unavailable")
    st.text(item["display_path"])
    for reason in item["reasons"]:
        st.text("• " + describe_reason(reason))
    key = f"keep::{job_id}::{path}::{default}"
    effective = st.session_state.overrides.get(path, (default, group))[0]
    st.checkbox("Keep this image", value=(effective == "keep"), key=key,
                on_change=on_toggle, args=(path, group, default, key))


def render_gallery(job_id: str, group: str, summary: dict,
                   apply_removals: bool, review_default: str) -> None:
    pick = st.selectbox("Class", ["All"] + sorted(summary.get("by_class", {})),
                        key=f"cls::{job_id}::{group}")
    name = None if pick == "All" else pick
    off_key = f"off::{job_id}::{group}::{pick}"
    offset = st.session_state.get(off_key, 0)
    try:
        page = client.images(job_id, verdict=group, class_name=name,
                             limit=PAGE_SIZE, offset=offset)
        if page["total"] and not page["items"]:
            offset = 0
            page = client.images(job_id, verdict=group, class_name=name,
                                 limit=PAGE_SIZE, offset=0)
    except ApiError as exc:
        show_error(exc)
        return
    if not page["total"]:
        st.info("No images in this group.")
        return
    columns = st.columns(4)
    for i, item in enumerate(page["items"]):
        with columns[i % 4]:
            render_card(job_id, item, apply_removals, review_default)
    prev_col, mid_col, next_col = st.columns([1, 2, 1])
    prev_col.button("◀ Previous", disabled=offset == 0,
                    key=f"prev::{off_key}", on_click=set_offset,
                    args=(off_key, max(0, offset - PAGE_SIZE)))
    mid_col.caption(f"Showing {offset + 1}–{offset + len(page['items'])} "
                    f"of {page['total']}")
    next_col.button("Next ▶", disabled=offset + PAGE_SIZE >= page["total"],
                    key=f"next::{off_key}", on_click=set_offset,
                    args=(off_key, offset + PAGE_SIZE))


def render_approval(job_id: str, totals: dict, by_class: dict,
                    apply_removals, review_default: str) -> None:
    st.subheader("4 · Build your cleaned dataset")
    overrides = st.session_state.overrides
    blocked = apply_removals is None
    if apply_removals is None:
        st.info("Choose what to do with the proposed removals above.")
    else:
        preview = preview_counts(totals, apply_removals, review_default,
                                 overrides)
        st.write(f"Your cleaned dataset will contain **{preview['keep']}** "
                 f"images; **{preview['remove']}** will be left out. "
                 f"({len(overrides)} individual choices.) "
                 "Your original upload is never modified.")
        per_class = class_preview(by_class, apply_removals, review_default,
                                  overrides)
        st.caption("Images kept per class: " + ", ".join(
            f"{name}: {n}" for name, n in sorted(per_class.items())))
        empty = empty_classes(per_class)
        if preview["keep"] <= 0:
            st.error("These choices would leave no images at all.")
            blocked = True
        elif empty:
            st.warning("These choices leave no images for: "
                       + ", ".join(empty)
                       + ". A dataset missing a class is usually not "
                       "usable for training.")
            blocked = not st.checkbox(
                "I understand and want to build it anyway.",
                key=f"confirm_empty::{job_id}::{'|'.join(empty)}")
    if st.button("Build cleaned dataset", type="primary", disabled=blocked):
        try:
            payload = build_approval_payload(apply_removals, review_default,
                                             overrides)
            with st.spinner("Building your cleaned dataset…"):
                st.session_state.approval = client.approve(job_id, payload)
            st.session_state.pop("zip", None)
        except ValueError as exc:
            st.error(str(exc))
        except ApiError as exc:
            show_error(exc)

    approval = st.session_state.get("approval")
    if not approval:
        return
    st.success(f"Built: {approval['kept']} images kept, "
               f"{approval['removed']} left out "
               f"({approval['rescued_from_remove']} proposed removals "
               f"rescued).")
    for warning in approval.get("warnings", []):
        if warning.get("code") == "class_empty":
            st.warning(f"No images of class '{warning.get('class')}' remain.")
        elif warning.get("code") == "missing_files":
            st.warning(f"{warning.get('count')} approved images could not be "
                       "found and are not in the download.")
    render_download(job_id)


def render_download(job_id: str) -> None:
    st.subheader("5 · Download")
    ss = st.session_state
    if ss.get("zip") is None:
        if st.button("Prepare download"):
            with st.spinner("Packaging your dataset…"):
                try:
                    ss.zip = client.download(job_id)
                except ApiError as exc:
                    show_error(exc)
                    return
            st.rerun()
        return
    data, missing = ss.zip
    if missing:
        st.warning(f"{missing} approved images are missing from this ZIP.")
    st.download_button("Download cleaned dataset (ZIP)", data=data,
                       file_name=f"cleaned_dataset_{job_id[:8]}.zip",
                       mime="application/zip", type="primary")
    st.caption("The ZIP also contains approval_decisions.csv (every image, "
               "its verdict and your decision) and summary.json.")


def render_review(job_id: str) -> None:
    try:
        summary = client.results(job_id)
    except ApiError as exc:
        show_error(exc)
        return
    totals = summary["totals"]
    st.subheader("3 · Review the results")
    cols = st.columns(4)
    cols[0].metric("Images", totals["images"])
    cols[1].metric("Clean", totals["clean"])
    cols[2].metric("Needs review", totals["review"])
    cols[3].metric("Proposed removals", totals["remove"])
    st.caption(CHECKS_NOTE)
    warning = removal_warning(totals)
    if warning:
        st.error(
            f"⚠️ {warning['remove']} of {warning['images']} images "
            f"({warning['share_percent']}%) are proposed for removal. "
            "That is far more than usual. Check your dataset before "
            "approving, for example whether many images are near-identical "
            "(try running again without duplicate detection).")
    if summary.get("unmatched_report_rows"):
        st.warning(f"{summary['unmatched_report_rows']} report rows did not "
                   "match files in your dataset and are not shown.")

    removal_choice = st.radio(
        "Images the tool proposes to remove", ["Remove them", "Keep them"],
        index=None, key=f"apply::{job_id}", horizontal=True)
    review_choice = st.radio(
        "Images that need your review", ["Keep them", "Remove them"],
        index=0, key=f"review::{job_id}", horizontal=True)
    apply_removals = None if removal_choice is None else (
        removal_choice == "Remove them")
    review_default = "keep" if review_choice == "Keep them" else "remove"
    shown_apply = True if apply_removals is None else apply_removals

    tab_review, tab_remove, tab_clean = st.tabs([
        f"Needs review ({totals['review']})",
        f"Proposed removals ({totals['remove']})",
        f"Clean ({totals['clean']})"])
    for tab, group in ((tab_review, "REVIEW"), (tab_remove, "REMOVE"),
                       (tab_clean, "CLEAN")):
        with tab:
            render_gallery(job_id, group, summary, shown_apply, review_default)
        render_approval(job_id, totals, summary.get("by_class", {}),
                    apply_removals, review_default)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    init_state()
    with st.sidebar:
        st.header("Job")
        job_id = st.session_state.job_id
        st.code(job_id or "none yet")
        st.button("Start over", on_click=start_over)
        st.caption(CHECKS_NOTE)

    st.title("🧹 Clean your image dataset")
    if not job_id:
        render_upload()
        return
    try:
        job = client.job(job_id)
    except ApiError as exc:
        if exc.status in (400, 404):
            st.warning("This job no longer exists. Jobs are deleted after a "
                       "period of inactivity.")
        else:
            show_error(exc)
        st.button("Start over", on_click=start_over, key="over_main")
        return

    stage = stage_for(job["status"])
    if stage == "working":
        render_working(job)
    elif stage == "configure":
        render_configure(job)
    elif stage == "review":
        render_review(job_id)
    elif stage in ("failed", "interrupted"):
        st.error(job_error_text(job) if stage == "failed" else
                 "The service restarted while this job was running.")
        render_run_form(job_id)
        st.button("Start over with a different dataset",
                  on_click=start_over, key="over_failed")
    else:
        st.warning(f"Unknown job status: {job['status']}")


main()