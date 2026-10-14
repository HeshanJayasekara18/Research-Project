"""Collect public comments + replies for approved videos of one domain."""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import csvio
from .schema import COMMENT_FIELDS, video_url
from .youtube_api import ApiError, QuotaExceeded

log = logging.getLogger(__name__)

DONE_STATUSES = ("complete", "comments_disabled", "not_found")


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def comment_row(item, video_id, domain, thread_id, parent_id, collected_at, total_reply_count=""):
    """Map one YouTube comment resource to a research row. Author fields are dropped."""
    s = item.get("snippet", {})
    text = s.get("textOriginal")
    if text is None:
        text = s.get("textDisplay", "")
    return {
        "video_id": video_id,
        "comment_id": item["id"],
        "parent_id": parent_id,
        "thread_id": thread_id,
        "is_reply": "true" if parent_id else "false",
        "domain": domain,
        "comment_text": text,
        "published_at": s.get("publishedAt", ""),
        "updated_at": s.get("updatedAt", ""),
        "likes": str(s.get("likeCount", 0)),
        "total_reply_count": "" if parent_id else str(total_reply_count),
        "youtube_video_url": video_url(video_id),
        "collected_at": collected_at,
    }


class State:
    """Per-domain resume state: {video_id: {status, next_page_token, pages}}."""

    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def get(self, vid):
        return self.data.setdefault(vid, {"status": "pending", "next_page_token": None, "pages": 0})

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


class Stats(dict):
    def add(self, key, n=1):
        self[key] = self.get(key, 0) + n


def _thread_rows(client, thread, video_id, domain, collected_at, stats):
    """Rows for one thread: the top-level comment plus all of its replies."""
    snip = thread["snippet"]
    top = snip["topLevelComment"]
    thread_id = thread["id"]
    if top["id"] != thread_id:
        # YouTube uses the top-level comment id as the thread id; flag if not.
        stats.add("thread_id_mismatch")
    total = int(snip.get("totalReplyCount", 0) or 0)
    rows = [comment_row(top, video_id, domain, thread_id, "", collected_at, total)]

    replies = (thread.get("replies") or {}).get("comments", [])
    partial = False
    if total > len(replies):
        # commentThreads only embeds a subset of replies; fetch the full list.
        try:
            replies = list(client.iter_replies(top["id"]))
            stats.add("reply_list_calls")
        except QuotaExceeded:
            raise
        except ApiError as e:
            log.warning("replies for %s unavailable (%s); keeping %d embedded replies",
                        top["id"], e.reason, len(replies))
            stats.add("reply_fetch_errors")
            partial = True
    for rep in replies:
        parent = rep.get("snippet", {}).get("parentId", top["id"])
        if parent != top["id"]:
            stats.add("reply_parent_mismatch_skipped")
            continue
        rows.append(comment_row(rep, video_id, domain, thread_id, parent, collected_at))
    if len(replies) < total and not partial:
        # Normal when replies were deleted/held for review; recorded, not invented.
        stats.add("replies_missing_vs_total", total - len(replies))
    return rows, partial


def collect_domain(client, domain, videos, out_csv, state_path, refresh=False, max_pages=None):
    """Collect comments for the given approved manifest rows into out_csv.

    Returns (stats, per_video_status). Stops cleanly on quota exhaustion; the
    state file lets a later run continue from the last saved page.
    """
    _, existing = csvio.read_rows(out_csv)
    seen = {r["comment_id"] for r in existing}
    state = State(state_path)
    stats = Stats(existing_comments=len(seen))
    statuses = {}
    collected_at = utcnow()

    with csvio.AppendWriter(out_csv, COMMENT_FIELDS) as writer:
        for video in videos:
            vid = video["video_id"]
            if video["domain"] != domain or video["verification_status"] != "approved":
                raise ValueError(f"{vid} is not an approved {domain} video")
            st = state.get(vid)
            if st["status"] in DONE_STATUSES and not refresh:
                log.info("%s already %s; skipping (use --refresh to re-scan)", vid, st["status"])
                statuses[vid] = st["status"]
                continue
            if refresh and st["status"] != "in_progress":
                st.update(next_page_token=None, pages=0)
            st["status"] = "in_progress"
            video_partial = False
            new_for_video = 0
            pages_this_run = 0
            try:
                while True:
                    try:
                        page = client.comment_threads_page(vid, st["next_page_token"])
                    except ApiError as e:
                        if e.reason == "invalidPageToken" and st["next_page_token"]:
                            log.warning("%s: saved page token expired; restarting video", vid)
                            st.update(next_page_token=None, pages=0)
                            continue
                        raise
                    batch = []
                    for thread in page.get("items", []):
                        rows, partial = _thread_rows(client, thread, vid, domain, collected_at, stats)
                        video_partial |= partial
                        for row in rows:
                            if row["comment_id"] in seen:
                                stats.add("duplicates_skipped")
                                continue
                            seen.add(row["comment_id"])
                            batch.append(row)
                    writer.write(batch)  # rows are flushed before state advances
                    new_for_video += len(batch)
                    stats.add("new_comments", len(batch))
                    st["pages"] += 1
                    pages_this_run += 1
                    st["next_page_token"] = page.get("nextPageToken")
                    state.save()
                    if not st["next_page_token"]:
                        st["status"] = "partial" if video_partial else "complete"
                        break
                    if max_pages and pages_this_run >= max_pages:
                        st["status"] = "partial"
                        log.info("%s: stopped at --max-pages %d", vid, max_pages)
                        break
            except QuotaExceeded:
                state.save()
                statuses[vid] = "in_progress"
                log.error("Quota exhausted during %s; progress saved. Re-run later to resume.", vid)
                stats["stopped_on_quota"] = 1
                break
            except ApiError as e:
                if e.reason == "commentsDisabled":
                    st["status"] = "comments_disabled"
                elif e.reason in ("videoNotFound", "notFound") or e.status == 404:
                    st["status"] = "not_found"
                else:
                    st["status"] = "error"
                    log.error("%s failed: %s", vid, e)
                stats.add(f"videos_{st['status']}")
            state.save()
            statuses[vid] = st["status"]
            log.info("%s: +%d comments (%s)", vid, new_for_video, st["status"])
    stats["api_calls"] = client.calls
    stats["quota_units_used"] = client.quota_used
    return stats, statuses
