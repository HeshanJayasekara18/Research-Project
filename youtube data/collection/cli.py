"""Command line entry point: python -m collection <command> --help"""
import argparse
import collections
import logging
import sys
from datetime import datetime, timezone

from . import csvio, manifest
from .collector import State, collect_domain, utcnow
from .combine import build_combined, filter_rows
from .schema import (COMBINED_CSV, COMMENT_FIELDS, DOMAINS, ROOT, STATE_DIR, VIDEO_FIELDS,
                     VIDEOS_CSV, domain_csv)
from .validate import validate_combined, validate_domains
from .youtube_api import QuotaExceeded, YouTubeClient, load_api_key, redact, urllib_transport

log = logging.getLogger("collection")

# YouTube category ids used only as a *hint* when a human reviews a video.
CATEGORY_HINTS = {"food": {"26", "19", "22", "24"}, "gaming": {"20"}, "education": {"27", "28"}}
RETENTION_DAYS = 30


def _manifest_or_exit(path):
    header, rows = manifest.load(path)
    errors = manifest.check(header, rows)
    if errors:
        for e in errors:
            print("manifest:", e)
        raise SystemExit(2)
    return rows


def _client(args):
    key = load_api_key(ROOT / ".env")
    return YouTubeClient(urllib_transport(key), quota_budget=getattr(args, "quota_budget", None)), key


def cmd_check_manifest(args):
    rows = _manifest_or_exit(args.manifest)
    c = collections.Counter((r["domain"], r["verification_status"]) for r in rows)
    for d in DOMAINS:
        print(f"{d:10s} " + "  ".join(f"{s}={c[(d, s)]}" for s in ("approved", "candidate", "rejected")))
    print("manifest OK")


def cmd_fetch_metadata(args):
    rows = _manifest_or_exit(args.manifest)
    targets = [r for r in rows if not args.domain or r["domain"] == args.domain]
    client, key = _client(args)
    try:
        items = {i["id"]: i for i in client.videos([r["video_id"] for r in targets])}
    except Exception as e:  # never print the key
        raise SystemExit(f"metadata request failed: {redact(e, key)}")
    now = utcnow()
    for r in targets:
        item = items.get(r["video_id"])
        if not item:
            r["collection_status"] = r["collection_status"] or "not_found"
            print(f"{r['video_id']}: not returned by API (private/deleted?)")
            continue
        sn, stats = item["snippet"], item.get("statistics", {})
        r.update(title=sn.get("title", ""), channel_title=sn.get("channelTitle", ""),
                 video_published_at=sn.get("publishedAt", ""), api_category_id=sn.get("categoryId", ""),
                 default_language=sn.get("defaultAudioLanguage") or sn.get("defaultLanguage", ""),
                 api_comment_count=stats.get("commentCount", ""), metadata_fetched_at=now)
        if r["api_category_id"] not in CATEGORY_HINTS[r["domain"]]:
            print(f"REVIEW {r['video_id']}: manifest domain={r['domain']} but API category="
                  f"{r['api_category_id']} ({r['title'][:60]})")
    csvio.write_rows_atomic(args.manifest, VIDEO_FIELDS, rows)
    print(f"metadata updated for {len(items)}/{len(targets)} videos; quota used {client.quota_used}")


def cmd_collect(args):
    if not args.confirm_ethics_clearance and not args.dry_run:
        raise SystemExit("Refusing to collect: pass --confirm-ethics-clearance once the selected "
                         "videos are approved and ethical clearance is confirmed.")
    rows = _manifest_or_exit(args.manifest)
    videos = manifest.approved(rows, args.domain)
    if args.video_id:
        mby = manifest.by_id(rows)
        for vid in args.video_id:
            m = mby.get(vid)
            if not m or m["verification_status"] != "approved" or m["domain"] != args.domain:
                raise SystemExit(f"{vid} is not an approved {args.domain} video in the manifest")
        videos = [v for v in videos if v["video_id"] in set(args.video_id)]
    if not videos:
        raise SystemExit(f"No approved {args.domain} videos in {args.manifest}")
    out = domain_csv(args.domain)
    state_path = STATE_DIR / f"{args.domain}_state.json"
    if args.dry_run:
        st = State(state_path)
        for v in videos:
            print(f"{v['video_id']}  {st.get(v['video_id'])['status']:12s} {v['title'][:60]}")
        print(f"{len(videos)} videos would be collected into {out}")
        return
    client, key = _client(args)
    try:
        stats, statuses = collect_domain(client, args.domain, videos, out, state_path,
                                         refresh=args.refresh, max_pages=args.max_pages)
    except Exception as e:
        raise SystemExit(f"collection failed: {redact(e, key)}")
    now = utcnow()
    for r in rows:
        if r["video_id"] in statuses:
            r["collection_status"] = statuses[r["video_id"]]
            r["last_collected_at"] = now
    csvio.write_rows_atomic(args.manifest, VIDEO_FIELDS, rows)
    for k, v in sorted(stats.items()):
        print(f"{k}: {v}")
    if stats.get("stopped_on_quota"):
        sys.exit(3)


def cmd_validate(args):
    rows = _manifest_or_exit(args.manifest)
    if args.combined:
        report = validate_combined(COMBINED_CSV, rows)
    else:
        report = validate_domains(rows, domains=[args.domain] if args.domain else DOMAINS)
    print(report.render(limit=args.limit))
    sys.exit(0 if report.ok else 1)


def cmd_combine(args):
    rows = _manifest_or_exit(args.manifest)
    report, n = build_combined(rows, COMBINED_CSV)
    print(report.render(limit=args.limit))
    if not report.ok:
        print("combined corpus NOT built: fix the errors above")
        sys.exit(1)
    print(f"wrote {n} rows to {COMBINED_CSV}")


def cmd_filter(args):
    header, rows = csvio.read_rows(args.input)
    if header is None:
        raise SystemExit(f"{args.input} not found")
    out = filter_rows(rows, args.domain, args.video_id, args.thread_id)
    if args.out:
        csvio.write_rows_atomic(args.out, COMMENT_FIELDS, out)
        print(f"wrote {len(out)} rows to {args.out}")
    else:
        for r in out[:args.limit]:
            indent = "    " if r["is_reply"] == "true" else ""
            print(f"{indent}[{r['domain']}/{r['video_id']}/{r['comment_id']}] {r['comment_text'][:100]}")
        print(f"{len(out)} rows")


def cmd_coverage(args):
    _, mrows = manifest.load(args.manifest)
    now = datetime.now(timezone.utc)
    print(f"{'domain':10s} {'approved':>8s} {'collected':>9s} {'comments':>9s} {'replies':>8s} "
          f"{'threads':>8s} {'empty':>6s} {'oldest_days':>11s}")
    for d in DOMAINS:
        approved = manifest.approved(mrows, d)
        _, rows = csvio.read_rows(domain_csv(d))
        ages = [(now - datetime.fromisoformat(r["collected_at"].replace("Z", "+00:00"))).days
                for r in rows if r["collected_at"]]
        print(f"{d:10s} {len(approved):8d} {len({r['video_id'] for r in rows}):9d} {len(rows):9d} "
              f"{sum(r['is_reply'] == 'true' for r in rows):8d} {len({r['thread_id'] for r in rows}):8d} "
              f"{sum(not r['comment_text'].strip() for r in rows):6d} {max(ages, default=0):11d}")
        if ages and max(ages) > RETENTION_DAYS:
            print(f"  NOTE: {d} has data older than {RETENTION_DAYS} days; review YouTube API "
                  "retention obligations (refresh or delete).")
        if args.per_video:
            per = collections.Counter(r["video_id"] for r in rows)
            for v in approved:
                print(f"    {v['video_id']} {per[v['video_id']]:6d}  api_count={v['api_comment_count'] or '?':>6s} "
                      f"status={v['collection_status'] or 'not collected'}  {v['title'][:50]}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m collection")
    p.add_argument("--manifest", default=VIDEOS_CSV)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check-manifest").set_defaults(fn=cmd_check_manifest)

    s = sub.add_parser("fetch-metadata", help="fill title/category from videos.list (1 unit per 50 videos)")
    s.add_argument("--domain", choices=DOMAINS)
    s.set_defaults(fn=cmd_fetch_metadata)

    s = sub.add_parser("collect")
    s.add_argument("--domain", choices=DOMAINS, required=True)
    s.add_argument("--video-id", action="append", help="restrict to these approved ids (repeatable)")
    s.add_argument("--max-pages", type=int, help="max commentThreads pages per video this run (pilot)")
    s.add_argument("--quota-budget", type=int, help="stop after this many quota units")
    s.add_argument("--refresh", action="store_true", help="re-scan completed videos for new comments")
    s.add_argument("--dry-run", action="store_true", help="show what would be collected; no API calls")
    s.add_argument("--confirm-ethics-clearance", action="store_true")
    s.set_defaults(fn=cmd_collect)

    s = sub.add_parser("validate")
    s.add_argument("--domain", choices=DOMAINS)
    s.add_argument("--combined", action="store_true")
    s.add_argument("--limit", type=int, default=25)
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("combine")
    s.add_argument("--limit", type=int, default=25)
    s.set_defaults(fn=cmd_combine)

    s = sub.add_parser("filter")
    s.add_argument("--input", default=COMBINED_CSV)
    s.add_argument("--domain", choices=DOMAINS)
    s.add_argument("--video-id")
    s.add_argument("--thread-id")
    s.add_argument("--out")
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(fn=cmd_filter)

    s = sub.add_parser("coverage")
    s.add_argument("--per-video", action="store_true")
    s.set_defaults(fn=cmd_coverage)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args.fn(args)
