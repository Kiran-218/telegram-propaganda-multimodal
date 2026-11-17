"""
Local annotation server. Serves the annotation page and the dataset images
entirely on your own machine -- nothing is uploaded anywhere.

Usage:
    python annotate_server.py
then open the printed URL in your browser.

Expects, in the same folder:
    annotation_sample_translated.csv   (with ann_id, channel, post_id,
                                        image_path, caption, caption_en)
    media/<channel>/<post_id>.jpg      (the images, as downloaded)

Writes/updates:
    annotation_labels.csv   (ann_id, label, notes, timestamp)
"""

import csv
import json
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote

# CSV_IN   = "annotation_sample_translated.csv"
# CSV_OUT  = "annotation_labels.csv"
CSV_IN   = "annotation_sample_translated_1.3B.csv"
CSV_OUT  = "annotation_labels_full.csv"
PORT     = 8000

ROWS = []           # loaded annotation rows
LABELS = {}         # ann_id -> {label, notes, timestamp}


def load_rows():
    with open(CSV_IN, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ROWS.append(r)


def load_labels():
    if os.path.exists(CSV_OUT):
        with open(CSV_OUT, "r", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                LABELS[r["ann_id"]] = {"label": r.get("label", ""),
                                       "notes": r.get("notes", ""),
                                       "timestamp": r.get("timestamp", "")}


def save_labels():
    with open(CSV_OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ann_id", "label", "notes", "timestamp"])
        for aid, d in LABELS.items():
            w.writerow([aid, d["label"], d["notes"], d["timestamp"]])


def norm_image_path(p):
    """Accept Windows or posix style paths from the CSV."""
    return p.replace("\\", "/")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):       # quiet console
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            with open("annotate.html", "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/data":
            payload = {"rows": ROWS, "labels": LABELS}
            self._send(200, json.dumps(payload).encode("utf-8"))
        elif path.startswith("/img/"):
            rel = norm_image_path(unquote(path[len("/img/"):]))
            if os.path.exists(rel) and rel.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                with open(rel, "rb") as f:
                    self._send(200, f.read(), "image/jpeg")
            else:
                self._send(404, b"not found", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path == "/save":
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            aid = data.get("ann_id")
            if aid:
                LABELS[aid] = {"label": data.get("label", ""),
                               "notes": data.get("notes", ""),
                               "timestamp": datetime.now().isoformat(timespec="seconds")}
                save_labels()
                self._send(200, json.dumps({"ok": True}).encode("utf-8"))
            else:
                self._send(400, json.dumps({"ok": False}).encode("utf-8"))
        else:
            self._send(404, b"not found", "text/plain")


def main():
    if not os.path.exists(CSV_IN):
        raise SystemExit(f"Missing {CSV_IN} in this folder.")
    load_rows()
    load_labels()
    print(f"Loaded {len(ROWS)} posts. {sum(1 for v in LABELS.values() if v['label'])} already labelled.")
    print(f"\n  Open  http://localhost:{PORT}  in your browser\n")
    print("  Labels save automatically to annotation_labels.csv")
    print("  Press Ctrl+C here to stop the server.\n")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
