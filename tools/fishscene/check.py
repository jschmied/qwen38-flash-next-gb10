#!/usr/bin/env python3
"""Mechanical checks of the hard constraints on a generated scene. argv: <file.html>..."""
import json, re, sys, unicodedata
for f in sys.argv[1:]:
    s = open(f, encoding="utf-8", errors="replace").read()
    pict = [c for c in s if ord(c) > 0x2000 and (unicodedata.category(c) == "So" or 0x1F000 <= ord(c) <= 0x1FAFF)]
    r = {"file": f, "bytes": len(s.encode()), "under_18k": len(s.encode()) < 18 * 1024,
         "starts_doctype": s.lstrip().lower().startswith("<!doctype html>"),
         "markdown_fence": "```" in s,
         "link_tag": bool(re.search(r"<link\b", s, re.I)), "script_src": bool(re.search(r"<script[^>]*\bsrc=", s, re.I)),
         "fetch_or_xhr": bool(re.search(r"\bfetch\s*\(|XMLHttpRequest|import\s*\(", s)),
         "http_urls": sorted(set(u for u in re.findall(r"https?://[^\s\"'<>)]+", s) if "w3.org" not in u)),
         "pictographs": pict[:5], "linearGradient": "lineargradient" in s.lower(),
         "svg_elems": {t: len(re.findall(rf"<{t}\b", s, re.I)) for t in ("svg", "ellipse", "circle", "path", "polygon", "rect")}}
    print(json.dumps(r, ensure_ascii=False))
