"""Export the three Markdown reports with embedded images for local browsers.

Run: python export_html.py
Optional dependency: mistune>=3,<4 (not required for stereo experiments).
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlsplit

import mistune

ROOT = Path(__file__).resolve().parent
SOURCES = [ROOT / 'README.md', ROOT / 'ANALYSIS.md', ROOT / 'results/sgbm_final/REPORT.md']
EXPORTED = {p.resolve() for p in SOURCES}
STYLE = '''
html { color-scheme: light; background: #eef2f6; }
body { margin: 0; color: #182433; font: 16px/1.85 "Segoe UI", "Microsoft YaHei", sans-serif; }
main { max-width: 1120px; margin: 28px auto; padding: 32px 42px; background: white; border-radius: 12px; }
h1,h2,h3 { line-height: 1.4; color: #153956; margin-top: 1.8em; }
h1 { font-size: 28px; margin-top: 0; }
h2 { border-bottom: 1px solid #d9e3eb; padding-bottom: 8px; }
a { color: #146eaa; overflow-wrap: anywhere; }
table { border-collapse: collapse; width: 100%; font-size: 14px; margin: 18px 0; }
th,td { border: 1px solid #d8e0e8; padding: 9px 12px; text-align: left; }
th { background: #edf4f9; } tr:nth-child(even) { background: #f8fafc; }
figure { margin: 24px 0; padding: 10px; border: 1px solid #e0e6ec; border-radius: 8px; }
img { display: block; width: 100%; height: auto; }
figcaption { font-size: 13px; color: #576b7c; margin-top: 8px; text-align: center; }
pre { overflow-x: auto; background: #f3f6f9; padding: 16px; border-radius: 7px; font-size: 13px; line-height: 1.6; }
code { font-family: Consolas, monospace; font-size: .9em; }
blockquote { border-left: 4px solid #5b9fca; background: #f5f9fc; margin: 20px 0; padding: 12px 20px; }
.notice { background: #eaf6ef; color: #23593d; padding: 12px 16px; border-radius: 7px; margin-bottom: 26px; }
@media(max-width: 700px) { main { margin: 0; padding: 22px 16px; } table { font-size: 12px; } th,td { padding: 5px; } }
@media print { html { background: white; } main { margin: 0; padding: 0; max-width: none; } figure, tr { break-inside: avoid; } }
'''


class EmbeddedRenderer(mistune.HTMLRenderer):
    def __init__(self, source):
        super().__init__(escape=True)
        self.source = source
        self.images = []

    def image(self, text, url, title=None):
        parsed = urlsplit(url)
        if parsed.scheme:
            raise ValueError(f'Expected a local report image, got {url}')
        target = (self.source.parent / unquote(parsed.path)).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():
            raise FileNotFoundError(f'Missing or external report image: {target}')
        blob = target.read_bytes()
        mime = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
        encoded = base64.b64encode(blob).decode('ascii')
        label = html.escape(text or target.name)
        self.images.append({'path': str(target.relative_to(ROOT)),
                            'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)})
        return f'<figure><img src="data:{mime};base64,{encoded}" alt="{label}" loading="lazy"><figcaption>{label}</figcaption></figure>'

    def link(self, text, url, title=None):
        parsed = urlsplit(url)
        if not parsed.scheme and parsed.path:
            target = (self.source.parent / unquote(parsed.path)).resolve()
            if target in EXPORTED:
                url = str(Path(unquote(parsed.path)).with_suffix('.html')).replace('\\', '/')
                if parsed.fragment:
                    url += '#' + parsed.fragment
        return super().link(text, url, title)


def main():
    manifest = []
    for source in SOURCES:
        renderer = EmbeddedRenderer(source)
        md = mistune.create_markdown(renderer=renderer, plugins=['table'])
        body = md(source.read_text(encoding='utf-8'))
        title = 'Part 1 — ' + source.stem
        document = (f'<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
                    f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                    f'<title>{html.escape(title)}</title><style>{STYLE}</style></head><body><main>'
                    '<div class="notice">图片已内嵌，可离线查看。用 Edge、Chrome 等浏览器打开本文件即可。'
                    '点击图片后可通过浏览器放大页面查看细节。</div>'
                    f'{body}</main></body></html>')
        output = source.with_suffix('.html')
        output.write_text(document, encoding='utf-8')
        manifest.append({'source': str(source.relative_to(ROOT)),
                         'output': str(output.relative_to(ROOT)), 'images': renderer.images,
                         'html_bytes': output.stat().st_size})
        print(f'{output.name}: {len(renderer.images)} embedded images, {output.stat().st_size / 1e6:.2f} MB')
    (ROOT / 'html_export_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
