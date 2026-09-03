"""Human-readable evidence rendering: a candidate contact sheet and an HTML report.

Numbers in a terminal are hard to audit at a glance. A contact sheet showing every
candidate the pipeline examined, with its distance and decision printed on it, lets a
reviewer confirm in one look that the accepted match is the right person and the rejected
ones are not.

Both artifacts are local files. Nothing is hosted, and nothing here is required for
verification: the proof stands on the manifest and the chain record alone.
"""

from __future__ import annotations

import base64
import html
import io
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw

from sigil.models import DecisionStatus

if TYPE_CHECKING:
    from sigil.verify import CandidateVerification

logger = logging.getLogger(__name__)

TILE = 220
PADDING = 12
LABEL_HEIGHT = 46
COLUMNS = 4

COLOURS = {
    DecisionStatus.MATCH: (34, 197, 94),
    DecisionStatus.NON_MATCH: (239, 68, 68),
    DecisionStatus.INCONCLUSIVE: (234, 179, 8),
}
BACKGROUND = (17, 24, 39)
TEXT = (229, 231, 235)


def _thumbnail(media_bytes: bytes, size: int = TILE) -> Image.Image:
    """Decode candidate media into a fixed-size tile, or a placeholder if undecodable."""

    try:
        with Image.open(io.BytesIO(media_bytes)) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
    except Exception:
        thumb = Image.new("RGB", (size, size), (55, 65, 81))
        draw = ImageDraw.Draw(thumb)
        draw.text((size // 2 - 34, size // 2 - 6), "no media", fill=TEXT)
        return thumb

    canvas = Image.new("RGB", (size, size), BACKGROUND)
    canvas.paste(thumb, ((size - thumb.width) // 2, (size - thumb.height) // 2))
    return canvas


def contact_sheet(
    verifications: list[CandidateVerification],
    destination: str | Path,
    *,
    columns: int = COLUMNS,
) -> Path:
    """Render every examined candidate with its decision and distance."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not verifications:
        Image.new("RGB", (TILE, TILE), BACKGROUND).save(path)
        return path

    rows = (len(verifications) + columns - 1) // columns
    cell_w = TILE + PADDING
    cell_h = TILE + LABEL_HEIGHT + PADDING
    sheet = Image.new("RGB", (columns * cell_w + PADDING, rows * cell_h + PADDING), BACKGROUND)
    draw = ImageDraw.Draw(sheet)

    for index, item in enumerate(verifications):
        column, row = index % columns, index // columns
        x = PADDING + column * cell_w
        y = PADDING + row * cell_h

        sheet.paste(_thumbnail(item.media.data), (x, y))

        status = item.decision.status
        colour = COLOURS[status]
        # A thick border is readable in a screen recording at 1080p.
        draw.rectangle((x, y, x + TILE, y + TILE), outline=colour, width=4)

        distance = (
            f"d={item.decision.distance:.4f}" if item.decision.distance is not None else "d=-"
        )
        draw.text((x + 2, y + TILE + 4), f"{status}  {distance}", fill=colour)
        draw.text(
            (x + 2, y + TILE + 20),
            f"{item.candidate.platform} · {item.faces_detected} face(s) · "
            f"{str(item.media.quality).lower()}",
            fill=TEXT,
        )

    sheet.save(path, format="PNG", optimize=True)
    logger.info("contact sheet written to %s", path)
    return path


def write_thumbnails(
    verifications: list[CandidateVerification],
    destination: str | Path,
    *,
    size: int = 480,
) -> list[str]:
    """Write one downscaled JPEG per candidate for the results grid.

    These are presentation artifacts and are never part of the evidence root. The bytes
    the decision was computed on are digested in the manifest instead; a thumbnail is a
    lossy re-encode and must not be mistaken for the thing that was verified.
    """

    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for index, item in enumerate(verifications):
        if not item.media.ok:
            continue
        try:
            with Image.open(io.BytesIO(item.media.data)) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
                path = folder / f"{index}.jpg"
                thumb.save(path, format="JPEG", quality=82, optimize=True)
                written.append(path.name)
        except Exception as exc:
            logger.debug("no thumbnail for candidate %d: %s", index, exc)
    return written


def _data_uri(path: Path) -> str:
    try:
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{payload}"


def _row(label: str, value: str, *, mono: bool = False) -> str:
    css = ' class="mono"' if mono else ""
    return f"<tr><th>{html.escape(label)}</th><td{css}>{html.escape(value)}</td></tr>"


def html_report(
    bundle_dir: str | Path,
    *,
    manifest: dict[str, Any],
    root: str,
    verifications: list[CandidateVerification],
    receipt: Any = None,
    timings: dict[str, float] | None = None,
    search_records: list[dict[str, Any]] | None = None,
) -> Path:
    """Write a single self-contained HTML file summarising one run.

    Images are inlined as data URIs so the report can be moved or attached anywhere and
    still render, a report whose evidence lives in adjacent files is easy to separate
    from what it describes.
    """

    directory = Path(bundle_dir)
    sheet_path = directory / "contact_sheet.png"
    if verifications:
        contact_sheet(verifications, sheet_path)

    post = manifest.get("post", {})
    decision = manifest.get("decision", {})

    rows: list[str] = []
    for index, item in enumerate(verifications, start=1):
        status = item.decision.status
        colour = {"MATCH": "#22c55e", "NON_MATCH": "#ef4444", "INCONCLUSIVE": "#eab308"}[
            str(status)
        ]
        distance = f"{item.decision.distance:.4f}" if item.decision.distance is not None else "-"
        rows.append(
            f"<tr>"
            f"<td>{index}</td>"
            f'<td><span class="pill" style="background:{colour}">{status}</span></td>'
            f'<td class="mono">{distance}</td>'
            f"<td>{item.faces_detected}</td>"
            f"<td>{html.escape(str(item.media.quality).lower())}</td>"
            f'<td class="url"><a href="{html.escape(str(item.candidate.source_url))}" '
            f'rel="noopener noreferrer">{html.escape(str(item.candidate.source_url))}</a></td>'
            f"<td>{html.escape(item.decision.reason)}</td>"
            f"</tr>"
        )

    chain_block = "<p class='muted'>Not anchored yet, run <code>sigil anchor</code>.</p>"
    if receipt is not None:
        chain_block = (
            "<table>"
            + "".join(
                [
                    _row("Chain ID", str(receipt.chain_id)),
                    _row("Contract", receipt.contract_address, mono=True),
                    _row("Transaction", receipt.transaction_hash, mono=True),
                    _row("Block", str(receipt.block_number)),
                    _row("Submitter", receipt.submitter, mono=True),
                    _row("Anchored at", str(receipt.anchored_at)),
                    _row("Gas used", str(receipt.gas_used)),
                ]
            )
            + (
                f'</table><p><a href="{html.escape(receipt.explorer_url)}" '
                f'rel="noopener noreferrer">View on the block explorer →</a></p>'
                if receipt.explorer_url
                else "</table>"
            )
        )

    timing_block = ""
    if timings:
        cells = "".join(
            f"<tr><th>{html.escape(k)}</th><td class='mono'>{v:.0f} ms</td></tr>"
            for k, v in timings.items()
        )
        timing_block = f"<h2>Stage timings</h2><table>{cells}</table>"

    search_block = ""
    if search_records:
        cells = "".join(
            f"<tr><td>{html.escape(str(r.get('route', '')))}</td>"
            f"<td class='mono'>{html.escape(str(r.get('search_id', '')))}</td>"
            f"<td>{'cached' if r.get('from_cache') else 'live'}</td>"
            f"<td>{html.escape(str(r.get('status', '')))}</td></tr>"
            for r in search_records
        )
        search_block = (
            "<h2>Search audit trail</h2>"
            "<p class='muted'>Each row is a real provider call. The search ID can be "
            "checked against the provider's own dashboard.</p>"
            "<table><thead><tr><th>Route</th><th>Search ID</th><th>Source</th>"
            "<th>Status</th></tr></thead><tbody>" + cells + "</tbody></table>"
        )

    sheet_uri = _data_uri(sheet_path) if sheet_path.is_file() else ""
    crop_uri = _data_uri(directory / "media" / "aligned_crop.jpg")
    annotated_uri = _data_uri(directory / "media" / "annotated_input.jpg")

    document = f"""<!doctype html>
<meta charset="utf-8">
<title>Sigil evidence, {html.escape(root[:18])}</title>
<style>
  /* Same palette and type scale as the app that opens it. The report used to be dark
     while the rest of the product was light, so opening it read as leaving the tool. */
  :root {{ color-scheme: light; }}
  body {{ font: 500 16px/1.65 "Noto Sans", -apple-system, BlinkMacSystemFont,
                "Segoe UI", Roboto, sans-serif;
         background: #fbfbff; color: #312758; margin: 0; padding: 40px 24px; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; }}
  h1 {{ font-size: 34px; line-height: 1.25; margin: 0 0 6px; color: #1c1e25;
        letter-spacing: -.02em; }}
  h2 {{ font-size: 21px; margin: 40px 0 12px; color: #1c1e25; }}
  .tag {{ color: #6c6c9c; margin: 0 0 28px; font-size: 15px; }}
  .mono {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 14px;
           word-break: break-all; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0;
           background: #fff; border-radius: 12px; overflow: hidden;
           box-shadow: 0 2px 8px rgba(89,99,168,.14); }}
  th, td {{ text-align: left; padding: 11px 14px; border-bottom: 1px solid #ececf6;
            vertical-align: top; font-size: 15px; }}
  tr:last-child th, tr:last-child td {{ border-bottom: none; }}
  th {{ color: #6c6c9c; font-weight: 500; white-space: nowrap; }}
  .pill {{ padding: 3px 11px; border-radius: 999px; color: #fff; font-weight: 700;
           font-size: 13px; }}
  .root {{ background: #e9eafd; border: 1px solid #d9dbfa; border-radius: 12px;
           padding: 16px 18px; margin: 14px 0; }}
  .url {{ max-width: 280px; overflow-wrap: anywhere; }}
  .muted {{ color: #6c6c9c; font-size: 15px; }}
  img {{ max-width: 100%; border-radius: 10px; border: 1px solid #e6e6f2; }}
  .side {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .side img {{ max-width: 260px; }}
  a {{ color: #d67419; }}
  .scroll {{ overflow-x: auto; }}
</style>
<div class="wrap">
  <h1>Sigil evidence report</h1>
  <p class="tag">A face, sealed · generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC}</p>

  <div class="root">
    <div class="muted">Evidence root (Merkle, anchored on-chain)</div>
    <div class="mono">{html.escape(root)}</div>
  </div>

  <h2>Input</h2>
  <div class="side">
    {f'<img src="{annotated_uri}" alt="detected faces">' if annotated_uri else ""}
    {f'<img src="{crop_uri}" alt="aligned crop">' if crop_uri else ""}
  </div>

  <h2>Verified post</h2>
  <table>
    {_row("URL", str(post.get("canonical_url", "")))}
    {_row("Platform", str(post.get("platform", "")))}
    {_row("Post ID", str(post.get("post_id", "")), mono=True)}
    {_row("Media quality", str(post.get("media_quality", "")))}
    {_row("Discovered at", str(post.get("discovered_at", "")))}
    {_row("Search routes", ", ".join(post.get("search_routes", [])))}
    {_row("Decision", str(decision.get("status", "")))}
    {_row("Cosine distance", str(decision.get("distance", "")), mono=True)}
    {_row("Threshold", str(decision.get("threshold", "")), mono=True)}
    {_row("Model", str(manifest.get("model", "")), mono=True)}
  </table>

  <h2>Candidates examined</h2>
  <p class="muted">Every candidate the pipeline looked at, including the ones it rejected.
     Search rank cannot promote a candidate that failed the face gate.</p>
  {f'<img src="{sheet_uri}" alt="candidate contact sheet">' if sheet_uri else ""}
  <div class="scroll"><table>
    <thead><tr><th>#</th><th>Decision</th><th>Distance</th><th>Faces</th><th>Media</th>
    <th>Post</th><th>Reason</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>

  {search_block}

  <h2>Blockchain anchor</h2>
  {chain_block}

  {timing_block}

  <h2>Verify this independently</h2>
  <p class="muted">No private key is needed, verification is a read-only call.</p>
  <div class="root mono">sigil verify --bundle {html.escape(directory.name)}</div>
</div>
"""

    path = directory / "report.html"
    path.write_text(document, encoding="utf-8")
    logger.info("HTML report written to %s", path)
    return path


__all__ = ["contact_sheet", "html_report"]
