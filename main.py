from __future__ import annotations
import asyncio
import csv
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urljoin

import tkinter as tk
from tkinter import filedialog, messagebox

import ttkbootstrap as tb
from ttkbootstrap.constants import BOTH, LEFT, RIGHT, X

from curl_cffi.requests import AsyncSession


from scraper import BASE, scrape_search
from process_990 import parse_990
from process_990PF import parse_990PF
from process_historical_data import (
    extract_xml_link_from_html_text,
    is_990pf_from_html_text,
    parse_historical_data_from_html_text,
)


"""
Modes:
1) Single query:
   - User types a query and clicks Run
   - File name is ALWAYS derived automatically:
       <query>_990.csv or <query>_990pf.csv (depending on detected filing type)
   - If file already exists, the program APPENDS a new row to it.

2) Multi query:
   - Reads queries from ./queries.txt (project root)
   - Outputs TWO CSVs:
       <base>_990.csv (only 990 rows)
       <base>_990pf.csv (only 990-PF rows)
   - User can edit <base> in UI
   - If output files exist, the program APPENDS new rows to them.
   - Wait time between queries is configurable (helps avoid rate limiting).

Per query pipeline:
- Query -> scrape_search -> pick first result
- Fetch org HTML
- Parse historical data from HTML
- Extract XML link from first filing section
- Download XML
- Detect 990 vs 990-PF
- Parse with correct parser
- Merge rows
"""

# ============================================================
# Helpers
# ============================================================


def _safe_merge_rows(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dict rows. If a key collides, b's value gets a suffix."""
    out = dict(a)
    for k, v in b.items():
        if k not in out:
            out[k] = v
            continue
        i = 2
        nk = f"{k}__{i}"
        while nk in out:
            i += 1
            nk = f"{k}__{i}"
        out[nk] = v
    return out


def _sanitize_filename_stem(s: str) -> str:
    """Make a readable, filesystem-safe filename stem."""
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s[:80] or "query"


def _read_queries_file(path: Path) -> List[str]:
    """
    Read queries from a text file:
    - one query per line
    - ignore blank lines
    - ignore lines starting with '#'
    """
    if not path.exists():
        raise RuntimeError(f"Queries file not found: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    if not out:
        raise RuntimeError("Queries file is empty (no valid queries found).")
    return out


def _append_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    Append rows to CSV.
    - If file doesn't exist or is empty: write header first.
    - If file exists: append rows using the existing header.
    - Missing/blank cells become '-'.

    NOTE:
      If new columns appear in later runs, we do NOT rewrite the existing header
      (that would require rewriting the whole file). We append using the existing header.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    file_exists = path.exists()
    existing_header: List[str] = []

    if file_exists:
        try:
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_header = next(reader, [])
        except Exception:
            existing_header = []

    incoming_cols = sorted({k for r in rows for k in r.keys()})
    all_cols = sorted(set(existing_header) | set(incoming_cols))

    write_header = (not file_exists) or (path.stat().st_size == 0) or (not existing_header)

    with path.open("a", newline="", encoding="utf-8") as f:
        if write_header:
            writer_cols = all_cols
            w = csv.DictWriter(f, fieldnames=writer_cols)
            w.writeheader()
        else:
            writer_cols = existing_header
            w = csv.DictWriter(f, fieldnames=writer_cols)

        for r in rows:
            w.writerow({k: ("-" if r.get(k) in (None, "") else r.get(k)) for k in writer_cols})


async def _fetch_text(session: AsyncSession, url: str) -> str:
    r = await session.get(url, headers={"Accept": "text/html,application/xhtml+xml"})
    r.raise_for_status()
    return r.text


async def _fetch_bytes(session: AsyncSession, url: str) -> bytes:
    r = await session.get(url, headers={"Accept": "*/*"})
    r.raise_for_status()
    return r.content


async def run_one_query_to_row(
    *,
    query: str,
    output_dir: Path,
) -> Tuple[Dict[str, Any], str]:
    """
    Run pipeline for ONE query and return:
      - combined_row
      - filing_type: "990" or "990-PF"
    """
    results = await scrape_search(query, max_pages=1, delay_s=0.0)
    if not results:
        raise RuntimeError(f"No results found for query: {query!r}")

    org = results[0]
    org_url = org.org_url

    async with AsyncSession(
        impersonate="chrome",
        timeout=60,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": BASE + "/nonprofits/",
        },
    ) as session:
        html_text = await _fetch_text(session, org_url)

        hist_row, _hist_meta = parse_historical_data_from_html_text(html_text)

        xml_href = extract_xml_link_from_html_text(html_text)
        xml_url = urljoin(BASE, xml_href)

        is_pf = is_990pf_from_html_text(html_text)
        filing_type = "990-PF" if is_pf else "990"

        xml_bytes = await _fetch_bytes(session, xml_url)

    tmp_xml_path = output_dir / f"__tmp_download_{_sanitize_filename_stem(query)}.xml"
    tmp_xml_path.write_bytes(xml_bytes)

    try:
        if filing_type == "990-PF":
            xml_row, _xml_meta = parse_990PF(tmp_xml_path)
        else:
            xml_row, _xml_meta = parse_990(tmp_xml_path)

        combined = _safe_merge_rows(hist_row, xml_row)

        combined = _safe_merge_rows(
            {
                "input_query": query,
                "org_name": org.org_name,
                "org_url": org.org_url,
                "org_id": org.org_id or "",
                "city": org.city or "",
                "state": org.state or "",
                "category_text": org.category_text or "",
                "revenue_year": org.revenue_year or "",
                "revenue_value": org.revenue_value or "",
                "filing_type": filing_type,
                "xml_url": xml_url,
            },
            combined,
        )
        return combined, filing_type
    finally:
        try:
            tmp_xml_path.unlink(missing_ok=True)
        except Exception:
            pass


# ============================================================
# UI
# ============================================================


class App(tb.Frame):
    def __init__(self, master: tb.Window) -> None:
        super().__init__(master, padding=18)
        self.master = master

        self.project_root = Path(__file__).resolve().parent
        self.default_data_dir = self.project_root / "data"
        self.default_queries_file = self.project_root / "queries.txt"

        # mode
        self.var_mode = tk.StringVar(value="single")  # "single" or "multi"

        # single
        self.var_query = tk.StringVar(value="")
        self.var_single_preview_990 = tk.StringVar(value="(type a query)")
        self.var_single_preview_990pf = tk.StringVar(value="(type a query)")

        # output dir (shared)
        self.var_output_dir = tk.StringVar(value=str(self.default_data_dir))

        # multi
        self.var_queries_file = tk.StringVar(value=str(self.default_queries_file))
        self.var_multi_base_name = tk.StringVar(value="results")  # stem only
        self.var_wait_seconds = tk.StringVar(value="5")  # configurable wait

        self.var_status = tk.StringVar(value="Ready.")
        self._run_btn: tb.Button | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.master.title("Nonprofit Filing Exporter")
        self.master.minsize(920, 590)
        self.pack(fill=BOTH, expand=True)

        header = tb.Frame(self)
        header.pack(fill=X, pady=(0, 12))

        tb.Label(header, text="Search & Export", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        tb.Label(
            header,
            text="Single exports one CSV (auto-named, appends if exists). Multi reads ./queries.txt and exports two CSVs (appends).",
            font=("Segoe UI", 10),
            bootstyle="secondary",
        ).pack(anchor="w", pady=(4, 0))

        # Mode
        mode_card = tb.Labelframe(self, text="Mode", padding=14)
        mode_card.pack(fill=X, pady=(0, 12))

        row_mode = tb.Frame(mode_card)
        row_mode.pack(fill=X)

        tb.Radiobutton(
            row_mode,
            text="Single query",
            value="single",
            variable=self.var_mode,
            bootstyle="success-outline-toolbutton",
            command=self._refresh_mode,
        ).pack(side=LEFT, padx=(0, 10))

        tb.Radiobutton(
            row_mode,
            text="Multi query (from file)",
            value="multi",
            variable=self.var_mode,
            bootstyle="success-outline-toolbutton",
            command=self._refresh_mode,
        ).pack(side=LEFT)

        # Content container
        self.content = tb.Frame(self)
        self.content.pack(fill=BOTH, expand=True)

        # ---- Single frame
        self.single_frame = tb.Frame(self.content)
        self.single_frame.pack(fill=BOTH, expand=True)

        card_search = tb.Labelframe(self.single_frame, text="Single query", padding=14)
        card_search.pack(fill=X, pady=(0, 12))

        row_query = tb.Frame(card_search)
        row_query.pack(fill=X)

        tb.Label(row_query, text="Query", width=10).pack(side=LEFT, padx=(0, 10))
        ent_query = tb.Entry(row_query, textvariable=self.var_query)
        ent_query.pack(side=LEFT, fill=X, expand=True)
        ent_query.bind("<KeyRelease>", lambda _e: self._update_single_previews())
        ent_query.focus_set()

        # output folder (shared)
        card_out_single = tb.Labelframe(self.single_frame, text="Output", padding=14)
        card_out_single.pack(fill=X)

        row_folder = tb.Frame(card_out_single)
        row_folder.pack(fill=X, pady=(0, 10))

        tb.Label(row_folder, text="Folder", width=10).pack(side=LEFT, padx=(0, 10))
        tb.Entry(row_folder, textvariable=self.var_output_dir).pack(side=LEFT, fill=X, expand=True)
        tb.Button(row_folder, text="Choose…", bootstyle="secondary", command=self.on_choose_folder).pack(
            side=LEFT, padx=(10, 0)
        )

        # preview names (read-only labels)
        preview_box = tb.Frame(card_out_single)
        preview_box.pack(fill=X)

        tb.Label(preview_box, text="Will save as:", width=10).pack(side=LEFT, padx=(0, 10))
        prev_text = tb.Frame(preview_box)
        prev_text.pack(side=LEFT, fill=X, expand=True)

        tb.Label(prev_text, textvariable=self.var_single_preview_990, bootstyle="secondary").pack(anchor="w")
        tb.Label(prev_text, textvariable=self.var_single_preview_990pf, bootstyle="secondary").pack(anchor="w")

        tb.Label(
            card_out_single,
            text="The program chooses 990 vs 990-PF automatically and APPENDS to the file if it already exists.",
            bootstyle="secondary",
        ).pack(anchor="w", pady=(10, 0))

        # ---- Multi frame
        self.multi_frame = tb.Frame(self.content)

        card_multi = tb.Labelframe(self.multi_frame, text="Multi query", padding=14)
        card_multi.pack(fill=X, pady=(0, 12))

        row_qfile = tb.Frame(card_multi)
        row_qfile.pack(fill=X, pady=(0, 10))

        tb.Label(row_qfile, text="Queries file", width=12).pack(side=LEFT, padx=(0, 10))
        tb.Entry(row_qfile, textvariable=self.var_queries_file).pack(side=LEFT, fill=X, expand=True)
        tb.Button(row_qfile, text="Choose…", bootstyle="secondary", command=self.on_choose_queries_file).pack(
            side=LEFT, padx=(10, 0)
        )

        tb.Label(
            card_multi,
            text="Format: one query per line. Blank lines and lines starting with # are ignored.",
            bootstyle="secondary",
        ).pack(anchor="w")

        card_out_multi = tb.Labelframe(self.multi_frame, text="Output", padding=14)
        card_out_multi.pack(fill=X)

        row_folder2 = tb.Frame(card_out_multi)
        row_folder2.pack(fill=X, pady=(0, 10))

        tb.Label(row_folder2, text="Folder", width=10).pack(side=LEFT, padx=(0, 10))
        tb.Entry(row_folder2, textvariable=self.var_output_dir).pack(side=LEFT, fill=X, expand=True)
        tb.Button(row_folder2, text="Choose…", bootstyle="secondary", command=self.on_choose_folder).pack(
            side=LEFT, padx=(10, 0)
        )

        row_base = tb.Frame(card_out_multi)
        row_base.pack(fill=X, pady=(0, 8))
        tb.Label(row_base, text="Base name", width=10).pack(side=LEFT, padx=(0, 10))
        tb.Entry(row_base, textvariable=self.var_multi_base_name).pack(side=LEFT, fill=X, expand=True)

        row_wait = tb.Frame(card_out_multi)
        row_wait.pack(fill=X, pady=(0, 8))
        tb.Label(row_wait, text="Wait (sec)", width=10).pack(side=LEFT, padx=(0, 10))
        tb.Entry(row_wait, textvariable=self.var_wait_seconds, width=10).pack(side=LEFT)
        tb.Label(row_wait, text="between queries", bootstyle="secondary").pack(side=LEFT, padx=(8, 0))

        tb.Label(
            card_out_multi,
            text="This produces TWO files: <base>_990.csv and <base>_990pf.csv. It APPENDS if files already exist.",
            bootstyle="secondary",
        ).pack(anchor="w", pady=(10, 0))

        # Bottom bar
        bottom = tb.Frame(self)
        bottom.pack(fill=X, pady=(16, 0))

        tb.Label(bottom, textvariable=self.var_status, bootstyle="secondary").pack(side=LEFT)

        self._run_btn = tb.Button(bottom, text="Run", bootstyle="primary", width=12, command=self.on_run)
        self._run_btn.pack(side=RIGHT)

        self.master.bind("<Return>", lambda _e: self.on_run())

        # init
        self._refresh_mode()
        self._update_single_previews()

    # ============================================================
    # Mode helpers
    # ============================================================
    def _refresh_mode(self) -> None:
        if self.var_mode.get() == "single":
            self.multi_frame.pack_forget()
            self.single_frame.pack(fill=BOTH, expand=True)
            self.var_status.set("Ready (Single query).")
        else:
            self.single_frame.pack_forget()
            self.multi_frame.pack(fill=BOTH, expand=True)
            self.var_status.set("Ready (Multi query).")

    def _update_single_previews(self) -> None:
        stem = _sanitize_filename_stem(self.var_query.get())
        self.var_single_preview_990.set(f"{stem}_990.csv")
        self.var_single_preview_990pf.set(f"{stem}_990pf.csv")

    # ============================================================
    # UI actions
    # ============================================================
    def on_choose_folder(self) -> None:
        chosen = filedialog.askdirectory(title="Choose output folder")
        if chosen:
            self.var_output_dir.set(chosen)
            self.var_status.set("Output folder updated.")

    def on_choose_queries_file(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Choose queries file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if chosen:
            self.var_queries_file.set(chosen)
            self.var_status.set("Queries file updated.")

    def on_run(self) -> None:
        settings = self._collect_settings()

        if settings["mode"] == "single":
            if not settings["query"].strip():
                messagebox.showwarning("Missing query", "Please type an organization name.")
                return
        else:
            qfile = Path(settings["queries_file"])
            if not qfile.exists():
                messagebox.showwarning("Missing queries file", f"Queries file not found:\n\n{qfile}")
                return

        if self._run_btn is not None:
            self._run_btn.configure(state="disabled")
        self.var_status.set("Working...")

        threading.Thread(target=self._run_in_thread, args=(settings,), daemon=True).start()

    def _run_in_thread(self, settings: dict[str, str]) -> None:
        try:
            output_dir = Path(settings["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)

            if settings["mode"] == "single":
                csv_path, filing_type = asyncio.run(
                    self._run_single(query=settings["query"], output_dir=output_dir)
                )
                self.master.after(0, lambda p=csv_path, t=filing_type: self._on_success_single(p, t))
            else:
                wait_s = self._parse_wait_seconds(settings["wait_seconds"])
                csv_990, csv_990pf = asyncio.run(
                    self._run_multi(
                        queries_file=Path(settings["queries_file"]),
                        output_dir=output_dir,
                        base_name=settings["multi_base_name"],
                        wait_seconds=wait_s,
                    )
                )
                self.master.after(0, lambda p1=csv_990, p2=csv_990pf: self._on_success_multi(p1, p2))

        except Exception as err:
            # capture err safely for later execution
            self.master.after(0, lambda e=err: self._on_error(e))

    def _parse_wait_seconds(self, s: str) -> float:
        try:
            v = float(s)
            return max(0.0, v)
        except Exception:
            return 0.0

    async def _run_single(self, *, query: str, output_dir: Path) -> Tuple[Path, str]:
        row, filing_type = await run_one_query_to_row(query=query, output_dir=output_dir)

        stem = _sanitize_filename_stem(query)
        filename = f"{stem}_990pf.csv" if filing_type == "990-PF" else f"{stem}_990.csv"
        csv_path = output_dir / filename

        _append_rows_csv(csv_path, [row])  # APPEND
        return csv_path, filing_type

    async def _run_multi(
        self,
        *,
        queries_file: Path,
        output_dir: Path,
        base_name: str,
        wait_seconds: float,
    ) -> Tuple[Path, Path]:
        queries = _read_queries_file(queries_file)

        rows_990: List[Dict[str, Any]] = []
        rows_990pf: List[Dict[str, Any]] = []

        for i, q in enumerate(queries, start=1):
            self.master.after(0, lambda q=q, i=i, n=len(queries): self.var_status.set(f"Working... ({i}/{n}) {q}"))

            row, filing_type = await run_one_query_to_row(query=q, output_dir=output_dir)

            if filing_type == "990-PF":
                rows_990pf.append(row)
            else:
                rows_990.append(row)

            # wait between queries to reduce rate limiting
            if i < len(queries) and wait_seconds > 0:
                await asyncio.sleep(wait_seconds)

        stem = _sanitize_filename_stem(base_name) or "results"
        csv_990 = output_dir / f"{stem}_990.csv"
        csv_990pf = output_dir / f"{stem}_990pf.csv"

        # APPEND
        _append_rows_csv(csv_990, rows_990)
        _append_rows_csv(csv_990pf, rows_990pf)

        return csv_990, csv_990pf

    def _on_success_single(self, csv_path: Path, filing_type: str) -> None:
        self.var_status.set(f"Done. Exported {filing_type} → {csv_path.name}")
        if self._run_btn is not None:
            self._run_btn.configure(state="normal")
        messagebox.showinfo("Success", f"Export complete.\n\nSaved/appended: {csv_path}")

    def _on_success_multi(self, csv_990: Path, csv_990pf: Path) -> None:
        self.var_status.set("Done. Exported multi-query CSVs.")
        if self._run_btn is not None:
            self._run_btn.configure(state="normal")
        messagebox.showinfo(
            "Success",
            "Multi export complete.\n\nSaved/appended:\n"
            f"- {csv_990}\n"
            f"- {csv_990pf}",
        )

    def _on_error(self, err: Exception) -> None:
        self.var_status.set("Error.")
        if self._run_btn is not None:
            self._run_btn.configure(state="normal")
        messagebox.showerror("Error", str(err))

    def _collect_settings(self) -> dict[str, str]:
        return {
            "mode": self.var_mode.get(),
            "query": self.var_query.get().strip(),
            "output_dir": self.var_output_dir.get().strip(),
            "queries_file": self.var_queries_file.get().strip(),
            "multi_base_name": (self.var_multi_base_name.get().strip() or "results"),
            "wait_seconds": self.var_wait_seconds.get().strip(),
        }


def main() -> None:
    app = tb.Window(themename="flatly")
    App(app)
    app.mainloop()


if __name__ == "__main__":
    main()
