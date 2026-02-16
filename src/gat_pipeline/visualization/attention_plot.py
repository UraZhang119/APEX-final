from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ..config import PipelineConfig
from .common import ensure_dir, prepare_sequence_artifacts

RepresentativeInterval = Tuple[int, int, str]
RepresentativeAnnotations = Dict[str, List[RepresentativeInterval]]

CATEGORY_STYLES = {
    "domain": {
        "palette": ["#f4b6c2", "#cdb4ff", "#a9def9", "#f7d6e0", "#c2e7da", "#d0bdf4"],
        "dedicated": True,
    },
    "homologous_superfamily": {
        "palette": ["#ffd6a5", "#ffb5a7", "#e9ff70", "#ffd3b6", "#f9dcc4"],
        "dedicated": True,
    },
    "family": {
        "palette": ["#a3c948", "#7fb069", "#c0d860", "#8cc286"],
        "dedicated": True,
    },
}
DEFAULT_CATEGORY_STYLE = {
    "palette": ["#9ad0c2", "#f7aef8", "#fbe7c6", "#c0fdff", "#fec5bb", "#e4c1f9"],
    "dedicated": True,
}

ACTIVE_SITE_COLOURS = ["#d62839", "#f77f00", "#2a9d8f", "#6a4c93", "#ffba08", "#4361ee", "#ff006e"]


def plot_attention_and_importance(
    sequence: str,
    protein_name: str,
    checkpoint_path: Path,
    config: PipelineConfig,
    fold_number: Optional[int] = None,
    inference_dir: Path | str = Path("inference_results"),
    explain_dir: Path | str = Path("gnn_results"),
    output_dir: Path | str = Path("graficos"),
    top_fraction: float = 0.1,
    explainer_steps: int = 11,
    explainer_epochs: Optional[int] = None,
    explainer_seed: int = 42,
    annotation_path: Optional[Path | str] = None,
    annotation_name: Optional[str] = None,
    active_site_zip: Optional[Path | str] = None,
    active_site_pockets: Optional[List[int]] = None,
    seq_start_offset: int = 1,
) -> tuple[Path, Path]:
    """Generate separate figures for attention/importance lines and the contact map."""

    sequence = sequence.strip()
    checkpoint_path = Path(checkpoint_path)
    charts_root = ensure_dir(output_dir)

    attention_df, importance_df, contact_map = prepare_sequence_artifacts(
        sequence=sequence,
        protein_name=protein_name,
        checkpoint_path=checkpoint_path,
        config=config,
        fold_number=fold_number,
        inference_dir=inference_dir,
        explain_dir=explain_dir,
        top_fraction=top_fraction,
        explainer_steps=explainer_steps,
        explainer_epochs=explainer_epochs,
        explainer_seed=explainer_seed,
    )
    offset = max(int(seq_start_offset), 1) - 1
    attention_df["position"] = attention_df["position"] + offset
    importance_df["position"] = importance_df["position"] + offset

    chart_dir = ensure_dir(Path(charts_root) / protein_name)
    chart_lines_path = chart_dir / f"{protein_name}_attention_importance.png"
    chart_contact_path = chart_dir / f"{protein_name}_contact_map_teal.png"
    chart_contact_alt_path = chart_dir / f"{protein_name}_contact_map_diverging.png"

    annotations = _load_representative_annotations(
        annotation_path,
        annotation_name,
        seq_offset=offset,
        seq_length=len(sequence),
    )
    active_site_residues = _load_active_site_residues(active_site_zip, active_site_pockets)
    if active_site_residues:
        filtered = {}
        region_start = offset + 1
        region_end = offset + len(sequence)
        for pid, residues in active_site_residues.items():
            adj = [pos for pos in residues if region_start <= pos <= region_end]
            if adj:
                filtered[pid] = sorted(dict.fromkeys(adj))
        active_site_residues = filtered or None

    _style_plot()
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 9.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 0.6]},
    )

    colours = {
        "sand": "#dfc07b",
        "bronze": "#b06b32",
        "pearl": "#ced1cf",
        "mist": "#dfe4e1",
        "slate": "#808c86",
        "drift": "#8ea59d",
        "teal": "#2f6a6a",
        "graphite": "#222222",
    }

    ax_att = axes[0]
    ax_att.fill_between(
        attention_df["position"],
        attention_df["total_attention"],
        color=colours["drift"],
        alpha=0.6,
        linewidth=0,
    )
    ax_att.plot(
        attention_df["position"],
        attention_df["total_attention"],
        color=colours["teal"],
        linewidth=2.2,
    )
    ax_att.set_ylabel("Attention score", color=colours["graphite"])
    _style_axis(ax_att, colours)
    ax_att.set_title(f"{protein_name} · Attention & GNNExplainer profile", color=colours["graphite"], pad=18)

    ax_imp = axes[1]
    ax_imp.fill_between(
        importance_df["position"],
        importance_df["node_importance"],
        color=colours["sand"],
        alpha=0.6,
        linewidth=0,
    )
    ax_imp.plot(
        importance_df["position"],
        importance_df["node_importance"],
        color=colours["bronze"],
        linewidth=2.0,
    )
    ax_imp.set_ylabel("Node importance", color=colours["graphite"])
    _style_axis(ax_imp, colours)

    ax_domains = axes[2]
    sequence_start = int(attention_df["position"].iloc[0])
    sequence_end = int(attention_df["position"].iloc[-1])
    _plot_domains(
        ax_domains,
        colours,
        sequence_start,
        sequence_end,
        annotations,
        active_site_data=active_site_residues,
    )
    ax_domains.set_xlabel("Residue position", color=colours["graphite"])
    _style_axis(ax_domains, colours, show_ticks=False)

    plt.tight_layout(h_pad=2.2, pad=1.6)
    for ax in axes:
        ax.set_facecolor("#FFFFFF")
    fig.savefig(chart_lines_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)

    fig_cm, ax_cm = plt.subplots(figsize=(6.5, 5.5))
    teal_cmap = mcolors.LinearSegmentedColormap.from_list(
        "contact_teal",
        ["#ffffff", colours["pearl"], colours["slate"], colours["teal"]],
    )
    im = ax_cm.imshow(contact_map, cmap=teal_cmap, origin="lower", aspect="equal")
    ax_cm.set_xlabel("Residue index", color=colours["graphite"])
    ax_cm.set_ylabel("Residue index", color=colours["graphite"])
    ax_cm.tick_params(axis="both", colors=colours["graphite"])
    for spine in ax_cm.spines.values():
        spine.set_color("#c0c2c1")
        spine.set_linewidth(0.8)
    ax_cm.set_facecolor("#FFFFFF")
    cbar = fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Contact probability", color=colours["graphite"])
    cbar.ax.tick_params(color=colours["graphite"], labelcolor=colours["graphite"])
    cbar.outline.set_edgecolor("#c0c2c1")
    cbar.outline.set_linewidth(0.8)
    fig_cm.tight_layout()
    fig_cm.savefig(chart_contact_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig_cm)

    fig_alt, ax_alt = plt.subplots(figsize=(6.5, 5.5))
    diverging_cmap = mcolors.LinearSegmentedColormap.from_list(
        "contact_diverging",
        ["#b7d4f4", "#ffffff", "#d04f46"],
    )
    vmax_value = max(0.6, float(contact_map.max()))
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=0.2, vmax=vmax_value)
    im_alt = ax_alt.imshow(contact_map, cmap=diverging_cmap, norm=norm, origin="lower", aspect="equal")
    ax_alt.set_xlabel("Residue index", color=colours["graphite"])
    ax_alt.set_ylabel("Residue index", color=colours["graphite"])
    ax_alt.tick_params(axis="both", colors=colours["graphite"])
    for spine in ax_alt.spines.values():
        spine.set_color("#c0c2c1")
        spine.set_linewidth(0.8)
    ax_alt.set_facecolor("#FFFFFF")
    cbar_alt = fig_alt.colorbar(im_alt, ax=ax_alt, fraction=0.046, pad=0.04)
    cbar_alt.ax.set_ylabel("Contact probability", color=colours["graphite"])
    cbar_alt.ax.tick_params(color=colours["graphite"], labelcolor=colours["graphite"])
    cbar_alt.outline.set_edgecolor("#c0c2c1")
    cbar_alt.outline.set_linewidth(0.8)
    fig_alt.tight_layout()
    fig_alt.savefig(chart_contact_alt_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig_alt)

    return chart_lines_path, chart_contact_path, chart_contact_alt_path


def _style_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.edgecolor": "#c0c2c1",
            "axes.linewidth": 0.8,
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "figure.facecolor": "#FFFFFF",
        }
    )


def _style_axis(ax, colours, show_ticks: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    if show_ticks:
        ax.tick_params(axis="both", which="both", colors=colours["graphite"])
    else:
        ax.tick_params(axis="both", which="both", length=0, labelleft=False, labelright=False)


def _plot_domains(
    ax,
    colours,
    sequence_start: int,
    sequence_end: int,
    annotations: Optional[RepresentativeAnnotations] = None,
    max_position: Optional[int] = None,
    active_site_data: Optional[Dict[int, List[int]]] = None,
) -> None:
    annotations_map = annotations or {}

    sequence_limit = sequence_end if sequence_end is not None else 1
    try:
        sequence_limit = int(sequence_limit)
    except (TypeError, ValueError):
        sequence_limit = 1
    sequence_origin = max(1, int(sequence_start))
    if max_position is None:
        limit = max(sequence_origin, sequence_limit)
    else:
        limit = max(sequence_origin, min(sequence_limit, max_position))

    lane_height = 0.22
    lane_gap = 0.08

    preferred_order = ["domain", "homologous_superfamily", "family"]
    ordered_categories = [cat for cat in preferred_order if annotations_map.get(cat)]
    remaining_categories = sorted(
        cat for cat in annotations_map.keys() if cat not in preferred_order and annotations_map.get(cat)
    )
    ordered_categories.extend(remaining_categories)

    section_configs: List[dict] = []
    for category in ordered_categories:
        intervals = annotations_map.get(category, [])
        if not intervals:
            continue
        style = CATEGORY_STYLES.get(category, DEFAULT_CATEGORY_STYLE)
        section_configs.append(
            {
                "category": category,
                "title": _format_category_label(category),
                "intervals": intervals,
                "palette": style["palette"],
                "dedicated": style["dedicated"],
            }
        )

    plotted_sections: List[dict] = []
    total_height = lane_gap
    for config in section_configs:
        tracks = _stack_intervals(config["intervals"], limit, dedicated=config["dedicated"])
        track_count = len(tracks)
        section_height = track_count * lane_height + max(track_count - 1, 0) * lane_gap if track_count else 0.0
        start_y = total_height
        if track_count:
            total_height += section_height + lane_gap
        plotted_sections.append(
            {
                "title": config["title"],
                "tracks": tracks,
                "palette": config["palette"],
                "start_y": start_y,
            }
        )

    active_sections: list[dict] = []
    if active_site_data:
        for pocket_id in sorted(active_site_data):
            residues = sorted(
                {
                    pos
                    for pos in (int(residue) for residue in active_site_data[pocket_id])
                    if 1 <= pos <= limit
                }
            )
            if not residues:
                continue
            active_sections.append(
                {
                    "pocket": pocket_id,
                    "residues": residues,
                    "start_y": total_height,
                }
            )
            total_height += lane_height + lane_gap

    ax.set_ylim(0, max(total_height, 1.0))
    ax.set_xlim(sequence_origin, limit)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#c0c2c1")
        spine.set_linewidth(0.8)

    legend_entries: list[tuple[str, str]] = []
    for section in plotted_sections:
        entries = _draw_track_section(
            ax=ax,
            tracks=section["tracks"],
            start_y=section["start_y"],
            lane_height=lane_height,
            lane_gap=lane_gap,
            palette=section["palette"],
        )
        legend_entries.extend(entries)

    if active_sections:
        for idx, section in enumerate(active_sections):
            colour = ACTIVE_SITE_COLOURS[idx % len(ACTIVE_SITE_COLOURS)]
            label = f"Pocket {section['pocket']}"
            entries = _draw_active_site_lane(
                ax=ax,
                residues=section["residues"],
                baseline=section["start_y"],
                lane_height=lane_height,
                colour=colour,
                label=label,
            )
            legend_entries.extend(entries)

    if legend_entries:
        handles = [
            Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none", alpha=0.9, label=label)
            for label, color in legend_entries
        ]
        ncols = min(2, len(handles))
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.35),
            frameon=False,
            ncol=ncols,
            columnspacing=1.8,
            handletextpad=0.8,
        )


def _load_representative_annotations(
    annotation_path: Optional[Path | str],
    target_name: Optional[str] = None,
    seq_offset: int = 0,
    seq_length: Optional[int] = None,
) -> Optional[RepresentativeAnnotations]:
    if annotation_path is None:
        return None

    path = Path(annotation_path)
    with path.open() as f:
        payload = json.load(f)

    results = payload.get("results") or []
    if not results:
        return {}

    target_result = None
    if target_name:
        normalized = target_name.strip().lower()
        for result in results:
            identifiers = []
            xrefs = result.get("xref") or []
            for xref in xrefs:
                for key in ("name", "id"):
                    value = xref.get(key)
                    if value:
                        identifiers.append(value.strip().lower())
            if any(identifier == normalized for identifier in identifiers):
                target_result = result
                break
    if target_result is None:
        if target_name and target_name.strip():
            return {}
        target_result = results[0]

    buckets: Dict[str, List[RepresentativeInterval]] = defaultdict(list)
    region_start = max(1, int(seq_offset) + 1)
    region_end = region_start + int(seq_length) - 1 if seq_length is not None else None

    matches = target_result.get("matches", [])
    for match in matches:
        signature = match.get("signature") or {}
        bucket_name = _determine_annotation_bucket(signature)
        target_bucket = buckets[bucket_name]
        entry = signature.get("entry") or {}
        label = entry.get("name") or signature.get("description") or signature.get("name") or signature.get("accession")
        label = label or "Unknown"
        locations = match.get("locations") or []
        for location in locations:
            if not location.get("representative"):
                continue
            fragments = location.get("location-fragments") or []
            if fragments:
                for fragment in fragments:
                    start = fragment.get("start")
                    end = fragment.get("end")
                    if start is None or end is None:
                        continue
                    adj_start = int(start)
                    adj_end = int(end)
                    if region_end is not None:
                        if adj_end < region_start or adj_start > region_end:
                            continue
                        adj_start = max(region_start, adj_start)
                        adj_end = min(region_end, adj_end)
                    target_bucket.append((adj_start, adj_end, label))
            else:
                start = location.get("start")
                end = location.get("end")
                if start is None or end is None:
                    continue
                adj_start = int(start)
                adj_end = int(end)
                if region_end is not None:
                    if adj_end < region_start or adj_start > region_end:
                        continue
                    adj_start = max(region_start, adj_start)
                    adj_end = min(region_end, adj_end)
                target_bucket.append((adj_start, adj_end, label))

    for bucket_values in buckets.values():
        bucket_values.sort(key=lambda item: item[0])
    return dict(buckets)


def _determine_annotation_bucket(signature: Dict) -> str:
    entry = signature.get("entry") or {}
    entry_type = (entry.get("type") or "").strip().lower()
    signature_type = (signature.get("type") or "").strip().lower()
    library = (signature.get("signatureLibraryRelease") or {}).get("library", "").strip().lower()

    if entry_type == "family" or signature_type == "family" or library.startswith("panther"):
        return "family"
    if entry_type == "homologous_superfamily" or signature_type == "homologous_superfamily":
        return "homologous_superfamily"
    if entry_type == "domain" or signature_type == "domain":
        return "domain"
    if entry_type:
        return entry_type
    if signature_type:
        return signature_type
    if library:
        return library
    return "other"


def _load_active_site_residues(
    zip_path: Optional[Path | str],
    pockets: Optional[List[int]] = None,
) -> Optional[Dict[int, List[int]]]:
    if zip_path is None:
        return None

    path = Path(zip_path)
    if not path.exists():
        return None

    pocket_set: set[int] = set()
    if pockets:
        for value in pockets:
            try:
                pocket_set.add(int(value))
            except (TypeError, ValueError):
                continue
    if not pocket_set:
        pocket_set = {1}
    if not pocket_set:
        pocket_set = {1}

    residues_by_pocket: Dict[int, List[int]] = defaultdict(list)

    try:
        with zipfile.ZipFile(path) as archive:
            candidates = [
                n for n in archive.namelist()
                if n.endswith(("structure.cif_residues.csv", "structure.pdb_residues.csv"))
            ]
            target_name = candidates[0] if candidates else None
            if target_name is None:
                return None


            with archive.open(target_name) as handle:
                reader = csv.DictReader(io.TextIOWrapper(handle))
                for raw_row in reader:
                    row = { (k.strip() if k else k): (v.strip() if isinstance(v, str) else v) for k, v in raw_row.items() }
                    pocket_value = row.get("pocket")
                    try:
                        pocket_flag = int(float(pocket_value))
                    except (TypeError, ValueError):
                        continue
                    if pocket_flag not in pocket_set:
                        continue
                    label = row.get("residue_label") or row.get("residue_number")
                    try:
                        residue_index = int(label)
                    except (TypeError, ValueError):
                        continue
                    residues_by_pocket[pocket_flag].append(residue_index)
                if not residues_by_pocket:
                    return None
                return {
                    pocket: sorted(dict.fromkeys(indices))
                    for pocket, indices in residues_by_pocket.items()
                }
    except (zipfile.BadZipFile, FileNotFoundError):
        return None



def _draw_track_section(
    ax,
    tracks: List[List[RepresentativeInterval]],
    start_y: float,
    lane_height: float,
    lane_gap: float,
    palette: List[str],
    legend_prefix: Optional[str] = None,
) -> list[tuple[str, str]]:
    legend_entries: list[tuple[str, str]] = []
    if not tracks:
        return legend_entries

    color_map: Dict[str, str] = {}
    for idx, track in enumerate(tracks):
        baseline = start_y + idx * (lane_height + lane_gap)
        for start_i, end_i, label in track:
            colour = color_map.get(label)
            if colour is None:
                colour = palette[len(color_map) % len(palette)]
                color_map[label] = colour
                legend_label = f"{legend_prefix}: {label}" if legend_prefix else label
                legend_entries.append((legend_label, colour))
            width = end_i - start_i + 1
            rect = Rectangle(
                (start_i, baseline),
                width,
                lane_height,
                facecolor=colour,
                alpha=0.9,
                edgecolor="none",
            )
            ax.add_patch(rect)
    return legend_entries


def _stack_intervals(
    intervals: List[RepresentativeInterval],
    limit: int,
    dedicated: bool = False,
) -> List[List[RepresentativeInterval]]:
    clipped: List[RepresentativeInterval] = []
    for start, end, label in intervals:
        start_i = max(1, int(start))
        end_i = min(limit, int(end))
        if end_i < start_i:
            continue
        clipped.append((start_i, end_i, label))

    clipped.sort(key=lambda item: (item[0], -item[1]))

    if dedicated:
        return [[interval] for interval in clipped]

    tracks: List[List[RepresentativeInterval]] = []
    track_ends: List[int] = []

    for interval in clipped:
        placed = False
        for idx, last_end in enumerate(track_ends):
            if interval[0] > last_end:
                tracks[idx].append(interval)
                track_ends[idx] = interval[1]
                placed = True
                break
        if not placed:
            tracks.append([interval])
            track_ends.append(interval[1])

    return tracks


def _draw_active_site_lane(
    ax,
    residues: List[int],
    baseline: float,
    lane_height: float,
    colour: str,
    label: str,
) -> list[tuple[str, str]]:
    if not residues:
        return []
    unique_positions = sorted(dict.fromkeys(residues))
    for residue in unique_positions:
        start = residue - 0.5
        rect = Rectangle(
            (start, baseline),
            1.0,
            lane_height,
            facecolor=colour,
            alpha=0.95,
            edgecolor="none",
        )
        ax.add_patch(rect)
    return [(label, colour)]


def _format_category_label(category: str) -> str:
    words = category.replace("_", " ").strip().lower()
    if not words:
        words = "feature"
    plural = _pluralize_last_word(words)
    return f"Representative {plural.title()}"


def _pluralize_last_word(phrase: str) -> str:
    tokens = phrase.split()
    if not tokens:
        return "features"
    last = tokens[-1]
    tokens[-1] = _pluralize_word(last)
    return " ".join(tokens)


def _pluralize_word(word: str) -> str:
    if not word:
        return "features"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("f"):
        return word[:-1] + "ves"
    if word.endswith("fe"):
        return word[:-2] + "ves"
    return word + "s"
