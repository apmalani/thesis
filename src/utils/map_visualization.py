"""
Utilities to visualize best redistricting maps.
"""

from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from utils.paths import get_data_dir, get_outputs_dir


def visualize_best_map(
    state: str,
    map_file: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    show: bool = True
) -> Optional[Path]:
    """
    Visualize a redistricting map, optionally side-by-side with the baseline.

    Args:
        state: State abbreviation (e.g., 'az')
        map_file: Path to CSV file with district assignments. If None, uses latest.
        figsize: Figure size for the plot
        show: Whether to display the plot

    Returns:
        Path to saved visualization PNG, or None on failure.
    """
    data_dir = get_data_dir(state, 'processed')

    possible_shapefiles = [
        f"{state}_precincts.shp",
        "precincts_with_vap.shp",
        "precincts_with_districts.shp"
    ]

    shapefile = None
    for sf_name in possible_shapefiles:
        sf_path = data_dir / sf_name
        if sf_path.exists():
            shapefile = sf_path
            break

    if shapefile is None:
        return None

    gdf = gpd.read_file(shapefile)
    gdf = gdf.reset_index(drop=True)
    gdf['graph_node_id'] = gdf.index.astype(str)

    if map_file is None:
        best_maps_dir = get_outputs_dir(state, 'best_maps')
        map_files = list(best_maps_dir.glob("best_map_*.csv"))
        if not map_files:
            return None
        map_file = max(map_files, key=lambda p: p.stat().st_mtime)
    else:
        map_file = Path(map_file)
        if not map_file.exists():
            return None

    assignments = pd.read_csv(map_file)
    merge_key_shapefile = 'graph_node_id'
    merge_key_csv = 'precinct_id'

    if merge_key_csv not in assignments.columns:
        return None

    gdf[merge_key_shapefile] = gdf[merge_key_shapefile].astype(str)
    assignments[merge_key_csv] = assignments[merge_key_csv].astype(str)

    gdf = gdf.merge(assignments, left_on=merge_key_shapefile, right_on=merge_key_csv, how='left')

    if 'district' not in gdf.columns:
        return None

    if gdf['district'].isna().all():
        return None

    score_str = "N/A"
    if 'score' in map_file.stem:
        try:
            parts = map_file.stem.split('_score')
            if len(parts) > 1:
                score_part = parts[1].replace('p', '.')
                score_str = score_part
        except Exception:
            pass

    has_current_map = 'CONG_DIST' in gdf.columns

    if has_current_map:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None

    gdf_best = gdf[gdf['district'].notna()].copy()
    if len(gdf_best) == 0:
        return None

    gdf_best['district'] = pd.to_numeric(gdf_best['district'], errors='coerce')
    best_districts = sorted(gdf_best['district'].dropna().unique())

    if has_current_map:
        current_districts = sorted(gdf['CONG_DIST'].dropna().unique())
        all_districts = sorted(set(best_districts + [int(d) for d in current_districts if pd.notna(d)]))
        n_districts = len(all_districts)
    else:
        n_districts = len(best_districts)
        all_districts = best_districts

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    if n_districts <= 10:
        cmap = plt.colormaps['tab10']
    elif n_districts <= 20:
        cmap_base = plt.colormaps['tab20']
        colors = [cmap_base(i / max(n_districts - 1, 1)) for i in range(n_districts)]
        cmap = ListedColormap(colors)
    else:
        cmap = plt.colormaps['tab20']

    if isinstance(cmap, ListedColormap):
        colors_list = cmap.colors
    else:
        colors_list = [cmap(i / max(n_districts - 1, 1)) for i in range(n_districts)]

    district_to_color_idx = {d: i for i, d in enumerate(all_districts)}

    if has_current_map:
        gdf_current = gdf.copy()
        gdf_current['CONG_DIST_num'] = pd.to_numeric(gdf_current['CONG_DIST'], errors='coerce')
        gdf_current.plot(column='CONG_DIST_num', cmap=cmap, ax=ax1,
                        edgecolor='black', linewidth=0.1, legend=False,
                        vmin=min(all_districts), vmax=max(all_districts))
        ax1.set_title(f"{state.upper()} Current Map", fontsize=14, fontweight='bold')
        ax1.axis('off')

    gdf_best['district_num'] = pd.to_numeric(gdf_best['district'], errors='coerce')
    gdf_best.plot(column='district_num', cmap=cmap, ax=ax2 if ax2 is not None else ax1,
                  edgecolor='black', linewidth=0.1, legend=False,
                  vmin=min(all_districts), vmax=max(all_districts))

    best_title = f"{state.upper()} Best Map - Score: {score_str}"
    if ax2 is not None:
        ax2.set_title(best_title, fontsize=14, fontweight='bold')
        ax2.axis('off')
    else:
        ax1.set_title(best_title, fontsize=14, fontweight='bold')
        ax1.axis('off')

    legend_elements = [
        Patch(facecolor=colors_list[district_to_color_idx.get(d, 0)],
             edgecolor='black', label=f'District {int(d)}')
        for d in all_districts
    ]

    legend_ax = ax2 if ax2 is not None else ax1
    legend_ax.legend(handles=legend_elements, title='District',
                     bbox_to_anchor=(1.02, 1), loc='upper left',
                     frameon=True, fancybox=True, shadow=True,
                     ncol=1 if n_districts <= 15 else 2)

    plt.tight_layout()

    output_dir = get_outputs_dir(state, 'best_maps')
    output_file = output_dir / f"{map_file.stem}_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return output_file
