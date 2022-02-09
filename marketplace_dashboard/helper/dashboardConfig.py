from __future__ import annotations
from typing import Dict, List

class DashboardConfig:
    def __init__(
        self,
        title: str,
        meta_tags: Dict,
        html_header: str,
        lookback_window_options: Dict,
        date_interval_options: Dict,
        pricing_unit_options: Dict,
        plot_color_palette: List[str],
        dropdown_menu_style: Dict,
        collection_attributes: Dict):

        self.title = title
        self.meta_tags = meta_tags
        self.html_header = html_header
        self.lookback_window_options = lookback_window_options
        self.date_interval_options = date_interval_options,
        self.pricing_unit_options = pricing_unit_options,
        self.plot_color_palette = plot_color_palette
        self.dropdown_menu_style = dropdown_menu_style
        self.collection_attributes = collection_attributes