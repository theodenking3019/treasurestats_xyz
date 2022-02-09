import numpy as np

title = "Treasure NFT Sales"
meta_tags = [{
    'name': 'viewport', 
    'content': 'width=device-width, initial-scale=1.0'
    }]
html_header = """
<!DOCTYPE html>
<html>
    <head>
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-DR5F0RFK4C"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());
        
          gtag('config', 'G-DR5F0RFK4C');
        </script>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""
lookback_window_options = {
    '1 day': 1,
    '7 day': 7,
    '30 day': 30,
    'All time': 100000
}
date_interval_options = {
    '15 min': '15min',
    '30 min': '30min',
    '1 hour': '1h',
    '6 hour': '6h',
    '12 hour': '12h',
    '1 day': '1d'
}
plot_color_palette = [
    '#ff0063',
    '#8601fe',
    '#05ff9c',
    '#fefe00',
    '#1601ff',
]
dropdown_style = {
    'color':'#FFFFFF',
    'background-color':'#374251', 
    'border-color':'rgb(229 231 235)', 
    'border-radius':'0.375rem',
    'white-space': 'nowrap'
}
pricing_unit_options = {
    'MAGIC': 'sale_amt_magic',
    'USD': 'sale_amt_usd',
    'ETH': 'sale_amt_eth'
}
collection_attributes = {
    'treasures': ['nft_subcategory'],
    'smol_brains': ['gender','body','hat','glasses','mouth','clothes', 'is_one_of_one'], 
    'legacy_legions_genesis': ['nft_subcategory'],
    'smol_cars': ['background','base_color','spots','tire_color','window_color','tip_color','lights_color','door_color','wheel_color', 'is_one_of_one'],
    'life': [np.nan],
    'smol_brains_land': [np.nan],
    'smol_bodies': ['gender','background', 'body','clothes','feet','hands','head'],
    'quest_keys': [np.nan],
    'legacy_legions': ['nft_subcategory'],
    'extra_life': [np.nan],
    'smol_brains_pets': [np.nan],
    'smol_bodies_pets': [np.nan],
    'legions': [np.nan],
    'consumables': ['nft_subcategory']
}