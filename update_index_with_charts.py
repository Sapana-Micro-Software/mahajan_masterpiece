"""
Update index.html to include benchmark visualizations inline.
This ensures the charts work even without Jekyll includes.
"""

import re
import os

def update_index_html():
    """Update index.html to include benchmark charts inline."""
    
    # Read current index.html
    with open('index.html', 'r') as f:
        content = f.read()
    
    # Check if benchmark charts exist
    charts_path = '_includes/benchmark_charts.html'
    if os.path.exists(charts_path):
        with open(charts_path, 'r') as f:
            charts_html = f.read()
        
        # Replace the placeholder with actual charts
        placeholder = r'<div id="benchmark-charts-container"></div>.*?</script>'
        replacement = f'<div id="benchmark-charts-container">{charts_html}</div>'
        
        content = re.sub(
            r'<div id="benchmark-charts-container"></div>\s*<script>.*?</script>',
            replacement,
            content,
            flags=re.DOTALL
        )
        
        print("✅ Updated index.html with benchmark charts")
    else:
        print("⚠️  Benchmark charts not found, keeping placeholder")
    
    # Write updated content
    with open('index.html', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    update_index_html()
