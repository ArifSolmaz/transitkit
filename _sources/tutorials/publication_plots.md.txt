# Publication-Quality Plots

Create journal-ready figures for your papers.

## Setup Publication Style

```python
from transitkit.visualization import setup_publication_style

# Apply AAS journal style
setup_publication_style(style='aas', dpi=300)
```

## Transit Report Figure

```python
from transitkit.visualization import create_transit_report_figure
from transitkit.core import TransitParameters

params = TransitParameters(
    period=5.0, period_err=0.001,
    t0=2.5, t0_err=0.01,
    depth=0.01, depth_err=0.001,
    duration=0.15, duration_err=0.01,
    snr=50.0
)

fig = create_transit_report_figure(time, flux, params)
fig.savefig('transit_report.pdf', bbox_inches='tight', dpi=300)
```

## Tips for Publication

1. Use PDF format for vector graphics
2. Set DPI to 300 for raster elements
3. Use consistent fonts (serif for text)
4. Include error bars where possible
5. Label all axes with units
