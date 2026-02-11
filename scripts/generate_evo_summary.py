#!/usr/bin/env python3
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

base = Path(__file__).parent.parent
csv_path = base / 'results' / 'trajectory_metrics_evo.csv'
pdf_path = base / 'results' / 'trajectory_metrics_evo_summary.pdf'

rows = []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

headers = ['Method', 'ATE RMSE (m)', 'RTE@2m RMSE (m)', 'RTE@5m RMSE (m)', 'RTE@10m RMSE (m)']
cell_text = []
for r in rows:
    cell_text.append([
        r['method'],
        f"{float(r['ate_evo_rmse']):.3f}",
        f"{float(r['rte2_evo_rmse']):.3f}",
        f"{float(r['rte5_evo_rmse']):.3f}",
        f"{float(r['rte10_evo_rmse']):.3f}"
    ])

fig, ax = plt.subplots(figsize=(11.7, 8.27))
ax.axis('off')
ax.text(0.5, 0.95, 'EVO Metrics Summary (RMSE)', ha='center', va='center', fontsize=14, weight='bold')

table = ax.table(cellText=cell_text, colLabels=headers, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

pp = PdfPages(pdf_path)
pp.savefig(fig, bbox_inches='tight')
pp.close()
plt.close(fig)
print(f"Wrote {pdf_path}")
