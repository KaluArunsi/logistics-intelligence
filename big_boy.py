# --------------------------------------------------------------
# Bitcoin (BTC) daily‑close price chart – Jan 2 2026 through Mar 3 2026
# --------------------------------------------------------------
# 1) Install the required library (if you don’t have it yet):
#    pip install matplotlib pandas
# 2) Run the script – it will display the chart and save it as btc_2026.png
# --------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# ---- 1️⃣ CSV data (copy‑paste from Stooq) ---------------------------------
csv_data = """Date,Close
2026-01-02,89748
2026-01-05,94181.4
2026-01-06,93341
2026-01-07,91126.9
2026-01-08,91092.9
2026-01-09,90229
2026-01-12,91211.7
2026-01-13,95650.7
2026-01-14,96878.5
2026-01-15,95561.1
2026-01-16,95396.1
2026-01-19,92529.1
2026-01-20,88429.8
2026-01-21,89759.3
2026-01-22,89316
2026-01-23,89453.4
2026-01-26,88083.2
2026-01-27,89172.6
2026-01-28,89298.4
2026-01-29,84468.7
2026-01-30,83842.6
2026-02-02,78756.2
2026-02-03,75662.9
2026-02-04,72414.4
2026-02-05,63798.7
2026-02-06,70048.1
2026-02-09,70358.1
2026-02-10,68641.7
2026-02-11,67401.3
2026-02-12,66287.5
2026-02-13,68780.9
2026-02-16,68570.6
2026-02-17,67530.6
2026-02-18,66403.7
2026-02-19,66936.6
2026-02-20,67692.4
2026-02-23,64725.7
2026-02-24,64154.3
2026-02-25,68485.6
2026-02-26,67629.7
2026-02-27,65590.9
2026-03-02,69336.1
2026-03-03,68340
"""

# ---- 2️⃣ Load into a DataFrame -------------------------------------------
df = pd.read_csv(StringIO(csv_data), parse_dates=['Date'])

# ---- 3️⃣ Plot -------------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Close'], marker='o', color='#1f78b4', linewidth=2)

# Title & labels
plt.title('Bitcoin Daily Close Prices – 2026', fontsize=14, weight='bold')
plt.xlabel('Date (2026)', fontsize=12)
plt.ylabel('Close (USD)', fontsize=12)

# Nice grid and date formatting
plt.grid(alpha=0.3)
plt.tight_layout()

# ---- 4️⃣ Save & show ------------------------------------------------------
output_file = 'btc_2026.png'
plt.savefig(output_file, dpi=300)
print(f'Chart saved as {output_file}')
plt.show()
