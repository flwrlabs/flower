
### Slow = 20%

| Strategy | Polling Calls | Polling Overhead (%) | Server CPU Util. (%) | Agg. Rate (msgs/s) |
| :--- | :---: | :---: | :---: | :---: |
| FedSaSync_100 (M = 65) | 5018.7 $\pm$ 952.9 | {overhead_mean:.1f} $\\pm$ {overhead_std:.1f}\\\% | {cpu_util_mean:.1f} $\\pm$ {cpu_util_std:.1f}\\\% | 8.95 $\pm$ 1.82 |
| FedSaSync_100 (M = 75) | **5011.0 $\pm$ 130.6** | {overhead_mean:.1f} $\\pm$ {overhead_std:.1f}\\\% | {cpu_util_mean:.1f} $\\pm$ {cpu_util_std:.1f}\\\% | **10.82 $\pm$ 0.37** |
| FedSaSync_100 (M = 80) | 5532.7 $\pm$ 380.5 | {overhead_mean:.1f} $\\pm$ {overhead_std:.1f}\\\% | {cpu_util_mean:.1f} $\\pm$ {cpu_util_std:.1f}\\\% | 10.45 $\pm$ 0.71 |
| FedSaSync_100 (M = 85) | 6107.0 $\pm$ 124.4 | {overhead_mean:.1f} $\\pm$ {overhead_std:.1f}\\\% | {cpu_util_mean:.1f} $\\pm$ {cpu_util_std:.1f}\\\% | 9.97 $\pm$ 0.24 |
| FedSaSync_100 (M = 95) | 13137.0 $\pm$ 108.7 | {overhead_mean:.1f} $\\pm$ {overhead_std:.1f}\\\% | {cpu_util_mean:.1f} $\\pm$ {cpu_util_std:.1f}\\\% | 5.64 $\pm$ 0.02 |
