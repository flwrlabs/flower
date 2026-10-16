# Comprehensive Performance Metrics: cifar10_dirichlet

### Slow = 20%

| Strategy | Delta_loss/time | Final Acc. | Max Acc. | TTA 50% | TTA 58% | Acc@15s | Acc@25s |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FedSaSync_100 (M = 65) | **0.0050 $\pm$ 0.0004** | {acc_values[final_idx]:.1f} $\\pm$ {acc_std_values[final_idx]:.1f}\\\% | {acc_values[max_idx]:.1f} $\\pm$ {acc_std_values[max_idx]:.1f}\\\% | **214.0 $\pm$ 15.6s** | N/A | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% |
| FedSaSync_100 (M = 75) | 0.0041 $\pm$ 0.0003 | {acc_values[final_idx]:.1f} $\\pm$ {acc_std_values[final_idx]:.1f}\\\% | {acc_values[max_idx]:.1f} $\\pm$ {acc_std_values[max_idx]:.1f}\\\% | 234.8 $\pm$ 20.5s | N/A | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% |
| FedSaSync_100 (M = 80) | 0.0038 $\pm$ 0.0002 | {acc_values[final_idx]:.1f} $\\pm$ {acc_std_values[final_idx]:.1f}\\\% | {acc_values[max_idx]:.1f} $\\pm$ {acc_std_values[max_idx]:.1f}\\\% | 224.8 $\pm$ 27.4s | N/A | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% |
| FedSaSync_100 (M = 85) | 0.0036 $\pm$ 0.0002 | {acc_values[final_idx]:.1f} $\\pm$ {acc_std_values[final_idx]:.1f}\\\% | {acc_values[max_idx]:.1f} $\\pm$ {acc_std_values[max_idx]:.1f}\\\% | 224.0 $\pm$ 23.4s | N/A | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% |
| FedSaSync_100 (M = 95) | 0.0019 $\pm$ 0.0000 | {acc_values[final_idx]:.1f} $\\pm$ {acc_std_values[final_idx]:.1f}\\\% | {acc_values[max_idx]:.1f} $\\pm$ {acc_std_values[max_idx]:.1f}\\\% | 413.6 $\pm$ 37.2s | N/A | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% |

---
*Note:* Superscripts indicate the number of successful runs out of 5 repetitions when not all runs finished within the target.