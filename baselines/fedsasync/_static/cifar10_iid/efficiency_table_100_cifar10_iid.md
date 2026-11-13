# Comprehensive Performance Metrics: cifar10_iid

### Slow = 20%

| Strategy | Delta_loss/time | Final Acc. | Max Acc. | TTA 50% | TTA 58% | Acc@15s | Acc@25s |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FedSaSync_100 (M = 65) | 0.0039 $\pm$ 0.0005 | {acc_values[final_idx]:.1f} $\\pm$ {acc_std_values[final_idx]:.1f}\\\% | {acc_values[max_idx]:.1f} $\\pm$ {acc_std_values[max_idx]:.1f}\\\% | 251.5 $\pm$ 40.2s | N/A | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% |
| FedSaSync_100 (M = 75) | **0.0040 $\pm$ 0.0001** | {acc_values[final_idx]:.1f} $\\pm$ {acc_std_values[final_idx]:.1f}\\\% | {acc_values[max_idx]:.1f} $\\pm$ {acc_std_values[max_idx]:.1f}\\\% | **209.9 $\pm$ 18.7s** | N/A | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% |
| FedSaSync_100 (M = 80) | 0.0037 $\pm$ 0.0002 | {acc_values[final_idx]:.1f} $\\pm$ {acc_std_values[final_idx]:.1f}\\\% | {acc_values[max_idx]:.1f} $\\pm$ {acc_std_values[max_idx]:.1f}\\\% | 222.7 $\pm$ 9.9s | N/A | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% |
| FedSaSync_100 (M = 85) | 0.0034 $\pm$ 0.0001 | {acc_values[final_idx]:.1f} $\\pm$ {acc_std_values[final_idx]:.1f}\\\% | {acc_values[max_idx]:.1f} $\\pm$ {acc_std_values[max_idx]:.1f}\\\% | 224.7 $\pm$ 19.4s | N/A | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% |
| FedSaSync_100 (M = 95) | 0.0018 $\pm$ 0.0000 | {acc_values[final_idx]:.1f} $\\pm$ {acc_std_values[final_idx]:.1f}\\\% | {acc_values[max_idx]:.1f} $\\pm$ {acc_std_values[max_idx]:.1f}\\\% | 393.9 $\pm$ 25.0s | N/A | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% | {acc_mean:.1f} $\\pm$ {acc_std:.1f}\\\% |

---
*Note:* Superscripts indicate the number of successful runs out of 5 repetitions when not all runs finished within the target.