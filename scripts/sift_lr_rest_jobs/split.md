| script | jobs | GPU-hours (est., H100) | jobs |
|---|---|---|---|
| gpu_00.sh | 1 | 0.35 | gas-drift_R30_c1_s0 (tabsift_sgd_lr1e4) |
| gpu_01.sh | 1 | 0.35 | gas-drift_R30_c3_s0 (tabsift_adam_lr0.5) |
| gpu_02.sh | 1 | 0.35 | gas-drift_R30_c3_s0 (tabsift_sgd_lr1e5) |
| gpu_03.sh | 1 | 0.35 | gas-drift_R30_c5_s0 (tabsift_adam_lr5) |

4 jobs, about 1.4 H100-hours in all, 0.35 on the busiest script (on a GPU that another busy job shares, two to four times that).
