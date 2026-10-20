| script | jobs | GPU-hours (est.) | larger-V jobs | tiny-V jobs |
|---|---|---|---|---|
| gpu_00.sh | 15 | 0.75 | 8 | 7 |
| gpu_01.sh | 11 | 0.75 | 7 | 4 |
| gpu_02.sh | 15 | 0.75 | 7 | 8 |
| gpu_03.sh | 12 | 0.75 | 4 | 8 |
| gpu_04.sh | 13 | 0.75 | 4 | 9 |

66 jobs, about 3.7 H100-hours in all, 0.75 on the busiest script. Tiny-V: soft form 4 updates at lr 0.2, M1 300 updates (M1 rerun as attn_topk_it300).
