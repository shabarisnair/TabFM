| script | units | minutes (est., free H100) | gas-drift / coil2000 units |
|---|---|---|---|
| gpu_00.sh | 10 | 11 | 3 |
| gpu_01.sh | 10 | 11 | 4 |
| gpu_02.sh | 10 | 11 | 4 |
| gpu_03.sh | 11 | 11 | 3 |
| gpu_04.sh | 10 | 11 | 3 |
| gpu_05.sh | 11 | 11 | 3 |
| gpu_06.sh | 10 | 11 | 4 |

72 units, about 78 H100-minutes in all, 11 on the busiest script (a GPU shared with another busy job: two to four times that). Inputs to copy to each machine: results/vp_tune/draws (72 draws), datasets/, results/attn_form/preflight_reference, results/bench_largev/draws.
