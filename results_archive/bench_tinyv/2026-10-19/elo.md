# Tiny-V benchmark: TabArena-style Elo (54 tasks)

Error: 1 - ROC-AUC on the binary datasets (heloc, ecommerce, good_customer, coil2000), log loss on the multi-class ones (gas-drift, students). Task = one instance (dataset x c x R). Bradley-Terry by maximum likelihood over all pairwise battles (ties = half a win), Elo = 400 log10-odds; calibrated so poisoned + V = 1000. 95% CI: TabArena's, from 200 bootstrap resamples of the tasks. Gap to poisoned + V: the 95% interval of Elo - its Elo within each resample, over the tasks and over the 6 datasets (all 9 instances of a drawn dataset; wider, since the instances of one dataset are not independent); an interval above 0 means it beats the anchor beyond the noise. Improvability (TabArena's): mean over tasks of 1 - best error / error (%, lower is better). W-T-L: wins, ties and losses against poisoned + V.

| # | entry | Elo | 95% CI | gap to poisoned + V, over tasks | over datasets | win rate | improvability (%) | tasks | W-T-L |
|---|---|---|---|---|---|---|---|---|---|
| 1 | emlc (default) | 1020 | 981 to 1065 | -77 to +101 | -225 to +240 | 0.70 | 22.0 | 54 | 30-0-24 |
| 2 | attention weights, soft (A3) | 1013 | 955 to 1073 | -40 to +67 | -97 to +109 | 0.69 | 21.4 | 54 | 21-0-33 |
| 3 | ieg (default) | 1000 | 953 to 1056 | -73 to +71 | -205 to +166 | 0.67 | 23.2 | 54 | 26-0-28 |
| 4 | poisoned + V | 1000 | 945 to 1076 | . | . | 0.67 | 20.7 | 54 | . |
| 5 | NaN gate, top-k vote | 989 | 941 to 1066 | -53 to +32 | -71 to +64 | 0.66 | 21.6 | 54 | 22-0-32 |
| 6 | knn_shapley (default) | 957 | 913 to 1006 | -111 to +8 | -212 to +66 | 0.62 | 23.6 | 54 | 24-0-30 |
| 7 | attention weights, top-k (M1) | 955 | 921 to 997 | -131 to +26 | -269 to +132 | 0.62 | 24.8 | 54 | 22-0-32 |
| 8 | V-filter | 938 | 906 to 979 | -155 to +17 | -309 to +135 | 0.60 | 25.4 | 54 | 23-0-31 |
| 9 | knn_shapley (V-selected) | 927 | 877 to 976 | -132 to -19 | -242 to +51 | 0.58 | 25.5 | 54 | 19-0-35 |
| 10 | ieg (default, relabel) | 894 | 840 to 954 | -205 to -2 | -414 to +149 | 0.54 | 31.2 | 54 | 22-0-32 |
| 11 | A3, soft weights + V | 893 | 842 to 953 | -189 to -34 | -303 to +65 | 0.54 | 27.1 | 54 | 15-0-39 |
| 12 | greedy, 1% per step | 878 | 813 to 958 | -238 to -25 | -435 to +241 | 0.52 | 21.8 | 54 | 18-0-36 |
| 13 | greedy, 1 row per step | 876 | 800 to 957 | -234 to -15 | -438 to +223 | 0.51 | 23.3 | 54 | 20-0-34 |
| 14 | label_spread (default) | 876 | 826 to 921 | -193 to -81 | -286 to -25 | 0.51 | 28.2 | 54 | 14-0-40 |
| 15 | greedy, 10% per step | 871 | 817 to 940 | -212 to -48 | -344 to +69 | 0.51 | 26.2 | 54 | 15-0-39 |
| 16 | ieg (V-selected) | 849 | 803 to 906 | -265 to -40 | -488 to +112 | 0.48 | 35.0 | 54 | 19-0-35 |
| 17 | V only | 712 | 649 to 767 | -396 to -203 | -547 to -96 | 0.31 | 38.3 | 54 | 11-0-43 |
| 18 | emlc (V-selected) | 647 | 566 to 718 | -490 to -243 | -829 to -101 | 0.24 | 46.0 | 54 | 10-0-44 |
| 19 | emlc (default, relabel) | 642 | 553 to 712 | -512 to -244 | -868 to -91 | 0.24 | 46.5 | 54 | 10-0-44 |
| 20 | label_spread (V-selected) | 565 | 452 to 654 | -618 to -304 | -1103 to -122 | 0.17 | 52.2 | 54 | 8-0-46 |
| 21 | label_spread (default, relabel) | 484 | 372 to 573 | -658 to -411 | -1002 to -288 | 0.12 | 53.4 | 54 | 4-0-50 |

## Sensitivity

Elo under other choices, each calibrated the same way. imputed: the unfinished attention instances filled with the anchor's result (TabArena's handling of failed tasks). binary only: the 36 binary tasks (1 - AUC). multi-class only: the 18 multi-class tasks (log loss). multi-class by accuracy: as the main fit, with 1 - accuracy instead of log loss on gas-drift and students. without ecommerce: the 45 tasks left without ecommerce_shipping, where the poisoning costs nothing (oracle + V is no better than poisoned + V there).

| entry | main | imputed | binary only | multi-class only | multi-class by accuracy | without ecommerce |
|---|---|---|---|---|---|---|
| emlc (default) | 1020 | 1020 | 1052 | 956 | 1054 | 1091 |
| attention weights, soft (A3) | 1013 | 1013 | 986 | 1076 | 1021 | 1040 |
| ieg (default) | 1000 | 1000 | 1059 | 882 | 1039 | 1032 |
| poisoned + V | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| NaN gate, top-k vote | 989 | 989 | 1000 | 966 | 1024 | 1000 |
| knn_shapley (default) | 957 | 957 | 950 | 969 | 995 | 1014 |
| attention weights, top-k (M1) | 955 | 955 | 995 | 865 | 1000 | 1006 |
| V-filter | 938 | 938 | 989 | 821 | 991 | 993 |
| knn_shapley (V-selected) | 927 | 927 | 910 | 957 | 962 | 979 |
| ieg (default, relabel) | 894 | 894 | 907 | 853 | 948 | 979 |
| A3, soft weights + V | 893 | 893 | 876 | 920 | 931 | 912 |
| greedy, 1% per step | 878 | 878 | 784 | 1082 | 980 | 945 |
| greedy, 1 row per step | 876 | 876 | 820 | 990 | 984 | 923 |
| label_spread (default) | 876 | 876 | 872 | 870 | 912 | 913 |
| greedy, 10% per step | 871 | 871 | 849 | 908 | 1012 | 913 |
| ieg (V-selected) | 849 | 849 | 867 | 790 | 937 | 939 |
| V only | 712 | 712 | 656 | 792 | 674 | 798 |
| emlc (V-selected) | 647 | 647 | 716 | 227 | 717 | 722 |
| emlc (default, relabel) | 642 | 642 | 723 | 136 | 710 | 719 |
| label_spread (V-selected) | 565 | 565 | 651 | -23 | 652 | 661 |
| label_spread (default, relabel) | 484 | 484 | 519 | 145 | 545 | 581 |
