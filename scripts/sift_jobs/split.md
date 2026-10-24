| script | jobs | GPU-hours (est., H100) | jobs |
|---|---|---|---|
| gpu_00.sh | 2 | 0.40 | gas-drift_R30_c1_s0 tabsift, is_this_a_good_customer_R30_c3_s0 metasift |
| gpu_01.sh | 2 | 0.40 | gas-drift_R30_c3_s0 tabsift, is_this_a_good_customer_R30_c5_s0 metasift |
| gpu_02.sh | 2 | 0.37 | gas-drift_R30_c5_s0 tabsift, is_this_a_good_customer_R30_c1_s0 tabsift |
| gpu_03.sh | 3 | 0.39 | coil2000_insurance_policies_R30_c1_s0 tabsift, students_dropout_R30_c5_s0 tabsift, is_this_a_good_customer_R30_c1_s0 metasift |
| gpu_04.sh | 3 | 0.40 | coil2000_insurance_policies_R30_c3_s0 tabsift, students_dropout_R30_c1_s0 metasift, ecommerce_shipping_R30_c3_s0 tabsift |
| gpu_05.sh | 3 | 0.40 | coil2000_insurance_policies_R30_c5_s0 tabsift, students_dropout_R30_c3_s0 metasift, ecommerce_shipping_R30_c5_s0 tabsift |
| gpu_06.sh | 4 | 0.40 | gas-drift_R30_c1_s0 metasift, heloc_R30_c3_s0 tabsift, students_dropout_R30_c1_s0 tabsift, ecommerce_shipping_R30_c5_s0 metasift |
| gpu_07.sh | 4 | 0.39 | gas-drift_R30_c3_s0 metasift, heloc_R30_c5_s0 tabsift, students_dropout_R30_c3_s0 tabsift, ecommerce_shipping_R30_c1_s0 tabsift |
| gpu_08.sh | 4 | 0.39 | gas-drift_R30_c5_s0 metasift, heloc_R30_c1_s0 metasift, heloc_R30_c5_s0 metasift, ecommerce_shipping_R30_c3_s0 metasift |
| gpu_09.sh | 4 | 0.38 | coil2000_insurance_policies_R30_c1_s0 metasift, coil2000_insurance_policies_R30_c5_s0 metasift, students_dropout_R30_c5_s0 metasift, is_this_a_good_customer_R30_c3_s0 tabsift |
| gpu_10.sh | 5 | 0.40 | coil2000_insurance_policies_R30_c3_s0 metasift, heloc_R30_c1_s0 tabsift, heloc_R30_c3_s0 metasift, ecommerce_shipping_R30_c1_s0 metasift, is_this_a_good_customer_R30_c5_s0 tabsift |

36 jobs, about 4.3 H100-hours in all, 0.40 on the busiest script (on a GPU that another busy job shares, two to four times that).
