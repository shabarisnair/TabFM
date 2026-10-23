| script | jobs | GPU-hours (est., H100) | jobs |
|---|---|---|---|
| gpu_00.sh | 2 | 0.73 | gas-drift_R30_c1_s0 tabsift, is_this_a_good_customer_R30_c1_s0 metasift |
| gpu_01.sh | 2 | 0.73 | gas-drift_R30_c3_s0 tabsift, is_this_a_good_customer_R30_c3_s0 metasift |
| gpu_02.sh | 2 | 0.73 | gas-drift_R30_c5_s0 tabsift, is_this_a_good_customer_R30_c5_s0 metasift |
| gpu_03.sh | 4 | 0.72 | coil2000_insurance_policies_R30_c1_s0 tabsift, ecommerce_shipping_R30_c1_s0 tabsift, students_dropout_R30_c1_s0 metasift, is_this_a_good_customer_R30_c1_s0 tabsift |
| gpu_04.sh | 4 | 0.72 | coil2000_insurance_policies_R30_c3_s0 tabsift, ecommerce_shipping_R30_c3_s0 tabsift, students_dropout_R30_c3_s0 metasift, is_this_a_good_customer_R30_c3_s0 tabsift |
| gpu_05.sh | 4 | 0.72 | coil2000_insurance_policies_R30_c5_s0 tabsift, ecommerce_shipping_R30_c5_s0 tabsift, students_dropout_R30_c5_s0 metasift, is_this_a_good_customer_R30_c5_s0 tabsift |
| gpu_06.sh | 6 | 0.74 | gas-drift_R30_c1_s0 metasift, heloc_R30_c1_s0 tabsift, coil2000_insurance_policies_R30_c1_s0 metasift, students_dropout_R30_c1_s0 tabsift, heloc_R30_c1_s0 metasift, ecommerce_shipping_R30_c1_s0 metasift |
| gpu_07.sh | 6 | 0.74 | gas-drift_R30_c3_s0 metasift, heloc_R30_c3_s0 tabsift, coil2000_insurance_policies_R30_c3_s0 metasift, students_dropout_R30_c3_s0 tabsift, heloc_R30_c3_s0 metasift, ecommerce_shipping_R30_c3_s0 metasift |
| gpu_08.sh | 6 | 0.74 | gas-drift_R30_c5_s0 metasift, heloc_R30_c5_s0 tabsift, coil2000_insurance_policies_R30_c5_s0 metasift, students_dropout_R30_c5_s0 tabsift, heloc_R30_c5_s0 metasift, ecommerce_shipping_R30_c5_s0 metasift |

36 jobs, about 6.6 H100-hours in all, 0.74 on the busiest script (on a GPU that another busy job shares, two to four times that).
