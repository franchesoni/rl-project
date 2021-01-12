### [direct] ###
# python3 main.py --config-file direct/config_1_direct.json
# python3 main.py --config-file direct/config_2_direct.json
# python3 main.py --config-file direct/config_3_direct.json
# python3 main.py --config-file direct/config_4_direct.json
# python3 main.py --config-file direct/config_5_direct.json
# python3 main.py --config-file direct/config_6_direct.json
# python3 main.py --config-file direct/config_7_direct.json
# python3 main.py --config-file direct/config_8_direct.json
# python3 main.py --config-file direct/config_9_direct.json

### [baseline]
# python3 main.py --config-file baseline/config_1_baseline.json
# python3 main.py --config-file baseline/config_2_baseline.json
# python3 main.py --config-file baseline/config_3_baseline.json
# python3 main.py --config-file baseline/config_4_baseline.json
# python3 main.py --config-file baseline/config_5_baseline.json
# python3 main.py --config-file baseline/config_6_baseline.json
# python3 main.py --config-file baseline/config_7_baseline.json
# python3 main.py --config-file baseline/config_8_baseline.json
# python3 main.py --config-file baseline/config_9_baseline.json

### [combined]
# python3 main.py --config-file combined/config_1_combined.json
# python3 main.py --config-file combined/config_2_combined.json
# python3 main.py --config-file combined/config_3_combined.json
# python3 main.py --config-file combined/config_4_combined.json
# python3 main.py --config-file combined/config_5_combined.json
# python3 main.py --config-file combined/config_6_combined.json
# python3 main.py --config-file combined/config_7_combined.json
# python3 main.py --config-file combined/config_8_combined.json
# python3 main.py --config-file combined/config_9_combined.json

### [online]
# python3 main.py --config-file online/config_1_online.json
# python3 main.py --config-file online/config_2_online.json
# python3 main.py --config-file online/config_3_online.json
# python3 main.py --config-file online/config_4_online.json
# python3 main.py --config-file online/config_5_online.json
# python3 main.py --config-file online/config_6_online.json
# python3 main.py --config-file online/config_7_online.json
# python3 main.py --config-file online/config_8_online.json
# python3 main.py --config-file online/config_9_online.json

### [sampling]
# python3 main.py --config-file sampling/config_1_sampling.json
# python3 main.py --config-file sampling/config_2_sampling.json
# python3 main.py --config-file sampling/config_3_sampling.json
# python3 main.py --config-file sampling/config_4_sampling.json
# python3 main.py --config-file sampling/config_5_sampling.json
# python3 main.py --config-file sampling/config_6_sampling.json
# python3 main.py --config-file sampling/config_7_sampling.json
# python3 main.py --config-file sampling/config_8_sampling.json
# python3 main.py --config-file sampling/config_9_sampling.json

### [raw]
# python3 main.py --config-file raw/config_1_raw.json
# python3 main.py --config-file raw/config_2_raw.json
# python3 main.py --config-file raw/config_3_raw.json
# python3 main.py --config-file raw/config_4_raw.json
# python3 main.py --config-file raw/config_5_raw.json
# python3 main.py --config-file raw/config_6_raw.json
# python3 main.py --config-file raw/config_7_raw.json
# python3 main.py --config-file raw/config_8_raw.json
# python3 main.py --config-file raw/config_9_raw.json

### [franco1]
rm -rf runs/franco_2/
python3 main.py --config-file franco_1/config_1_franco.json
echo -e "config_1\n" >> config/franco_1/one_digit_errors1.txt
python3 main.py --config-file franco_1/config_2_franco.json
echo -e "config_2\n" >> config/franco_1/one_digit_errors2.txt
python3 main.py --config-file franco_1/config_3_franco.json
python3 main.py --config-file franco_1/config_4_franco.json

### [franco2]
rm -rf runs/franco_2/
python3 main.py --config-file franco_2/config_1_franco.json
python3 main.py --config-file franco_2/config_2_franco.json
python3 main.py --config-file franco_2/config_3_franco.json
python3 main.py --config-file franco_2/config_4_franco.json
python3 main.py --config-file franco_2/config_5_franco.json


