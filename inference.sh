#!/bin/bash
cd model

python run_SLPF.py \
--dataset=LA \
--seed=6 \
--num_unsensed_locs=150 \
--mode=test \
--adp_model_path=../runs/LA/08-07-12h34m47s_LA_embed32_lyr3_lr0.001_wd0.0003_s_6_m_150/adp_best_model.pth \
--forecast_model_path=../runs/LA/08-07-12h34m47s_LA_embed32_lyr3_lr0.001_wd0.0003_s_6_m_150/forecast_best_model.pth \
--agg_model_path=../runs/LA/08-07-12h34m47s_LA_embed32_lyr3_lr0.001_wd0.0003_s_6_m_150/agg_best_model.pth