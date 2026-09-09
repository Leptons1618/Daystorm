# Modality ablation

Each row forces one modality offline for the whole window by zeroing its
validity mask, exactly as a dead sensor would. Deltas are against `full`.

```
run                    exact match    manoeuvre accuracy           numeric mae    hallucination rate
----------------------------------------------------------------------------------------------------
full                   0.000                 1.000                 0.290                 0.042      
no_camera              0.062 (+0.06)           1.000                 0.471 (+0.18)           0.021 (-0.02)
no_can                 0.000                 1.000                 1.079 (+0.79)           0.250 (+0.21)
no_radar               0.000                 1.000                 3.460 (+3.17)           0.091 (+0.05)
no_audio               0.000                 1.000                 3.826 (+3.54)           0.140 (+0.10)
```
