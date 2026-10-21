# Training runs

Forest-mask IoU for the forest class at threshold 0.5. "All-forest" is the IoU a model would get by predicting forest everywhere (the forest share of the pixels), which is the bar to beat in mostly-forest regions. Labels: ESA WorldCover 2021 (class 10). Test region Hasdeo is never used for training or model selection.

| Run | Train regions | Hasdeo Jan 2021 | Hasdeo Apr 2021 | Hasdeo overall | Western Ghats (val) |
|---|---|---:|---:|---:|---:|
| all-forest baseline | – | 0.618 | 0.618 | 0.618 | 0.787 |
| [run 1](run1_metrics.json) | Rondônia, Mato Grosso, Prey Lang | 0.868 | 0.157 | 0.526 | 0.932 |
| **[run 2](run2_metrics.json)** (current) | + Kanha (Jan, Apr) | **0.879** | **0.731** | **0.806** | **0.933** |

Both runs: Kaggle T4, 30 epochs, batch 16, about 10 s/epoch, checkpoint chosen by best validation IoU (run 1: epoch 6, run 2: epoch 30). ONNX parity with PyTorch: max |diff| 9.5e-7. Re-scoring the downloaded ONNX model locally (`python -m training.evaluate`) reproduces every number above.

**What run 1 taught us.** On leaf-on imagery the model already exceeded the 0.85 target on unseen Hasdeo, but it found only 16 % of the forest in April, when Hasdeo's dry deciduous sal is leafless; no training region had that forest type. Adding Kanha (central Indian sal, January and April) lifted April from 0.157 to 0.731 without hurting January or the validation region.

**Where it stands against the target (IoU > 0.85).** Met on leaf-on imagery (Jan 0.879); not met on leaf-off imagery (Apr 0.731), so the two-month average is 0.806. Leaf-off canopy is genuinely ambiguous (WorldCover's "tree cover" includes 10 %+ canopy), so the monitoring pipeline uses leaf-on months for Indian AOIs and compares the same month year-on-year rather than consecutive months.
