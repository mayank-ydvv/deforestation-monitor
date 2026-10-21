# Class balance: training tiles

Labels: ESA WorldCover 2021 (class 10 = forest). Tiles: 256x256 px, stride 256, dropped if > 10% NaN. Split is by region.

## Per region-month

| Region | Split | Month | Scenes | Clear | Tiles kept | Forest (tiles) | Forest (AOI) |
|---|---|---|---:|---:|---:|---:|---:|
| hasdeo | test | 2021-01 | 6 | 100.0% | 132/132 | 61.8% | 62.6% |
| hasdeo | test | 2021-04 | 6 | 100.0% | 132/132 | 61.8% | 62.6% |
| kanha | train | 2021-01 | 4 | 100.0% | 156/156 | 61.9% | 61.8% |
| kanha | train | 2021-04 | 5 | 100.0% | 156/156 | 61.9% | 61.8% |
| mato_grosso | train | 2021-06 | 6 | 100.0% | 156/156 | 38.6% | 38.8% |
| mato_grosso | train | 2021-08 | 6 | 100.0% | 156/156 | 38.6% | 38.8% |
| prey_lang | train | 2021-03 | 6 | 100.0% | 144/144 | 91.7% | 92.5% |
| prey_lang | train | 2021-12 | 6 | 100.0% | 144/144 | 91.7% | 92.5% |
| rondonia | train | 2021-07 | 6 | 100.0% | 255/255 | 85.1% | 84.6% |
| rondonia | train | 2021-08 | 6 | 100.0% | 255/255 | 85.1% | 84.6% |
| western_ghats | val | 2021-01 | 5 | 100.0% | 144/144 | 78.7% | 79.8% |
| western_ghats | val | 2021-02 | 5 | 100.0% | 144/144 | 78.7% | 79.8% |

## Per split

| Split | Tiles | Forest pixels | All non-forest | (0, .25] | (.25, .5] | (.5, .75] | (.75, 1) | All forest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1422 | 71.2% | 0 | 206 | 174 | 216 | 586 | 240 |
| val | 288 | 78.7% | 0 | 6 | 36 | 44 | 202 | 0 |
| test | 264 | 61.8% | 0 | 38 | 44 | 74 | 108 | 0 |
