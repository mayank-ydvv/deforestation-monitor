# Class balance: training tiles

Labels: ESA WorldCover 2021 (class 10 = forest). Tiles: 256x256 px, stride 256, dropped if > 10% NaN. Split is by region.

## Per region-month

| Region | Split | Month | Scenes | Clear | Tiles kept | Forest (tiles) | Forest (AOI) |
|---|---|---|---:|---:|---:|---:|---:|
| hasdeo | test | 2021-01 | 6 | 100.0% | 132/132 | 61.8% | 62.6% |
| hasdeo | test | 2021-04 | 6 | 100.0% | 132/132 | 61.8% | 62.6% |
| kalimantan | train | 2021-06 | 4 | 39.4% | 51/156 | 91.9% | 87.7% |
| kalimantan | train | 2021-07 | 6 | 40.7% | 57/156 | 90.5% | 87.7% |
| mato_grosso | train | 2021-06 | 6 | 100.0% | 156/156 | 38.6% | 38.8% |
| mato_grosso | train | 2021-08 | 6 | 100.0% | 156/156 | 38.6% | 38.8% |
| rondonia | train | 2021-07 | 6 | 100.0% | 255/255 | 85.1% | 84.6% |
| rondonia | train | 2021-08 | 6 | 100.0% | 255/255 | 85.1% | 84.6% |
| western_ghats | val | 2021-01 | 5 | 100.0% | 144/144 | 78.7% | 79.8% |
| western_ghats | val | 2021-02 | 5 | 100.0% | 144/144 | 78.7% | 79.8% |

## Per split

| Split | Tiles | Forest pixels | All non-forest | (0, .25] | (.25, .5] | (.5, .75] | (.75, 1) | All forest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 930 | 70.2% | 0 | 126 | 137 | 155 | 341 | 171 |
| val | 288 | 78.7% | 0 | 6 | 36 | 44 | 202 | 0 |
| test | 264 | 61.8% | 0 | 38 | 44 | 74 | 108 | 0 |
