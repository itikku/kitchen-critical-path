# Phase 2a — First PyTorch Training Run (generated)

Train: 1137 videos, 299,651 frames (Breakfast persons s2–s4, camera IDs other than `webcam02`).
T1: 237 videos, 59,023 frames, 13 persons (P03–P15).
T2: 15 videos, 4,334 frames, 2 persons (P03–P04).

## Macro F1 (8 classes)

| Model | T1: new people | T2: new people + unseen camera ID |
|---|---:|---:|
| `majority_class` | 0.088 | 0.082 |
| `frame_linear` | 0.221 | 0.141 |
| `temporal_head` | 0.159 | 0.206 |

## Per-class results — T1, `temporal_head` vs `frame_linear`

| Class | Support | Temporal P | Temporal R | Temporal F1 | Frame F1 |
|---|---:|---:|---:|---:|---:|
| `cut` | 5,775 | 0.29 | 0.16 | 0.21 | 0.37 |
| `peel` | 2,252 | 0.24 | 0.05 | 0.08 | 0.12 |
| `pour` | 5,294 | 0.17 | 0.24 | 0.20 | 0.22 |
| `stir` | 3,705 | 0.05 | 0.02 | 0.02 | 0.13 |
| `transfer` | 2,230 | 0.04 | 0.05 | 0.04 | 0.07 |
| `take_object` | 3,590 | 0.07 | 0.05 | 0.06 | 0.15 |
| `background` | 4,147 | 0.10 | 0.43 | 0.16 | 0.18 |
| `other` | 32,030 | 0.62 | 0.43 | 0.51 | 0.53 |

Confusion matrix (T1, `temporal_head`; rows = true, columns = predicted):

| true \ pred | cut | peel | pour | stir | transfer | take_object | background | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cut` | 941 | 137 | 788 | 43 | 173 | 188 | 1,705 | 1,800 |
| `peel` | 344 | 103 | 261 | 63 | 139 | 111 | 616 | 615 |
| `pour` | 95 | 9 | 1,286 | 137 | 252 | 226 | 1,588 | 1,701 |
| `stir` | 131 | 27 | 561 | 60 | 239 | 154 | 993 | 1,540 |
| `transfer` | 197 | 13 | 509 | 46 | 113 | 82 | 584 | 686 |
| `take_object` | 148 | 3 | 450 | 50 | 116 | 179 | 1,653 | 991 |
| `background` | 125 | 4 | 571 | 37 | 143 | 220 | 1,767 | 1,280 |
| `other` | 1,240 | 127 | 3,334 | 681 | 2,046 | 1,324 | 9,378 | 13,900 |

## Per-class results — T2, `temporal_head` vs `frame_linear`

| Class | Support | Temporal P | Temporal R | Temporal F1 | Frame F1 |
|---|---:|---:|---:|---:|---:|
| `cut` | 610 | 0.46 | 0.18 | 0.26 | 0.24 |
| `peel` | 159 | 0.00 | 0.00 | 0.00 | 0.00 |
| `pour` | 295 | 0.37 | 0.52 | 0.43 | 0.30 |
| `stir` | 285 | 0.17 | 0.10 | 0.13 | 0.01 |
| `transfer` | 165 | 0.19 | 0.15 | 0.17 | 0.00 |
| `take_object` | 269 | 0.10 | 0.01 | 0.02 | 0.10 |
| `background` | 423 | 0.14 | 0.57 | 0.23 | 0.21 |
| `other` | 2,128 | 0.48 | 0.37 | 0.42 | 0.28 |

Confusion matrix (T2, `temporal_head`; rows = true, columns = predicted):

| true \ pred | cut | peel | pour | stir | transfer | take_object | background | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cut` | 111 | 0 | 1 | 52 | 0 | 0 | 141 | 305 |
| `peel` | 3 | 0 | 0 | 42 | 0 | 0 | 12 | 102 |
| `pour` | 3 | 0 | 152 | 8 | 8 | 1 | 54 | 69 |
| `stir` | 0 | 0 | 34 | 29 | 0 | 5 | 13 | 204 |
| `transfer` | 4 | 0 | 10 | 13 | 24 | 0 | 44 | 70 |
| `take_object` | 0 | 0 | 67 | 1 | 17 | 3 | 119 | 62 |
| `background` | 5 | 0 | 83 | 22 | 25 | 4 | 243 | 41 |
| `other` | 114 | 0 | 64 | 7 | 50 | 17 | 1,089 | 787 |
