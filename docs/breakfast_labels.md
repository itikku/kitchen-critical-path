# Breakfast Actions — coarse label inventory

Generated 2026-11-01 from `segmentation_coarse.tar.gz` (SHA-256 `b9929b59…7945`, verified against the mirror's SHA256SUMS).

- 1712 annotation files (one per camera view of a session), 52 persons, 10 recipes, 48 labels.
- Format: one line per segment, `<start_frame>-<end_frame> <label>`, at 15 fps.
- Hours count every camera view, so one real-world minute appears up to 5 times (cameras: cam01, cam02, webcam01, webcam02, stereo01).
- The fine annotations (178 labels, 257 sessions) are not used in v1: they split actions into reach/carry steps that a prep-zone camera does not need.

| Label | Hours (all views) | Segments | Recipes |
|---|---:|---:|---|
| `fry_pancake` | 7.99 | 173 | pancake |
| `fry_egg` | 5.90 | 235 | friedegg |
| `cut_fruit` | 5.68 | 515 | salat |
| `SIL` | 4.88 | 3424 | cereals, coffee, friedegg, juice, milk, pancake, salat, sandwich, scrambledegg, tea |
| `stirfry_egg` | 3.95 | 234 | scrambledegg |
| `crack_egg` | 3.21 | 511 | friedegg, pancake, scrambledegg |
| `stir_dough` | 3.12 | 252 | pancake |
| `peel_fruit` | 3.03 | 223 | salat |
| `pour_milk` | 2.63 | 687 | cereals, coffee, milk, pancake |
| `squeeze_orange` | 2.51 | 197 | juice |
| `butter_pan` | 2.21 | 234 | friedegg, pancake, scrambledegg |
| `smear_butter` | 1.85 | 160 | sandwich |
| `put_egg2plate` | 1.40 | 322 | friedegg, scrambledegg |
| `add_saltnpepper` | 1.37 | 285 | friedegg, scrambledegg |
| `take_plate` | 1.34 | 431 | friedegg, juice, pancake, salat, sandwich, scrambledegg |
| `put_toppingOnTop` | 1.20 | 176 | sandwich |
| `pour_dough2pan` | 1.16 | 157 | pancake |
| `spoon_powder` | 1.08 | 195 | milk |
| `spoon_flour` | 1.00 | 156 | pancake |
| `pour_cereals` | 0.96 | 188 | cereals |
| `cut_bun` | 0.85 | 165 | sandwich |
| `put_fruit2bowl` | 0.79 | 408 | salat |
| `pour_water` | 0.77 | 184 | tea |
| `pour_coffee` | 0.76 | 167 | coffee |
| `stir_milk` | 0.76 | 187 | milk |
| `pour_juice` | 0.70 | 189 | juice |
| `add_teabag` | 0.70 | 184 | tea |
| `pour_oil` | 0.62 | 158 | friedegg, pancake, scrambledegg |
| `stir_egg` | 0.59 | 95 | scrambledegg |
| `put_pancake2plate` | 0.53 | 127 | pancake |
| `cut_orange` | 0.43 | 154 | juice |
| `pour_egg2pan` | 0.43 | 74 | scrambledegg |
| `take_cup` | 0.40 | 212 | coffee, milk, tea |
| `take_bowl` | 0.38 | 192 | cereals, pancake, salat, scrambledegg |
| `take_glass` | 0.28 | 128 | juice |
| `spoon_sugar` | 0.20 | 40 | coffee, tea |
| `stir_cereals` | 0.15 | 71 | cereals |
| `take_knife` | 0.15 | 83 | juice, salat, sandwich |
| `take_topping` | 0.13 | 36 | sandwich |
| `stir_fruit` | 0.11 | 33 | salat |
| `stir_coffee` | 0.11 | 39 | coffee |
| `pour_flour` | 0.10 | 18 | pancake |
| `put_bunTogether` | 0.10 | 83 | sandwich |
| `pour_sugar` | 0.07 | 25 | coffee, tea |
| `take_eggs` | 0.06 | 17 | friedegg, pancake, scrambledegg |
| `take_squeezer` | 0.05 | 33 | juice |
| `stir_tea` | 0.02 | 10 | tea |
| `take_butter` | 0.02 | 13 | sandwich |
