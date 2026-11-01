# Label Mapping — Breakfast Actions → Kitchen Critical Path (v0)

Last updated: 2026-11-01. Source inventory: `docs/breakfast_labels.md`.

## Camera labels (classifier classes)

The prep-zone classifier predicts **actions**, not tasks. The scheduler turns actions into task progress (see "From actions to tasks").

| Our label | Breakfast coarse labels | Hours (all views) | Supports these tasks |
|---|---|---:|---|
| `cut` | `cut_fruit`, `cut_bun`, `cut_orange` | 6.96 | `cut_aromatics`, `cut_firm_veg`, `cut_leafy_veg`, `cut_tomato`, `cube_tofu` |
| `peel` | `peel_fruit` | 3.03 | carrot peeling, inside `cut_firm_veg` |
| `pour` | `pour_milk`, `pour_water`, `pour_juice`, `pour_coffee`, `pour_cereals`, `pour_flour`, `pour_sugar` | 5.99 | `mix_sauce` |
| `stir` | `stir_dough`, `stir_milk`, `stir_egg`, `stir_fruit`, `stir_coffee`, `stir_cereals`, `stir_tea` | 4.86 | `mix_sauce` |
| `transfer` | `put_fruit2bowl`, `spoon_flour`, `spoon_powder`, `spoon_sugar` | 3.07 | end of each cutting task (food into a bowl); `mix_sauce` |
| `take_object` | `take_bowl`, `take_plate`, `take_knife`, `take_cup`, `take_glass`, `take_topping`, `take_eggs`, `take_squeezer`, `take_butter` | 2.81 | setup; no task by itself |
| `background` | `SIL` | 4.88 | no activity |
| `other` | all other labels (stove actions such as `fry_egg`, `fry_pancake`, `stirfry_egg`, `butter_pan`, `pour_dough2pan`, `pour_egg2pan`, `pour_oil`; and `crack_egg`, `smear_butter`, `squeeze_orange`, `add_saltnpepper`, `add_teabag`, `put_egg2plate`, `put_pancake2plate`, `put_toppingOnTop`, `put_bunTogether`) | rest | none; absorbs actions outside our task set |

Choices:
- `pour_oil` goes to `other`, because in Breakfast it happens at the pan, outside our prep zone.
- `stir_egg` goes to `stir`, because it happens in a bowl before the egg goes into the pan.
- `pour` and `stir` stay separate classes. They look different, and both map to `mix_sauce`.

## Tasks with no public data (tap in v1)

| Task | Why no camera | v1 input |
|---|---|---|
| `wash_greens` | Happens at the sink, outside the prep zone | tap |
| `press_tofu` | No pressing action in Breakfast | tap |
| Rice tasks, all wok tasks, `plate` | Outside the prep zone | tap |

## From actions to tasks (risk 1)

**Finding:** all Breakfast cutting is fruit or bread. The model can learn "someone is cutting," but not *what* is cut. Our graph has five cutting tasks.

**v1 rule (simple and testable):**
1. When the camera reports `cut`, `peel`, or `transfer`, credit the time to the **currently recommended** cutting task. The user usually follows the recommendation.
2. If no cutting task is recommended, credit the open cutting task with the longest remaining path.
3. When the credited time reaches the task's estimated duration, or a `transfer` follows a run of `cut`, ask "Done with <task>?" One tap confirms it.
4. If the user taps a different task, move the credited time to that task.

This rule belongs to Phase 2b (connecting the model to the scheduler). Phase 1 replays task-level events and does not need it.

## Test splits (v1, no user labels)

- **train:** s2–s3 persons; **val:** s4 persons (model selection only). Both exclude the held-out camera ID.
- **T1 — new people:** s1 persons (P03–P15), all camera IDs except the held-out one.
- **T2 — new people + unseen camera ID:** s1 persons, camera ID `webcam01`, which is removed from training for all persons. Chosen by coverage: it appears in every person split (s1: 97 videos, 11 persons).
- Camera IDs do not mean the same position in every kitchen (`docs/figures/`), so T2 tests "a camera ID never trained on", not one fixed angle.
- Run 1 used `webcam02`. Among test persons it exists only for P03 and P04 (15 videos), so T2 was too small; it was replaced. (It also has 323 videos in s2–s4, which run 1 removed from training.) See `reports/phase2a.md`.
- **50 Salads:** planned as a second, unseen dataset. **Blocked on 2026-11-01:** the official server (`cvip.computing.dundee.ac.uk`) does not resolve, and the only online copy (`dinggd/50salads` on Hugging Face) has precomputed features, not raw video. T2 replaces it until the server returns.
