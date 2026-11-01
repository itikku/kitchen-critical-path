"""Breakfast Actions coarse labels -> prep-zone camera labels (see docs/label_mapping.md)."""

CLASSES: tuple[str, ...] = (
    "cut",
    "peel",
    "pour",
    "stir",
    "transfer",
    "take_object",
    "background",
    "other",
)

_SOURCES: dict[str, tuple[str, ...]] = {
    "cut": ("cut_fruit", "cut_bun", "cut_orange"),
    "peel": ("peel_fruit",),
    "pour": (
        "pour_milk",
        "pour_water",
        "pour_juice",
        "pour_coffee",
        "pour_cereals",
        "pour_flour",
        "pour_sugar",
    ),
    "stir": (
        "stir_dough",
        "stir_milk",
        "stir_egg",
        "stir_fruit",
        "stir_coffee",
        "stir_cereals",
        "stir_tea",
    ),
    "transfer": ("put_fruit2bowl", "spoon_flour", "spoon_powder", "spoon_sugar"),
    "take_object": (
        "take_bowl",
        "take_plate",
        "take_knife",
        "take_cup",
        "take_glass",
        "take_topping",
        "take_eggs",
        "take_squeezer",
        "take_butter",
    ),
    "background": ("SIL",),
}

# The 48 coarse labels in the Breakfast Actions release (docs/breakfast_labels.md).
BREAKFAST_LABELS: frozenset[str] = frozenset(
    {
        *(lab for labs in _SOURCES.values() for lab in labs),
        "fry_pancake",
        "fry_egg",
        "stirfry_egg",
        "crack_egg",
        "squeeze_orange",
        "butter_pan",
        "smear_butter",
        "put_egg2plate",
        "add_saltnpepper",
        "put_toppingOnTop",
        "pour_dough2pan",
        "add_teabag",
        "pour_oil",
        "put_pancake2plate",
        "pour_egg2pan",
        "put_bunTogether",
    }
)

_TO_CLASS: dict[str, str] = {lab: cls for cls, labs in _SOURCES.items() for lab in labs}


def map_label(breakfast_label: str) -> str:
    """Map one Breakfast coarse label to a camera label. Unknown labels raise KeyError."""
    if breakfast_label not in BREAKFAST_LABELS:
        raise KeyError(f"not a Breakfast coarse label: {breakfast_label!r}")
    return _TO_CLASS.get(breakfast_label, "other")


def class_index(camera_label: str) -> int:
    return CLASSES.index(camera_label)
