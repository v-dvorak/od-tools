# Configuration file

Configuration file stores setting for dataset processing that would be near impossible to pass to the provided scripts via command line arguments or settings that do not change often.

- [Class mapping](#class-mapping)
- [Window size](#window-size)
- [Overlap ratio](#overlap-ratio)

## Class mapping

Makes it possible to map class IDs to class names and the other way. Class ID has to be a number, other variables have to be strings. Three formats are available:

### Class mapped to itself

```json
{
    "class_id": "class_name"
}
```

This class wil be retrieved from input annotations by its name `class_name` and will be further processed and saved under this name.

### Many classes mapped to the first one

```json
{
    "class_id": ["class1", "class2", "class3"]
}
```

These classes will be retrieved from input annotations by their name but will be further processed and saved under the name of the first class specified.

#### Example

```json
{
    "1": [
        "noteheadHalf",
        "noteheadHalfSmall"
    ]
}
```

Both `noteheadHalf` and `noteheadHalfSmall` will be processed and saved under the name of the first class `noteheadHalf`, their ID is `1`.

### Many classes mapped to one

```json
{
    "class_id": [
        [
            "class1",
            "class2",
            "class3"
        ],
        "class_name"
    ]
}
```

These classes will be retrieved from input annotations by their name but will be further processed and saved under the specified `class_name`.

#### Example

```json
{
    "4": [
        [
            "accidentalFlat",
            "accidentalNatural",
            "accidentalSharp"
        ],
        "accidental"
    ]
}
```

All three types of accidental will be processed and saved under the name `accidental`, their ID is `4`.

## Window size

Specified the `(width, height)` size of sliding window from which tiles for further processing are generated.

## Overlap ratio

Defines the minimal overlap ratio of sliding window for both vertical and horizontal overlaps.

### Example

Sliding window size set to `(640, 640)` and overlap ratio to `0.10` and `0.25`:

<p float="middle">
  <img src="../docs/splitviz/overlap10.jpg" width="300" />
  <img src="../docs/splitviz/overlap25.jpg" width="300" />
</p>