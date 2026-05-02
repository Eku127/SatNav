# SatNav Examples

Small scripts showing how to use SatNav navigation helpers with the shared
example resources in `applications/resources/`.

## Reference Path Follower

`reference_follower_example.py` runs one VLN episode with
`ReferencePathFollower`. It follows the episode reference path waypoint by
waypoint and saves the final observation.

```bash
python examples/reference_follower_example.py
```

Use a custom task config:

```bash
python examples/reference_follower_example.py --config path/to/task.yaml
```

## SatNav Path Follower

`satnav_path_follower_example.py` runs all episodes with `SatNavPathFollower`.
It greedily navigates through waypoints, writes per-episode metrics, and can
generate mp4 videos.

```bash
python examples/satnav_path_follower_example.py
```

Skip video generation:

```bash
python examples/satnav_path_follower_example.py --no-video
```

Use a custom task config:

```bash
python examples/satnav_path_follower_example.py --config path/to/task.yaml
```
