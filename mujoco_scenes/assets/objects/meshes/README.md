# Prepared kitchen meshes

These files are runtime-ready copies of the selected YCB and Google Scanned
Objects visuals. Every OBJ is normalized to a centred object frame and every
texture is limited to 1024x1024 for portable MuJoCo rendering.

Regenerate all downloaded assets from the repository root:

```bash
python mujoco_scenes/scripts/prepare_object_assets.py --force
```

`manifest.json` is the machine-readable source of truth for semantic names,
dataset IDs, canonical URLs, preparation URLs, and file hashes. The custom
folder contains project-authored stirrer, folded-napkin, and tong meshes.
The coffee-jar folder contains both the normalized closed source derivative
and `gso_coffee_jar_open.obj`, whose top faces are clipped by the preparation
script for the open, powder-visible runtime model.

See `../../../THIRD_PARTY_NOTICES.md` for attribution and redistribution notes.
