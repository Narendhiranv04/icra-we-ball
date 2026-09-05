"""Cabinet geometry regressions; execution is checked by direct variant runs."""
import mujoco
import numpy as np
import pytest
import xml.etree.ElementTree as ET

from mujoco_scenes.workshop_scene import WorkshopScene, build_workshop_xml


@pytest.mark.parametrize('variant', ['F4_MANUAL_FIRST_THREE_REGIONS', 'F7_POWER_ONLY'])
def test_cabinet_power_driver_fits_its_supported_slot(variant):
    scene = WorkshopScene(robot='none', variant=variant)
    model, data = scene.model, scene.data
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'workshop_power_driver')
    bounds = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body:
            continue
        rotation = data.geom_xmat[geom].reshape(3, 3)
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start = model.mesh_vertadr[mesh]
            vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
            bounds.extend(vertices @ rotation.T + data.geom_xpos[geom])
            continue
        centre = data.geom_xpos[geom] + rotation @ model.geom_aabb[geom, :3]
        extent = np.abs(rotation) @ model.geom_aabb[geom, 3:]
        bounds.extend((centre - extent, centre + extent))
    bounds = np.asarray(bounds)
    lower, upper = bounds.min(axis=0), bounds.max(axis=0)
    assert lower[0] > .233 and upper[0] < .647
    assert lower[1] > .466 and upper[1] < .644
    support = .696 if variant.startswith('F4_') else .766
    assert support - .001 <= lower[2] <= support + .003
    assert upper[2] < (.754 if variant.startswith('F4_') else 1.322)
    weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'storage_weld_workshop_power_driver')
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.eq_obj1id[weld]) == 'tool_cabinet'


@pytest.mark.parametrize('variant,object_name,position,quaternion', [
    ('F0_MANUAL_FIRST_ONE_REGION', 'workshop_wooden_hammer', (.44, .535, .767), (1, 0, 0, 0)),
    ('F1_POWER_FIRST_ONE_REGION', 'workshop_wooden_hammer', (.44, .535, .767), (1, 0, 0, 0)),
    ('F2_MANUAL_FIRST_TWO_REGIONS', 'workshop_power_driver', (.44, .53, .79), (.7071, -.7071, 0, 0)),
    ('F3_POWER_FIRST_TWO_REGIONS', 'workshop_long_phillips_driver', (.34, .535, .767), (1, 0, 0, 0)),
])
def test_reference_variants_keep_original_cabinet_storage(variant, object_name, position, quaternion):
    root = ET.fromstring(build_workshop_xml(robot='none', variant=variant))
    body = root.find(f".//body[@name='{object_name}']")
    assert np.allclose(np.fromstring(body.get('pos'), sep=' '), position)
    assert np.allclose(np.fromstring(body.get('quat'), sep=' '), quaternion)
