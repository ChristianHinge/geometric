import os

from pip._internal.operations import freeze

from src.settings.paths import MODULE_PATH


def test_env():
    with open(os.path.join(MODULE_PATH, 'requirements.txt')) as f:
        req_lines = f.readlines()
    with open(os.path.join(MODULE_PATH, 'requirements-geometric.txt')) as f:
        req_geo_lines = f.readlines()
    req_geo = [i.split(' \n')[0] for i in req_geo_lines]

    req_lines.extend(req_geo)

    freeze_lines = []

    x = freeze.freeze()
    for p in x:
        if '-e ' not in p: 
            freeze_lines.append(p)

    req_lines = [i for i in req_lines if not ('#' in i)]
    reqs = set(req_lines)
    frz = set(freeze_lines)

    diff_list = [x for x in freeze_lines if x not in reqs]
    diff_list2 = [x for x in reqs if x.split('\n')[0] not in frz]
    diff_list.extend(diff_list2)
    nono_list = [x for x in diff_list if x in reqs]

    assert len(nono_list) == 0