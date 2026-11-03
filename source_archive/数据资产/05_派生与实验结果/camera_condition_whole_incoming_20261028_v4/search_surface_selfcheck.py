"""A4 搜索域结构测试；只使用合成数组，不读取真实图像或连接设备。"""
from pathlib import Path
import hashlib
import json
import platform
import sys

import cv2
import numpy as np

from whole_path import build_search_surface


def _walk_until_exit(mask, row=80, width=12.):
    """重现实际路径器的 entered/outside 规则，固定中心线便于查明因果。"""
    entered=False
    outside=0.
    entered_at=None
    for x in range(10,mask.shape[1]):
        inside=bool(mask[row,x])
        entered=entered or inside
        if inside and entered_at is None:
            entered_at=x
        outside=0. if inside else outside+1.
        if entered and not inside and outside>=width*.8:
            return entered_at,x
    return entered_at,None


def main():
    # 主卷面在 x=130 开始。自由绳方向上的两块小碎片必须不能
    # 提前触发 entered；正常自由段应抵达主卷面并在真正离开后停止。
    source=np.zeros((160,300),np.uint8)
    source[50:120,130:260]=255
    source[79:82,50:53]=255
    source[79:82,64:68]=255
    before=source.copy()
    search,diag=build_search_surface(source)
    old_entry,old_stop=_walk_until_exit(source)
    entry,stop=_walk_until_exit(search)
    assert old_entry==50 and old_stop<130,(old_entry,old_stop)
    assert entry==130 and stop>=260,(entry,stop)
    assert not search[80,51] and search[80,150]
    assert diag['discarded_fragment_pixels']==21,diag
    assert np.array_equal(source,before),'input was modified'
    assert set(np.unique(search)).issubset({0,255})

    # 两个相距很远的显著组件，凸包过大时不能擅自填充中间背景。
    sparse=np.zeros((220,360),np.uint8)
    sparse[20:70,20:70]=255
    sparse[150:200,280:330]=255
    sparse[99:102,149:152]=255
    limited,limited_diag=build_search_surface(sparse)
    assert limited_diag['search_surface_method']=='significant_components_hull_area_limit'
    assert limited_diag['convex_hull_candidate_area_ratio']>1.8
    assert limited[40,40] and limited[175,305]
    assert not limited[100,150] and not limited[110,180]
    assert limited_diag['discarded_fragment_pixels']==9

    # 无显著卷面时保留空域，不能把散落的像素扩大成整个搜索区。
    tiny=np.zeros((60,60),np.uint8)
    tiny[10:13,10:13]=255
    tiny[45:48,45:48]=255
    absent,absent_diag=build_search_surface(tiny)
    assert not np.any(absent)
    assert absent_diag['search_surface_method']=='no_significant_component'
    assert absent_diag['discarded_fragment_pixels']==18
    empty,empty_diag=build_search_surface(np.zeros((60,60),np.uint8))
    assert not np.any(empty) and empty_diag['original_surface_pixels']==0

    root=Path(__file__).resolve().parent
    receipt={
        'scope':'synthetic search-domain/state-transition regression; not real-image accuracy',
        'passed':True,
        'tests':[
            {'name':'fragments_do_not_trigger_entered','old_entry_x':old_entry,
             'old_stop_x':old_stop,'new_entry_x':entry,'new_stop_x':stop,'diagnostics':diag},
            {'name':'oversized_hull_does_not_fill_background','diagnostics':limited_diag},
            {'name':'only_fragments_returns_empty_domain','diagnostics':absent_diag},
            {'name':'empty_input_stays_empty','diagnostics':empty_diag},
            {'name':'input_unmodified_and_binary_output','passed':True},
        ],
        'python':sys.version,'executable':sys.executable,
        'platform':platform.platform(),'numpy':np.__version__,'opencv':cv2.__version__,
        'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                         for name in ('whole_path.py','search_surface_selfcheck.py')},
        'command':'vision_prototype/.venv/Scripts/python.exe -B camera_condition_whole_incoming_20261028_v4/search_surface_selfcheck.py',
        'exit_code':0,'equipment_connected':False,'real_images_read':False,
    }
    output=root/'search_surface_selfcheck_001.json'
    if output.exists():
        raise FileExistsError('Preserve the previous receipt: '+str(output))
    output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':True,'receipt':str(output),'whole_path_sha256':receipt['source_sha256']['whole_path.py']}))


if __name__=='__main__':
    main()
