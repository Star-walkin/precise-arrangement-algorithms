"""Record completed assistant visual review after viewing all26 fixed composites."""
from common import HERE,load,dump,sha

def main():
    ids=load(HERE/'protocol.json')['gallery_fixed_diagnostic_ids']
    first=[
        '左入斜带仍可见，C卷面细条与外围纹理增多，卷面上的弱段未被唯一中心响应替代。',
        '自由直段保留，贴卷面位置仍有相近多条响应，C不能证明完整接续改善。',
        '原图直绳明显，C在卷面背景上的响应仍与周期纹理竞争，弱段存在。',
        '斜向结构仍可辨，C布面/邻圈细响应更显著，目标末端身份未得到验证。',
        '跨卷面上部的斜带得到保留，邻圈仍强，不能以更亮的局部线认定正确延伸。',
        '左入长条B/C均存在，C仍有双侧细条及卷面中心竞争响应，非唯一目标热带。',
        '同E2条件下主斜带保留，工具及卷面邻圈响应均在；没有额外入口提示。',
        '右入直带较清楚，C邻近细条/工具背景更复杂，后段身份仍未知。',
        '右入长带较连续，C大片背景响应有所降低而邻近细纹更显著，属于混合变化。',
        '长斜带保留，C卷面两侧周期响应增强，不能据热图验证末端选圈。',
        '同E2条件下高位斜带可见，卷面上缘与斜带靠近，远端身份仍需参考。',
        '同E3条件下直带可见，手套和工具轮廓也产生明显响应，目标专一性不足。',
        '明显斜向直带保留，C布面、卷面旁侧条纹也增强，非普遍清晰化。',
    ]
    last=[
        '用户图1：同位置局部实际放大查看，部分旁峰减弱但叠卷面仍有双线/多峰与弱区；未解决。',
        '用户图2：同位置局部实际放大查看，主斜带仍明显，但两侧响应和邻纹同在；不能据亮线确认完整同根。',
    ]
    prior=load(HERE/'visual_review_root.json');notes=first+[r['observation'] for r in prior['rows']]+last
    assert len(ids)==len(notes)==26
    rows=[]
    for ident,note in zip(ids,notes):
        path=HERE/'postprocess/pilot_001/原图_旧B_新热力图'/(ident+'.jpg')
        rows.append(dict(id=ident,image=str(path),sha256=sha(path),observation=note,
                         actual_image_tool_view=True,identity_verified=False))
    closeups=[]
    for ident in ids[-2:]:
        path=HERE/'comparison_responses'/(ident+'__local_same_crop.png')
        closeups.append(dict(image=str(path),sha256=sha(path),actual_image_tool_view=True))
    dump(HERE/'visual_review.json',dict(reviewer='root assistant',actual_viewed_images=26,rows=rows,additional_user_case_closeups=closeups,
        displayed_composite_resolution=[2048,552],scope='Fixed26 original/oldB/C overview composites plus2 detailed user-case crops. Not a pixel-level human annotation or all292 image review.',
        all292_visually_reviewed=False,no_complete_identity_truth=True,parameters_changed=False,used_for_model_selection=False,
        preservation='All mixed or failed examples retained; old images/models unchanged.',
        conclusion='Partial strengthening and changed background distribution, with multiple responses and unresolved weak regions. No near-universal target-identity improvement established.',
        source_complete_review_script_sha256=sha(__file__)))
    print('Actual fixed26 visual review recorded')

if __name__=='__main__':main()
