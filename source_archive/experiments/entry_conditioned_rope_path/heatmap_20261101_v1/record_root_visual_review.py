"""Record the root assistant's actual image-tool inspection; no automated quality label."""
from common import HERE,load,dump,sha

def main():
    ids=load(HERE/'protocol.json')['gallery_fixed_diagnostic_ids'][13:24]
    notes=[
        '斜向长条在B/C均可见，C卷面及外围纹理更碎；不足以确认贴面末端唯一身份。',
        '斜直段保留；C外缘、布面和工具附近响应更复杂，不能把中心热条增强等同末端选对。',
        '直段主方向可辨，靠近卷面左侧仍与密集邻圈响应竞争；没有明确的完整路径改善证据。',
        '左入段明亮，但叠卷面后仍有弱段/多线，C未形成可唯一跟随的连续中心响应。',
        '左侧长直带保留，卷面中部仍呈旁侧多条细响应，无法据此确认整根目标。',
        '入口到卷面附近条带可辨，进入密集卷面后仍难与邻圈区分；C未解决用户同类左视角问题。',
        '接近卷面上缘的斜带保留，热响应与上缘及邻圈贴近，无法从热图单独确定接续身份。',
        '右入长条仍存在，C没有消除卷面周期响应及工具邻近干扰；尾端不能视作完整识别。',
        '两版斜直带都较连续；C保留可见结构，但手套、外围背景及邻圈响应也突出。',
        '主斜带在两版均清楚，C周围纹理响应增加，未证明卷面末端的目标身份。',
        '明显直带得到保留，C布面及卷面纹理更显著，属于混合变化而非所有区域改善。',
    ]
    assert len(ids)==len(notes)==11
    rows=[]
    for ident,note in zip(ids,notes):
        path=HERE/'postprocess/pilot_001/原图_旧B_新热力图'/(ident+'.jpg')
        rows.append(dict(id=ident,image=str(path),sha256=sha(path),observation=note,
            scope='Actual original/oldB/C composite viewed through image tool at2048x552 preview. Qualitative overview, not pixel-level identity annotation.',
            classification='mixed_or_unresolved',identity_verified=False))
    dump(HERE/'visual_review_root.json',dict(reviewer='root assistant',actual_viewed_images=11,rows=rows,
        delegated_visual_review_failed='path_model visual follow-up errored Bad Request; root re-viewed all11 itself, no model results altered.',
        all292_visually_reviewed=False,parameters_changed=False,used_for_model_selection=False,
        conclusion='Long diagonal structures are often retained, while neighboring/background responses and weak tails remain. No near-universal identity claim.'))

if __name__=='__main__':main()
