"""Build a local review index from completed, verified experiment outputs."""
import html,shutil
from common import *
from compare_results import main as compare
E=html.escape

def pct(n,total):return f'{n}/{total} ({100*n/total:.1f}%)'
def main():
    runs_name='runs_bg9';compare(runs_name,'comparison_bg9.json');plan=read(ROOT/'plan.json')
    comparisons={r['id']:r for r in read(ROOT/'comparison_bg9.json')['videos']}
    priority=['dual_20260923_113029__left','dual_20260924_111500__left','dual_20260923_113305__right',
        'dual_20260923_113745__left','dual_20260924_111143__right','dual_20260923_101643__right']
    items=sorted(plan['videos'],key=lambda x:(priority.index(x['id']) if x['id'] in priority else 9,
        not x['visual_role'].startswith('active'),x['id']))
    cards=[];table=[];manifest=[];total=0;errors=0
    for item in items:
        id=item['id'];run=ROOT/runs_name/id;receipt=read(run/'receipt.json');stats=comparisons[id]
        assert receipt['status']=='complete' and receipt['frames']==item['planned_frames']
        assert receipt['source_unchanged'] and receipt['core_unchanged']
        total+=receipt['frames'];errors+=receipt['algorithm_error_frames']
        v=Path(receipt['video']);assert v.exists() and sha(v)==receipt['video_sha256']
        manifest.append(dict(id=id,video=str(v),video_sha256=receipt['video_sha256'],frames=receipt['frames']))
        t=stats['changes']['tracking_valid'];b=stats['changes']['boundary_fit'];n=stats['frames']
        active=item['visual_role'].startswith('active');kind='含活动绳' if active else '另一侧相机对照'
        if item.get('partial_source'):kind+=' · 截断源文件'
        if id=='dual_20260923_113029__right':kind+=' · 活动绳后段才进入画面'
        detail=[]
        for label,s in [('活动绳',t),('卷面边界',b)]:
            intervals=s['longest_missing_intervals_s'][:3]
            text='；'.join(f'{a:.1f}–{z:.1f}s ({k} 帧)' for a,z,k in intervals) or '无'
            detail.append(f'<p>{label}未通过检查的最长时段：{E(text)}</p>')
        poster=f'{runs_name}/{id}/preview_{min(10,n-1):06d}.jpg';video=f'{runs_name}/{id}/{v.name}'
        cards.append(f'''<article data-role="{'active' if active else 'control'}"><h2>{E(id)}</h2><p>{kind} · {n} 帧 / {n/item['fps']:.1f} 秒</p>
        <video controls preload="none" poster="{poster}" src="{video}"></video>
        <p>活动绳检查通过：{pct(t['old'],n)} → <b>{pct(t['new'],n)}</b><br>边界检查通过：{pct(b['old'],n)} → <b>{pct(b['new'],n)}</b></p>
        {''.join(detail)}<p><a href="{video}">打开完整视频</a> · <a href="{runs_name}/{id}/receipt.json">完整性记录</a></p></article>''')
        table.append(f'<tr><td>{E(id)}</td><td>{kind}</td><td>{pct(t["old"],n)} → {pct(t["new"],n)}</td><td>{pct(b["old"],n)} → {pct(b["new"],n)}</td></tr>')
    report=f'''<!doctype html><meta charset="utf-8"><title>RES-1.5 复测结果</title><style>
    body{{font:16px/1.7 "Microsoft YaHei",sans-serif;background:#101b26;color:#e7edf5;margin:30px auto;max-width:1480px;padding:0 24px}}
    a{{color:#7ccef2}} h1{{font-size:30px}} h2{{font-size:21px}} article{{background:#1b2b3b;border:1px solid #405567;padding:20px;border-radius:12px}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(570px,1fr));gap:24px}} video{{width:100%;background:black}} table{{width:100%;border-collapse:collapse;font-size:14px}}
    td,th{{border:1px solid #405567;padding:8px;text-align:left}} .notice{{background:#3d3523;padding:18px}} button{{padding:9px 18px;margin:8px;background:#29465b;color:white;border:0;border-radius:7px;cursor:pointer}}
    @media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}
    </style><h1>RES-1.5：失败诊断、修改与完整视频复测</h1>
    <p>基准：RES-1.4。当前修改：SG3 活动绳分段跟踪、BG9 边界分组拟合、CG2 背景稳定性检查。共 {len(manifest)} 个可解码视频、{total} 帧；程序异常帧 {errors}。</p>
    <div class="notice">表内数字是算法内部检查的通过帧数，不是人工逐帧标注的检测准确率。本轮针对活动绳跟踪与卷面边界；没有重新运行整套 F3 / 剩余根数 / 实体贴边判定。参考 d 沿用 2238.7 px，未对每个视频重新标定。青线表示有可见边缘支持，橙线是模型推断，绿线仅表示当前实测绳段。橙圈及竖向范围表示 d 处模型估计与敏感性范围，非标定置信区间。</div>
    <h2>确认的失效原因及修改</h2><ol>
    <li>113029 左相机：原入口检测无候选，引导器把进绳分成不完全共线的可见段。新增传统边缘/直线候选，并关联引导器前后两个当前可见段；由 S1 验证绳宽、双边缘和当前点支持，保留 P1 独立重试。</li>
    <li>111500 左相机：少数缺口端点的宽度或亮暗极性变化会否决整条线。改为从当前实测点筛选有足够长度、覆盖率与低残差的直线段；显示时保留真实缺口，不把遮挡补成测量点。</li>
    <li>113305 右相机：d 处缺少边缘及形状偏离旧模型，使边界形状长期无法更新。改为验证未遮挡空间分区，重大形状变化需连续三帧当前证据一致后才更新。</li>
    <li>101643 右相机：背景匹配内点比例不足会连带否决正确边界。补充内点空间分布和残差检查；未确认时暂停本帧输出，短暂未确认不再清空形状参考。真实位移仍触发失效。</li>
    <li>111143 工具干扰：颜色相近的工具形成第二组连贯边缘，普通鲁棒拟合仍可能被干扰。BG9 先用有空间跨度的当前锚点共识选出卷面分组，再拟合；只在该组上测试模型敏感性，并在拟合后的边界重新检查遮挡。对可见范围以外的位置不伪造实测线。</li></ol>
    <p>实验包含多个失败窗口的冷启动复测及全部可解码视频从首帧顺序重跑。BG9 迭代只复用已经验证图像哈希的同帧 BG5 / SG3 / CG2 输出，重新按时间顺序拟合边界；没有用上一帧替代当前检测。代码快照见 <a href="frozen_final_BG9/hashes.json">版本哈希</a>；数字对照见 <a href="comparison_bg9.json">逐视频统计</a>；相机位移测试见 <a href="experiments/camera_test.json">测试记录</a>。人工查看范围另列于 <a href="visual_review.json">画面复核记录</a>。</p>
    <p>源文件限制：191147 左文件为空，右文件唯一可解码帧严重损坏，继续排除；其余 9 月 25 日文件只有 4–9 帧可恢复，不代表完整一分钟视频。视频没有声轨。</p>
    <details><summary>展开全部数字对照</summary><table><tr><th>视频</th><th>范围</th><th>活动绳：基准 → 修改后</th><th>卷面边界：基准 → 修改后</th></tr>{''.join(table)}</table></details>
    <p><button onclick="filter('all')">全部</button><button onclick="filter('active')">含活动绳</button><button onclick="filter('control')">对照相机</button></p>
    <div class="grid">{''.join(cards)}</div><script>function filter(v){{document.querySelectorAll('article').forEach(x=>x.hidden=v!=='all'&&x.dataset.role!==v)}}</script>'''
    (ROOT/'结果索引.html').write_text(report,encoding='utf8')
    dump(ROOT/'delivery_manifest.json',dict(algorithm='RES-1.5/SG3-BG9-CG2',frames=total,videos=manifest,algorithm_error_frames=errors,
        metric_semantics='gate availability, not externally labelled accuracy',F3_run=False,count_run=False))
    assert errors==0,'Do not call a batch with algorithm errors complete'
    print('Built index',len(manifest),total)
if __name__=='__main__':main()
