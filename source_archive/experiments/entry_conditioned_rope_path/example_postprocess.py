"""Read-only pair0075 diagnostic: frozen A/B responses and P0/P1 outputs."""
from pathlib import Path
import argparse,gzip,hashlib,json,math,sys
import cv2,numpy as np

CASE='dual_20260923_101517__pair0075__right'
COLORS={'entry_condition':(255,255,0),'heatmap_supported':(40,220,40),
        'image_supported':(235,170,20),'legacy_supported_proxy':(40,220,40),
        'bridge':(220,40,220),'ambiguous':(25,80,245),'competitor':(245,160,240)}
def load(p):
    p=Path(p);data=gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes()
    return json.loads(data.decode('utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def save(p,im):cv2.imencode('.png',im)[1].tofile(str(p))
def text(im,s,xy,color=(30,30,30),size=.65):cv2.putText(im,s,xy,cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
def heat(a):
    out=cv2.applyColorMap(np.round(np.clip(np.nan_to_num(a),0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
    out[~np.isfinite(a)]=(110,110,110);return out
def compatible_states(c):
    if 'segment_status' in c:
        assert len(c['segment_status'])==len(c['points_xy'])-1
        return c['segment_status']
    return [{'visible_supported':'legacy_supported_proxy','inferred':'bridge'}.get(s,s) for s in c['point_status'][1:]]
def lengths(c):
    if c is None:return {'entry_condition':0.,'supported_new':0.,'bridge':0.,'ambiguous':0.,'total':0.}
    p=np.asarray(c['points_xy'],float);d=np.linalg.norm(np.diff(p,axis=0),axis=1);s=np.array(compatible_states(c))
    return {'entry_condition':float(d[s=='entry_condition'].sum()),
            'supported_new':float(d[np.isin(s,['heatmap_supported','image_supported','legacy_supported_proxy'])].sum()),
            'bridge':float(d[s=='bridge'].sum()),'ambiguous':float(d[s=='ambiguous'].sum()),'total':float(d.sum())}
def dash_parts(length,arc,on=18.,off=10.):
    """Arc phase persists across all short polyline edges; never resets at4px."""
    parts=[];t=0.;period=on+off
    while t<length-1e-8:
        phase=(arc+t)%period;visible=phase<on;remaining=(on-phase) if visible else (period-phase)
        end=min(length,t+max(remaining,1e-7))
        if visible:parts.append((t,end))
        t=end
    return parts
def render_candidate(canvas,c,origin,scale,competitor=False):
    p=np.asarray(c['points_xy'],float);states=compatible_states(c);arc=0.
    for a,b,state in zip(p[:-1],p[1:],states):
        length=float(np.linalg.norm(b-a));color=COLORS['competitor'] if competitor else COLORS.get(state,COLORS['ambiguous'])
        dashed=competitor or state in ('bridge','ambiguous')
        parts=dash_parts(length,arc,18,10) if dashed else [(0.,length)]
        for lo,hi in parts:
            aa=a+(b-a)*lo/max(length,1e-8);bb=a+(b-a)*hi/max(length,1e-8)
            q=np.rint((np.stack([aa,bb])-origin)*scale).astype(int)
            cv2.line(canvas,tuple(q[0]),tuple(q[1]),color,2 if competitor else 3,cv2.LINE_AA)
        arc+=length
def seed_overlay(canvas,seed,origin,scale):
    p=np.rint((np.asarray(seed)-origin)*scale).astype(int)
    cv2.polylines(canvas,[p],False,COLORS['entry_condition'],3,cv2.LINE_AA)
    cv2.arrowedLine(canvas,tuple(p[0]),tuple(p[-1]),COLORS['entry_condition'],2,cv2.LINE_AA,tipLength=.25)
def crop_box(center,shape,width=760,height=450):
    h,w=shape;cx,cy=center;x0=int(np.clip(cx-width/2,0,max(0,w-width)));y0=int(np.clip(cy-height/2,0,max(0,h-height)))
    return [x0,y0,min(w,x0+width),min(h,y0+height)]
def pane(gray,box,width):
    x0,y0,x1,y1=box;scale=width/(x1-x0);height=round((y1-y0)*scale)
    im=cv2.cvtColor(np.round(gray[y0:y1,x0:x1]*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    return cv2.resize(im,(width,height)),np.array([x0,y0]),scale
def paths_panel(gray,row,box,width):
    im,origin,scale=pane(gray,box,width);selected=row['selected_candidate_id']
    candidates=row['candidates'];primary=next((c for c in candidates if c['id']==selected),candidates[0] if candidates else None)
    for c in candidates:
        if c is not primary:render_candidate(im,c,origin,scale,True)
    if primary is not None:render_candidate(im,primary,origin,scale)
    seed_overlay(im,row['actual_seed_xy'],origin,scale)
    return im
def legend(canvas,y):
    labels=[('entry_condition','entry / direction'),('heatmap_supported','local support (not identity)'),
            ('image_supported','weak heat + image support'),('bridge','bridge: dashed'),('ambiguous','ambiguous: dashed'),('competitor','other candidate: dashed')]
    x=20
    for key,label in labels:
        cv2.line(canvas,(x,y),(x+32,y),COLORS[key],3)
        text(canvas,label,(x+40,y+5),size=.48);x+=300
def colorbar(canvas,x,y,w=800,title='Raw response intensity: fixed 0-1; gray = uncomputed NaN'):
    bar=heat(np.linspace(0,1,w,dtype=np.float32)[None,:]);canvas[y:y+18,x:x+w]=np.repeat(bar,18,axis=0)
    for value in (0,.5,1):text(canvas,str(value),(x+int(value*w)-5,y+40),size=.5)
    text(canvas,title,(x+w+30,y+16),size=.5)


def entry_figures(out,entry,gray,rows,responses,bundles,derived):
    h,w=gray.shape;seed=np.array(rows[('A','P0')]['actual_seed_xy']);box=[0,0,w,h]
    title='Entry01: recovered_01 points to hand-held metal tool' if entry=='entry_01' else 'Entry02: recovered_00 points to visible free rope; continuation unverified'
    width=1024;height=round(h*width/w)
    fig=np.full((height+170,width*3,3),250,np.uint8);text(fig,title,(20,32),size=.8)
    orig,origin,scale=pane(gray,box,width);seed_overlay(orig,seed,origin,scale)
    for col,(im,label) in enumerate([(orig,'Original gray + supplied entry'),(cv2.resize(heat(responses['A']),(width,height)),'A / fixed checkpoint step750'),(cv2.resize(heat(responses['B']),(width,height)),'B / fixed checkpoint step750')]):
        text(fig,label,(col*width+15,70));fig[90:90+height,col*width:(col+1)*width]=im
    colorbar(fig,20,height+110,900)
    save(out/'original_A_B_fixed.png',fig)
    combo=[('A','P0'),('A','P1'),('B','P0'),('B','P1')]
    fig=np.full((2*(height+100)+100,width*2,3),250,np.uint8)
    for i,key in enumerate(combo):
        col=i%2;rr=i//2;y=rr*(height+100);r=rows[key];selected=r['selected_candidate_id'];candidate=next((c for c in r['candidates'] if c['id']==selected),r['candidates'][0] if r['candidates'] else None)
        stat=lengths(candidate);state='SELECTED provisional' if selected else 'REJECTED: candidates only' if candidate else 'NO OUTPUT'
        text(fig,f'{key[0]}/{key[1]}  {state}',(col*width+15,y+28),color=(30,30,30) if selected else (25,70,200),size=.7)
        text(fig,f'candidate support {stat["supported_new"]:.1f}px | bridge {stat["bridge"]:.1f}px | ambiguous {stat["ambiguous"]:.1f}px',(col*width+15,y+58),size=.54)
        fig[y+80:y+80+height,col*width:(col+1)*width]=paths_panel(gray,r,box,width)
    legend(fig,2*(height+100)+35)
    text(fig,'Thin centerline schematic only. Rejected candidates are shown for diagnosis, not as accepted detections. No complete identity ground truth.',(20,2*(height+100)+78),size=.56)
    save(out/'paths_four_combinations.png',fig)
    scene=rows[('A','P0')]['edge_target'];direction=np.asarray(scene['entry_direction_xy']);contact=np.asarray(scene['contact_proxy']['xy'])
    locations={'entry_free_or_tool':seed[-1]+140*direction,'transition_visual_proxy':contact,'post_transition_visual_region':contact+350*direction}
    if rows[('B','P1')]['candidates']:
        locations['P1_B_stop_diagnostic']=np.asarray(rows[('B','P1')]['candidates'][0]['points_xy'][-1])
    windows=[]
    for name,center in locations.items():
        crop=crop_box(center,gray.shape);cw=720;ch=round((crop[3]-crop[1])*cw/(crop[2]-crop[0]));fig=np.full((2*(ch+75)+100,3*cw,3),250,np.uint8)
        text(fig,f'{name} | original xyxy={crop}; same window for all panels',(15,30),size=.67)
        im,org,sc=pane(gray,crop,cw);seed_overlay(im,seed,org,sc)
        x0,y0,x1,y1=crop;top=[(im,'Original + supplied entry'),(cv2.resize(heat(responses['A'][y0:y1,x0:x1]),(cw,ch)),'A raw 0-1'),(cv2.resize(heat(responses['B'][y0:y1,x0:x1]),(cw,ch)),'B raw 0-1')]
        for i,(im,label) in enumerate(top):text(fig,label,(i*cw+15,65));fig[80:80+ch,i*cw:(i+1)*cw]=im
        # Four pathway panels are separate files below to preserve full crop resolution.
        for i,m in enumerate(['A','B']):
            overlay,org,sc=pane(gray,crop,cw)
            for s in bundles[m]['segments']:cv2.polylines(overlay,[np.rint((np.array(s['points_xy'])-org)*sc).astype(np.int32)],False,(0,210,245),1,cv2.LINE_AA)
            seed_overlay(overlay,seed,org,sc);text(fig,f'{m} derived ridge segments, no bridges',(i*cw+15,ch+125),size=.6)
            fig[ch+140:ch+140+ch,i*cw:(i+1)*cw]=overlay
        text(fig,'Yellow = extracted center fragments',(2*cw+15,ch+175),size=.62)
        text(fig,'No ground-truth path corridor used.',(2*cw+15,ch+215),size=.62)
        text(fig,'Post-transition is a visual region proxy,',(2*cw+15,ch+255),size=.56)
        text(fig,'not confirmed physical contact/identity.',(2*cw+15,ch+290),size=.56)
        save(out/f'{name}__gray_heat_ridges.png',fig)
        pathfig=np.full((2*(ch+70)+75,2*cw,3),250,np.uint8)
        for i,key in enumerate(combo):
            col=i%2;rr=i//2;y=rr*(ch+70);r=rows[key];state='selected' if r['selected_candidate_id'] else 'rejected candidate / empty'
            text(pathfig,f'{key[0]}/{key[1]} {state}',(col*cw+12,y+30),size=.6)
            pathfig[y+50:y+50+ch,col*cw:(col+1)*cw]=paths_panel(gray,r,crop,cw)
        text(pathfig,'Green/cyan local support; magenta bridge; orange-red ambiguity. Dashed phase accumulates over short edges.',(15,2*(ch+70)+40),size=.53)
        save(out/f'{name}__paths_four_combinations.png',pathfig)
        windows.append({'name':name,'center_xy':center.tolist(),'bbox_xyxy':crop,'selection_basis':'seed/scene only except explicitly named P1 stop diagnostic; no truth or scoring use'})
    # Cached derivatives have their own names and their own fixed contrast scale.
    workshape=derived['A']['filtered_response'].shape;dw=900;dh=round(workshape[0]*dw/workshape[1]);fig=np.full((2*(dh+65)+120,dw*3,3),250,np.uint8)
    for rr,m in enumerate(['A','B']):
        data=derived[m];strength=np.zeros(data['working_shape'],np.float32);yx=data['ridge_yx_working'];strength[yx[:,0],yx[:,1]]=data['ridge_strength']
        mats=[heat(data['filtered_response']),cv2.cvtColor(data['candidate_mask'].astype(np.uint8)*255,cv2.COLOR_GRAY2BGR),heat(strength)]
        titles=[f'{m}: filtered response 0-1',f'{m}: binary peak candidates (white)',f'{m}: derived ridge contrast 0-1']
        for col,(im,label) in enumerate(zip(mats,titles)):
            y=rr*(dh+65);text(fig,label,(col*dw+15,y+30));fig[y+50:y+50+dh,col*dw:(col+1)*dw]=cv2.resize(im,(dw,dh),interpolation=cv2.INTER_NEAREST if col>0 else cv2.INTER_LINEAR)
    colorbar(fig,20,2*(dh+65)+20,800,'Fixed 0-1, derived quantities; binary mask is separate. Raw response unchanged.')
    save(out/'derived_filtered_candidates_ridge_contrast.png',fig)
    # Tight entrance view makes fixed projection rejection inspectable.
    crop=crop_box(seed[-1],gray.shape,360,240);cw=720;ch=480;fig=np.full((2*(ch+100)+80,2*cw,3),250,np.uint8)
    for rr,m in enumerate(['A','B']):
        bundle=bundles[m];distances=[]
        for s in bundle['segments']:
            p=np.array(s['points_xy']);j=int(np.argmin(np.linalg.norm(p-seed[-1],axis=1)));distances.append((float(np.linalg.norm(p[j]-seed[-1])),s['id'],p[j]))
        nearest=min(distances,key=lambda x:x[0]) if distances else None
        im,org,sc=pane(gray,crop,cw)
        for s in bundle['segments']:cv2.polylines(im,[np.rint((np.array(s['points_xy'])-org)*sc).astype(np.int32)],False,(0,210,245),2)
        seed_overlay(im,seed,org,sc)
        if nearest:
            q=np.rint((nearest[2]-org)*sc).astype(int);cv2.circle(im,tuple(q),7,(20,30,240),2)
        y=rr*(ch+100);text(fig,f'{m}: seed + cached ridge; nearest {nearest[0]:.3f}px' if nearest else f'{m}: no segment',(12,y+30),size=.6)
        fig[y+65:y+65+ch,:cw]=im
        x0,y0,x1,y1=crop;fig[y+65:y+65+ch,cw:]=cv2.resize(heat(responses[m][y0:y1,x0:x1]),(cw,ch))
        why=rows[(m,'P1')]['search'].get('selection_reason','unknown')
        text(fig,f'P1: {why}',(cw+12,y+30),size=.48)
    text(fig,'Red circle = nearest extracted sample, not truth. Projection/forward-gate diagnostics retain frozen decisions.',(15,2*(ch+100)+40),size=.54)
    save(out/'entry_projection_diagnostic.png',fig)
    return windows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    run=args.run.resolve();out=run/'example_pair0075';out.mkdir(exist_ok=False)
    base=run/'batch/conditions'/CASE;summary={'case':CASE,'source_group':'dual_20260923_101517','split_scope':'historically used training source; development display only, not independent evaluation','entries':{},'source_hashes':{}}
    # A 32px line split into 4px pieces must have 22px visible dashes at18/10.
    drawn=sum(hi-lo for i in range(8) for lo,hi in dash_parts(4.,4.*i));assert abs(drawn-22)<1e-7
    for entry in ['entry_01','entry_02']:
        dest=out/entry;dest.mkdir();rows={(m,p):load(base/entry/m/p/'paths.json') for m in ['A','B'] for p in ['P0','P1']}
        first=rows[('A','P0')];image=Path(first['image_path']);assert sha(image)==first['image_sha256']
        gray=cv2.imdecode(np.fromfile(image,np.uint8),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
        responses={};bundles={};derived={};entrysum={'visible_entry_type':'hand-held metal tool' if entry=='entry_01' else 'visible free rope; distal identity unverified','combinations':{},'metadata':{}}
        for m in ['A','B']:
            inf=load(base/entry/m/'inference.json');assert inf['step']==750
            p=Path(first['response_path']) if m=='A' else Path(rows[(m,'P0')]['response_path'])
            digest=sha(p);assert digest==inf['response_sha256']==rows[(m,'P0')]['response_sha256']==rows[(m,'P1')]['response_sha256']
            responses[m]=np.load(p,mmap_mode='r');bundles[m]=load(rows[(m,'P1')]['segments_path'])
            with np.load(rows[(m,'P1')]['derived_arrays_path']) as z:derived[m]={k:z[k] for k in z.files}
            entrysum['metadata'][m]={'response_path':str(p),'sha256':digest,'step':750,'checkpoint':inf['checkpoint'],'actual_seed_xy':inf['entry_xy'],'input_tensor_sha256':inf['input_tensor_sha256']}
        windows=entry_figures(dest,entry,gray,rows,responses,bundles,derived);entrysum['windows']=windows
        for key,r in rows.items():
            selected=next((c for c in r['candidates'] if c['id']==r['selected_candidate_id']),None)
            candidates=[{'id':c['id'],'lengths_px':lengths(c),'termination':c['termination'],'end_xy':c['points_xy'][-1]} for c in r['candidates']]
            entrysum['combinations']['/'.join(key)]={'selected':r['selected_candidate_id'],'main_lengths_px':lengths(selected),'candidates':candidates,
                'stop_or_rejection':selected['termination'] if selected else r.get('search',{}).get('selection_reason','rejected'),
                'entry_candidate_checks':r.get('search',{}).get('entry_candidate_checks'),
                'scope':'P0 support is legacy local image+response proxy; P1 heatmap/image evidence is not independent target identity.'}
            path=base/entry/key[0]/key[1]/'paths.json';summary['source_hashes'][str(path)]=sha(path)
        summary['entries'][entry]=entrysum;print(entry,'rendered',flush=True)
    summary.update(exit_code=0,command=sys.argv,source_sha256=sha(__file__),dashes_accumulate_across_4px_edges=True,
        original_gray_display=[0,1],raw_response_display=[0,1],derived_contrast_display=[0,1],
        no_model_or_postprocessor_execution=True,no_algorithms_or_predictions_modified=True,no_full_identity_reference=True)
    summary['outputs']=[{'path':str(p),'sha256':sha(p)} for p in sorted(out.rglob('*.png'))]
    dump(out/'diagnostics.json',summary)
    report=['# pair0075 双入口冻结结果诊断','',
        '本例属于历史训练来源 dual_20260923_101517，只作开发展示。A/B均使用共同750步权重与各自对应入口；没有训练、重算网络或修改后处理。两个入口分别保留，未择优。','',
        '## 入口1：左下手持金属工具','',
        '原图可见入口1（recovered_01）落在手持金属工具上，不是右侧活动自由绳。P0的A/B均暂选一条长路径并碰到观察边缘，不能称为目标绳识别。P1两组都没有入口相容片段，新增输出支持与桥接均为0；A/B最近片段分别距入口提示末端约40.48/48.32px，超出冻结40px接入范围。该拒识保留，不把它解释为已解决自动入口分类。','',
        '## 入口2：右侧可见自由绳','',
        '入口2（recovered_00）位于右侧清楚可见的自由绳。P0两组主输出均拒识，图中仅展示候选诊断；长候选的连续性不代表接续正确。','',
        'A/P1已提取靠近入口的片段，但最近采样点距提示末端4.353px，稍大于固定4px直接投影容差；连接方向约122.57°触发前向gap限制，尽管片段切向差只有约12.22°。因此它是在入口接入规则处被拒绝，不能归结为热图完全没有线。正式冻结后没有放宽该条件。','',
        'B/P1只暂选了一条短局部路径：排除120px已给入口后，新增局部支持540.70px，桥接23.63px，停止点约[3426.5,2138.5]。原图中自由绳仍可继续辨认；该输出未到视觉过渡区，更没有证明贴面后身份保持或完整到边缘。它的停止原因是短缺口片段图无后续可接受连接。','',
        '## 图示与统计口径','',
        '原灰度和原始A/B响应统一固定0～1；域外NaN为灰色。派生滤波响应、二值峰候选和脊线对比分别命名另图，未覆盖原响应。绿色/蓝色实线为局部证据，紫红虚线为桥，橙红虚线为歧义，其他候选另色。虚线按整条路径累计弧长，跨4px小边连续计相位。中心线只是细线示意，不是真实绳体宽度mask。','',
        '主输出统计将拒识记为0；另外保留候选长度，不能混为已识别长度。局部窗口由入口、同一场景视觉过渡代理和明确标出的P1停止位置确定，没有用完整目标真值裁走廊或反画标签。','',
        '|入口|组合|暂选|新增支持px（主输出）|桥接px（主输出）|候选数|','|---|---|---|---:|---:|---:|']
    for en,e in summary['entries'].items():
        for key,r in e['combinations'].items():
            report.append(f'|{en}|{key}|{r["selected"] or "拒识"}|{r["main_lengths_px"]["supported_new"]:.2f}|{r["main_lengths_px"]["bridge"]:.2f}|{len(r["candidates"])}|')
    report+=['','全部逐候选停止原因、支持/桥/歧义长度、输入哈希与路径见 diagnostics.json。没有独立完整身份参考，不计算本例完整成功率。']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({'exit_code':0,'figures':len(summary['outputs']),'out':str(out)}))


if __name__=='__main__':main()
