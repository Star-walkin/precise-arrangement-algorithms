"""Synthetic descriptor/coordinate contract; no scene data or target labels."""
import json
from pathlib import Path
import numpy as np
import r1_pipeline as pipeline

HERE=Path(__file__).resolve().parent
def main():
    pipeline.cv2.setNumThreads(1)
    _,spatial,_=pipeline.original_modules()
    feature=np.broadcast_to(np.arange(8,dtype=np.float16)[:,None,None],(8,248,512)).copy()
    heat=np.full((975,2048),.4,np.float32);gray=np.full((3000,4096),.2,np.float32)
    seed=[[500,1500],[620,1500]]
    context=spatial.Context(gray,heat,feature,seed)
    history=[[620,1500],[644,1500],[668,1500],[692,1500]]
    proposed=context.propose(history);points=np.asarray([p for p,_ in proposed])
    descriptor,evidence=context.descriptor(history,points)
    assert descriptor.shape==(len(points),197)
    assert np.array_equal(descriptor[:,:48],np.tile(np.arange(8),(len(points),6)))
    assert np.allclose(descriptor[:,48:54],.4)
    assert np.all(descriptor[:,-8:]==0) and np.isfinite(descriptor).all()
    assert len(points)<=spatial.P['max_proposals']
    distance=pipeline.distance_to_segments([[5,3],[12,0]],[ [[0,0],[10,0]] ])
    assert np.allclose(distance,[3,2])
    segments=[{'kind':'target_center','points_xy':[[100,0],[50,0]]},
              {'kind':'target_center','points_xy':[[20,0],[40,0]]}]
    ordered=pipeline.ordered_centers(segments,[[0,0],[10,0]])
    assert np.array_equal(ordered[0],[[20,0],[40,0]]) and np.array_equal(ordered[1],[[50,0],[100,0]])
    candidate=dict(id='synthetic',points_xy=[[0,0],[1,1],[2,2]],point_status=['entry_condition','visible_supported','inferred'],
                   rank_score=.3,termination='synthetic',length_px=2.8,status_length_px={})
    mapped=pipeline.flat_candidate(candidate)
    assert mapped['supported']==[False,True,False] and mapped['inferred']==[False,False,True]
    result=dict(passed=True,synthetic_only=True,descriptor_dimensions=197,dec1_six_spatial_positions_verified=True,
                heat_mapping_verified=True,missing_tracks_explicitly_zero=True,proposal_budget_verified=True,
                polyline_distance_verified=True,fragment_orientation_verified=True,entry_not_recognition_verified=True,
                source_sha256=pipeline.sha(HERE/'r1_pipeline.py'))
    pipeline.dump(HERE/'r1_numerical_checks.json',result)
    print(json.dumps(result))
if __name__=='__main__':main()
