"""Real optional-environment integration check using an existing operator prompt."""
from pathlib import Path
import json
import cv2
import numpy as np
from .geometry import read_image, write_image, analyze_frame, draw_overlay, dump_json
from .sam_bridge import sam_rope_proposal


def main():
    root = Path(__file__).resolve().parents[1]
    base = root/'vision_training/task_dual_midlayer_v1/holdout'
    name = 'dual_20260923_101643__pair0100__right'
    frame = read_image(base/'frames'/(name+'.jpg'))
    annotation = json.loads((base/'labels'/name/'annotation.json').read_text(encoding='utf-8'))
    prompts = annotation['classes']['incoming_rope_free_span']['prompts']
    prior = read_image(base/'labels'/name/'incoming_rope_free_span.png')[:,:,0] > 0
    y,x = np.where(prior)
    roi = [max(0,int(x.min())-80),max(0,int(y.min())-80),
           min(frame.shape[1],int(x.max())+81),min(frame.shape[0],int(y.max())+81)]
    points = [item['point'] for item in prompts]
    labels = [item['label'] for item in prompts]
    mask, metadata = sam_rope_proposal(frame,points,labels,roi)
    result = analyze_frame(frame,rope_mask=mask,seed_points=[p for p,l in zip(points,labels) if l])
    result.update(sam2=metadata, prompt_source='existing_operator_annotation',
                  roi_source='existing_mask_bounding_box_for_integration_check_only',
                  accuracy_status='not_independent_accuracy_validation')
    output = Path(__file__).parent/'outputs/sam_integration'
    dump_json(output/'result.json',result)
    write_image(output/'mask.png',mask.astype(np.uint8)*255)
    write_image(output/'overlay.jpg',draw_overlay(frame,result))
    print(json.dumps(dict(sam2=metadata,arc=result['arc']['valid'],rope=result['rope']['valid'],
                         rope_reason=result['rope']['reason']),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
