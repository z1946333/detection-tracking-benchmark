"""Deep OC-SORT + OSNet-x0.25, adapted online configuration, not paper reproduction."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from src.deep_ocsort_adapter import OSNetEmbedder, make_tracker


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequences', nargs='+', default=['191', '192', '193'])
    p.add_argument('--data-root', type=Path, default=Path('D:/Datasets/dataset_B/PIG_MOT_TEST_DATA'))
    p.add_argument('--detector', type=Path, default=ROOT/'experiments/detection/yolov8n/weights/best.pt')
    p.add_argument('--reid-weights', type=Path, default=ROOT/'third_party/StrongSORT/osnet_x0_25_msmt17.pt')
    p.add_argument('--output', type=Path, default=ROOT/'evaluation/tracking/deep_ocsort_osnet_x025')
    p.add_argument('--max-frames', type=int, default=0, help='0: all frames; positive: smoke test only')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.max_frames < 0 or len(set(args.sequences)) != len(args.sequences):
        p.error('Invalid frame count or duplicate sequences')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Use a new --output, refusing overwrite: {args.output}')
    sequences = {}
    for seq in args.sequences:
        paths = sorted((args.data_root/seq/'img1').glob('*.jpg'), key=lambda f:int(f.stem))
        if not paths or [int(f.stem) for f in paths] != list(range(1,len(paths)+1)):
            raise ValueError(f'Frames missing or not contiguous from 1: {seq}')
        sequences[seq] = paths[:args.max_frames] if args.max_frames else paths
    torch.set_num_threads(4)
    embedder = OSNetEmbedder(ROOT,args.reid_weights,args.device)
    model = YOLO(str(args.detector))
    (args.output/'data').mkdir(parents=True)
    manifest = {'status':'running','algorithm':'Deep OC-SORT + OSNet-x0.25 (adapted)',
        'smoke_test':bool(args.max_frames),'original_paper_configuration':False,
        'detector_sha256':sha256(args.detector),'reid_sha256':sha256(args.reid_weights),
        'settings':{'det_conf':.1,'det_imgsz':640,'det_iou':.7,'track_thresh':.3,
            'reid':'osnet_x0_25','reid_crop_hw':[256,128],'reid_normalization':'RGB ImageNet',
            'strict_loading':True,'grid_off':True,'new_kf_off':True,'embedding_off':False,
            'aw_off':False,'cmc':'sparse optical flow','postprocessing':False},
        'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'source_hashes':{str(f.relative_to(ROOT)):sha256(f) for f in
            [Path(__file__),ROOT/'src/deep_ocsort_adapter.py',*sorted((ROOT/'src/vendor/deep_ocsort').glob('*.py'))]},
        'sequences':{}}
    def save():
        (args.output/'run.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    try:
        for seq,paths in sequences.items():
            tracker=make_tracker(ROOT,embedder)
            lines=[]; started=time.perf_counter(); calls_before=embedder.calls
            for frame_id,path in enumerate(paths,1):
                img=cv2.imread(str(path))
                if img is None: raise ValueError(f'Cannot read {path}')
                h,w=img.shape[:2]
                pred=model.predict(img,conf=.1,iou=.7,imgsz=640,device=args.device,verbose=False)[0]
                dets=pred.boxes.data.cpu().numpy()
                dets=dets[dets[:,5]==0,:5].copy()
                dets[:,[0,2]]=np.clip(dets[:,[0,2]],0,w)
                dets[:,[1,3]]=np.clip(dets[:,[1,3]],0,h)
                dets=dets[(dets[:,2]>dets[:,0])&(dets[:,3]>dets[:,1])]
                # Already in original image coordinates: scale must remain 1.
                shape=torch.empty((1,3,h,w),device='meta')
                tracks=tracker.update(dets,shape,img,f'{seq}:{frame_id}')
                if not np.isfinite(tracks).all(): raise ValueError('Non-finite tracks')
                for x1,y1,x2,y2,tid in tracks:
                    lines.append(f'{frame_id},{int(tid)},{x1:.3f},{y1:.3f},{x2-x1:.3f},{y2-y1:.3f},1,-1,-1,-1\n')
                if frame_id%100==0 or frame_id==len(paths):
                    print(f'{seq}: {frame_id}/{len(paths)} frames, ReID calls={embedder.calls-calls_before}',flush=True)
            dest=args.output/'data'/f'{seq}.txt'
            dest.write_text(''.join(lines),encoding='utf-8')
            manifest['sequences'][seq]={'frames':len(paths),'rows':len(lines),
                'reid_calls':embedder.calls-calls_before,'elapsed_seconds':time.perf_counter()-started,
                'output_sha256':sha256(dest)}
            save()
        manifest['status']='completed';save()
        print(f'Completed: {args.output}',flush=True)
    except BaseException as exc:
        manifest['status']='failed';manifest['error']=repr(exc);save();raise


if __name__=='__main__':
    main()
