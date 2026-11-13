"""Official UniDepthV2 RGB-only inference; no intrinsics or GT passed in."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
import numpy as np
import runtime
os.environ.setdefault('WANDB_MODE','disabled')
os.environ.setdefault('HF_HUB_OFFLINE','1')


class MetricModel:
    def __init__(self, model_dir, device='auto', threads=8, resolution_level=0):
        import torch
        from unidepth.models import UniDepthV2
        torch.set_num_threads(threads)
        self.torch=torch
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else torch.device(device)
        if self.device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        config=json.loads((Path(model_dir)/'config.json').read_text(encoding='utf-8'))
        # Exact official class/architecture and a strict full checkpoint load.
        self.model=UniDepthV2.from_pretrained(str(Path(model_dir).resolve()), config=config,
                                            local_files_only=True,strict=True,map_location='cpu')
        self.model.eval().to(self.device)
        self.model.resolution_level=resolution_level
        self.encoder_shape=None
        self.model.pixel_encoder.register_forward_pre_hook(self._shape_hook)
        self.info={'model':'UniDepthV2 ViT-S', 'device':str(self.device),'torch':torch.__version__,
                   'precision':'CPU FP32' if self.device.type=='cpu' else 'official infer CUDA FP16 autocast',
                   'threads':threads,'resolution_level':resolution_level,
                   'parameters':sum(p.numel() for p in self.model.parameters()),
                   'checkpoint_load':'official from_pretrained(local directory), strict=True', 'RGB_only':True,
                   'preprocessing':'Official model.infer(uint8 CHW RGB); normalize/pad/resize internally',
                   'hardware':torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else __import__('platform').processor()}

    def _shape_hook(self,module,inputs): self.encoder_shape=list(inputs[0].shape)

    def infer(self,rgb):
        torch=self.torch
        if rgb.dtype!=np.uint8 or rgb.shape!=(480,640,3): raise ValueError('Expected uint8 RGB 480x640x3')
        tensor=torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)
        if self.device.type=='cuda':
            torch.cuda.synchronize(self.device);torch.cuda.reset_peak_memory_stats(self.device)
        start=time.perf_counter()
        with torch.inference_mode(): output=self.model.infer(tensor)
        if self.device.type=='cuda': torch.cuda.synchronize(self.device)
        ms=(time.perf_counter()-start)*1000
        depth=output['depth'].detach().float().cpu().numpy().squeeze().copy()
        if depth.ndim!=2 or not np.isfinite(depth).all(): raise ValueError('Invalid depth output')
        peak=torch.cuda.max_memory_allocated(self.device)/1024**2 if self.device.type=='cuda' else None
        info={'inference_ms':ms,'gpu_peak_allocated_mb':peak,'encoder_shape':self.encoder_shape,
              'output_shape':list(depth.shape)}
        try:
            import psutil
            info['process_rss_mb']=psutil.Process().memory_info().rss/1024**2
        except ImportError: pass
        return depth,info
