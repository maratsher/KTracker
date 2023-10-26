import os
import time
from utils import clip_box, sample_target
import onnxruntime
import multiprocessing
import numpy as np


class Krolik():
    def __init__(self, params, debug=False):
        self.debug = debug
        self.params = params
        self.visdom = None
        num_gpu = 2
        print("total number of GPUs is %d, change it if it is not matched with your machine." % num_gpu)
        try:
            worker_name = multiprocessing.current_process().name
            worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
            gpu_id = worker_id % num_gpu
            print(gpu_id)
        except:
            gpu_id = 0
            
        providers = ["CUDAExecutionProvider"]
        provider_options = [{"device_id": str(gpu_id)}]
        self.ort_sess_z = onnxruntime.InferenceSession("weights/backbone.onnx", providers=providers,
                                                       provider_options=provider_options)
        self.ort_sess_x = onnxruntime.InferenceSession("weights/head.onnx", providers=providers,
                                                       provider_options=provider_options)
        self.state = None
        # for debug
        self.frame_id = 0
        # if self.debug:
        #     self.save_dir = "debug"
        #     if not os.path.exists(self.save_dir):
        #         os.makedirs(self.save_dir)
        self.ort_outs_z = []

    def preprocessing(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        """img_arr: (H,W,3), amask_arr: (H,W)"""
        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
        # Deal with the image patch
        img_arr_4d = img_arr[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        img_arr_4d = (img_arr_4d / 255.0 - mean) / std  # (1, 3, H, W)
        # Deal with the attention mask
        amask_arr_3d = amask_arr[np.newaxis, :, :]  # (1,H,W)
        return img_arr_4d.astype(np.float32), amask_arr_3d.astype(np.bool_) 

    def initialize(self, image, info: dict):

        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        template, template_mask = self.preprocessing(z_patch_arr, z_amask_arr)
        # forward the template once
        ort_inputs = {'img_z': template, 'mask_z': template_mask}
        t1 = time.time()
        self.ort_outs_z = self.ort_sess_z.run(None, ort_inputs)
        t2 = time.time()

        if self.debug:
            print(f"Получение фич: {t2 - t1}")

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        t1 = time.time()
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        
        search, search_mask = self.preprocessing(x_patch_arr, x_amask_arr)

        t2 = time.time()

        if self.debug:
            print(f"Предобработка:: {t2 - t1}")

        t1 = time.time()

        ort_inputs = {'img_x': search,
                      'mask_x': search_mask,
                      'feat_vec_z': self.ort_outs_z[0],
                      'mask_vec_z': self.ort_outs_z[1],
                      'pos_vec_z': self.ort_outs_z[2],
                      }

        ort_outs = self.ort_sess_x.run(None, ort_inputs)

        t2 = time.time()

        if self.debug:
            print(f"Получение позиции: {t2 - t1}")

        t1 = time.time()
        pred_box = (ort_outs[0].reshape(4) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        t2 = time.time()

        if self.debug:
            print(f"Постпроцессинг: {t2 - t1}")
        # for debug
        if self.debug:
            pass

        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

