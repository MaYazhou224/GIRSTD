import os.path as osp
from typing import Dict, Sequence

import numpy as np
from mmengine.logging import MMLogger, print_log
from PIL import Image

from mmseg.registry import METRICS
from mmseg.evaluation.metrics.iou_metric import IoUMetric
from collections import defaultdict
import scipy.ndimage as ndi  # 用于目标统计
from skimage import measure



# @METRICS.register_module()
# class DGIoUMetric(IoUMetric):
#     def __init__(self, dataset_keys=[], mean_used_keys=[], **kwargs):
#         super().__init__(**kwargs)
#         self.dataset_keys = dataset_keys
#         if mean_used_keys:
#             self.mean_used_keys = mean_used_keys
#         else:
#             self.mean_used_keys = dataset_keys

#     def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
#         """Process one batch of data and data_samples.

#         The processed results should be stored in ``self.results``, which will
#         be used to compute the metrics when all batches have been processed.

#         Args:
#             data_batch (dict): A batch of data from the dataloader.
#             data_samples (Sequence[dict]): A batch of outputs from the model.
#         """
#         num_classes = len(self.dataset_meta["classes"])
#         for data_sample in data_samples:
#             pred_label = data_sample["pred_sem_seg"]["data"].squeeze()
#             # format_only always for test dataset without ground truth
#             if not self.format_only:
#                 label = data_sample["gt_sem_seg"]["data"].squeeze().to(pred_label)
#                 res1, res2, res3, res4 = self.intersect_and_union(
#                     pred_label, label, num_classes, self.ignore_index
#                 )
#                 dataset_key = "unknown"
#                 for key in self.dataset_keys:
#                     if key in data_samples[0]["seg_map_path"]:
#                         dataset_key = key
#                         break
#                 self.results.append([dataset_key, res1, res2, res3, res4])
#             # format_result
#             if self.output_dir is not None:
#                 basename = osp.splitext(osp.basename(data_sample["img_path"]))[0]
#                 png_filename = osp.abspath(osp.join(self.output_dir, f"{basename}.png"))
#                 output_mask = pred_label.cpu().numpy()
#                 # The index range of official ADE20k dataset is from 0 to 150.
#                 # But the index range of output is from 0 to 149.
#                 # That is because we set reduce_zero_label=True.
#                 if data_sample.get("reduce_zero_label", False):
#                     output_mask = output_mask + 1
#                 output = Image.fromarray(output_mask.astype(np.uint8))
#                 output.save(png_filename)

#     def compute_metrics(self, results: list) -> Dict[str, float]:
#         """Compute the metrics from processed results.

#         Args:
#             results (list): The processed results of each batch.

#         Returns:
#             Dict[str, float]: The computed metrics. The keys are the names of
#                 the metrics, and the values are corresponding results. The key
#                 mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
#                 mRecall.
#         """
#         dataset_results = defaultdict(list)
#         metrics = {}
#         for result in results:
#             dataset_results[result[0]].append(result[1:])
#         metrics_type2mean = defaultdict(list)
#         for key, key_result in dataset_results.items():
#             logger: MMLogger = MMLogger.get_current_instance()
#             print_log(f"----------metrics for {key}------------", logger)
#             key_metrics = super().compute_metrics(key_result)
#             print_log(f"number of samples for {key}: {len(key_result)}")
#             for k, v in key_metrics.items():
#                 metrics[f"{key}_{k}"] = v
#                 if key in self.mean_used_keys:
#                     metrics_type2mean[k].append(v)
#         for k, v in metrics_type2mean.items():
#             metrics[f"mean_{k}"] = sum(v) / len(v)
#         return metrics

@METRICS.register_module()
class DGIoUMetric(IoUMetric):
    def __init__(self, dataset_keys=[], mean_used_keys=[], **kwargs):
        super().__init__(**kwargs)
        self.dataset_keys = dataset_keys
        if mean_used_keys:
            self.mean_used_keys = mean_used_keys
        else:
            self.mean_used_keys = dataset_keys

        # 累加器初始化
        self.total_targets = 0.0
        self.total_fa = 0.0
        self.total_pd = 0.0
        self.total_pixels = 0.0
        self.total_nIoU = 0.0
        self.valid_nIoU_count = 0  # 用于计算 nIoU 的有效样本数

    def cal_Pd_Fa(self, pred_mask, true_mask, distance_thresh=3):
        """计算 Pd 和 Fa。"""
        pred_label = measure.label(pred_mask, connectivity=2)
        pred_props = measure.regionprops(pred_label)

        true_label = measure.label(true_mask, connectivity=2)
        true_props = measure.regionprops(true_label)

        total_targets = len(true_props)
        matched_preds = []
        matched_area = []

        all_pred_area = sum([p.area for p in pred_props])

        for true_obj in true_props:
            true_centroid = np.array(true_obj.centroid)
            for i, pred_obj in enumerate(pred_props):
                pred_centroid = np.array(pred_obj.centroid)
                dist = np.linalg.norm(pred_centroid - true_centroid)
                if dist < distance_thresh:
                    matched_preds.append(pred_obj)
                    matched_area.append(pred_obj.area)
                    del pred_props[i]
                    break

        pd = len(matched_preds)   # 成功匹配的目标数
        fa = all_pred_area - sum(matched_area) # 误报的像素（所有预测像素 - 成功匹配的像素）

        return total_targets, fa, pd   # 真实目标总数、误报像素数、成功预测目标数。

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        num_classes = len(self.dataset_meta["classes"])
        for data_sample in data_samples:
            pred_label = data_sample["pred_sem_seg"]["data"].squeeze()  #预测值，二维

            if not self.format_only:
                label = data_sample["gt_sem_seg"]["data"].squeeze().to(pred_label) # 确保标签与预测标签在同一设备上,数据类型保持一致 gt值，二维

                res1, res2, res3, res4 = self.intersect_and_union(
                    pred_label, label, num_classes, self.ignore_index
                )   # 	每类的 TP（预测正确像素数）,每类的 Union,每类预测为该类别的像素数（TP + FP）,每类真实的像素数（TP + FN）                
                #  IoU = TP / Union
                # Accuracy = TP / area_label
                # Precision = TP / area_pred_label
                # Recall = TP / area_label

                dataset_key = "unknown"
                for key in self.dataset_keys:
                    if key in data_sample["seg_map_path"]:
                        dataset_key = key
                        break

                # 二值化目标（假设目标类为1，背景为0）
                pred_np = pred_label.cpu().numpy()
                label_np = label.cpu().numpy()
                pred_bin = (pred_np == 1).astype(np.uint8)
                label_bin = (label_np == 1).astype(np.uint8)

                target, fa, pd = self.cal_Pd_Fa(pred_bin, label_bin)

                self.total_targets += target
                self.total_fa += fa
                self.total_pd += pd
                self.total_pixels += label_np.size
                if res2[1] > 0:
                    self.total_nIoU += res1[1] / res2[1]  # 计算 nIoU
                    self.valid_nIoU_count += 1  # 有效样本计数


                self.results.append([dataset_key, res1, res2, res3, res4])

            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(data_sample["img_path"]))[0]
                png_filename = osp.abspath(osp.join(self.output_dir, f"{basename}.png"))
                output_mask = pred_label.cpu().numpy()
                if data_sample.get("reduce_zero_label", False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        dataset_results = defaultdict(list)
        metrics = {}
        for result in results:
            dataset_results[result[0]].append(result[1:])

        metrics_type2mean = defaultdict(list)

        for key, key_result in dataset_results.items():
            logger: MMLogger = MMLogger.get_current_instance()
            print_log(f"----------metrics for {key}------------", logger)
            key_metrics = super().compute_metrics(key_result)
            print_log(f"number of samples for {key}: {len(key_result)}")

            for k, v in key_metrics.items():
                metrics[f"{key}_{k}"] = v
                if key in self.mean_used_keys:
                    metrics_type2mean[k].append(v)

        for k, v in metrics_type2mean.items():
            metrics[f"mean_{k}"] = sum(v) / len(v)

        # 计算Pd和Fa
        Pd = self.total_pd / self.total_targets if self.total_targets > 0 else 0.0
        Fa = self.total_fa / self.total_pixels if self.total_pixels > 0 else 0.0

        metrics["Pd"] = Pd * 100
        metrics["Fa"] = Fa * 1000000
        metrics["nIoU"] = self.total_nIoU * 100 / self.valid_nIoU_count if self.valid_nIoU_count > 0 else 0.0

        return metrics


# 增加nIoU


# @METRICS.register_module()
# class DGIoUMetric(IoUMetric):
#     def __init__(self, dataset_keys=[], mean_used_keys=[], **kwargs):
#         super().__init__(**kwargs)
#         self.dataset_keys = dataset_keys
#         self.mean_used_keys = mean_used_keys if mean_used_keys else dataset_keys

#         # Pd / Fa 统计累加器
#         self.total_targets = 0
#         self.total_fa = 0
#         self.total_pd = 0
#         self.total_pixels = 0

#         # 用于 nIoU 统计
#         self.sample_ious = []

#     def cal_Pd_Fa(self, pred_mask, true_mask, distance_thresh=3):
#         """计算检测率 Pd 和虚警率 Fa。"""
#         pred_label = measure.label(pred_mask, connectivity=2)
#         pred_props = measure.regionprops(pred_label)

#         true_label = measure.label(true_mask, connectivity=2)
#         true_props = measure.regionprops(true_label)

#         total_targets = len(true_props)
#         matched_preds = []
#         matched_area = []

#         all_pred_area = sum([p.area for p in pred_props])

#         for true_obj in true_props:
#             true_centroid = np.array(true_obj.centroid)
#             for i, pred_obj in enumerate(pred_props):
#                 pred_centroid = np.array(pred_obj.centroid)
#                 dist = np.linalg.norm(pred_centroid - true_centroid)
#                 if dist < distance_thresh:
#                     matched_preds.append(pred_obj)
#                     matched_area.append(pred_obj.area)
#                     del pred_props[i]
#                     break

#         pd = len(matched_preds)
#         fa = all_pred_area - sum(matched_area)
#         return total_targets, fa, pd

#     def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
#         num_classes = len(self.dataset_meta["classes"])
#         for data_sample in data_samples:
#             pred_label = data_sample["pred_sem_seg"]["data"].squeeze()

#             if not self.format_only:
#                 label = data_sample["gt_sem_seg"]["data"].squeeze().to(pred_label)

#                 res1, res2, res3, res4 = self.intersect_and_union(
#                     pred_label, label, num_classes, self.ignore_index
#                 )

#                 dataset_key = "unknown"
#                 for key in self.dataset_keys:
#                     if key in data_sample["seg_map_path"]:
#                         dataset_key = key
#                         break

#                 # === IoU per sample ===
#                 class_idx = 1  # 假设前景类是 1
#                 inter = res1[class_idx]
#                 union = res3[class_idx]
#                 if union > 0:
#                     self.sample_ious.append(inter / union)

#                 # === Pd 和 Fa ===
#                 pred_np = pred_label.cpu().numpy()
#                 label_np = label.cpu().numpy()
#                 pred_bin = (pred_np == class_idx).astype(np.uint8)
#                 label_bin = (label_np == class_idx).astype(np.uint8)

#                 target, fa, pd = self.cal_Pd_Fa(pred_bin, label_bin)

#                 self.total_targets += target
#                 self.total_fa += fa
#                 self.total_pd += pd
#                 self.total_pixels += label_np.size

#                 self.results.append([dataset_key, res1, res2, res3, res4])

#             if self.output_dir is not None:
#                 basename = osp.splitext(osp.basename(data_sample["img_path"]))[0]
#                 png_filename = osp.abspath(osp.join(self.output_dir, f"{basename}.png"))
#                 output_mask = pred_label.cpu().numpy()
#                 if data_sample.get("reduce_zero_label", False):
#                     output_mask = output_mask + 1
#                 output = Image.fromarray(output_mask.astype(np.uint8))
#                 output.save(png_filename)

#     def compute_metrics(self, results: list) -> Dict[str, float]:
#         dataset_results = defaultdict(list)
#         metrics = {}

#         for result in results:
#             dataset_results[result[0]].append(result[1:])

#         metrics_type2mean = defaultdict(list)
#         for key, key_result in dataset_results.items():
#             logger: MMLogger = MMLogger.get_current_instance()
#             print_log(f"----------metrics for {key}------------", logger)
#             key_metrics = super().compute_metrics(key_result)
#             print_log(f"number of samples for {key}: {len(key_result)}")

#             for k, v in key_metrics.items():
#                 metrics[f"{key}_{k}"] = v
#                 if key in self.mean_used_keys:
#                     metrics_type2mean[k].append(v)

#         for k, v in metrics_type2mean.items():
#             metrics[f"mean_{k}"] = sum(v) / len(v)

#         # === Pd 和 Fa ===
#         Pd = self.total_pd / self.total_targets if self.total_targets > 0 else 0.0
#         Fa = self.total_fa / self.total_pixels if self.total_pixels > 0 else 0.0
#         metrics["Pd"] = round(Pd * 100, 2)
#         metrics["Fa"] = round(Fa * 1e6, 2)

#         # === nIoU ===
#         if self.sample_ious:
#             metrics["nIoU"] = round(np.mean(self.sample_ious) * 100, 2)
#         else:
#             metrics["nIoU"] = 0.0

#         return metrics
