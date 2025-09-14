import torch
import numpy as np

def get_metric(metric_name):    
    """
    Get the metric class by name.
    """

    if 'pcp' in metric_name:
        thresh = float(metric_name[7:-1]) # 'pcp_at_{thresh}m'
        return PCPMetric(thresh)

    metric_classes = {
        'distance': DistanceMetric,
        'angle': AngleMetric,
        'translation_xy': TranslationXYMetric,
        'translation_xyz': TranslationXYZMetric
    }
    
    if metric_name not in metric_classes:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    return metric_classes[metric_name]()

class BaseMetric:

    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, pred, label):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def aggregate(self):
        raise NotImplementedError("Subclasses should implement this method.")
    

class DistanceMetric(BaseMetric):

    def __init__(self):
        super().__init__()

    def reset(self):
        self.distances = np.empty((0))
        self.count = 0

    def __call__(self, pred, label):
        pred_norm_uv = pred['norm_uv']
        label_norm_uv = label['norm_uv'].to(pred_norm_uv.device)
        coverage = label['coverage'].to(pred_norm_uv.device)
        distance = torch.norm(pred_norm_uv - label_norm_uv, dim=-1)
        distance = distance * coverage * 0.5
        self.distances = np.concatenate((self.distances, distance.cpu().numpy()))
        self.count += len(distance)
        return distance
    
    def aggregate(self):
        if self.count == 0:
            return 0
        return np.mean(self.distances)

class AngleMetric(BaseMetric):
    def __init__(self):
        super().__init__()

    def reset(self):
        self.errors = np.empty((0))
        self.count = 0

    def __call__(self, pred, label):
        pred_angle = pred['yaw']
        label_angle = label['yaw'].to(pred_angle.device)
        error = torch.abs(pred_angle % (2*np.pi) - label_angle % (2*np.pi))
        error = torch.min(error, 2*np.pi - error)
        self.errors = np.concatenate((self.errors, error.cpu().numpy()))
        self.count += len(error)
        return error
    
    def aggregate(self):
        if self.count == 0:
            return 0
        return np.mean(self.errors)

class TranslationXYMetric(BaseMetric):
    def __init__(self):
        super().__init__()

    def reset(self):
        self.errors = torch.empty([0, 2])
        self.count = 0

    def __call__(self, pred, label):

        t_left2world_pred = pred['t_left2world']
        t_left2world_label = label['t_left2world'].to(t_left2world_pred.device)
        error = torch.abs(t_left2world_pred - t_left2world_label)[:, 0:2]

        self.errors = torch.cat([self.errors.to(error.device), error], dim=0)
        self.count += error.shape[0]
        return error
    
    def aggregate(self):
        if self.count == 0:
            return 0
        return torch.norm(self.errors, dim=-1).mean()

class TranslationXYZMetric(BaseMetric):
    def __init__(self):
        super().__init__()

    def reset(self):
        self.errors = torch.empty([0, 3])
        self.count = 0

    def __call__(self, pred, label):

        t_left2world_pred = pred['t_left2world']
        t_left2world_label = label['t_left2world'].to(t_left2world_pred.device)
        error = torch.abs(t_left2world_pred - t_left2world_label)

        self.errors = torch.cat([self.errors.to(error.device), error], dim=0)
        self.count += error.shape[0]
        #print(torch.norm(error, dim=-1).item())
        return error
    
    def aggregate(self):
        if self.count == 0:
            return 0
        return torch.norm(self.errors, dim=-1).mean()

class PCPMetric(BaseMetric):
    def __init__(self, thresh):
        super().__init__()
        self.thresh = thresh
    
    def reset(self):
        self.correct = 0
        self.total = 0

    def __call__(self, pred, label):

        t_left2world_pred = pred['t_left2world']
        t_left2world_label = label['t_left2world'].to(t_left2world_pred.device)
        error = torch.norm(t_left2world_pred - t_left2world_label, dim=-1)
        mask = error < self.thresh
        self.correct = self.correct + torch.sum(mask).item()
        self.total = self.total + error.shape[0]
        #print(torch.norm(error, dim=-1).item())
        return error
    
    def aggregate(self):
        if self.total == 0:
            return 0
        return self.correct / self.total
