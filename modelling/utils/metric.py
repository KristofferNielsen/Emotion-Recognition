import torch
class metric():
    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, y_pred, y_true):
        #predicted = torch.argmax(y_pred, dim=1)
        self.correct += (y_pred == y_true).all().item()
        self.total += len(y_true)

    def eval(self):
        return self.correct / self.total
    

class F2Score():
    def __init__(self, beta=2):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.beta_squared = beta ** 2

    def __call__(self, y_pred, y_true):
        self.tp += ((y_pred == 1) & (y_true == 1)).sum().item()
        self.fp += ((y_pred == 1) & (y_true == 0)).sum().item()
        self.fn += ((y_pred == 0) & (y_true == 1)).sum().item()

    def precision(self):
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    def recall(self):
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

    def eval(self):
        p = self.precision()
        r = self.recall()
        f2_score = ((1 + self.beta_squared) * p * r) / ((self.beta_squared * p) + r + 1e-15)
        return f2_score
