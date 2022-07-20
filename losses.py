# modified from https://github.com/facebookresearch/deit
"""
Implements the knowledge distillation loss
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion, teacher_model, cnn_teacher,
                 distillation_type, lambda_token, lambda_logits, 
                 lambda_cnn, tau):
        super().__init__()
        self.base_criterion = base_criterion
        self.mse_loss = nn.MSELoss()
        self.teacher_model = teacher_model
        self.cnn_teacher = cnn_teacher
        assert distillation_type in ['none', 'frd']
        self.distillation_type = distillation_type
        self.lambda_token = lambda_token
        self.lambda_logits = lambda_logits
        self.lambda_cnn = lambda_cnn
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        if self.distillation_type == 'none':
            if not isinstance(outputs, torch.Tensor):
                outputs = outputs[0]
            base_loss = self.base_criterion(outputs, labels)
            return base_loss, base_loss, torch.zeros_like(base_loss), torch.zeros_like(base_loss), torch.zeros_like(base_loss)
        else:
            student_logits, student_dist, student_tokens = outputs
            loss_cls = self.base_criterion(student_logits, labels)

            # don't backprop throught the teacher
            with torch.no_grad():
                teacher_logits, teacher_tokens = self.teacher_model(inputs)
                if self.cnn_teacher:
                    cnn_teacher_logits = self.cnn_teacher(inputs)

            loss_token = 0
            for i in range(len(student_tokens)):
                loss_token += (self.mse_loss(student_tokens[i], teacher_tokens[i]) / len(student_tokens))
            loss_token *= self.lambda_token

            loss_logits = F.kl_div(
                F.log_softmax(student_logits / self.tau, dim=1),
                F.log_softmax(teacher_logits / self.tau, dim=1),
                reduction='batchmean',
                log_target=True
            ) * (self.tau * self.tau) * self.lambda_logits

            loss_cnn = torch.zeros_like(loss_logits)
            if self.cnn_teacher:
                loss_cnn += F.cross_entropy(student_dist, cnn_teacher_logits.argmax(dim=1))

        loss = loss_cls + loss_token + loss_logits + loss_cnn
        return loss, loss_cls, loss_token, loss_logits, loss_cnn
