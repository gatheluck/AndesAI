import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

from collections import OrderedDict

import torch
import torchvision

class Evaluator(object):
    def __init__(self, prediction_out, t, selection_out=None, get_top5=True, prediction_threshold:float=0.0, selection_threshold:float=0.0):
        """
        evaluator of classification and rejection.
        if len(prediction_out.shape)==1, it is considered as binary classification.
        if len(prediction_out.shape)==2, it is considered as multi-class classification.
        if selection_out==None, only classification result is evaluated.

        Args:
            prediction_out (B, #class): prediction logit.
            t (B):                      target label. if binary classification, t is in [0,1] else t is in [0,#class-1]
            selection_out (B, 1):       selection (rejection) logit.
            get_top5:                   if True return top_5 accuracy too. (only multi-classification case.)
            prediction_threshold:       prediction threshold. it is used if binary classification.      
            selection_threshold:        selection threshold.
        """
        # automatically decide it is binary or multi classification
        if len(prediction_out.shape) == 1:
            self.is_binary_classification = True
        elif len(prediction_out.shape) == 2:
            self.is_binary_classification = False
        else:
            raise ValueError('shape of prediction_out is invalid.')

        # classification results
        # if binary classification, prediction_result is 0,1 binary tensor.
        # if multi-class classification, prediction_result represent predicted class index in [0,#class-1].
        if self.is_binary_classification:
            condition = (prediction_out>=prediction_threshold)
            self.prediction_result = torch.where(condition, torch.ones_like(prediction_out), torch.zeros_like(prediction_out)).long()
        else:
            self.prediction_result = prediction_out.topk(5, dim=1, sorted=True).indices # (B,5)

        # target label is already adjusted for binary or multi-class
        self.t = t.detach() # (B)

        # selection results
        if selection_out is not None:
            condition = (selection_out >= selection_threshold)
            self.selection_result = torch.where(condition, torch.ones_like(selection_out), torch.zeros_like(selection_out)).view(-1) # (B)
        else:
            self.selection_result = None

        # get top5 accuracy or not
        self.get_top5 = get_top5

    def __call__(self):
        """
        compute 'accuracy (Acc)', 'precision (Pre)', 'recall (Rec)'. 
        if selection_out is not None, 'rejection rate (RR)' and 'rejection precision (PR)' are added.

        Return:
            eval_dict: dict which include evaluation result.
        """
        eval_dict = OrderedDict()
        
        # evaluate classification
        if self.is_binary_classification:
            eval_dict_cls = self._evaluate_binary_classification_with_rejection(self.prediction_result, self.t, self.selection_result)
        else:
            eval_dict_cls = self._evaluate_multi_classification_with_rejection(self.prediction_result, self.t, self.selection_result, self.get_top5)
        eval_dict.update(eval_dict_cls)

        # evaluate slection
        if self.selection_result is not None:
            eval_dict_rjc = self._evaluate_rejection(self.prediction_result, self.t, self.selection_result)
            eval_dict.update(eval_dict_rjc)
        else:
            pass

        return eval_dict

    def _compute_binary_classification_accuracy(self, h:torch.tensor, t_binary:torch.tensor):
        """
        compute accuracy, recall, precision of binary classification. 
        if h.size(0) == t_binary.size(0) == 0, return zeros.

        Args:
            h (B): binary prediction which indicates 'positive:1' and 'negative:0'
            t_binary (B): labels which indicates 'true1:' and 'false:0'
        Return:
            OrderedDict: accuracy, precision, recall
        """
        assert h.size(0) == t_binary.size(0) >= 0
        assert len(h.size()) == len(t_binary.size()) == 1
        # assert 0 <= h.max().item() <= 1
        # assert 0 <= t_binary.max().item() <= 1

        if h.size(0) == t_binary.size(0) == 0:
            # if all samples are rejected, return zeros.
            acc = 0.0
            pre = 0.0
            rec = 0.0
        else:
            # conditions (true,false,positive,negative)
            condition_true  = (h==t_binary)
            condition_false = (h!=t_binary)
            condition_pos = (h==torch.ones_like(h))
            condition_neg = (h==torch.zeros_like(h))

            # TP, TN, FP, FN
            true_pos = torch.where(condition_true & condition_pos, torch.ones_like(h), torch.zeros_like(h))
            true_neg = torch.where(condition_true & condition_neg, torch.ones_like(h), torch.zeros_like(h))
            false_pos = torch.where(condition_false & condition_pos, torch.ones_like(h), torch.zeros_like(h))
            false_neg = torch.where(condition_false & condition_neg, torch.ones_like(h), torch.zeros_like(h))

            tp = float(true_pos.sum())
            tn = float(true_neg.sum())
            fp = float(false_pos.sum())
            fn = float(false_neg.sum())

            # accuracy, precision, recall
            acc = float((tp+tn)/(tp+tn+fp+fn+1e-12))
            pre = float(tp/(tp+fp+1e-12))
            rec = float(tp/(tp+fn+1e-12))

        return acc, pre, rec

    def _compute_multi_classification_accuracy(self, h:torch.tensor, t:torch.tensor):
        """
        compute accuracy of multi-class classification. 

        Args:
            h (B,5): prediction which indicates class index from 0 to #class-1
            t (B): labels which indicates true label form 0 to #class-1
        Return:
            acc_top1 (float): top 1 accuracy
            acc_top5 (float): top 5 accuracy
        """
        assert h.size(0) == t.size(0) > 0
        assert h.size(1)  == 5
        assert len(h.size()) == 2
        assert len(t.size()) == 1

        # sort h through dim 1
        h = h.sort(descending=True, dim=1).values # (B,5)

        # compute topk
        true_top1 = float(torch.where(h[:,0]==t, torch.ones_like(h[:,0]), torch.zeros_like(h[:,0])).sum())
        true_top5 = float(torch.where(h==t.view(-1,1).repeat(1,5), torch.ones_like(h), torch.zeros_like(h)).sum())
        
        # compute accuracy
        num_samples = h.size(0)
        acc_top1 = float(true_top1/(num_samples+1e-12))
        acc_top5 = float(true_top5/(num_samples+1e-12))

        return acc_top1, acc_top5

    def _evaluate_binary_classification_with_rejection(self, h:torch.tensor, t:torch.tensor, r_binary:torch.tensor):
        """
        evaluate result of binary classification with rejection.
        if r_binary == None, it evalutates standard classification result.

        Args:
            h (B): prediction result 0 or 1
            t (B): labels which indicates true result 0 or 1
            r_binary (B): labels which indicates 'accept:1' and 'reject:0'
        Return:
            OrderedDict: 'accuracy', 'recall', 'precision' / 'raw accuracy', 'raw recall', 'raw accuracy'
        """
        assert h.size(0) == t.size(0) > 0
        assert len(h.size()) == len(t.size()) == 1

        eval_dict = OrderedDict()

        # raw accuracy (before rejection)
        raw_acc, raw_pre, raw_rec = self._compute_binary_classification_accuracy(h, t)
        eval_dict['{key}'.format(key='raw accuracy' if r_binary is not None else 'accuracy')] = raw_acc
        eval_dict['{key}'.format(key='raw precision' if r_binary is not None else 'precision')] = raw_pre
        eval_dict['{key}'.format(key='raw recall' if r_binary is not None else 'recall')] = raw_rec

        # if r_binary != None, then compute accuracy with rejection.
        if r_binary is not None:
            assert h.size(0) == t.size(0) == r_binary.size(0)> 0
            assert len(h.size()) == len(t.size()) == len(r_binary.size()) == 1
            
            # execute rejection
            h_rjc = torch.masked_select(h, r_binary.bool())
            t_rjc = torch.masked_select(t, r_binary.bool())

            # accuracy (after rejection). if all samples are rejected, get zeros. 
            acc, pre, rec = self._compute_binary_classification_accuracy(h_rjc, t_rjc)
            eval_dict['accuracy'] = acc
            eval_dict['precision'] = pre
            eval_dict['recall'] = rec
        else:
            pass

        return eval_dict

    def _evaluate_multi_classification_with_rejection(self, h:torch.tensor, t:torch.tensor, r_binary:torch.tensor, get_top5=True):
        """
        evaluate result of multi classification with rejection.
        if r_binary == None, it evalutates standard classification result.

        Args:
            h (B,5): prediction which indicates class index from 0 to #class-1
            t (B):   labels which indicates true label form 0 to #class-1
            r_binary (B): labels which indicates 'accept:1' and 'reject:0'
            get_top5:     get top5 accuracy
        Return:
            eval_dict
        """
        assert h.size(0) == t.size(0)> 0
        assert h.size(1) == 5
        assert len(h.size()) == 2
        assert len(t.size()) == 1

        eval_dict = OrderedDict()

        # raw accuracy (before rejection)
        raw_acc_top1, raw_acc_top5 = self._compute_multi_classification_accuracy(h, t)
        if get_top5:
            eval_dict['{key}'.format(key='raw accuracy top5' if r_binary is not None else 'accuracy top5')] = raw_acc_top5
            eval_dict['{key}'.format(key='raw accuracy top1' if r_binary is not None else 'accuracy top1')] = raw_acc_top1
        else:
            eval_dict['{key}'.format(key='raw accuracy' if r_binary is not None else 'accuracy')] = raw_acc_top1

        # if r_binary != None, then compute accuracy with rejection.
        if r_binary is not None:
            assert h.size(0) == t.size(0) == r_binary.size(0)> 0
            assert len(h.size()) == len(t.size()) == len(r_binary.size()) == 1

            # execute rejection
            h_rjc = torch.masked_select(h, r_binary.bool())
            t_rjc = torch.masked_select(t, r_binary.bool())

            # accuracy (after rejection)
            acc_top1, acc_top5 = self._compute_multi_classification_accuracy(h_rjc, t_rjc)
            if get_top5:
                eval_dict['accuracy top5'] = acc_top5
                eval_dict['accuracy top1'] = acc_top1
            else:
                eval_dict['accuracy'] = acc_top1
        else:
            pass

        return eval_dict

    def _evaluate_rejection(self, h:torch.tensor, t:torch.tensor, r_binary:torch.tensor):
        """
        evaluate result of rejection.
        this function is able to use both binary-classification and multi-classification. 

        Args:
            h (B): prediction which indicates class index from 0 to #class-1 
            t (B): labels which indicates true class index from 0 to #class-1
            r_binary (B): labels which indicates 'accept:1' and 'reject:0'
        Return:
            OrderedDict: rejection_rate, rejection_precision
        """
        assert h.size(0) == t.size(0) == r_binary.size(0)> 0
        assert len(h.size()) == len(t.size()) == len(r_binary.size()) == 1

        # conditions (true,false,positive,negative)
        condition_true  = (h==t)
        condition_false = (h!=t)
        
        condition_acc = (r_binary==torch.ones_like(r_binary))
        condition_rjc = (r_binary==torch.zeros_like(r_binary))

        # TP, TN, FP, FN
        ta = float(torch.where(condition_true & condition_acc, torch.ones_like(h), torch.zeros_like(h)).sum())
        tr = float(torch.where(condition_true & condition_rjc, torch.ones_like(h), torch.zeros_like(h)).sum())
        fa = float(torch.where(condition_false & condition_acc, torch.ones_like(h), torch.zeros_like(h)).sum())
        fr = float(torch.where(condition_false & condition_rjc, torch.ones_like(h), torch.zeros_like(h)).sum())

        # accuracy, precision, recall
        rejection_rate = float((tr+fr)/(ta+tr+fa+fr+1e-12))
        rejection_pre  = float(tr/(tr+fr+1e-12))

        return OrderedDict({'rejection rate':rejection_rate, 'rejection precision':rejection_pre}) 

if __name__ == '__main__':
    from andesai.data import DatasetBuilder
    from andesai.models.models import ModelBuilder

    # dataset
    dataset_builder = DatasetBuilder(name='cifar10', root_path='../data')
    test_dataset    = dataset_builder(train=False, normalize=True)
    test_loader     = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=16, pin_memory=True)

    # model
    model = ModelBuilder(num_classes=10)['resnet18'].cuda()
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)

    # test
    with torch.autograd.no_grad():
        for i, (x,t) in enumerate(test_loader):
            model.eval()
            x = x.to('cuda', non_blocking=True)
            t = t.to('cuda', non_blocking=True)

            # forward
            out_class = model(x)

            # evaluator
            evaluator = Evaluator(out_class.detach(), t.detach())

            # compute selective loss
            eval_dict = OrderedDict()
            eval_dict.update(evaluator())
            print(eval_dict)
            raise NotImplementedError
            


            
            

