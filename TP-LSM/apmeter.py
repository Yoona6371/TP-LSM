import math
import torch


class Meter(object):
    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass


class APMeter(Meter):
    """
    The APMeter measures the average precision per class.

    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
APMeter 测量每个class的平均精度。
APMeter被设计为在“NxK”张量“输出”和
    “目标”，以及可选的“Nx1”张量权重，其中 （1） “输出”
    包含“N”个示例和“K”类的模型输出分数，这些分数应该
    当模型更确信示例应该是
    正标记，当模型认为示例应该时，标记较小
    被负面标记（例如，sigmoid 函数的输出）;(2)
    “目标”仅包含值 0（对于负示例）和 1
    （正面例子）;（3）“重量”（>0）表示重量
    每个样本。
    """
    def __init__(self,weighted=False):
        """
        __init__ 函数将 weighted 设为 False,
        并调用其父类的构造函数初始化 Meter。同时,也调用了 reset 函数，以初始化该类的成员变量。
        """
        super(APMeter, self).__init__()
        self.reset()
        self.weighted=weighted

    def reset(self):
        """
        它会将scores、targets和weights三个成员变量分别设置为空的FloatTensor和LongTensor类型,
        以准备开始下一轮的计算。这样做可以确保每一轮计算都从一个空状态开始,
        避免上一轮的结果对当前结果造成干扰。
        """
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())
        self.weights = torch.FloatTensor(torch.FloatStorage())

    def add(self, output, target, weight=None):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        
        APMeter中的add函数用于向计量器中添加新的预测结果和真实标签对。
        具体来说，它接收两个张量：一个是模型对一批输入数据的预测结果，另一个是相应的真实标签。
        它使用这些张量来计算精确率和召回率，
        并将这些度量值添加到计量器的内部状态中,以便稍后计算平均精确率(Average Precision,AP)。
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if weight is not None:
            if not torch.is_tensor(weight):
                weight = torch.from_numpy(weight)
            weight = weight.squeeze()  # 如果不指定维度，那么将删除所有维度为1的维度
        if output.dim() == 1:
            output = output.view(-1, 1)  # view()相当于reshape、resize，重新调整Tensor的形状
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if weight is not None:
            assert weight.dim() == 1, 'Weight dimension should be 1'
            # numel 获取tensor中一共包含多少个元素
            assert weight.numel() == target.size(0), \
                'Weight dimension 1 should be the same as that of target'
            assert torch.min(weight) >= 0, 'Weight should be non-negative only'
        assert torch.equal(target**2, target), \
            'targets should be binary (0 or 1)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            new_weight_size = math.ceil(self.weights.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))
            if weight is not None:
                self.weights.storage().resize_(int(new_weight_size
                                               + output.size(0)))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)  # narrow函数返回tensor的第dim维切片start: start+length的数据
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

        if weight is not None:
            self.weights.resize_(offset + weight.size(0))
            self.weights.narrow(0, offset, weight.size(0)).copy_(weight)

    def value(self):
        """Returns the model's average precision for each class

        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
            1xK 张量，每个类 k 的平均精度
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)+1).float()
        if self.weights.numel() > 0:
            weight = self.weights.new(self.weights.size())
            weighted_truth = self.weights.new(self.weights.size())

        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            _, sortind = torch.sort(scores, 0, True)
            truth = targets[sortind]
            if self.weights.numel() > 0:
                weight = self.weights[sortind]
                weighted_truth = truth.float() * weight
                rg = weight.cumsum(0)

            # compute true positive sums
            if self.weights.numel() > 0:
                tp = weighted_truth.cumsum(0)
            else:
                tp = truth.float().cumsum(0)

            # compute precision curve
            precision = tp.div(rg)

            # compute average precision
            ap[k] = precision[truth.bool()].sum() / max(truth.sum(), 1)

        return ap
