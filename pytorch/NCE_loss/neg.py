#! usr/bin/python
#coding=utf-8
import torch as t
import math
import torchnet as tnt
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
import datautil
batch_size = 2
window_size = 1
num_sample = 3
epoch = 100
nclass = 6
embedd_dim = 3
class NEG_loss(nn.Module):
    def __init__(self, num_classes, embed_size, weights=None):
        super(NEG_loss, self).__init__()
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.in_embed = nn.Embedding(self.num_classes, self.embed_size)
        self.out_embed = self.in_embed #nn.Embedding(self.num_classes, self.embed_size)
        #self.out_embed = nn.Embedding(self.num_classes, self.embed_size, sparse=True)
        self.out_embed.weight = Parameter(t.FloatTensor(self.num_classes, self.embed_size).uniform_(-1, 1))
        #self.in_embed = nn.Embedding(self.num_classes, self.embed_size, sparse=True)
        self.in_embed.weight = Parameter(t.FloatTensor(self.num_classes, self.embed_size).uniform_(-1, 1))
        self.weights = weights
        if self.weights is not None:
            assert min(self.weights) >= 0, "Each weight should be >= 0"
            self.weights = Variable(t.from_numpy(weights)).float()
    def sample(self, num_sample):
        return t.multinomial(self.weights, batch_size * window_size * num_sample, replacement=True)

    def forward(self, input_labes, out_labels, num_sampled, is_Test=False):
        use_cuda = self.out_embed.weight.is_cuda
        #[batch_size, window_size] = out_labels.size()
        [batch_size] = out_labels.size()
        input = self.in_embed(input_labes.repeat(1, window_size).contiguous().view(-1))
        output = self.out_embed(out_labels.contiguous().view(-1))

        if self.weights is not None:
            noise_sample_count = batch_size * window_size * num_sampled
            draw = self.sample(noise_sample_count)
            noise = draw.view(batch_size * window_size, num_sampled)
        else:
            noise = Variable(t.Tensor(batch_size * window_size, num_sampled).
                             uniform_(0, self.num_classes - 1).long())
        if use_cuda:
            noise = noise.cuda()
        noise = self.out_embed(noise).neg()

        #log_target = (input * output).sum(1).squeeze().sigmoid().log()

        if not is_Test:
            log_target = (t.mul(input, output)).sum(1).squeeze().sigmoid().log()
            sum_log_sampled = t.bmm(noise, input.unsqueeze(2)).sigmoid().log().sum(1).squeeze()
            #score = t.cat(((t.mul(input, output)).sum(1).unsqueeze(1), t.bmm(noise, input.unsqueeze(2)).squeeze(2)), 1).view(-1)
            score =  t.cat(((t.mul(input, output)).sum(1).view(-1),t.bmm(noise, input.unsqueeze(2)).squeeze(2).view(-1)),0)
            loss = log_target + sum_log_sampled
            return  -loss.sum() / batch_size, score
        else:
            return (t.mul(input, output)).sum(1).squeeze()

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()


model = NEG_loss(nclass, embedd_dim)
use_cuda = t.cuda.is_available()
if use_cuda:
    model.cuda()
optimizer = t.optim.Adam(model.parameters(), lr=0.01)

def test():
    batch_sampler = datautil.BatchSampleFromFile('data_test.txt')
    auc_meter = tnt.meter.AUCMeter()
    while True:
        feas, labels, gts = batch_sampler.sample_batch_test(batch_size)
        if len(labels)==0:
            break
        score = model(Variable(t.LongTensor(feas)), Variable(t.LongTensor(labels)), num_sample, is_Test=True)
        #print score.data.numpy()[0],gts[0]
        auc_meter.add(score.data, t.LongTensor(gts))
    auc,tpr,fpr=auc_meter.value()
    batch_sampler.close()
    return auc

for i in range(epoch):
    batch_sampler = datautil.BatchSampleFromFile('data.txt')
    auc_meter = tnt.meter.AUCMeter()
    while True:
        feas, labels = batch_sampler.sample_batch(batch_size)
        batch_sam_num = len(labels)
        if len(labels) <= 0:
            break
        loss,score = model(Variable(t.LongTensor(feas)), Variable(t.LongTensor(labels)), num_sample)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print score.data
        fake_label =1
        auc_meter.add(score.data, t.LongTensor([1 for j in range(batch_sam_num)] + [0 for j in range(batch_sam_num * num_sample)] ))
        #print loss.data[0]
    batch_sampler.close()
    if i % 10 ==0:
        auc, tpr, fpr = auc_meter.value()
        auc_meter.reset()
        print 'AUC', test(),auc
word_embeddings = model.input_embeddings()
print word_embeddings
def dot_product(v1, v2):
    return sum(map(lambda x: x[0] * x[1], zip(v1, v2)))

def cosine_measure(v1, v2):
    prod = dot_product(v1, v2)
    len1 = math.sqrt(dot_product(v1, v1))
    len2 = math.sqrt(dot_product(v2, v2))
    return prod / (len1 * len2)
    #return prod
for i in range(nclass):
    for j in range(nclass):
        print '%.4f' % (cosine_measure(word_embeddings[i,:],word_embeddings[j,:])),
    print ''
