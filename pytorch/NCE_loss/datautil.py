
class BatchSampleFromFile:
    def __init__(self, file, ):
        self.f = open(file, 'r')
    def sample_batch(self, batch_size):
        sample_num = 0
        feasA = []
        labelsA = []
        for line in self.f:
            if line=='':
                self.f.close()
                break
            fea, label = line.strip().split(',')
            labelsA.append(int(label) - 1)
            feasA.append(int(fea) - 1)
            sample_num += 1
            if sample_num >= batch_size:
                break
        return feasA,labelsA
    def sample_batch_test(self, batch_size):
        sample_num = 0
        feasA = []
        labelsA = []
        gts = []
        for line in self.f:
            if line=='':
                self.f.close()
                break
            fea, label, gt = line.strip().split(',')
            labelsA.append(int(label) - 1)
            feasA.append(int(fea) - 1)
            gts.append(int(gt))
            sample_num += 1
            if sample_num >= batch_size:
                break
        return feasA,labelsA,gts
    def close(self):
        self.f.close()
