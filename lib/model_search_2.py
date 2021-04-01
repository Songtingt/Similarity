import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
import numpy as np
def get_xy_ctr_np(score_size, score_offset, total_stride):
    """ generate coordinates on image plane for score map pixels (in numpy)
    """
    fm_height, fm_width = score_size, score_size

    y_list = np.linspace(0., fm_height - 1., fm_height).reshape(1, 1, fm_height, 1)
    y_list = y_list.repeat(fm_width, axis=3)
    x_list = np.linspace(0., fm_width - 1., fm_width).reshape(1, 1, 1, fm_width)
    x_list = x_list.repeat(fm_height, axis=2)
    xy_list = score_offset + np.concatenate((y_list, x_list), 1) * total_stride
    xy_ctr = torch.tensor(xy_list.astype(np.float32), requires_grad=False)
    return xy_ctr
def get_left_top_point(fm_ctr,score_size,small_size):
    sar_size = torch.tensor([(small_size-1)/2]) #255
    # print(sar_size.shape)
    sar_point = sar_size.repeat(score_size * score_size).reshape(1, score_size, score_size).repeat(2, 1, 1) #[2, score_size, score_size]

    # print(sar_size.shape)
    xy0 = fm_ctr - sar_point #[1, 2, score_size, score_size]
    return xy0
class Similarity(nn.Module):
    def __init__(self, model_file=None, use_cuda=True,z_size=512,
                 x_size=800):
        super(Similarity, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
        self.similarity = search()

        self.stride=2*2*2*2  #8
        self.score_size = x_size // self.stride - z_size // self.stride + 1 #19
        self.score_offset = (x_size - 1 - (self.score_size - 1) * self.stride) / 2  # /2
        self.fm_ctr=get_xy_ctr_np(self.score_size, self.score_offset, self.stride)
        self.bbox_corner=get_left_top_point(self.fm_ctr,self.score_size,z_size) #1,2,19,19

    def forward(self, domainA_small, domainB_big):
        f_domainA_small = self.feature(domainA_small)  # b,64,32,32
        f_domainB_big = self.feature(domainB_big)  # b,64,50,50
        score = self.similarity(f_domainA_small, f_domainB_big)  # b,19,19
        return score
    
            
            
            


class search(nn.Module):
    def __init__(self):
        super(search, self).__init__()

    def forward(self, domainA_small, domainB_big):

        search_size = domainB_big.shape[2] - domainA_small.shape[2] + 1

        b, c, h, w = domainA_small.shape
        similarity_matrix = torch.zeros((b, search_size, search_size)).cuda()

        '''
        torch.einsum('bdhw,bdhw->b', [sar, opt_patch])
        和torch.sum(torch.mul(a,a),dim=(1,2,3)) 的效果是一样的
        '''

        for i in range(search_size):
            for j in range(search_size):
                domainB_big_patch = domainB_big[:, :, i:i + h, j:j + w]  # b,64,h,w
                # opt_patch = opt_patch.reshape(b, -1)  # b,128,1024
                # opt_patch = F.normalize(opt_patch, dim=1)
                # similarity_matrix[:, i, j] = torch.cosine_similarity(sar, opt_patch, dim=1)
                # print(similarity_matrix[:, i, j])
                domainA_small_sum=torch.sum(torch.mul(domainA_small,domainA_small),dim=(1,2,3)) #b
                domainB_big_sum=torch.sum(torch.mul(domainB_big_patch,domainB_big_patch),dim=(1,2,3)) #b

                total_sum=torch.sqrt(domainA_small_sum*domainB_big_sum) #b
                similarity_matrix[:, i, j] = torch.einsum('bdhw,bdhw->b', [domainA_small, domainB_big_patch])/total_sum #归一化相关系数 所有值都差不多。。

        #TODO
        # print(similarity_matrix)
        # exp_sum=torch.sum(torch.exp(similarity_matrix),dim=(1,2)) #b,1
        # similarity_matrix=torch.exp(similarity_matrix)/exp_sum.unsqueeze(1).unsqueeze(2)
        # similarity_matrix = similarity_matrix.reshape(b, -1)  # bx|S|
        # similarity_matrix = F.softmax(similarity_matrix, dim=1).reshape(b, search_size, search_size)
        # print(similarity_matrix)
        # similarity_matrix=similarity_matrix.detach().numpy()
        # similar_min=np.min(similarity_matrix,axis=(1,2),keepdims=True)
        # similar_max=np.max(similarity_matrix,axis=(1,2),keepdims=True)
        # similarity_matrix=(similarity_matrix-similar_min)/similar_max

        return similarity_matrix



if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:2" if use_cuda else "cpu")
    model = Similarity().to(device)
    summary(model, input_size=[(1, 512, 512), (1, 800, 800)], batch_size=1)

