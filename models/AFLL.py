import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models.cal import cal
from models import pvt
from torch.autograd import Variable
from torch.nn import init
from utils.genLD import genLD
from utils.cdf import cjs
from utils.cdf import hellinger


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight, mode='fan_out')
        if m.bias is not None:
            init.constant(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias, 0)

class single_feature(nn.Module):
    def __init__(self, dim):
        super(single_feature, self).__init__()
        self.base = pvt.feature_pvt_v2_b3()
        ckpt = torch.load('/home/zbf/ord2seq/data/models/pvt_v2/pvt_v2_b3.pth')
        self.base.load_state_dict(ckpt)

    def forward(self, x):
        x = self.base(x)
        return x

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.name = 'pvt'
        dim = 512
        self.cls = args.num_classes
        self.need_mask_in_training = True
        self.feature = single_feature(dim=dim)
        self.nhead = 8
        self.lamada = args.lamada
        self.beta = args.beta
        self.transformer = nn.Transformer()
        encoder_layer = nn.TransformerEncoderLayer(512, 8, 2048, 0.1, 'relu')
        encoder_norm = nn.LayerNorm(512)
        self.encoder = nn.TransformerEncoder(encoder_layer, 6, encoder_norm)
        decoder_layer = nn.TransformerDecoderLayer(512, 8, 2048, 0.1, 'relu')
        decoder_norm = nn.LayerNorm(512)
        self.decoder = nn.TransformerDecoder(decoder_layer, 6, decoder_norm)
        self.Embed = nn.Embedding(10, dim)

        self.fc1 = nn.Linear(dim, self.cls)
        self.fc2 = nn.Linear(dim, self.cls)
        self.fc3 = nn.Linear(dim, self.cls)
        self.fc = [self.fc1, self.fc2, self.fc3]
        self.acti = nn.Identity()
        self.weight = [1., 1., 1.]

        self.grade = nn.Linear(512, 4)
        self.grade.apply(weights_init)

        self.avgPool = nn.AdaptiveAvgPool2d(1)



    def model_name(self):
        return self.name

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def make_binary_gt(self, tgt):
        # 起始符 = 14
        if self.cls == 4:
            mapping = {
                0: [0, 2, 5],
                1: [1, 3, 6],
                2: [1, 4, 7],
                3: [1, 4, 8],
            }
        else:
            print('This mapping has not been set up')
            raise ValueError
        bs, tgt_len = tgt.shape
        start_token = (torch.ones([bs, 1]) * 9).long()
        if tgt_len == 1:
            converted_tgt = self.Embed(start_token.cuda())
        else:
            if tgt_len == 2:
                cls = torch.tensor([mapping[x.item()][0] for x in tgt[:, 1]]).unsqueeze(1)
            elif tgt_len == 3:
                cls = torch.tensor(
                    [[mapping[x.item()][0] for x in tgt[:, 1]],
                     [mapping[x.item()][1] for x in tgt[:, 2]]],
                ).permute((1, 0))
            else:
                cls = torch.tensor([mapping[x.item()] for x in tgt[:, 1]])

            converted_tgt = torch.cat([start_token, cls], dim=1)
            converted_tgt = self.Embed(converted_tgt.long().cuda())
        return converted_tgt.permute((1, 0, 2))

    def make_tgt(self, tgt):
        tgt = tgt.unsqueeze(dim=1)
        BS = tgt.size(0)
        start_token = torch.ones([BS, 1]) * self.cls
        tgt = torch.cat([start_token.cuda(), tgt, tgt, tgt], dim=1).long()
        tgt = self.make_binary_gt(tgt)
        return tgt

    def make_mask_l1(self, out1):
        '''
        1000 --- 0111这样的mask
        '''
        if len(out1.shape) == 3:
            out1 = out1.squeeze(0)
        if self.cls == 4:
            left, right = out1.split([1, 3], dim=1)
            left = left.mean(dim=1)
            right = right.mean(dim=1)
            comparison = (left > right)
            mask = torch.stack([comparison, ~comparison, ~comparison, ~comparison]).float().permute(1, 0)
        else:
            print('Not Supported')
            raise ValueError

        return mask.float()

    def make_mask_l2(self, out2):
        '''
        1000 --- 0100 --- 0011这样的mask
        '''
        bs = out2.size(0)
        if len(out2.shape) == 3:
            out2 = out2.squeeze(0)
        if self.cls == 4:
            tmp = torch.zeros([bs, 3]).cuda()
            tmp[:, 0] = out2[:, 0]
            tmp[:, 1] = out2[:, 1]
            tmp[:, 2] = out2[:, 2] + out2[:, 3]
            _, indices = torch.max(tmp, dim=1)
            indices = F.one_hot(indices, num_classes=3)
            mask = torch.cat([indices[:, 0].unsqueeze(1),
                              indices[:, 1].unsqueeze(1),
                              indices[:, 2].unsqueeze(1),
                              indices[:, 2].unsqueeze(1)], dim=1)

        else:
            print('Not Supported')
            raise ValueError

        return mask.float()

    def make_mask_l3(self, out3):
        '''
        1000 --- 0100 --- 0010 --- 0001这样的mask
        '''
        bs = out3.size(0)
        if len(out3.shape) == 3:
            out3 = out3.squeeze(0)

        if self.cls == 4:
            tmp = torch.zeros([bs, 4]).cuda()
            tmp[:, 0] = out3[:, 0]
            tmp[:, 1] = out3[:, 1]
            tmp[:, 2] = out3[:, 2]
            tmp[:, 3] = out3[:, 3]
            _, indices = torch.max(tmp, dim=1)
            mask = F.one_hot(indices, num_classes=4)

        else:
            print('Not Supported')
            raise ValueError

        return mask.float()

    def forward(self, x, tgt):
        tgt = tgt.long()
        BS = x.size(0)
        lam = None

        ld = genLD(tgt.cpu().data.numpy(), 3.0, 'klloss', 4)
        ld = torch.from_numpy(ld).cuda().float()

        feature = self.feature(x)
        memory = self.encoder(feature)

        temp = self.avgPool(memory.reshape(memory.shape[1], memory.shape[2], 7, 7))
        temp = temp.view(temp.size(0), -1)
        grade = self.grade(temp)

        # CJH
        grade = F.softmax(grade, dim=1)
        loss_grade = cjs(grade, ld) + (1 - self.beta) * hellinger(grade, ld);


        label_3 = F.one_hot(tgt, num_classes=self.cls).float()
        label_1 = self.make_mask_l1(label_3)
        label_2 = self.make_mask_l2(label_3)

        if self.training:
            tgt_mask = self.generate_square_subsequent_mask(4).cuda()
            tgt = cal(self.make_tgt, tgt, lam, add=True)
            output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            out1, out2, out3, _ = output.split(1, dim=0)
            out1 = self.fc[0](self.acti(out1)).squeeze(0)
            out2 = self.fc[1](self.acti(out2)).squeeze(0)
            out3 = self.fc[2](self.acti(out3)).squeeze(0)

            loss1 = nn.BCEWithLogitsLoss()(out1, label_1)
            if self.need_mask_in_training:
                mask1 = self.make_mask_l1(torch.sigmoid(out1))
                mask2 = self.make_mask_l2(torch.sigmoid(out2))
                ones = torch.ones_like(mask1)
                mask1 = torch.where(torch.sum(mask1 * mask2, dim=-1).unsqueeze(1) == 0, ones, mask1)
                out2 = mask1 * out2

            loss2 = nn.BCEWithLogitsLoss()(out2, label_2)
            if self.need_mask_in_training:
                mask2 = self.make_mask_l2(torch.sigmoid(out2))
                mask3 = self.make_mask_l3(torch.sigmoid(out3))
                ones = torch.ones_like(mask2)
                mask2 = torch.where(torch.sum(mask2 * mask3, dim=-1).unsqueeze(1) == 0, ones, mask2)
                out3 = mask2 * out3

            loss3 = nn.BCEWithLogitsLoss()(out3, label_3)

            loss = loss1 + loss2 + loss3

        else:
            dec_input = torch.zeros(BS, 0).long().cuda()
            next_symbol = (torch.ones([BS, 1]) * self.cls).long().cuda()
            output_hard = []
            for i in range(3):
                dec_input = torch.cat([dec_input, next_symbol], -1)
                tgt_mask = self.generate_square_subsequent_mask(i + 1).cuda()

                tgt = self.make_binary_gt(dec_input)

                output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
                projected = self.fc[i](self.acti(output))

                projected = projected[-1]
                if i == 0:
                    mask = self.make_mask_l1(torch.sigmoid(projected))
                    loss1 = nn.BCEWithLogitsLoss()(projected, label_1)
                    out1 = projected
                if i == 1:
                    projected = mask * projected
                    mask = self.make_mask_l2(torch.sigmoid(projected))
                    loss2 = nn.BCEWithLogitsLoss()(projected, label_2)
                    out2 = projected
                if i == 2:
                    projected = mask * projected
                    loss3 = nn.BCEWithLogitsLoss()(projected, label_3)
                    out3 = projected
                if i in [0, 1]:
                    prob = mask.max(dim=-1, keepdim=False)[1]
                else:
                    prob = projected.max(dim=-1, keepdim=False)[1]

                next_word = prob.data[-1].unsqueeze(dim=-1) if len(prob.shape) > 1 else prob.unsqueeze(
                    dim=-1).data
                next_symbol = next_word.clone()
                output_hard.append(next_symbol)


            loss = loss1 * self.weight[0] + loss2 * self.weight[1] + loss3 * self.weight[2]
        loss = loss + self.lamada * loss_grade
        return out3, grade, loss
