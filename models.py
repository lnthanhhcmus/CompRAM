from helper import *
from CompRAM import CompRAM
from special_spmm_func import SpecialSpmmFunc
import torch.nn as nn


class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p = params
        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

class CompRAMBase(BaseModel):
    def __init__(self, edge_index, edge_type, num_rel, params=None):
        super(CompRAMBase, self).__init__(params)

        self.edge_index = edge_index
        self.edge_type = edge_type
        self.p.gcn_dim = self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
        

        if self.p.init_e == 'u' or self.p.init_r == 'u':
            self.gamma = nn.Parameter(
                torch.Tensor([self.p.g]), 
                requires_grad=False
            )
            self.epsilon = 2.0
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.gamma.item() + self.epsilon) / self.p.init_dim]), 
                requires_grad=False
            )

        if self.p.init_e == 'u':
            self.init_embed_real = nn.Parameter(torch.zeros(self.p.num_ent, self.p.init_dim))
            self.init_embed_imaginary = nn.Parameter(torch.zeros(self.p.num_ent, self.p.init_dim))
            nn.init.uniform_(
                tensor=self.init_embed_real, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.init_embed_imaginary, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
        else:
            self.init_embed_real = get_param((self.p.num_ent, self.p.init_dim))
            self.init_embed_imaginary = get_param((self.p.num_ent, self.p.init_dim))


        self.device = self.edge_index.device

        if self.p.init_r == 'u':
            self.init_rel_real = nn.Parameter(torch.zeros(num_rel * 2, self.p.init_dim))
            self.init_rel_imaginary = nn.Parameter(torch.zeros(num_rel * 2, self.p.init_dim))
            nn.init.uniform_(
                tensor=self.init_rel_real, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.init_rel_imaginary, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
        else:
            self.init_rel_real = get_param((num_rel * 2, self.p.init_dim))
            self.init_rel_imaginary = get_param((num_rel * 2, self.p.init_dim))
            

        self.conv1 = CompRAM(self.edge_index, self.edge_type, self.p.init_dim, self.p.gcn_dim, num_rel,
                               act=self.act, params=self.p, head_num=self.p.head_num)
        self.conv2 = CompRAM(self.edge_index, self.edge_type, self.p.gcn_dim, self.p.embed_dim, num_rel,
                               act=self.act, params=self.p, head_num=1) if self.p.gcn_layer >= 2 else None
        self.conv3 = CompRAM(self.edge_index, self.edge_type, self.p.gcn_dim, self.p.embed_dim, num_rel,
                               act=self.act, params=self.p, head_num=1) if self.p.gcn_layer >= 3 else None
        self.conv4 = CompRAM(self.edge_index, self.edge_type, self.p.gcn_dim, self.p.embed_dim, num_rel,
                               act=self.act, params=self.p, head_num=1) if self.p.gcn_layer >= 4 else None

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        self.special_spmm = SpecialSpmmFunc()
        self.rel_drop = nn.Dropout(0.1)

        self.im_proj = get_param((self.p.init_dim, self.p.init_dim))

    def forward_base(self, sub, rel, drop1, drop2):
        init_rel_real, init_rel_imaginary = self.init_rel_real, self.init_rel_imaginary
  
        init_rel_imaginary = init_rel_imaginary.mm(self.im_proj)

        init_embed_imaginary = self.init_embed_imaginary.mm(self.im_proj)
        ent_embed1_real, ent_embed1_imaginary, rel_embed1_real, rel_embed1_imaginary = \
           self.conv1(self.init_embed_real, init_embed_imaginary, init_rel_real, init_rel_imaginary)

        ent_embed1_real = drop1(ent_embed1_real)
        ent_embed1_imaginary = drop1(ent_embed1_imaginary)

        ent_embed2_real, ent_embed2_imaginary, rel_embed2_real, rel_embed2_imaginary = self.conv2(ent_embed1_real, ent_embed1_imaginary, rel_embed1_real, rel_embed1_imaginary) if self.p.gcn_layer >= 2 \
            else (ent_embed1_real, ent_embed1_imaginary, rel_embed1_real, rel_embed1_imaginary)
        ent_embed2_real = drop2(ent_embed2_real) if self.p.gcn_layer >= 2 else ent_embed1_real
        ent_embed2_imaginary = drop2(ent_embed2_imaginary) if self.p.gcn_layer >= 2 else ent_embed1_imaginary

        ent_embed3_real, ent_embed3_imaginary, rel_embed3_real, rel_embed3_imaginary = self.conv3(ent_embed2_real, ent_embed2_imaginary, rel_embed2_real, rel_embed2_imaginary) if self.p.gcn_layer >= 3 \
            else (ent_embed1_real, ent_embed1_imaginary, rel_embed1_real, rel_embed1_imaginary)
        ent_embed3_real = drop2(ent_embed3_real) if self.p.gcn_layer >= 3 else ent_embed1_real
        ent_embed3_imaginary = drop2(ent_embed3_imaginary) if self.p.gcn_layer >= 3 else ent_embed1_imaginary

        ent_embed4_real, ent_embed4_imaginary, rel_embed4_real, rel_embed4_imaginary = self.conv4(ent_embed3_real, ent_embed3_imaginary, rel_embed3_real, rel_embed3_imaginary) if self.p.gcn_layer == 4 \
            else (ent_embed1_real, ent_embed1_imaginary, rel_embed1_real, rel_embed1_imaginary)
        ent_embed4_real = drop2(ent_embed4_real) if self.p.gcn_layer >= 4 else ent_embed1_real
        ent_embed4_imaginary = drop2(ent_embed4_imaginary) if self.p.gcn_layer >= 4 else ent_embed1_imaginary

        if self.p.gcn_layer == 1:
            final_ent_real = ent_embed1_real
            final_ent_imaginary = ent_embed1_imaginary
            final_rel_real = rel_embed1_real
            final_rel_imaginary = rel_embed1_imaginary
        elif self.p.gcn_layer == 2:
            final_ent_real = ent_embed2_real
            final_ent_imaginary = ent_embed2_imaginary
            final_rel_real = rel_embed2_real
            final_rel_imaginary = rel_embed2_imaginary
        elif self.p.gcn_layer == 3:
            final_ent_real = ent_embed3_real
            final_ent_imaginary = ent_embed3_imaginary
            final_rel_real = rel_embed3_real
            final_rel_imaginary = rel_embed3_imaginary
        elif self.p.gcn_layer == 4:
            final_ent_real = ent_embed4_real
            final_ent_imaginary = ent_embed4_imaginary
            final_rel_real = rel_embed4_real
            final_rel_imaginary = rel_embed4_imaginary

        sub_emb_real = torch.index_select(final_ent_real, 0, sub)
        sub_emb_imaginary = torch.index_select(final_ent_imaginary, 0, sub)

        rel_emb_real = torch.index_select(final_rel_real, 0, rel)
        rel_emb_imaginary = torch.index_select(final_rel_imaginary, 0, rel)

        return sub_emb_real, sub_emb_imaginary, rel_emb_real, rel_emb_imaginary, final_ent_real, final_ent_imaginary

    def gather_neighbours(self):
        edge_weight = torch.ones_like(self.edge_type).float().unsqueeze(1)
        deg = self.special_spmm(self.edge_index, edge_weight, self.p.num_ent, self.p.num_ent, 1,
                                dim=1)
        deg[deg == 0.0] = 1.0
        entity_neighbours = self.init_embed[self.edge_index[1, :], :]
        entity_gathered = self.special_spmm(
            self.edge_index, entity_neighbours, self.p.num_ent, self.p.num_ent, self.p.init_dim,
            dim=1).div(deg)
        relation_neighbours = torch.index_select(self.init_rel, 0, self.edge_type)
        relation_gathered = self.special_spmm(
            self.edge_index, relation_neighbours, self.p.num_ent, self.p.num_ent, self.p.init_dim, dim=1).div(deg)
        return entity_gathered, relation_gathered

class CompRAMDistMult(CompRAMBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel):
        sub_emb_real, sub_emb_imaginary, rel_emb_real, rel_emb_imaginary, all_ent_real, all_ent_imaginary = self.forward_base(sub, rel, self.drop, self.drop)
        obj_emb = sub_emb_real * rel_emb_real

        x = torch.mm(obj_emb, all_ent_real.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score

class CompRAMComplEx(CompRAMBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)
        self.drop2 = torch.nn.Dropout(self.p.hid_drop)

        self.register_parameter(
            'bias', torch.nn.Parameter(torch.zeros(self.p.num_ent)))

    def mat_vec_mul(self, v, m):
        vm = torch.mm(v, m.view(v.size(1), -1))
        vm = vm.view(-1, v.size(1))
        vm = self.drop2(vm)
        return vm

    def forward(self, sub, rel):
        sub_emb_real, sub_emb_imaginary, rel_emb_real, rel_emb_imaginary, all_ent_real, all_ent_imaginary = self.forward_base(sub, rel, self.drop, self.drop)
       
        x = torch.mm(sub_emb_real*rel_emb_real, all_ent_real.transpose(1, 0)) +\
            torch.mm(sub_emb_real*rel_emb_imaginary, all_ent_imaginary.transpose(1, 0)) +\
            torch.mm(sub_emb_imaginary*rel_emb_real, all_ent_imaginary.transpose(1, 0)) -\
            torch.mm(sub_emb_imaginary*rel_emb_imaginary,
                    all_ent_real.transpose(1, 0))

        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)

        return score