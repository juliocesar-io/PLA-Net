import torch
import logging
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from gcn_lib.sparse.torch_nn import norm_layer, MLP, MM_AtomEncoder
from model.model_encoder import AtomEncoder, BondEncoder

from gcn_lib.sparse.torch_nn import MLP, norm_layer, BondEncoder, MM_BondEncoder
from gcn_lib.sparse.torch_message import GenMessagePassing, MsgNorm

class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers=20, mlp_layers=3,
                 msg_norm=False, learn_msg_scale=False, conv_encode_edge=True, 
                 dropout=0.0, block='res+', add_virtual_node=False, 
                 conv='gen', aggr='softmax', t=1.0, learn_t=True, p=1.0, 
                 learn_p=False, norm='batch', graph_pooling='mean', nclasses=2, 
                 is_prot=False, hidden_channels_prot=128, 
                 num_layers_prot=20, mlp_layers_prot=3, 
                 msg_norm_prot=False, learn_msg_scale_prot=False, 
                 conv_encode_edge_prot=False, saliency=False):

        super(DeeperGCN, self).__init__()

        # Set PM configuration
        if is_prot:
            self.num_layers = num_layers_prot
            mlp_layers = mlp_layers_prot
            hidden_channels = hidden_channels_prot
            self.msg_norm = msg_norm_prot
            learn_msg_scale = learn_msg_scale_prot
            self.conv_encode_edge = conv_encode_edge_prot
        # Set LM configuration
        else:
            self.num_layers = num_layers
            hidden_channels = hidden_channels
            self.msg_norm = msg_norm
            learn_msg_scale = learn_msg_scale
            self.conv_encode_edge = conv_encode_edge

        # Set overall model configuration
        self.dropout = dropout
        self.block = block
        self.add_virtual_node = add_virtual_node
        self.training = True

        self.learn_t = learn_t
        self.learn_p = learn_p

        # Set GCN layers configuration
        print(
            "The number of layers {}".format(self.num_layers),
            "Aggr aggregation method {}".format(aggr),
            "block: {}".format(self.block),
        )
        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        if self.add_virtual_node:
            self.virtualnode_embedding = torch.nn.Embedding(1, hidden_channels)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
            self.mlp_virtualnode_list = torch.nn.ModuleList()
            for layer in range(self.num_layers - 1):
                self.mlp_virtualnode_list.append(MLP([hidden_channels] * 3, norm=norm))

        for layer in range(self.num_layers):
            if conv == "gen":
                gcn = GENConv(
                    hidden_channels,
                    hidden_channels,
                    advs=False,
                    aggr=aggr,
                    t=t,
                    learn_t=self.learn_t,
                    p=p,
                    learn_p=self.learn_p,
                    msg_norm=self.msg_norm,
                    learn_msg_scale=learn_msg_scale,
                    encode_edge=self.conv_encode_edge,
                    bond_encoder=True,
                    norm=norm,
                    mlp_layers=mlp_layers,
                )
            else:
                raise Exception("Unknown Conv Type")
            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

        # Set embbeding layers
        if saliency:
            self.atom_encoder = MM_AtomEncoder(emb_dim=hidden_channels)
        else:
            self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)

        if not self.conv_encode_edge:
            self.bond_encoder = BondEncoder(emb_dim=hidden_channels)

        # Set type of pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise Exception("Unknown Pool Type")

        # Set classification layer
        self.graph_pred_linear = torch.nn.Linear(hidden_channels, nclasses)

    def forward(self, input_batch, dropout=True, embeddings=False):

        x = input_batch.x
        edge_index = input_batch.edge_index
        edge_attr = input_batch.edge_attr
        batch = input_batch.batch

        h = self.atom_encoder(x)

        if self.add_virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1)
                .to(edge_index.dtype)
                .to(edge_index.device)
            )
            h = h + virtualnode_embedding[batch]

        if self.conv_encode_edge:
            edge_emb = edge_attr
        else:
            edge_emb = self.bond_encoder(edge_attr)

        if self.block == "res+":

            h = self.gcns[0](h, edge_index, edge_emb)

            for layer in range(1, self.num_layers):
                h1 = self.norms[layer - 1](h)
                h2 = F.relu(h1)
                if dropout:
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)

                if self.add_virtual_node:
                    virtualnode_embedding_temp = (
                        global_add_pool(h2, batch) + virtualnode_embedding
                    )
                    if dropout:
                        virtualnode_embedding = F.dropout(
                            self.mlp_virtualnode_list[layer - 1](
                                virtualnode_embedding_temp
                            ),
                            self.dropout,
                            training=self.training,
                        )

                    h2 = h2 + virtualnode_embedding[batch]

                h = self.gcns[layer](h2, edge_index, edge_emb) + h

            h = self.norms[self.num_layers - 1](h)
            if dropout:
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == "res":
            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)
            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                h = F.relu(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)

        h_graph = self.pool(h, batch)
        if embeddings:
            return h_graph
        else:
            return self.graph_pred_linear(h_graph)

    def print_params(self, epoch=None, final=False):
        if self.learn_t:
            ts = [gcn.t.item() for gcn in self.gcns]
            if final:
                print("Final t {}".format(ts))
            else:
                logging.info("Epoch {}, t {}".format(epoch, ts))
        if self.learn_p:
            ps = [gcn.p.item() for gcn in self.gcns]
            if final:
                print("Final p {}".format(ps))
            else:
                logging.info("Epoch {}, p {}".format(epoch, ps))
        if self.msg_norm:
            ss = [gcn.msg_norm.msg_scale.item() for gcn in self.gcns]
            if final:
                print("Final s {}".format(ss))
            else:
                logging.info("Epoch {}, s {}".format(epoch, ss))


class GENConv(GenMessagePassing):
    """
     GENeralized Graph Convolution (GENConv): https://arxiv.org/pdf/2006.07739.pdf
     SoftMax  &  PowerMean Aggregation
    """
    def __init__(self, in_dim, emb_dim, advs=False,
                 aggr='softmax',
                 t=1.0, learn_t=False,
                 p=1.0, learn_p=False,
                 y=0.0, learn_y=False,
                 msg_norm=False, learn_msg_scale=True,
                 encode_edge=False, bond_encoder=False,
                 edge_feat_dim=None,
                 norm='batch', mlp_layers=2,
                 eps=1e-7):

        super(GENConv, self).__init__(aggr=aggr,
                                      t=t, learn_t=learn_t,
                                      p=p, learn_p=learn_p, 
                                      y=y, learn_y=learn_y)

        channels_list = [in_dim]

        for i in range(mlp_layers-1):
            channels_list.append(in_dim*2)

        channels_list.append(emb_dim)

        self.mlp = MLP(channels=channels_list,
                       norm=norm,
                       last_lin=True)

        self.msg_encoder = torch.nn.ReLU()
        self.eps = eps

        self.msg_norm = msg_norm
        self.encode_edge = encode_edge
        self.bond_encoder = bond_encoder
        self.advs = advs
        if msg_norm:
            self.msg_norm = MsgNorm(learn_msg_scale=learn_msg_scale)
        else:
            self.msg_norm = None

        if self.encode_edge:
            if self.bond_encoder:
                if self.advs:
                    self.edge_encoder = MM_BondEncoder(emb_dim=in_dim)
                else:
                    self.edge_encoder = BondEncoder(emb_dim=in_dim)
            else:
                self.edge_encoder = torch.nn.Linear(edge_feat_dim, in_dim)