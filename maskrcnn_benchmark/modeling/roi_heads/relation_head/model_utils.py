import copy
import glob
import json
import math
import os
import random
import re
import time
import PIL
from PIL import Image
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from maskrcnn_benchmark.data.datasets.visual_genome import load_info
from maskrcnn_benchmark.modeling.roi_heads.relation_head.llava_llama import LlavaLlamaForCausalLM
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_motifs import FrequencyBias
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_transformer import MultiHeadAttention, PositionwiseFeedForward
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_vctree import VCTreeLSTMContext
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_relation import layer_init
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.utils.comm import all_gather_with_grad, concat_all_gather, get_rank,find_linear_layers
from .utils_motifs import rel_vectors, obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info 
from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.modeling.roi_heads.relation_head.conversation import conv_templates
from maskrcnn_benchmark.modeling.roi_heads.relation_head.llava_arch import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX, UNION_IMAGE_INDEX,UNION_IMAGE_TOKEN
import transformers
import logging

class BiRef(nn.Module):
    def __init__(self, config, in_channels):
        super(BiRef, self).__init__()

        self.logger = logging.getLogger(__name__)
        embed_dim = config.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        roi_dim = config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        inner_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM

        num_head = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        rel_layer = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        
        if config.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.config=config
        self.nms_thresh = config.TEST.RELATION.LATER_NMS_PREDICTION_THRES
        
        statistics = get_dataset_statistics(config)
        
        obj_classes, rel_classes,fg_matrix = statistics['obj_classes'], statistics['rel_classes'],statistics['fg_matrix']
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=config.GLOVE_DIR, wv_dim=embed_dim)  # load Glove for objects
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=embed_dim)   # load Glove for predicates
        self.obj_embed = nn.Embedding(self.num_obj_cls, embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
        
        ##### refine image/text features
        pretrain_clip_model='/data/sdb/pretrain_ckpt/CLIP/clip-vit-base-patch32'
        self.clip_processor=transformers.AutoProcessor.from_pretrained(pretrain_clip_model)
        self.clip_tokenizer=transformers.AutoTokenizer.from_pretrained(pretrain_clip_model)
        self.clip_vision_model=transformers.CLIPVisionModel.from_pretrained(pretrain_clip_model)

        self.align_img=make_fc(self.clip_vision_model.config.hidden_size,self.hidden_dim)
                
        if self.mode == 'predcls' or self.mode=='sgdet':
            ##### refine object labels
            self.pos_embed = nn.Sequential(*[
                nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
                nn.Linear(32, 128), nn.ReLU(inplace=True),
            ])
            
            self.out_obj = make_fc(self.hidden_dim, self.num_obj_cls) 
            self.lin_obj_cyx = make_fc(in_channels + embed_dim + 128, self.hidden_dim)
            self.p_pos=make_fc(128,self.hidden_dim)
            self.p_entity=make_fc(in_channels,self.hidden_dim*2)
        elif self.mode=='sgcls':
            # init contextual lstm encoding
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        else:
            raise ValueError(f'Unknow mode: {self.mode}')
            
        self.rel_quary=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.hidden_dim,)))
        self.sem_rel_quary=nn.Parameter(torch.normal(mean=0, std=0.1, size=(self.hidden_dim,)))
        self.p_img_rep=nn.Sequential(
            nn.Conv2d(in_channels,self.hidden_dim,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=1,padding=0,stride=1)
        )
        
        self.rel_query_init=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        
        self.rel_query_refine=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        self.p_sub = MLP(embed_dim, self.hidden_dim // 2, self.hidden_dim, 2)
        self.p_obj = MLP(embed_dim, self.hidden_dim // 2, self.hidden_dim, 2)
        self.p_pred = MLP(embed_dim, self.hidden_dim // 2, self.hidden_dim, 2)
        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.hidden_dim*2), nn.ReLU(True),
            nn.Dropout(dropout_rate), nn.Linear(self.hidden_dim*2, self.hidden_dim)
        ])
        
        self.gate_sub=make_fc(self.hidden_dim*2,self.hidden_dim)
        self.gate_obj=make_fc(self.hidden_dim*2,self.hidden_dim)
        self.gate_pred=make_fc(self.hidden_dim*2,self.hidden_dim)
        
        self.sample_union_rep=MLP(in_channels,self.hidden_dim,self.hidden_dim,2)
        
        self.filter_rel_rep=nn.Sequential(
            nn.Linear(self.hidden_dim,self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.filter_rel_norm=nn.LayerNorm(self.hidden_dim)
        self.drop_rel_rep=nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.refine_union_vis=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        
        self.init_sem_rel_query=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        self.refine_sem_rel_query=nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.MultiheadAttention(self.hidden_dim,num_head,dropout=dropout_rate,batch_first=True),
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(
                    nn.Linear(self.hidden_dim,inner_dim),
                    nn.ReLU(),
                    nn.Linear(inner_dim,self.hidden_dim),
                    nn.Dropout(dropout_rate)
                ),
                nn.LayerNorm(self.hidden_dim)
            ]) for _ in range(rel_layer)
        ])
        self.geo_rel_pre=nn.Linear(self.hidden_dim,self.num_rel_cls)
        self.sem_rel_pre=nn.Linear(self.hidden_dim,self.num_rel_cls)
        
        self.fusion_triple_sem_rep=MLP(self.hidden_dim*3,self.hidden_dim,self.hidden_dim,2)
        self.triple_sem_pre=nn.Linear(self.hidden_dim,self.num_rel_cls)
        
        self.proj_head=MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim*2, 2)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # **************** loss ********************
        self.gamma,self.total_iters=1,config.SOLVER.MAX_ITER
        bata=0.9999
        
        per_predicate_num=np.sum(fg_matrix.numpy(),axis=(0,1))
        self.per_predicate_weight=torch.tensor([(1-bata)/(1-bata**pre_num) for pre_num in per_predicate_num],dtype=torch.float)
        self.rel_ce_loss=nn.CrossEntropyLoss(self.per_predicate_weight)
        
        # **************** predicate prediction weights ********************
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)
        if self.mode=='sgcls':
            self.geo_rel_w,self.sem_rel_w,self.sem_rel_sim_w,self.triple_sem_rel_w=nn.Parameter(torch.ones((self.num_rel_cls,))),nn.Parameter(torch.ones((self.num_rel_cls,))),nn.Parameter(torch.ones((self.num_rel_cls,))),nn.Parameter(torch.ones((self.num_rel_cls,)))
            self.freq_weights=nn.Parameter(torch.ones((self.num_rel_cls,)))
            
    def calculate_loss(self,proposals,refine_logits,relation_logits,rel_labels):
        # ************************ relation loss ****************************
        relation_logits,rel_labels=torch.cat(relation_logits,dim=0) if isinstance(relation_logits,(list,tuple)) else relation_logits,torch.cat(rel_labels,dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
        rel_ce_loss=self.rel_ce_loss(relation_logits,rel_labels)
        
        rel_log_softmax = torch.log_softmax(relation_logits, dim=1)
        rel_logpt = torch.gather(rel_log_softmax, dim=1, index=rel_labels.view(-1, 1)).view(-1)
        
        rel_loss=(1-torch.exp(rel_logpt))**self.gamma*rel_ce_loss
        rel_loss=torch.mean(rel_loss)  # torch.sum(f_loss)
        
        # **************************** object loss ***************************
        refine_obj_logits = cat(refine_logits, dim=0) if isinstance(refine_logits,(list,tuple)) else refine_logits
        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        
        obj_loss = F.cross_entropy(refine_obj_logits, fg_labels.long())
        
        # ********************************************************************
        
        return rel_loss,obj_loss
      
    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,**kwargs):
        current_device,add_losses,add_data=torch.device(f'cuda:{torch.cuda.current_device()}'),dict(),dict()
        
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        # refine object labels        
        if self.mode == 'predcls' or self.mode=='sgdet':
            entity_dists, entity_preds, pos_embeds = self.refine_obj_labels(roi_features, proposals)
            entity_vis_rep=self.p_entity(roi_features)
            pos_embeds=pos_embeds.split(num_objs,dim=0)
        elif self.mode=='sgcls':
            entity_dists, entity_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)
            entity_vis_rep=F.relu(self.post_emb(edge_ctx))
            pos_embeds=[None]*len(num_objs)
            
            if self.training:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] =add_losses.get("binary_loss",0.0) + (sum(binary_loss) / len(binary_loss))
        else:
            raise ValueError(f'Unknow mode: {self.mode}') 
        
        entity_vis_rep = entity_vis_rep.view(entity_vis_rep.size(0), 2, self.hidden_dim) # entity representation
        
        sub_vis_reps = entity_vis_rep[:, 1].contiguous().view(-1, self.hidden_dim).split(num_objs,dim=0)
        obj_vis_reps = entity_vis_rep[:, 0].contiguous().view(-1, self.hidden_dim).split(num_objs,dim=0)
        
        entity_dists = entity_dists.split(num_objs, dim=0)
        entity_sem_reps= self.obj_embed(entity_preds.long()).split(num_objs,dim=0)
        entity_preds= entity_preds.split(num_objs, dim=0)
        # union_features=union_features.split(num_rels,dim=0)
        
        rel_sem_vector=self.p_pred(self.rel_embed.weight)
        rel_vis_reps,sub_sem_reps,obj_sem_reps,img_reps,pair_preds=[],[],[],[],[]
        for batch_idx,(proposal,sub_vis_rep,obj_vis_rep,entity_sem_rep,rel_pair_idx,entity_pred,pos_embed) in enumerate(zip(proposals,sub_vis_reps,obj_vis_reps,entity_sem_reps,rel_pair_idxs,entity_preds,pos_embeds)):
            image = Image.open(proposal.get_field('file_name'))
            image_inputs = self.clip_processor(images=image, return_tensors="pt").to(current_device)
            img_encode_out=self.clip_vision_model(**image_inputs)
            img_rep = self.align_img(img_encode_out.last_hidden_state[:,1:,:])  # without cls token
            
            if self.mode == 'predcls' or self.mode=='sgdet':
                sub_pos_embed,obj_pos_embed=self.p_pos(pos_embed[rel_pair_idx[:,0]]),self.p_pos(pos_embed[rel_pair_idx[:,1]])
            sub_vis_rep,obj_vis_rep=sub_vis_rep[rel_pair_idx[:,0]],obj_vis_rep[rel_pair_idx[:,1]]
            sub_sem_rep,obj_sem_rep=entity_sem_rep[rel_pair_idx[:,0]],entity_sem_rep[rel_pair_idx[:,1]]
        
            # ********************************************* refine visual features ***************************************************
            if self.mode == 'predcls' or self.mode=='sgdet':
                sub_geo_rep,obj_geo_rep=sub_vis_rep+F.relu(sub_pos_embed),obj_vis_rep+F.relu(obj_pos_embed)
            else:
                sub_geo_rep,obj_geo_rep=sub_vis_rep,obj_vis_rep
            rel_vis_rep,geo_vis_rep=self.rel_quary.expand(sub_geo_rep.shape[0],1,-1),torch.stack([sub_geo_rep,obj_geo_rep],dim=1)

            for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.rel_query_init:
                attn_output, _ =s_attn(query=rel_vis_rep,key=rel_vis_rep,value=rel_vis_rep)
                rel_vis_rep=s_norm(rel_vis_rep+attn_output)
                
                attn_output, _ =c_attn(query=rel_vis_rep,key=geo_vis_rep,value=geo_vis_rep)
                rel_vis_rep=c_norm(rel_vis_rep+attn_output)
                
                rel_vis_rep=ffn_norm(ffn(rel_vis_rep)+rel_vis_rep)
            
            expand_img_rep=img_rep.expand(sub_geo_rep.shape[0],-1,-1)
            for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.rel_query_refine:
                attn_output, _ =s_attn(query=rel_vis_rep,key=rel_vis_rep,value=rel_vis_rep)
                rel_vis_rep=s_norm(rel_vis_rep+attn_output)
                
                attn_output, _ =c_attn(query=rel_vis_rep,key=expand_img_rep,value=expand_img_rep)
                rel_vis_rep=c_norm(rel_vis_rep+attn_output)
                
                rel_vis_rep=ffn_norm(ffn(rel_vis_rep)+rel_vis_rep)

            rel_vis_rep=rel_vis_rep.squeeze(1)  # rel_num, hidden_dim
            rel_vis_reps.append(rel_vis_rep)
            # ********************************************* refine semantic features ***************************************************
            # refine object semantic features
            sub_sem_rep,obj_sem_rep=self.p_sub(sub_sem_rep),self.p_obj(obj_sem_rep)
            vis2sem_sub,vis2sem_obj,vis2sem_img=self.vis2sem(sub_vis_rep),self.vis2sem(obj_vis_rep),self.vis2sem(img_rep)
            
            gate_sub=F.sigmoid(self.gate_sub(torch.cat([sub_sem_rep,vis2sem_sub],dim=-1)))
            gate_obj=F.sigmoid(self.gate_obj(torch.cat([obj_sem_rep,vis2sem_obj],dim=-1)))
            
            sub_sem_rep,obj_sem_rep=sub_sem_rep+vis2sem_sub*gate_sub,obj_sem_rep+vis2sem_obj*gate_obj
            sub_sem_reps.append(sub_sem_rep)
            obj_sem_reps.append(obj_sem_rep)
            
            img_reps.append(vis2sem_img.expand(sub_geo_rep.shape[0],-1,-1))
            pair_preds.append(torch.stack([entity_pred[rel_pair_idx[:,0]],entity_pred[rel_pair_idx[:,1]]],dim=1))
        geo_rel_pre=self.geo_rel_pre(torch.cat(rel_vis_reps,dim=0))
        
        # refine predicate semantic features
        sub_sem_reps,obj_sem_reps=torch.cat(sub_sem_reps,dim=0),torch.cat(obj_sem_reps,dim=0)
        fusion_entity=F.relu(sub_sem_reps+obj_sem_reps)-(sub_sem_reps-obj_sem_reps)**2
        union_sem_reps=self.vis2sem(self.sample_union_rep(union_features))
        gate_union=F.sigmoid(self.gate_pred(torch.cat([fusion_entity,union_sem_reps],dim=-1)))
        
        rel_sem_reps=fusion_entity-union_sem_reps*gate_union
        rel_sem_reps=self.filter_rel_norm(self.filter_rel_rep(rel_sem_reps)+rel_sem_reps)
        rel_sem_reps=self.drop_rel_rep(rel_sem_reps)
        
        # refine triple semantic using image
        triple_sem_reps=torch.stack([sub_sem_reps,rel_sem_reps,obj_sem_reps],dim=1)
        img_reps=torch.cat(img_reps,dim=0)
        
        union_sem_reps,sem_rel_query=union_sem_reps.unsqueeze(1),self.sem_rel_quary.expand(img_reps.shape[0],1,-1)
        for (s_attn,s_norm,ffn,ffn_norm) in self.refine_union_vis:
            attn_output, _ =s_attn(query=img_reps,key=union_sem_reps,value=union_sem_reps)
            img_reps=s_norm(img_reps+attn_output)
            
            img_reps=ffn_norm(ffn(img_reps)+img_reps)
        
        for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.init_sem_rel_query:
            s_attn_output, _= s_attn(query=sem_rel_query,key=sem_rel_query,value=sem_rel_query)
            sem_rel_query=s_norm(sem_rel_query+s_attn_output)
            
            attn_output, _ =c_attn(query=sem_rel_query,key=triple_sem_reps,value=triple_sem_reps)
            sem_rel_query=c_norm(sem_rel_query+attn_output)
            
            sem_rel_query=ffn_norm(ffn(sem_rel_query)+sem_rel_query)
        
        for (s_attn,s_norm,c_attn,c_norm,ffn,ffn_norm) in self.refine_sem_rel_query:
            s_attn_output, _= s_attn(query=sem_rel_query,key=sem_rel_query,value=sem_rel_query)
            sem_rel_query=s_norm(sem_rel_query+s_attn_output)
            
            attn_output, _ =c_attn(query=sem_rel_query,key=img_reps,value=img_reps)
            sem_rel_query=c_norm(sem_rel_query+attn_output)
            
            sem_rel_query=ffn_norm(ffn(sem_rel_query)+sem_rel_query)
        
        sem_rel_pre=self.sem_rel_pre(sem_rel_query.squeeze(1))
        
        # triple semantic similarity
        triple_query_sem_reps=torch.cat([sub_sem_reps,sem_rel_query.squeeze(1),obj_sem_reps],dim=-1)  
        triple_query_sem_reps=self.fusion_triple_sem_rep(triple_query_sem_reps)
        triple_sem_rel_pre=self.triple_sem_pre(triple_query_sem_reps)
        
        # semantic similarity
        rel_sem_vec=self.proj_head(self.drop_rel_rep(rel_sem_vector))
        rel_sem_reps=self.proj_head(rel_sem_reps)
        
        rel_sem_reps_norm = rel_sem_reps / rel_sem_reps.norm(dim=1, keepdim=True)  # r_norm
        rel_sem_vec_norm = rel_sem_vec / rel_sem_vec.norm(dim=1, keepdim=True)  # c_norm

        sem_rel_sim=rel_sem_reps_norm @ rel_sem_vec_norm.t() * self.logit_scale.exp()
        # final predicate dists
        if self.mode=='sgcls':
            rel_dists=geo_rel_pre*self.geo_rel_w+sem_rel_pre*self.sem_rel_w+sem_rel_sim*self.sem_rel_sim_w+triple_sem_rel_pre*self.triple_sem_rel_w
        else:
            rel_dists=geo_rel_pre+sem_rel_pre+sem_rel_sim+triple_sem_rel_pre
        
        if self.use_bias:
            freq_dist=self.freq_bias.index_with_labels(torch.cat(pair_preds,dim=0).long())
            rel_dists=rel_dists+freq_dist*getattr(self,'freq_weights',1)
        
        if self.training:
            rel_labels=torch.cat(rel_labels,dim=0)

            add_losses=self.calculate_similar_loss(rel_sem_vec,rel_sem_reps,rel_labels,add_losses,loss_name="loss_dis")
            
            add_data['final_loss']=dict()
            loss_relation,loss_refine=self.calculate_loss(proposals=proposals,refine_logits=entity_dists,relation_logits=rel_dists,rel_labels=rel_labels)
            add_data['final_loss']['loss_relation'],add_data['final_loss']['loss_refine']=loss_relation,loss_refine
        
        rel_dists=rel_dists.split(num_rels,dim=0)
        return entity_dists, rel_dists, add_losses, add_data
    
    def calculate_semantic_loss(self,semantic_feature,semantic_feature_norm):
        add_losses=dict()
        
        ### Prototype Regularization  ---- cosine similarity
        target_rpredicate_proto_norm = semantic_feature_norm.clone().detach() 
        simil_mat = semantic_feature_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
        l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (51*51)  
        add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
        ### end
        
        ### Prototype Regularization  ---- Euclidean distance
        gamma2 = 7.0
        predicate_proto_a = semantic_feature.unsqueeze(dim=1).expand(-1, 51, -1) 
        predicate_proto_b = semantic_feature.detach().unsqueeze(dim=0).expand(51, -1, -1)
        proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
        sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
        topK_proto_dis = sorted_proto_dis_mat[:, :11].sum(dim=1) / 10   # obtain d-, where k2 = 1
        dist_loss = torch.max(torch.zeros(51).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
        add_losses.update({"dist_loss2": dist_loss})
        ### end
        
        return add_losses
        
    def calculate_similar_loss(self,semantic_feature,rel_rep,rel_labels,add_losses,loss_name="loss_dis"):
        ###  Prototype-based Learning  ---- Euclidean distance
        # rel_labels = cat(rel_labels, dim=0) if isinstance(rel_labels,(list,tuple)) else rel_labels
        gamma1 = 1.0
        rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, semantic_feature.shape[0], -1)  # r
        predicate_proto_expand = semantic_feature.unsqueeze(dim=0).expand(rel_rep.size(0), -1, -1)  # ci
        distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
        mask_neg = torch.ones(rel_rep.size(0), semantic_feature.shape[0]).cuda()  
        mask_neg[torch.arange(rel_rep.size(0)), rel_labels] = 0
        distance_set_neg = distance_set * mask_neg
        distance_set_pos = distance_set[torch.arange(rel_rep.size(0)), rel_labels]  # gt i.e., g+
        sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
        topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
        loss_sum = torch.max(torch.zeros(rel_rep.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
        add_losses[loss_name]=add_losses.get(loss_name,0.0)+loss_sum     # Le_euc = max(0, (g+) - (g-) + gamma1)
        ### end 
        
        return add_losses
    
    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 512 -> 151
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()
        
        return obj_dists, obj_preds, pos_embed

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds
