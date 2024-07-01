import json
import multiprocessing
import pathlib
import pdb,re
import random
import numpy as np
from torch import nn
import torch
import tqdm
import math
import os,h5py
from glob import glob
from PIL import Image

BOX_SCALE=1024

def load_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    if add_bg:
        info['label_to_idx']['__background__'] = 0
        info['predicate_to_idx']['__background__'] = 0
        info['attribute_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    attribute_to_ind = info['attribute_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])
    ind_to_attributes = sorted(attribute_to_ind, key=lambda k: attribute_to_ind[k])

    return ind_to_classes, ind_to_predicates, ind_to_attributes


def load_image_filenames(img_dir, image_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return: 
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
            img_info.append(img)
    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info


def load_graphs(roidb_file, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap=False):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
    Return: 
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground 
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of 
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    roi_h5 = h5py.File(roidb_file, 'r')
    data_split = roi_h5['split'][:]
    split_flag = 2 if split == 'test' else 0
    split_mask = data_split == split_flag

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if split == 'val':
            image_index = image_index[:num_val_im]
        elif split == 'train':
            image_index = image_index[num_val_im:]


    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_attributes = roi_h5['attributes'][:, :]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    gt_attributes = []
    relationships = []
    for i in range(len(image_index)):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]

        boxes_i = all_boxes[i_obj_start : i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start : i_obj_end + 1]
        gt_attributes_i = all_attributes[i_obj_start : i_obj_end + 1, :]

        if i_rel_start >= 0:
            predicates = _relation_predicates[i_rel_start : i_rel_end + 1]
            obj_idx = _relations[i_rel_start : i_rel_end + 1] - i_obj_start # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates)) # (num_rel, 3), representing sub, obj, and pred
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        gt_attributes.append(gt_attributes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, gt_attributes, relationships


def find_linear_layers(model, lora_target_modules):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            and all(
                [
                    x not in name
                    for x in [
                        "visual_model",
                        "vision_tower",
                        "mm_projector",
                        "text_hidden_fcs",
                    ]
                ]
            )
            and any([x in name for x in lora_target_modules])
        ):
            lora_module_names.add(name)
    return sorted(list(lora_module_names))

if __name__=='__main__':
    
    """
    zeroshot_load_path='maskrcnn_benchmark/data/datasets/evaluation/vg/zeroshot_triplet.pytorch'
    zeroshot_triplet = torch.load(zeroshot_load_path, map_location=torch.device("cpu")).long().numpy()
    if os.path.exists('maskrcnn_benchmark/data/datasets/evaluation/vg/zeroshot_seen_cls.json'):
        with open('maskrcnn_benchmark/data/datasets/evaluation/vg/zeroshot_seen_cls.json','r') as seen_cls_files:
            load_json=json.load(seen_cls_files)
            zeroshot_seen_cls=load_json['seen_cls']
            zeroshot_unseen_cls=load_json['unseen_cls']
    
    print(f'zeroshot seen class id: {zeroshot_seen_cls}')
    print(f'zeroshot unseen class id: {zeroshot_unseen_cls}')
    seen_rel_data,unseen_rel_data=[],[]    
    for per_rel in tqdm.tqdm(zeroshot_triplet):
        if per_rel[-1] in zeroshot_seen_cls:
            seen_rel_data.append(per_rel)

        if per_rel[-1] in zeroshot_unseen_cls:
            unseen_rel_data.append(per_rel)
    
    seen_rel_data=np.stack(seen_rel_data,axis=0)
    unseen_rel_data=np.stack(unseen_rel_data,axis=0)
    
    print(f'process seen rel data shape: {seen_rel_data.shape}, unseen rel data shape: {unseen_rel_data.shape}')
    torch.save(torch.from_numpy(seen_rel_data),'maskrcnn_benchmark/data/datasets/evaluation/vg/zeroshot_triplet_seen.pytorch')
    torch.save(torch.from_numpy(unseen_rel_data),'maskrcnn_benchmark/data/datasets/evaluation/vg/zeroshot_triplet_unseen.pytorch')
    raise
    """
    
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib import cm
    matplotlib.rcParams.update({'font.size': 26})
    
    """
    all_imgs=glob('/data/sdc/SGG_data/VG/VG_100K/*.jpg')


    # ----------------------------- load image files -------------------------------
    with open("/data/sdc/SGG_data/VG/image_data.json", 'r') as f:
        im_data = json.load(f)  # list length: 108073
    
    # ------------- load image attributes relationship and bounding box ------------
    roi_h5=h5py.File('/data/sdc/SGG_data/VG/VG-SGG-with-attri.h5','r')
    
    print(f'roi_h5 keys: {roi_h5.keys()}')  #  ['active_object_mask', 'attributes', 'boxes_1024', 'boxes_512', 'img_to_first_box', 'img_to_first_rel', 'img_to_last_box', 'img_to_last_rel', 'labels', 'predicates', 'relationships', 'split']
    
    all_imgs=glob('/data/sdc/SGG_data/VG/VG_100K/*.jpg')
    img_nums=len(all_imgs)  # 108249 ==> some image corrupted
    
    attributes=roi_h5['attributes']  # shape:(1145398, 10)  201 classes  attribute cls
    labels=roi_h5['labels'] # shape:(1145398, 1)  151 classes    object cls
    predicates=roi_h5['predicates'] # shape:(622705, 1)  51 classes  relationship cls
    relationships=roi_h5['relationships'] # shape:(622705, 2)   bounding box pair relation
    
    active_object_mask=roi_h5['active_object_mask'] # shape:(1145398, 1)  

    boxes_1024=roi_h5['boxes_1024'] # shape:(1145398, 4)
    boxes_512=roi_h5['boxes_512'] # shape:(1145398, 4)
    
    split=roi_h5['split']  # shape:(108073,)
    
    img_to_first_box=roi_h5['img_to_first_box'] # shape:(108073,)  ==> 索引==> boxes_1024[idx,]; boxes_512[idx,]
    img_to_first_rel=roi_h5['img_to_first_rel'] # shape:(108073,)
    
    img_to_last_box=roi_h5['img_to_last_box'] # shape:(108073,)
    img_to_last_rel=roi_h5['img_to_last_rel'] # shape:(108073,)
    """
    
    """
    # ------------------------------- generate subject-predict-object triples -------------------------------
    roidb_file,split,img_dir,image_file,dict_file="/data/sdc/SGG_data/VG/VG-SGG-with-attri.h5","test","/data/sdc/SGG_data/VG/VG_100K","/data/sdc/SGG_data/VG/image_data.json","/data/sdc/SGG_data/VG/VG-SGG-dicts-with-attri.json"
    ind_to_classes, ind_to_predicates, ind_to_attributes = load_info(dict_file)
    filenames, img_info = load_image_filenames(img_dir, image_file) # length equals to split_mask

    split_mask, gt_boxes, gt_classes, gt_attributes, relationships = load_graphs(
            roidb_file, split, num_im=-1, num_val_im=5000,
            filter_empty_rels=True
        )
    
    filenames = [filenames[i] for i in np.where(split_mask)[0]]
    img_info = [img_info[i] for i in np.where(split_mask)[0]]
    
    pdb.set_trace()
    for file in ['2321402.jpg','2324866.jpg','2342682.jpg','2342868.jpg','2343447.jpg','2343492.jpg']:
        img_path=f'{img_dir}/{file}'
        
        img_id=filenames.index(img_path)
        
        classes=gt_classes[img_id]
        relationship=relationships[img_id]
        
        relation_info=[]
        for sub_obj_rel in relationship:
            sub,obj,rel_id=sub_obj_rel
            
            sub_cls_id,obj_cls_id=classes[sub],classes[obj]
            
            relation_info.append(f'{ind_to_classes[sub_cls_id]} is {ind_to_predicates[rel_id]} {ind_to_classes[obj_cls_id]}')
        print(f'img_info: {img_info[img_id]}. In this image, relation info: {relation_info}')
    
    """
    
    
    # """
    # ---------------------------- generate data statistic picture ----------------------------
    # colors=['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b','#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d','#edf8e9', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b','#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']
    
    with open('/data/sdc/SGG_data/VG/common_json/train/relation/balanced_comm_2.json','r') as train_files:
        balances_file_datas=json.load(train_files)
        
    ind_to_predicates=balances_file_datas['ind_to_predicates'].values()

    final_all_cls_num=[0]*len(ind_to_predicates)
    for pair_info in balances_file_datas['pair_img_info']:
        for comm_rel in pair_info['common_rels']:
            final_all_cls_num[comm_rel-1]+=1
    
    final_cls_to_num_dict={name:num for name,num in zip(ind_to_predicates,final_all_cls_num)}

    print(final_cls_to_num_dict,len(balances_file_datas['pair_img_info']),len(final_cls_to_num_dict.keys()))

    
    fig=plt.figure(figsize=(30,30))
    
    
    colors_cool=cm.Set3(np.linspace(0,1,len(ind_to_predicates))) # tab20,tab20b
    # colors_hot=cm.Paired(np.linspace(0,1,len(ind_to_predicates)))  # Set2,Paired
    # cmap_merged=colors.ListedColormap(np.vstack((colors_cool,colors_hot)))
    cmap_merged=colors.ListedColormap(colors_cool)

    bars=plt.bar(ind_to_predicates, final_all_cls_num, color=cmap_merged.colors)
    for bar, value in zip(bars, final_all_cls_num):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha='center', va='bottom',weight='bold',fontsize=24)

    # wedges, texts, autotexts=plt.pie(final_all_cls_num,labels=ind_to_predicates,autopct='%1.1f%%',startangle=90,colors=cmap_merged.colors,textprops={'fontsize': 24,'weight': 'bold'},wedgeprops=dict(edgecolor='w'))
    
    plt.xticks(rotation=90,fontsize=27, ha='center',weight='bold')  # 90 度旋转
    plt.yticks(weight='bold',fontsize=27) 
    
    plt.title('Common relation class number distribution', weight='bold',pad=40)
    
    plt.xlabel('The name of the relationship category',weight='bold',labelpad=10)
    plt.ylabel('The number of relation categories', weight='bold',labelpad=30)
    
    plt.savefig('mac-rel-number.png')
    raise
    # """
    
    """
    # ---------------------------- generate question-answer pair statistic picture ----------------------------
    instruct_qa_lists,qa_num_dict=[],dict()
    with open('/data/sdc/SGG_data/VG/instruct_data.json','r') as instruct_files:
        for instruct_file in instruct_files:
            qa_infos=json.loads(instruct_file)['q_a']
            qa_num=len(qa_infos)
            instruct_qa_lists.append(qa_infos)

            qa_num_dict[qa_num]=qa_num_dict.get(qa_num,0)+1

    qa_num_key,qa_num_value=qa_num_dict.keys(),qa_num_dict.values()
    
    fig=plt.figure(figsize=(20,20))
    colors_cool=cm.tab20c(np.linspace(0,1,len(qa_num_key))) # tab20,tab20b          
    colors_hot=cm.Set3(np.linspace(0,1,len(qa_num_key)))  # Set2,Paired
    cmap_merged=colors.ListedColormap(np.vstack((colors_cool,colors_hot)))
    
    # plt.plot()
    bars=plt.bar(x=[num+3 for num in qa_num_key], height=qa_num_value, color=cmap_merged.colors)
    for bar, value in zip(bars, qa_num_value):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha='center', va='bottom',fontsize=22,weight='bold')

    
    plt.yscale('log')
    
    plt.xticks([num+3 for num in qa_num_key],[num+3 for num in qa_num_key])
    
    # plt.text(- 2.3, max(qa_num_value) * 2, 'log()')
    
    plt.xlabel('Number of instruction pairs generated per sample', weight='bold',labelpad=30)
    plt.ylabel('The number of instruction pairs generated for all samples', weight='bold',labelpad=30)
    plt.title('The number of instruction pairs generated for each sample is counted', weight='bold',pad=40)
    plt.savefig('qa_pair.png')
    
    """
    # -------------------------------- clean data to balance --------------------------------
    roidb_file,split,img_dir,image_file,dict_file="/data/sdc/SGG_data/VG/VG-SGG-with-attri.h5","test","/data/sdc/SGG_data/VG/VG_100K","/data/sdc/SGG_data/VG/image_data.json","/data/sdc/SGG_data/VG/VG-SGG-dicts-with-attri.json"
    ind_to_classes, ind_to_predicates, ind_to_attributes = load_info(dict_file)
    
    split_mask, gt_boxes, gt_classes, gt_attributes, relationships = load_graphs(
        roidb_file, split, -1, num_val_im=5000,
        filter_empty_rels=True,
        filter_non_overlap=False,
    )

    filenames, img_info = load_image_filenames(img_dir, image_file) # length equals to split_mask
    filenames = [filenames[i] for i in np.where(split_mask)[0]]
    img_ids=[img_info[i]['image_id']for i in np.where(split_mask)[0]]
    img_info = [img_info[i] for i in np.where(split_mask)[0]]
    
    
    """
    predict_map_id={idx:name for idx,name in enumerate(ind_to_classes)}
    print(predict_map_id)
    
    summ_cls_num={idx:0 for idx,name in enumerate(ind_to_classes)}
    with open('/data/sdc/SGG_data/VG/common_json/train/object/used_summary_comm_2.json','r') as train_10w_files:
        for line in tqdm.tqdm(train_10w_files):
            json_line=json.loads(line)
            for comm_i in json_line['common_rels']:
                summ_cls_num[comm_i]+=1

    new_summ_cls_num=dict()
    for cls_id,num in summ_cls_num.items():
        if num>500:
            new_summ_cls_num[cls_id]=num
    
    each_cls_num=100000/(len(new_summ_cls_num))
    print(new_summ_cls_num,len(new_summ_cls_num),each_cls_num)

    direct_in,direct_num,filter_in=[],0,[]
    for idx,nums in new_summ_cls_num.items():
        if nums<each_cls_num:
            direct_in.append(idx)
            direct_num+=nums
        else:
            filter_in.append(idx)
    
    final_data_dicts={idx:0 for idx,num in new_summ_cls_num.items()}
    print(f"{final_data_dicts}, direct insert: {direct_in}, filter insert: {filter_in}")
    
    each_cls_num=(100000-direct_num)/(len(filter_in))
    """
    with open('/data/sdc/SGG_data/VG/common_json/train/relation/balanced_comm_2.json','r') as train_balanced_files:
        balances_file_datas=json.load(train_balanced_files)
        
    new_ind_to_predicates=balances_file_datas['ind_to_predicates']
    
    new_ind_to_predicates_values=list(new_ind_to_predicates.values())
    print(new_ind_to_predicates,new_ind_to_predicates_values)
    
    pair_img_lists,un_match_file=[],[]
    # pdb.set_trace()
    with open('/data/sdc/SGG_data/VG/common_json/test/relation/cleaned_summary_comm_2.json','r') as all_test_files:
        for line in tqdm.tqdm(all_test_files):
            line_info=json.loads(line)
            img_1,img_2,comm_info=line_info['img_1'],line_info['img_2'],line_info['common_rels']
            
            if img_1 not in img_ids or img_2 not in img_ids:
                print(f'Error file info: {line_info}, image id not in img_ids')
                continue
            
            new_common_info=[]
            for comm_i in comm_info:
                if ind_to_predicates[comm_i] in new_ind_to_predicates_values:
                    new_common_info.append(new_ind_to_predicates_values.index(ind_to_predicates[comm_i])+1)
            if len(new_common_info)>0:
                pair_img_lists.append(dict(img_1=img_1,img_2=img_2,common_rels=new_common_info)) 
            else:
                un_match_file.append(line_info)
            
    """
    with open('/data/sdc/SGG_data/VG/common_json/test/relation/cleaned_summary_comm_2.json','r') as all_test_files:
        for line in tqdm.tqdm(all_test_files):
            json_line=json.loads(line)
            
            comm_infos=json_line['common_rels']
            
            insert_bool=True
            for comm_i in comm_infos:
                if comm_i in direct_in:
                    insert_bool=True
                    break
                elif comm_i in filter_in:
                    if final_data_dicts[comm_i]>each_cls_num+1500:
                        insert_bool=False
                
            if insert_bool:
                new_comm_info=[]
                for comm_i in comm_infos:
                    if comm_i in final_data_dicts:
                        final_data_dicts[comm_i]=final_data_dicts[comm_i]+1     
                        new_comm_info.append(comm_i)  
                
                if len(new_comm_info)>0: 
                    json_line['common_rels'] =new_comm_info
                    final_json_lists.append(json_line)
    
            
            if len(final_json_lists)==100000:
                break
    """
    
    with open('/data/sdc/SGG_data/VG/common_json/test/relation/cleaned_balanced_comm_2.json','w') as all_cleand_files:
        json.dump(dict(ind_to_predicates=new_ind_to_predicates,pair_img_info=pair_img_lists),all_cleand_files)
        
    print(f'clean test files number: {len(pair_img_lists)}, unmatch files number: {len(un_match_file)}')
    pair_img_lists=random.sample(pair_img_lists,min(10000,len(pair_img_lists)))
    
    print(f'save pair image list num: {len(pair_img_lists)}')
    if os.path.exists('/data/sdc/SGG_data/VG/common_json/test/relation/balanced_comm_2.json'):
        os.remove('/data/sdc/SGG_data/VG/common_json/test/relation/balanced_comm_2.json')
    
    """
    final_data_dict_predicates=final_data_dicts.keys()
    ori_predict_id_map,new_ind_to_predictes=dict(),dict()
    for new_id,ori_id in enumerate(final_data_dict_predicates):
        ori_predict_id_map[ori_id]=new_id+1
        new_ind_to_predictes[new_id+1]=ind_to_classes[ori_id]
    
    balanced_data=dict(ind_to_classes=new_ind_to_predictes)
    print(f'relation map: {ori_predict_id_map}, new ind to predictes: {new_ind_to_predictes}')
    new_ind_predict_json_lists=[]
    for final_json in final_json_lists:
        common_rels=final_json['common_rels']
        new_common_rels=[]
        for comm_id in common_rels:
            new_common_rels.append(ori_predict_id_map[comm_id])
        
        final_json['common_rels']=new_common_rels
        new_ind_predict_json_lists.append(final_json)
    """
    
    balanced_data=dict(ind_to_predicates=new_ind_to_predicates)
    balanced_data['pair_img_info']=pair_img_lists
    with open("/data/sdc/SGG_data/VG/common_json/test/relation/balanced_comm_2.json",'w') as final_files:
        json.dump(balanced_data,final_files)
        final_files.write('\n')                       
    