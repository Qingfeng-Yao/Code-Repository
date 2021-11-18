import os
import shutil

def compute_metrics(gt_dirs, gt_path, bag, query_items):
    ave_p = .0
    ave_r = .0

    cnt = 0
    for gt in gt_dirs:
        txt_id = gt.split("-")[1].split(".")[0]
        with open(gt_path+gt,'r',encoding='utf-8') as f_gt:
            ids = []
            for l in f_gt.readlines():
                if l.strip()[-3:] == "txt":
                    ids.append(l.strip().split(".")[0])
            assert len(ids) == int(gt[0])

            for i in ids:
                for kid, vlist in bag.items():
                    v_str_list = [query_items[vid][0] for vid in vlist]
                    if i in v_str_list:
                        assert len(ids) == len(set(ids)) and len(v_str_list) == len(set(v_str_list))
                        u = [j for j in ids if j in v_str_list]

                        if len(ids) > 1:
                            r = (len(u) - 1) / (len(ids) - 1)
                        else:
                            r = 1
                        ave_r += r
                    
                        if len(v_str_list) > 1:
                            p = (len(u) - 1) / (len(v_str_list) - 1)
                        else:
                            p = 1
                        ave_p += p
                        cnt += 1

    ave_p /= cnt
    ave_r /= cnt
    f1=(2*ave_p*ave_r/(ave_p+ave_r))
    return ave_p, ave_r, f1

def cmp_gt(bag, gt_dirs, out_path, gt_path, query_items):
    repeat_bag_ids = [k for k, v in bag.items() if len(v)>1]
    all_bag_ids = [k for k, _ in bag.items()]
    for gt in gt_dirs:
        txt_id = gt.split("-")[1].split(".")[0]
        if gt[0] == "1":
            if txt_id in bag.keys() and len(bag[txt_id])==1:
                new_dir = out_path+txt_id
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                    file_path = gt_path+gt
                    new_file_path = new_dir+"/gt-"+gt
                    shutil.copy(file_path,new_file_path)
                    new_file_path = new_dir+"/pred-1"+txt_id+".txt"
                    with open(new_file_path,'w') as f:
                        f.write(query_items[bag[txt_id][0]][1]["text"])
            else:
                for rid in repeat_bag_ids:
                    ids_sim = [query_items[r][0] for r in bag[rid]]
                    if txt_id in ids_sim:
                        new_dir = out_path+"*"+txt_id
                        if not os.path.exists(new_dir):
                            os.makedirs(new_dir)
                            file_path = gt_path+gt
                            new_file_path = new_dir+"/gt-"+gt
                            shutil.copy(file_path,new_file_path) 
                            new_file_path = new_dir+"/pred-"+str(len(ids_sim))+"-"+txt_id+".txt"
                            content = "\n\n".join([query_items[r][0]+"\n"+query_items[r][1]["text"] for r in bag[rid]])
                            with open(new_file_path,'w') as f:
                                f.write(content)

        else:
            with open(gt_path+gt,'r',encoding='utf-8') as f_gt:
                ids = []
                for l in f_gt.readlines():
                    if l.strip()[-3:] == "txt":
                        ids.append(l.strip().split(".")[0])
                assert len(ids) == int(gt[0])

                equal = 0
                for rid in repeat_bag_ids:
                    ids_sim = [query_items[r][0] for r in bag[rid]]

                    intersec_1 = [i for i in ids if i not in ids_sim]
                    intersec_2 = [i for i in ids_sim if i not in ids]

                    if len(intersec_1)==0 and len(intersec_2)==0:
                        equal += 1
                        new_dir = out_path+txt_id
                        if not os.path.exists(new_dir):
                            os.makedirs(new_dir)
                            file_path = gt_path+gt
                            new_file_path = new_dir+"/gt-"+gt
                            shutil.copy(file_path,new_file_path) 
                            new_file_path = new_dir+"/pred-"+str(len(ids_sim))+"-"+txt_id+".txt"
                            content = "\n\n".join([query_items[r][0]+"\n"+query_items[r][1]["text"]+"\n\n" for r in bag[rid]])
                            with open(new_file_path,'w') as f:
                                f.write(content)

                if equal == 0:
                    new_dir = out_path+"*"+txt_id
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                        file_path = gt_path+gt
                        new_file_path = new_dir+"/gt-"+gt
                        shutil.copy(file_path,new_file_path) 
                    for aid in all_bag_ids:
                        ids_sim = [query_items[a][0] for a in bag[aid]]
                        for i in ids_sim:
                            if i in ids:
                                new_file_path = new_dir+"/pred-"+str(len(ids_sim))+"-"+aid+".txt"
                                content = "\n\n".join([query_items[a][0]+"\n"+query_items[a][1]["text"]+"\n\n" for a in bag[aid]])
                                with open(new_file_path,'w') as f:
                                    f.write(content)