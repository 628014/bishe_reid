import torch
import numpy as np
import os
from prettytable import PrettyTable
from torch.nn import functional as F
import logging

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    """
    计算标准的 ReID 指标
    similarity: (num_query, num_gallery) 的相似度矩阵
    q_pids: query 身份标签
    g_pids: gallery 身份标签
    """
    if get_mAP:
        # 排序：相似度从大到小
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # 如果不需要mAP，只取前k个以加速
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk

    # 转到 CPU 进行指标计算
    indices = indices.cpu()
    q_pids = q_pids.cpu()
    g_pids = g_pids.cpu()

    # 获取预测的标签 [num_query, num_gallery]
    pred_labels = g_pids[indices]
    
    # 匹配矩阵：预测的ID是否等于Query的ID
    matches = pred_labels.eq(q_pids.view(-1, 1))  # [num_query, num_gallery]

    # --- 计算 CMC (Rank-K) ---
    # 计算累计匹配 (Cumulative Match Characteristic)
    # matches[:, :max_rank] 取前 max_rank 列
    # cumsum(1) 计算行维度的累加和
    # > 1 的地方设为 1（只要匹配到一个就算匹配成功）
    all_cmc = matches[:, :max_rank].cumsum(1) 
    all_cmc[all_cmc > 1] = 1
    
    # 对所有 query 取平均，并转换为百分比
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc 现在是一个向量，all_cmc[0] 是 Rank-1, all_cmc[4] 是 Rank-5

    if not get_mAP:
        return all_cmc, indices

    # --- 计算 mAP and mINP ---
    # num_rel: 每个 query 在 gallery 中真实的匹配总数
    num_rel = matches.sum(1)  # [num_query]
    
    # 这里的 tmp_cmc 不截断，计算完整的
    tmp_cmc = matches.cumsum(1)  # [num_query, num_gallery]

    # --- mINP 计算 ---
    # INP = Match_Rank / (Position of Hardest Match)
    # 找到所有匹配的位置
    inp_list = []
    for i in range(matches.size(0)):
        # 获取当前 query 所有匹配到的 gallery 索引
        valid_indices = matches[i].nonzero(as_tuple=False).squeeze()
        if valid_indices.numel() == 0:
            # 如果没有匹配到（理论上很少见），INP为0
            inp_list.append(0.0)
        else:
            # nonzero返回的是tensor，取最后一个元素即为 hardest match 的 rank (0-indexed)
            # 索引 + 1 才是 Rank
            if valid_indices.dim() == 0: # 只有一个匹配
                max_pos = valid_indices + 1.0
                correct_num = 1.0
            else:
                max_pos = valid_indices[-1] + 1.0 
                correct_num = tmp_cmc[i][valid_indices[-1]].float() # 此时累积了多少个正确匹配
            
            # INP = 正确匹配数 / 最难匹配的位置
            inp_list.append(correct_num / max_pos)
            
    mINP = torch.tensor(inp_list).mean() * 100

    # --- mAP 计算 ---
    # Precision at K = (TP @ K) / K
    ranks = torch.arange(1, matches.size(1) + 1).float() # [1, 2, ..., N]
    
    # tmp_cmc[:, i] 是第 i 列，表示截止到 Rank i 匹配到了多少个
    # AP 计算核心公式
    tmp_cmc = tmp_cmc.float() / ranks.unsqueeze(0) # Precision @ K
    
    # 只保留那些真正匹配到的位置的 Precision
    tmp_cmc = tmp_cmc * matches.float()
    
    # 对每个 query 求和得到 AP (sum precision / num_rel)
    AP = tmp_cmc.sum(1) / torch.clamp(num_rel.float(), min=1e-9) 
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("IRRA.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        # 获取模型所在的设备
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        
        # --- Extract Text Features (Query) ---
        # print("Extracting text features...")
        for batch in self.txt_loader:
            pid, caption = batch
            # pid handling
            if isinstance(pid, torch.Tensor):
                qids.append(pid.view(-1))
            else:
                qids.append(torch.tensor(pid))
            
            # caption handling (list of strings for Qwen)
            with torch.no_grad():
                # encode_text 应该已经在 model 内部处理了 device 转移
                # 但为了安全，如果 model 需要 inputs 在 device 上，model 内部会处理
                # 这里主要确保输出在 CPU 上以节省显存
                text_feat = model.encode_text(caption)
            
            qfeats.append(text_feat.cpu())

        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # --- Extract Image Features (Gallery) ---
        # print("Extracting image features...")
        for batch in self.img_loader:
            pid, img = batch
            if isinstance(pid, torch.Tensor):
                gids.append(pid.view(-1))
            else:
                gids.append(torch.tensor(pid))
            
            with torch.no_grad():
                img_feat = model.encode_image(img)
            
            gfeats.append(img_feat.cpu())

        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids
    
    def eval(self, model, i2t_metric=False):
        # 1. 计算特征
        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        # 2. 归一化特征 (Cosine Similarity 需要)
        qfeats = F.normalize(qfeats, p=2, dim=1).cuda() # 放回 GPU 计算矩阵乘法
        gfeats = F.normalize(gfeats, p=2, dim=1).cuda()

        # 3. 计算相似度矩阵
        # Query (Text) x Gallery (Image)
        similarity = qfeats @ gfeats.t()

        # 4. 计算指标 (T2I: Text to Image)
        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(
            similarity=similarity, 
            q_pids=qids, 
            g_pids=gids, 
            max_rank=10, 
            get_mAP=True
        )
        
        # 转换为 numpy 方便打印
        if isinstance(t2i_cmc, torch.Tensor):
            t2i_cmc = t2i_cmc.numpy()
        t2i_mAP = t2i_mAP.item() if isinstance(t2i_mAP, torch.Tensor) else t2i_mAP
        t2i_mINP = t2i_mINP.item() if isinstance(t2i_mINP, torch.Tensor) else t2i_mINP

        # 5. 构建表格
        table = PrettyTable(["Task", "R1", "R5", "R10", "mAP", "mINP"])
        table.float_format = '.2' # 保留两位小数
        
        # 添加 T2I 行
        table.add_row(['T2I', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        # 6. (可选) 计算 I2T: Image to Text
        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(
                similarity=similarity.t(), 
                q_pids=gids, 
                g_pids=qids, 
                max_rank=10, 
                get_mAP=True
            )
            if isinstance(i2t_cmc, torch.Tensor):
                i2t_cmc = i2t_cmc.numpy()
            i2t_mAP = i2t_mAP.item() if isinstance(i2t_mAP, torch.Tensor) else i2t_mAP
            i2t_mINP = i2t_mINP.item() if isinstance(i2t_mINP, torch.Tensor) else i2t_mINP
            
            table.add_row(['I2T', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])

        # 打印并记录日志
        self.logger.info('\n' + str(table))
        
        # 返回结果字典
        results = {
            "R1": t2i_cmc[0],
            "R5": t2i_cmc[4],
            "R10": t2i_cmc[9],
            "mAP": t2i_mAP,
            "mINP": t2i_mINP
        }
        return results