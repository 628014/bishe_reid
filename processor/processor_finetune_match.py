import logging
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter


def do_train_match(start_epoch, args, model, train_loader, evaluator0, evaluator1, evaluator2, optimizer,
             scheduler, checkpointer, trainset):

    log_period = args.log_period
    eval_period = args.eval_period
    num_epoch = args.num_epoch
    
    logger = logging.getLogger("IRRA.train")
    logger.info('Start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "match_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)
    scaler = torch.cuda.amp.GradScaler()

    best_top1_0 = 0.0 # CUHK-PEDES
    best_top1_1 = 0.0 # ICFG-PEDES
    best_top1_2 = 0.0 # RSTPReid

    # 验证前先打印一次参数状态，确保没全变0
    if get_rank() == 0:
        logger.info("Validation before training...")
        # 可选：先跑一次验证确保代码不崩
        # try:
        #     _eval_model(model, evaluator0)
        # except Exception as e:
        #     logger.warning(f"Validation check failed: {e}")

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                ret = model(batch)
                total_loss = sum([v for k, v in ret.items() if "loss" in k])

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            
            if 'sdm_loss' in ret: meters['sdm_loss'].update(ret['sdm_loss'].item(), batch_size)
            if 'itc_loss' in ret: meters['itc_loss'].update(ret['itc_loss'].item(), batch_size)
            if 'id_loss' in ret: meters['id_loss'].update(ret['id_loss'].item(), batch_size)
            if 'mlm_loss' in ret: meters['mlm_loss'].update(ret['mlm_loss'].item(), batch_size)
            if 'match_loss' in ret: meters['match_loss'].update(ret['match_loss'].item(), batch_size)
            
            if 'img_acc' in ret: meters['img_acc'].update(ret['img_acc'].item(), batch_size)
            if 'txt_acc' in ret: meters['txt_acc'].update(ret['txt_acc'].item(), batch_size)
            if 'mlm_acc' in ret: meters['mlm_acc'].update(ret['mlm_acc'].item(), 1)

            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iter[{n_iter + 1}/{len(train_loader)}]"
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        if get_rank() == 0:
            tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
            for k, v in meters.items():
                if v.avg > 0:
                    tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (len(train_loader) + 1e-5)
            logger.info(f"Epoch {epoch} done. Time: {time_per_batch:.3f}s/batch")

        # --- 验证逻辑 ---
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                
                # 修复：安全获取 model (解包 DDP)
                eval_model_ref = model.module if hasattr(model, 'module') else model
                
                # 评估
                top1_0 = evaluator0.eval(eval_model_ref.eval())
                logger.info(f"CUHK-PEDES R1: {top1_0}")
                
                top1_1 = evaluator1.eval(eval_model_ref.eval())
                logger.info(f"ICFG-PEDES R1: {top1_1}")
                
                top1_2 = evaluator2.eval(eval_model_ref.eval())
                logger.info(f"RSTPReid R1: {top1_2}")

                arguments = {"epoch": epoch}
                
                if top1_0 > best_top1_0:
                    best_top1_0 = top1_0
                    checkpointer.save("best_cuhk", **arguments)
                
                if top1_1 > best_top1_1:
                    best_top1_1 = top1_1
                    checkpointer.save("best_icfg", **arguments)

                if top1_2 > best_top1_2:
                    best_top1_2 = top1_2
                    checkpointer.save("best_rstp", **arguments)
                
                checkpointer.save("last", **arguments)
                
                model.train()
                torch.cuda.empty_cache()

    if get_rank() == 0:
        logger.info(f"Training finished. Best R1: CUHK:{best_top1_0}, ICFG:{best_top1_1}, RSTP:{best_top1_2}")