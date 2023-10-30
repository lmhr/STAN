import os
import torch
from model import get_model, get_loss_and_optimizer
from dataset import get_dataset
from torch.utils.data import DataLoader
from trainer import train,test,infer
from utils import set_random_seed, get_args, configure_logger, print_exp_info

def train_main(args):
    
    log_path, LOGGER = configure_logger(args, verbose=True, eval=False)
    print_exp_info(LOGGER, args)
    model = get_model(LOGGER, args)
    LOGGER.info(model)
    loss, optimizer = get_loss_and_optimizer(model, args)
    train_set = get_dataset('train', args)
    # print(len(train_set))
    test_set = get_dataset('test', args)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
    )
    # num_list = [0, 0]
    # for X, y in train_loader:
    #     for i in y:
    #         for index in i:
    #             num_list[int(index)] += 1
    # print(num_list)
    # return 
    best_pq = 0
    for epoch in range(args.epochs):
        LOGGER.info(f"Epoch {epoch+1}\n-------------------------------")
        train(LOGGER, train_loader, model, loss, optimizer)
        pq_test = test(LOGGER, test_loader, model, loss)
        if pq_test > best_pq:
            torch.save(model.state_dict(), os.path.join(log_path, args.model_best))
            best_pq = pq_test

    torch.save(model.state_dict(), os.path.join(log_path, args.model_final))
    LOGGER.info("Saved PyTorch Model State to model.pth \n")
    LOGGER.info("Final Evalaluation of the Best model")
    LOGGER.info("-------------------------------")
    model.load_state_dict(torch.load(os.path.join(log_path, args.model_best)))
    pq_test = test(LOGGER, test_loader, model, loss, log_path)
    LOGGER.info("Done!")

def test_main(args):
    test_set = get_dataset('test', args)
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )
    log_path, LOGGER = configure_logger(args, verbose=True, eval=True)
    model = get_model(LOGGER, args)
    loss, optimizer = get_loss_and_optimizer(model, args)
    test(LOGGER, test_loader, model, loss, log_path)
    LOGGER.info("Done!\n")

def infer_main(args):
    # 先test
    test_set = get_dataset('test', args)
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )
    log_path, LOGGER = configure_logger(args, verbose=True, eval=True)
    model = get_model(LOGGER, args)
    loss, optimizer = get_loss_and_optimizer(model, args)
    test(LOGGER, test_loader, model, loss, log_path)
    LOGGER.info("Done!\n")
    # 后infer
    infer_set = get_dataset('infer', args)
    infer_loader = DataLoader(
        infer_set,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )
    infer(LOGGER, infer_loader, model, log_path)
    LOGGER.info("Done!\n")


if __name__ == "__main__":
    # 建立在特征已经提取完成的基础上
    args = get_args()
    if args.debugging:
        import debugpy
        debugpy.listen(5556)
        print("====debugging====")
        debugpy.wait_for_client()
        debugpy.breakpoint()

    set_random_seed(args)
    if args.phase == 'train':
        train_main(args)
    elif args.phase == 'test':
        test_main(args)
    elif args.phase == 'infer':
        infer_main(args)
    else:
        print("no exist phase:{}".format(args.phase))