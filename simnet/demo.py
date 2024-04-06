import os

from torch import nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import MultiStepLR

from examples.domain_adaptation.image_classification.cluster_code.module import Learner_DensePairSimilarity


def run(args):
    # argv = '--loss DPS --dataset Omniglot --model_type vgg --model_name VGGS
    # --schedule 30 40 --epochs 50'.split(' ')
    if not os.path.exists('/simnet/outputs'):
        os.mkdir('/simnet/outputs')

    # Dense-Pair Similarity Learning
    LearnerClass = Learner_DensePairSimilarity
    criterion = nn.CrossEntropyLoss()
    args.out_dim = 2  # force it

    # Prepare dataloaders
    loaderFuncs = __import__('dataloaders.' + args.dataset_type)  # 默认是default
    loaderFuncs = loaderFuncs.__dict__[args.dataset_type]
    train_loader, eval_loader = loaderFuncs.__dict__[
        args.dataset](args.batch_size, args.workers)

    # Prepare the model
    model = LearnerClass.create_model(
        args.simnet_type, args.simnet_name, out_dim=2)  # type lenet model_name:LeNet   # 这里创建模型，创建的是一个完整的模型，包含backbone还有分类器

    # Load pre-trained model
    if args.pretrained_model != '':  # Load model weights only
        print('=> Load model weights:', args.pretrained_model)
        model_state = torch.load(
            args.pretrained_model,
            map_location=lambda storage,
            loc: storage)  # Load to CPU as the default!
        model.load_state_dict(model_state, strict=args.strict)
        print('=> Load Done')

    # Load the pre-trained Similarity Prediction Network (SPN, or the G
    # function in paper)
    if args.use_SPN:
        # To load a custom SPN, you can modify here.
        SPN = Learner_DensePairSimilarity.create_model(
            args.SPN_model_type, args.SPN_model_name, 2)  # 默认name为VGGS 类型为vgg
        print('=> Load SPN model weights:', args.SPN_pretrained_model)
        SPN_state = torch.load(
            args.SPN_pretrained_model,
            map_location=lambda storage,
            loc: storage)  # Load to CPU as the default!
        SPN.load_state_dict(SPN_state)
        print('=> Load SPN Done')
        print('SPN model:', SPN)
        # SPN.eval()  # Tips: Stay in train mode, so the BN layers of SPN adapt
        # to the new domain
        args.SPN = SPN  # It will be used in prepare_task_target()

    # GPU
    if args.use_gpu:
        torch.cuda.set_device(args.gpuid[0])
        cudnn.benchmark = True  # make it train faster
        model = model.cuda()
        criterion = criterion.cuda()
        if args.SPN is not None:
            args.SPN = args.SPN.cuda()

    # Multi-GPU
    if len(args.gpuid) > 1:
        model = torch.nn.DataParallel(
            model,
            device_ids=args.gpuid,
            output_device=args.gpuid[0])

    print('Main model:', model)
    print('Criterion:', criterion)

    # Evaluation Only
    if args.skip_train:
        cudnn.benchmark = False  # save warm-up time
        eval_loader = eval_loader if eval_loader is not None else train_loader
        KPI = evaluate(eval_loader, model, args)
        return KPI

    # Prepare the learner
    optim_args = {'lr': args.lr}
    if args.optimizer == 'SGD':
        optim_args['momentum'] = 0.9
    optimizer = torch.optim.__dict__[args.optimizer](
        model.parameters(), **optim_args)
    scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)
    learner = LearnerClass(model, criterion, optimizer, scheduler)

    # Start optimization
    if args.resume:
        args.start_epoch = learner.resume(
            args.resume) + 1  # Start from next epoch
    KPI = 0

    for epoch in range(args.start_epoch, args.epochs):
        train(epoch, train_loader, learner, args)
        if eval_loader is not None and (
            (not args.skip_eval) or (
                epoch == args.epochs - 1)):
            KPI = evaluate(eval_loader, model, args)
        # Save checkpoint at each LR steps and the end of optimization
        if epoch + 1 in args.schedule + [args.epochs]:
            learner.snapshot("outputs/%s_%s_%s" %
                             (args.dataset, args.model_name, args.saveid), KPI)
    return KPI