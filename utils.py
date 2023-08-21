import numpy as np
from data_utils.collations import SparseSegmentsCollation, SparseSegmentWithTrackCollation
from data_utils.datasets.SemanticKITTIDataLoader import SemanticKITTIDataLoader
from models.minkunet import *
from models.byol_pix import *
from models.blocks import ProjectionHead, SegmentationClassifierHead, PredictionHead
from data_utils.data_map import content, content_indoor

sparse_models = {
    'MinkUNet': MinkUNet,
}

data_loaders = {
    'SemanticKITTI': SemanticKITTIDataLoader,
}

data_class = {
    'SemanticKITTI': 20,
}

def set_deterministic():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

def list_parameters(models):
    optim_params = []
    for model in models:
        optim_params += list(models[model].parameters())

    return optim_params

def get_model(args, dtype, pre_training=False):
    return sparse_models[args.sparse_model](
        in_channels=4 if args.use_intensity else 3,
        out_channels=latent_features[args.sparse_model],
    ).type(dtype)

def get_projection_head(args, dtype):
    return ProjectionHead(in_channels=latent_features[args.sparse_model], out_channels=args.feature_size).type(dtype)

def get_moco_model(args, dtype):
    return MoCo(sparse_models[args.sparse_model], ProjectionHead, dtype, args)

def get_byol_model(args, dtype):
    return Byol(sparse_models[args.sparse_model], ProjectionHead, PredictionHead, dtype, args)  #

def get_byol_pix_model(args, dtype):
    return Byol_Pix(sparse_models[args.sparse_model], ProjectionHead, ProjectionHead, PredictionHead, dtype, args)  


def get_classifier_head(args, dtype):
    return SegmentationClassifierHead(
            in_channels=latent_features[args.sparse_model], out_channels=data_class[args.dataset_name]
        ).type(dtype)

def get_optimizer(optim_params, args):
    optimizer = torch.optim.SGD(optim_params, lr=args.lr, momentum=0.9, weight_decay=args.decay_lr)
    return optimizer

def get_class_weights(dataset):
    weights = list(content.values()) if dataset == 'SemanticKITTI' else list(content_indoor.values())

    weights = torch.from_numpy(np.asarray(weights)).float()
    if torch.cuda.is_available():
        weights = weights.cuda()

    return weights

def write_summary(writer, summary_id, report, epoch):
    writer.add_scalar(summary_id, report, epoch)

def get_dataset(args,):
    data_train = data_loaders[args.dataset_name](split='train',intensity_channel=args.use_intensity,resolution=args.sparse_resolution,args=args)
    return data_train

def get_data_loader(data_train, args, pin_memory=True):
    if args.stage==0:
        collate_fn = SparseSegmentsCollation(args.sparse_resolution, args.num_points)
    else:
        collate_fn = SparseSegmentWithTrackCollation(args.sparse_resolution, args.num_points)

    train_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )
    return train_loader
