import os
import torch
import shutil

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_checkpoint(state, is_best, model_path, name):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (model_path, name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (model_path, name),
            '%s/%s_best.pth.tar' % (model_path, name))

def resume_model(model, checkpoint):
    params_dict = torch.load(checkpoint)
    state_dict = params_dict['state_dict']
    model.load_state_dict(state_dict)

    epoch = params_dict['epoch']
    best = params_dict['best']
    print("Load model from {}: \n"
    "Epoch: {}\n"
    "Best: {:.3f}".format(checkpoint,epoch,best))
    return params_dict['epoch'], params_dict['best']

def resume_multigpu_model(model, checkpoint):
    params_dict = torch.load(checkpoint)
    state_dict = params_dict['state_dict']
    state_dict = {".".join(k.split(".")[1:]) : v for k,v in state_dict.items()}
    model.load_state_dict(state_dict)

    epoch = params_dict['epoch']
    best = params_dict['best']
    print("Load model from {}: \n"
    "Epoch: {}\n"
    "Best: {:.3f}%".format(checkpoint,epoch,best))
    return params_dict['epoch'], params_dict['best']

def resume_skeleton_model(model, checkpoint):
    params_dict = torch.load(checkpoint)
    state_dict = params_dict['state_dict']
    state_dict = {".".join(k.split(".")[2:]) : v for k,v in state_dict.items() if not 'att' in k}
    model.load_state_dict(state_dict)

    epoch = params_dict['epoch']
    best = params_dict['best_prec1']
    print("Load model from {}: \n"
    "Epoch: {}\n"
    "Best: {:.3f}%".format(checkpoint,epoch,best))
    return params_dict['epoch'], params_dict['best_prec1']

def resume_hcn_module(model, checkpoint):
    model_dict = model.state_dict()
    params_dict = torch.load(checkpoint)
    state_dict = params_dict['state_dict']
    state_dict = {"featureExtractor."+k : v for k,v in state_dict.items()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    epoch = params_dict['epoch']
    best = params_dict['best']
    print("Load HCN module from {}: \n"
    "Epoch: {}\n"
    "Best: {:.3f}%".format(checkpoint,epoch,best))
    return model

def resume_vae_module(model, checkpoint):
    model_dict = model.state_dict()
    params_dict = torch.load(checkpoint)
    state_dict = params_dict['state_dict']
    state_dict = {"vae."+k : v for k,v in state_dict.items()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    epoch = params_dict['epoch']
    best = params_dict['best']
    print("Load VAE module from {}: \n"
    "Epoch: {}\n"
    "Best: {:.3f}%".format(checkpoint,epoch,best))
    return model

def resume_hcn_lstm(model, checkpoint):
    model_dict = model.state_dict()
    params_dict = torch.load(checkpoint)
    state_dict = params_dict['state_dict']
    # Resume model except the final fully-connected layer
    state_dict = {k : v for k,v in state_dict.items() if not 'out' in k}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    epoch = params_dict['epoch']
    best = params_dict['best']
    print("Load HCN+LSTM model from {}: \n"
    "Epoch: {}\n"
    "Best: {:.3f}%".format(checkpoint,epoch,best))
    return model

def resume_main_part(model, checkpoint):
    model_dict = model.state_dict()
    params_dict = torch.load(checkpoint)
    state_dict = params_dict['state_dict']
    # Resume model don't have the bn layer
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    epoch = params_dict['epoch']
    best = params_dict['best']
    print("Load main part from {}: \n"
    "Epoch: {}\n"
    "Best: {:.3f}%".format(checkpoint,epoch,best))
    return model