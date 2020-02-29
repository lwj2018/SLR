import torch
import shutil

def save_checkpoint(state, is_best, model_path, name):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (model_path, name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (model_path, name),
            '%s/%s_best.pth.tar' % (model_path, name))

def resume_model(model, checkpoint):
    params_dict = torch.load(checkpoint)
    state_dict = params_dict['state_dict']
    state_dict = {".".join(k.split(".")[1:]) : v for k,v in state_dict.items()}
    model.load_state_dict(state_dict)

    epoch = params_dict['epoch']
    best_wer = params_dict['best_wer']
    print("Load model from {}: \n"
    "Epoch: {}\n"
    "Best_wer: {:.3f}%".format(checkpoint,epoch,best_wer*100))
    return params_dict['epoch'], params_dict['best_wer']