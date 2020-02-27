import torch
import time
from utils.textUtils import itos

# qualitative evaluation
def eval(model, epoch, testloader, device, logger, reverse_dict):
    # Set eval mode
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(testloader):
            # get the inputs and labels
            images, tgt = data['images'].to(device), data['sentence'].to(device)

            # forward
            outputs = model(images, tgt)

            # convert output & tgt to text format
            outputs = outputs.view(-1, outputs.shape[-1])
            outputs = outputs.argmax(1)
            tgt = tgt.view(-1)
            outputs = outputs.detach().data.cpu().numpy()
            tgt = tgt.detach().data.cpu().numpy()
            out_text = " ".join(itos(outputs, reverse_dict))
            tgt_text = " ".join(itos(tgt, reverse_dict))

            output = ('[Eval] Epoch: [{0}][{1}/{2}]\n'
                    'Output {3}\n'
                    'Target {4}\n'
                    .format(
                        epoch, i, len(testloader), 
                        out_text, tgt_text
                        ))

            logger.info(output)
        
