
import torch
from src.utils import loss_fn
from torch.autograd import Variable


def train(
        model: torch.nn.Module,
        optim, batch_size=128, show_every=250, num_epochs=10, train_loader=None, val_loader=None, device=None):
    """

    """

    iter_count = 0
    losses = []

    model.eval()

    for epoch in range(num_epochs):
        print('EPOCH: ', (epoch+1))

        for x1, x2, label in train_loader:

            # print(x.shape)
            # x1, x2, label = x

            v1 = get_vector(model, x1[0], device)
            v2 = get_vector(model, x2[0], device)

            # loss = None
            # labels = None # TODO

            # optim.zero_grad()

            # output = model(x) # predict

            # loss = loss_fn(output, labels)
            # loss.backward()

            # optim.step()

            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_sim = cos(v1.unsqueeze(0), v2.unsqueeze(0))

            predicted = 1 if cos_sim > 0.5 else 0
            print(f'cos: {cos_sim} | actual: {label} | predicted: {predicted}')


            if (iter_count % show_every == 0):
                # val_loss, val_acc = validate(model, val_loader, batches=4) # run validation

                # print('Iter: {}, train loss: {:.4}, val loss: {:.4}, val acc: {:.4}'.format(iter_count, loss.item(), val_loss, val_acc))
                print('Iter: {}'.format(iter_count))

                # losses.append(loss.item())

            iter_count += 1

    return losses


def validate(model, loader=None, batches=None):

    model.eval()
    loss = None
    acc = None
    correct = 0
    examples = 0

    for i, batch in enumerate(loader):

        output = model(x)
        x, target = batch  # TODO

        loss += loss_fn(output, target).cpu().item()

        # predict the argmax of the log-probabilities
        predicted = output.data.max(1, keepdim=True)[1]
        correct += predicted.eq(target.data.view_as(predicted)).cpu().sum()
        examples += predicted.size(0)

        if batches and (i >= batches):
            break

    loss /= examples
    acc = 100. * correct / examples

    return loss, acc


def get_vector(model, img, device):

    t_img = Variable(img.unsqueeze(0)).to(device)

    embedding = torch.zeros(512)

    def copy_data(m, i, o):
        embedding.copy_(o.data)

    layer = model.model._modules.get('avgpool')
    h = layer.register_forward_hook(copy_data)

    model(t_img)

    h.remove()

    return embedding
