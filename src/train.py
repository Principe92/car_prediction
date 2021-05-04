
from src.utils import loss_fn

def train(model, optim, batch_size=128, show_every=250, num_epochs=10, train_loader=None, val_loader=None, device=None):

    """

    """

    iter_count = 0
    losses = []

    for epoch in range(num_epochs):
        print('EPOCH: ', (epoch+1))

        for x, _ in train_loader:

            loss = None
            labels = None # TODO

            optim.zero_grad()

            output = model(x) # predict

            loss = loss_fn(output, labels)
            loss.backward()

            optim.step()


            if (iter_count % show_every == 0):
                val_loss, val_acc = validate(val_loader, batches=4) # run validation

                print('Iter: {}, train loss: {:.4}, val loss: {:.4}, val acc: {:.4}'.format(iter_count, loss.item(), val_loss, val_acc))

                losses.append(loss.item())

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
        x, target = batch # TODO

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

