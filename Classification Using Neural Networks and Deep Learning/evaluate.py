def evaluate(net, images, labels):
    acc = 0    
    loss = 0
    batch_size = 1

    pass
    for batch_index in range(0, images.shape[0], batch_size):
        x = images[batch_index]
        y = labels[batch_index]
        # forward pass
        for l in range(net.lay_num):
            output = net.layers[l].forward(x)
            x = output
        loss += cross_entropy(output, y)
        if np.argmax(output) == np.argmax(y):
            acc += 1
    return acc/len(labels), loss/len(labels)