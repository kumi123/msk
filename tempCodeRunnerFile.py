import mxnet as mx
mx.random.seed(1)
x = mx.nd.empty(shape = (1, 256), ctx = mx.cpu(0)) # not overwriting the memory and setting context to gpu
print(mx.nd.argmax(x, axis = 1)) # using argmax function along axis 1