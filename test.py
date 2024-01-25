from micrograd import MLP , draw_graph

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1], nonlin='tanh')

# gradient descent
for k in range(1000):
    # forward pass
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # backward pass
    model.zero_grad()
    loss.backward()

    # update parameters
    lr = 1 - 1.0**(k/100)
    for p in model.parameters():
        p.data += -lr * p.grad

    if k % 100 == 0:
        print(f"step {k} loss {loss.data}")

# code to vizualize the gradients.
# dot = draw_graph(loss)
# dot.view()
print(model.layers[0].neurons[0].w[0].grad)
print(model.layers[0].neurons[0].nonlin)