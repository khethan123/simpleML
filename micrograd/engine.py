import math

class Value:
    ''' stores a single scalar value and its gradient '''

    def __init__(self, data, _children=(), _op='', label=''):
        ''' Initializes a Value object with a scalar value and its gradient.

        Args:
            data (float): The scalar value.
            _children (tuple, optional): A tuple of child Value objects. Defaults to ().
            _op (str, optional): The operation that produced this node. Defaults to ''.
            label (str, optional): A label for this node. Defaults to ''.
            grad (float): Internal variables used for autograd graph construction.
            _op (str): The operation that produced this node, for graphviz / debugging / etc.
        '''
        self.data = data
        self.grad = 0.0
        self.label = label
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        ''' Returns a new Value object representing the sum of this Value and 
            another Value or scalar.
        '''
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        '''Returns a new Value object representing the product of this Value and
           another Value or scalar.
        '''
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        ''' Returns a new Value object representing this Value raised to the
            power of another value.
        '''
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        ''' Returns a new Value object representing the ReLU activation of this Value.
        '''
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        ''' Returns a new Value object representing the hyperbolic tangent 
           activation of this Value.
        '''
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        ''' Returns a new Value object representing the exponential of this Value.
        '''
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp') # it takes only a single value

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        ''' Returns a new Value object representing the natural logarithm of this Value.
        '''
        x = self.data
        out = Value(math.log(x), (self, ), 'log')  # it takes only a single value

        def _backward():
            self.grad += out.grad * x**-1
        out._backward = _backward

        return out

    def backward(self):
        ''' Performs backpropagation to calculate the gradient of all Value objects. '''
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __radd__(self, other): # other + self
        return self + other

    def __rsub__(self, other): # other - self
        return self + (-other)

    def __rmul__(self, other): # other * self
        return self * other

    def __rtruediv__(self, other): # other / self
        return self * other**-1

    def __repr__(self):
        return f"Value(data={self.data})"