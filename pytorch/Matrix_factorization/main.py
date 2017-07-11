import numpy as np
from scipy.sparse import rand as sprand
import torch
from torch.autograd import Variable

# Make up some random explicit feedback ratings
# and convert to a numpy array
n_users = 200
n_items = 200
ratings = sprand(n_users, n_items,
                 density=0.05, format='csr')
ratings.data = (np.random.randint(1, 5,
                                  size=ratings.nnz)
                          .astype(np.float64))
ratings = ratings.toarray()


class BiasedMatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super(BiasedMatrixFactorization,self).__init__()
        self.user_factors = torch.nn.Embedding(n_users,
                                               n_factors,
                                               sparse=True)
        self.item_factors = torch.nn.Embedding(n_items,
                                               n_factors,
                                               sparse=True)
        self.user_biases = torch.nn.Embedding(n_users,
                                              1,
                                              sparse=True)
        self.item_biases = torch.nn.Embedding(n_items,
                                              1,
                                              sparse=True)

    def forward(self, user, item):
        pred = self.user_biases(user) + self.item_biases(item)
        pred += (self.user_factors(user) * self.item_factors(item)).sum(1)
        return pred

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super(MatrixFactorization,self).__init__()
        self.user_factors = torch.nn.Embedding(n_users,
                                               n_factors,
                                               sparse=True)
        self.item_factors = torch.nn.Embedding(n_items,
                                               n_factors,
                                               sparse=True)

    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)

model = MatrixFactorization(n_users, n_items, n_factors=5)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5e-7) # learning rate

# Sort our data
rows, cols = ratings.nonzero()
p = np.random.permutation(len(rows))
rows, cols = rows[p], cols[p]

for i in range(10):
    loss_sum = 0
    for row, col in zip(*(rows, cols)):
        # Turn data into variables
        rating = Variable(torch.FloatTensor([ratings[row, col]]))
        row = Variable(torch.LongTensor([np.long(row)]))
        col = Variable(torch.LongTensor([np.long(col)]))
        # Predict and calculate loss
        prediction = model.forward(row, col)
        loss = loss_func(prediction, rating)
        loss_sum += loss.data[0]
        # Backpropagate
        loss.backward()
        # Update the parameters
        optimizer.step()
    print 'total loss',loss_sum

