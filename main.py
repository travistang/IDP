from database import Database
from solver import Solver
import matplotlib.pyplot as plt
if __name__ == '__main__':
    config = {
        'num_units': 128,
        'num_neighbours': 4,
        'input_length': 8,
        'output_length': 4,
    }

    optimizer_config = {
        'learning_rate': 1e-3,
    }

    database = Database()
    solver = Solver(**config)

    data_generator = database.get_social_lstm_train_data(num_data = 8)

    num_epoch = 10
    num_batches = 20
    def batch_generator():
        for i in range(num_batches):
            yield next(data_generator)

    def epoch_samples_generator():
        for epoch in range(num_epoch):
            yield batch_generator()

    training_history = solver.train(epoch_samples_generator = epoch_samples_generator())

    plt.plot(training_history)
    plt.show()
