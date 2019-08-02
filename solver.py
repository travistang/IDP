import tensorflow as tf
import numpy as np

class Solver:
    def __init__(self, *, num_units, num_neighbours, input_length, output_length, predict_gaussian = False):
        self.num_neighbours = num_neighbours
        self.input_length = input_length
        self.output_length = output_length
        self.num_units = num_units
        self.predict_gaussian = predict_gaussian

        self.social_lstm = Solver._create_social_lstm(num_units, num_neighbours, input_length, output_length, predict_gaussian)

        # input and output for this particular config
        self.input_placeholder = tf.placeholder(
            dtype = tf.float64,
            shape = (None, self.num_neighbours + 1, self.input_length, 2)
        )
        # for feeding the groundtruth
        self.output_placeholder = tf.placeholder(
            dtype = tf.float64,
            shape = (None, self.num_neighbours + 1, self.output_length, 2)
        )

        self.predictions, self.cell, self.W, self.b = self.social_lstm(self.input_placeholder)

    def train(self, *, epoch_samples_generator, optimizer_config = {}):

        '''
            Check the shape of the sample batch...
            ********
                expected epoch_samples_generator to have N elements, N = number of epochs,
            ********
                each epoch_sample is a generator that generates all batches
            ********
            expected sample dimension: (batch, neighbours + 1, input + output length, 2)

            Return the training history [epoch ]
        '''
        # sample_batch = next(sample_generator)
        # assert sample_batch.shape[1:] == (self.num_neighbours + 1, self.input_length + self.output_length, 2)

        # construct loss graph
        target_track_groundtruth = self.output_placeholder[:, 0]
        target_track_predicted = self.predictions[: , 0]
        # print('output placeholder', self.output_placeholder, 'predictions', self.predictions)

        # compute loss
        loss = tf.losses.huber_loss(target_track_groundtruth, target_track_predicted)

        # the optimizer
        optimizer = tf.train.AdamOptimizer(**optimizer_config).minimize(loss)

        with tf.Session() as sess:
            # initialize all variables
            sess.run(tf.global_variables_initializer())

            training_history = []
            for epoch, batch_generator in enumerate(epoch_samples_generator): # num epoch

                epoch_losses = []
                # print('--------- epoch {} --------'.format(epoch + 1))
                for batch in batch_generator:
                    input_batch = batch[:, :, :self.input_length, :]
                    output_batch = batch[:, :, self.input_length:, :]
                    batch_loss, _ = sess.run(
                        [loss, optimizer],
                        feed_dict = {
                            self.input_placeholder: input_batch,
                            self.output_placeholder: output_batch
                        }
                    )
                    # print('loss: ' , batch_loss)
                    epoch_losses.append(batch_loss)
                epoch_loss = np.mean(epoch_losses)
                print('--------- epoch {}: loss: {} --------'.format(epoch + 1, epoch_loss))
                training_history.append(epoch_loss)
            # TODO: return the final weights:
            return training_history

    @staticmethod
    def nll_loss(predictions, expected_outputs):
        # compute the likelihood
        def gaussian(ux, uy, sx, sy, sxy, px, py):
            '''
                Given a 2D normal defined by (ux, uy, sx, sy, sxy), and a sample at (px,py), return the prob.
            '''
            const = (2 * np.pi) ** (-0.5)
            cov_mat = tf.Variable([[sx, sxy], [sxy, sy]]) # (2,2)
            diff = tf.Variable([[px - ux], [py - uy]]) # (2, 1)
            exp = -0.5 * tf.exp(tf.transpose(diff) * tf.linalg.inv(cov_mat) * diff)
            return const * tf.linalg.det(cov_mat) ** (-1) * tf.linalg.exp(exp)
        # the loss is the NLL of gaussian along each batch
        losses = 0
        for batch_id in tf.shape(predictions)[0]:
            losses -= tf.log(gaussian(*expected_outputs[batch_id], *predictions[batch_id])) # -= because negative log

        return losses

    @staticmethod
    def _create_social_lstm(num_units, num_neighbours, input_length, pred_length, predict_gaussian = False):
        '''
            Shape is (batch, seq_length ,num_neighbours, coord)

            The input should be of size (
            num_data, <- the batch
            neighbours_to_considier + 1, <- num neighbours to consider, plus the car itself
            input_length + output_length, <- the total number of timestamps to consider
            2 <- X,Y of data)
        '''

        def social_layer(states, inputs):
            '''
                Merge all LSTMStateTuple
                input: list of (N + 1) states (cell_state, hidden_state)
                       list of (N + 1) inputs, each of shape (batch, 2)

            '''
            cell_states, hidden_states = [s.c for s in states_list], [s.h for s in states_list]
            hidden_states =  [
                tf.cond(tf.reduce_sum(coordinates) < 0, lambda: hidden_state * 0, lambda: hidden_state)
                for hidden_state, coordinates in hidden_states_and_input
            ]
        def make_graph(input_tensor):
            total_length = input_length + pred_length
            N = num_neighbours

            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units)
            # initial state for each track
            states = [
                lstm_cell.zero_state(tf.shape(input_tensor)[0], dtype = tf.float64)
                for _ in range(N + 1)
            ]
            # for each step...
            for step in range(input_length):
                outputs_and_states = [
                    lstm_cell(input_tensor[:,i, step, :], states[i])
                    for i in range(N + 1) # N neighbours and the trajectory itself
                ]
                # take out all the predicted stat
                states_list = [state for output, state in outputs_and_states]
                cell_states, hidden_states = [s.c for s in states_list], [s.h for s in states_list]

                hidden_states_and_input = [
                    (state, input_tensor[:, i, step, :])
                    for i, state in enumerate(hidden_states)
                ]

                # mask out the state of which the coordinates are paddings (sum(coord) < 0)
                # notice that the "state" in the for loop is an LSTMCellState!
                masked_states = [
                    tf.cond(tf.reduce_sum(coordinates) < 0, lambda: hidden_state * 0, lambda: hidden_state)
                    for hidden_state, coordinates in hidden_states_and_input
                ]

                # now get a combined state according to the social layer
                hidden_states = tf.stack(masked_states)
                hidden_states = tf.reduce_sum(hidden_states, 0)

                # finally merget the social LSTM hidden state to hidden states
                states = [
                    tf.nn.rnn_cell.LSTMStateTuple(c = cell_state, h = hidden_states)
                    for cell_state in cell_states
                ]
            # now we have the final state, compute the prediction...
            # we need the mapping from num_units to "2", which is a fully connected layer with ELU activation
            num_outputs = 2 if not predict_gaussian else 5 # coordinates if not outputing a gaussian; else ux,uy, sx,sy, sxy
            W = tf.random.normal((num_units, num_outputs), dtype = tf.float64)
            b = tf.zeros((num_outputs,), dtype = tf.float64)

            fully_connected = lambda hidden_layer: tf.nn.elu(tf.matmul(hidden_layer, W) + b)

            predictions = []
            for step in range(pred_length):
                predictions_of_step = []
                for i in range(N + 1):
                    output, state = lstm_cell(tf.zeros((tf.shape(input_tensor)[0], 2), dtype=tf.float64), states[i])

                    predictions_of_step.append(
                        fully_connected(output)
                    )
                    states[i] = state
                predictions.append(predictions_of_step)

            predictions = tf.stack(predictions, axis = 1) # so the predictions would be of shape (batch, pred_length, 2)
            predictions = tf.transpose(predictions, [2,0,1,3]) # (num_neighbours, step, batch, coord) => (batch, num_neighbours, step, coord)
            return predictions, lstm_cell, W, b
        return make_graph
