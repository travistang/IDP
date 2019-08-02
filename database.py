# read and parse database
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle, sample, choice

class Database:

    def __init__(self, filename = 'data_with_header.txt'):
        self.data = pd.read_csv(filename, '\t')
        self._preprocess_data()

    def is_trajectory_close(self, traj1, traj2, max_distance):
        '''
            If the trajectories are of shape (n, 2), where n is the enumber of timestamp
            then check pairwise if the trajectories are close to each other
            (||t_1i - t_2i|| <= max_distance)
        '''
        assert len(traj1.shape) == len(traj2.shape) == 2 # assert they are 2D matrix
        assert traj1.shape == traj2.shape # assert num timestamps are the same
        n = traj1.shape[0]
        return all([np.linalg.norm(traj1[i, : ] - traj2[i, :]) <= max_distance for i in range(n)])

    def padding_trajectory(self, num_steps):
        return [(-1, 0) for _ in range(num_steps)]

    def _preprocess_data(self):
        data = self.data # alias

        normalized_data = data.copy()

        # normalize X
        high = data['X'].max()
        low = data['X'].min()
        normalized_data['X'] = ((data['X'] - low) / (high - low)) * 10 # within [-0.1,0.1]

        # normalize Y
        high = data['Y'].max()
        low = data['Y'].min()
        normalized_data['Y'] = ((data['Y'] - low) / (high - low)) * 10 # within [-0.1,0.1]

        # timestamp / 40
        normalized_data['timestamp'] = (data['timestamp'] / 40).map(int)
        self.data = normalized_data

    '''
        Helper functions
    '''
    def get_trajectory_of_car(self, car_id, start_time, num_step):
        data = self.data
        return np.array(
            data[
                (data.ID == car_id) &
                (data.timestamp >= start_time) &
                (data.timestamp < start_time + num_step)]
            .groupby('ID')
            .apply(
                lambda trace: list(zip(trace.X, trace.Y))
            ).tolist()[0] # there is only one ID, assume it exists, it must be in the first record
        )

    def get_car_near_point_at_timestamp(self, start_time, num_step, point, distance):
        '''
            Given a particular point (x,y), a starting_time (T), and the distance to consider...
            return all the IDs of the vehicles that have a trajectory close to the given point within that time window (start_time + num_step)
        '''
        data = self.data
        target_X, target_Y = point
        return set(
            data[
                (data.timestamp >= start_time) &
                (data.timestamp < start_time + num_step) &
                ((data.X - target_X) ** 2 + ((data.Y - target_Y)) ** 2 < distance ** 2)
            ].ID
        )

    def get_cars_within_time_frame(self, start_time, num_step):
        '''
            Given the time window (start_time, start_time + step), return the IDs of all vehicles existing in that time window.
        '''
        data = self.data
        id_with_size = data[
            (data.timestamp >= start_time) &
            (data.timestamp < start_time + num_step)
        ].groupby('ID').size().reset_index()

        return set(id_with_size[id_with_size[0] == num_step].ID)

    '''
        Get training data
    '''
    def get_batch(self,num_data = 128,input_length = 8, output_length = 4):
        '''
            Get a batch for training REGULAR LSTM model
        '''
        data = self.data
        # evaluate the total length of series required
        total_length = input_length + output_length
        # filter out the series that has at least the number of `total_length` long
        id_counts = data.groupby('ID').ID.count()
        # get a table of candidate id, whose sequence is longer than (or eq. to) total_length
        candidate_id_counts = id_counts[id_counts >= total_length]
        # get the random sequence...
        # the candidate_id_counts is a series with ID as x and count as y
        # to get the usable indices, get list series as a list of tuple like (id, count),
        # then take the first one (list of id)
        # and make it a list, and shuffle on it
        random_ids_selected = list(map(lambda tup: tup[0],candidate_id_counts.items()))
        shuffle(random_ids_selected)

        selected_ids = []
        input_batch = []
        target_batch = []
        # retrieve the coordinates of the sequence (from the beginning to `total_length`)
        for i in random_ids_selected[:num_data]:
            selected_ids.append(i)
            # select X,Y from ID where ID == i order by timestamp...
            sequence_of_i = data[data.ID == i].sort_values(by = "timestamp")[["X","Y"]]
            # divide the sequence into two parts...
            input_sequence = sequence_of_i.iloc[:input_length]
            target_sequence = sequence_of_i.iloc[input_length:total_length]
            # and append the new sequence to existing arrays
            input_batch.append(np.array(input_sequence))
            target_batch.append(np.array(target_sequence))

        # return and array of selected ids as well as the batch...
        return np.stack(selected_ids), np.stack(input_batch), np.stack(target_batch)

    def get_social_lstm_train_data(self,
                                    num_data = 64,
                                    neighbours_to_consider = 4,
                                    input_length = 8,
                                    output_length = 4,
                                    distance = 2):
        while True:
            yield self.get_social_lstm_batch(num_data,
                neighbours_to_consider, input_length, output_length,
                distance)

    def get_social_lstm_batch(self,
                              num_data = 64,
                              neighbours_to_consider = 4,
                              input_length = 8,
                              output_length = 4,
                              distance = 2,
                              ignore_neighbours = []):
        '''
            Get a batch for the social LSTM input
            The input should be of size (
                num_data, <- the batch
                neighbours_to_considier + 1, <- num neighbours to consider, plus the car itself, @0 is the target, rest are it's neighbour
                input_length + output_length, <- the total number of timestamps to consider
                2 <- X,Y of data
            )

            num_data: badge to consider
            neighbours_to_consider: how many series of data to be summed
            input_length: number of steps as ground truth.
            output_length: number of steps as prediction

        '''
        data = self.data
        batch = []
        num_steps = input_length + output_length
        max_time = data.timestamp.max()
        possible_timestamps = [ts for ts in set(data.timestamp) if ts + input_length + output_length <= max_time]

        # sample time to start with (non-repeating)
        start_times = sample(possible_timestamps, num_data)
        # get all the vehicles that are in those timestamps, ID's may repeat...
        cars_id_in_time_frames = [
            (st, self.get_cars_within_time_frame(st, num_steps))
            for st in start_times]
        # for each starting time, choose 1 car from it
        target_car_ids = [
            (st, choice(list(cars_id_in_time_frame)))
            for st, cars_id_in_time_frame in cars_id_in_time_frames]
        # for each of the car and id, get all their neighbours
        # so (st, car_id) -> [(st, car_id, car_trajectory)]
        target_car_trajectories = [
            (st, car_id, self.get_trajectory_of_car(car_id, st, num_steps) )
            for st, car_id in target_car_ids]

        # now better use for-loop from here
        for i, (st, target_car_id, target_trajectory) in enumerate(target_car_trajectories):
            # get the list of ids of cards in that particular st provided that it's not target car's id
            possible_neighbours = [id for id in cars_id_in_time_frames[i][-1] if id != target_car_id]
            # then take out their trajectories
            neighbour_trajectories = [
                self.get_trajectory_of_car(neighbour_id, st, num_steps)
                for neighbour_id in possible_neighbours]

            # find all neighbours with trajectories close enough to the target
            neighbour_trajectories = list(
                filter(
                    lambda traj: self.is_trajectory_close(traj, target_trajectory, distance),
                    neighbour_trajectories
                )
            )

            # if there are this many neighbours, just take out random ones
            if len(neighbour_trajectories) >= neighbours_to_consider:
                neighbour_trajectories = sample(neighbour_trajectories, neighbours_to_consider)

            else:
                # there are not enough neighbours, add padding trajectories to maintain the shape
                neighbour_trajectories = neighbour_trajectories + [self.padding_trajectory(num_steps)] * (neighbours_to_consider - len(neighbour_trajectories))

            # merge all trajectories, make sure the target is the first one in the tensor
            tensor = np.array([target_trajectory] + neighbour_trajectories)
            # check the final shape of this data
            assert tensor.shape == (neighbours_to_consider + 1, num_steps, 2)
            batch.append(tensor)

        # finally merge all data into a batch
        batch = np.array(batch)
        # verify the shape
        assert batch.shape == (num_data, neighbours_to_consider + 1, num_steps, 2)
        # and we're done
        return batch
