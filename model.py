import random
from typing import Union

import numpy as np
import pandas as pd
import simpy
from faker import Faker

# Ranges in the form (min, max)
# to be used in random generation.

# Randomly generated values will be less than 'max',
# and greater than 'min'
# min < value < max

#################
# * Passenger * #
#################

# The amount of suitcases each passenger has
suitcase_range = (1, 3)
# The amount of hand luggage each passenger has
hand_luggage_range = (1, 3)
# The age of the passenger (Kids are excluded as they are less predictable)
age_range = (18, 75)
# Percentage chance that a passenger will trigger the metal detector
triggers_metal_range = (4, 15)
# Percentage chance a passengers suitcase will be overweight
overweight_suitcase_range = (2, 8)  # percent
# The amount of passengers in the given simulation
passenger_range = (20, 150)


################
# * Check In * #
################

# The time it takes to check in each suitcase
minutes_per_suitcase_range = (2, 6)
# The delay an overweight bag causes
overweight_delay_range = (5, 15)  # minutes
# The amount of check in workers on the booths
check_in_worker_range = (2, 4)


################
# * Security * #
################

# The amount each bag takes to x-ray
minutes_per_bag_range = (2, 4)
# The chance a passenger will be randomly chosen to be searched
random_search_range = (15, 30)  # percentage
# The time it takes to search a passenger
minutes_per_search_range = (3, 7)
# The amount of workers working in the security section
security_worker_range = (2, 8)


#####################
# * Boarding Gate * #
#####################

# The time it takes to check a passengers ticket
minutes_per_ticket_range = (0.1, 0.2)
# The amount of workers working on the gate
boarding_worker_range = (1, 2)


###############
# * Airport * #
###############

# The delay between the arrival of passengers
passenger_arrival_range = (5, 10)
# The amount of gates (flights) open at the airport
gate_amount_range = (5, 5)

"""
Airport Model:
    Agents: 
        Passenger

    Processes:
        Check-in
        Security
        Boarding Gate

        Note:
          Processes such as getting food or shopping are excluded as 
          they cannot be controlled by airport facilities / staff.

    Inputs:
        Passenger:
            Suitcase count - int
            Hand luggage count - int
            Age - int
            % chance of triggering metal detector - float
            % likelihood of overweight suitcase - float

        Check-in:
            minutes per suitcase - float
            Overweight suitcase delay - float

        Security:
            minutes per hand luggage - float
            % likelihood of random search - float
            minutes per search - float

        Boarding:
            minutes per ticket - float

        Airport:
            Amount of gates - int
            Passenger arrival delay - float

    Outputs:
        The time (in minutes) it took for controllable processes to complete.

        Overall expected happiness of the customer with processes.
"""

faker = Faker()


class Passenger:
    def __init__(
        self,
        suitcases: int,
        hand_luggage: int,
        age: int,
        metal_chance: float,
        overweight_chance: float,
    ) -> None:
        """
        Initialise a passenger class.
        """

        self.suitcases = suitcases
        self.hand_luggage = hand_luggage
        self.age = age
        self.metal_chance = metal_chance
        self.overweight_chance = overweight_chance

        self.had_overweight_luggage = False
        self.triggered_metal_detector = False
        self.was_randomly_searched = False

        self.total_wait_time = 0
        self.name = faker.name()

    def _triggers_metal_detector(self) -> bool:
        """
        Decide whether or not the passenger has metal in their hand luggage.
        """
        perc = self.metal_chance / 100
        return random.random() <= perc

    def _overweight_suitcase(self):
        """
        Decide whether the passenger has at least 1 overweight suitcase.
        """
        perc = (self.overweight_chance / 100) * self.suitcases
        return random.random() <= perc


class CheckIn:
    def __init__(
        self,
        env: simpy.Environment,
        minutes_per_suitcase: float,
        overweight_delay: float,
        worker_count: int,
    ):
        self.env = env
        self.minutes_per_suitcase = minutes_per_suitcase
        self.overweight_delay = overweight_delay

        self.workers = simpy.Resource(env, worker_count)

    def check_in(self, passenger: Passenger):
        arrival_time = self.env.now
        with self.workers.request() as request:
            # Wait for available worker.
            yield request

            # Simulate time to check in each suitcase.
            for _ in range(passenger.suitcases):
                yield self.env.timeout(self.minutes_per_suitcase)

            # If one of the suitcases is overweight, simulate time
            # it takes to resolve issue.
            if passenger._overweight_suitcase():
                yield self.env.timeout(self.overweight_delay)
                passenger.had_overweight_luggage = True

        passenger.total_wait_time += self.env.now - arrival_time


class Security:
    def __init__(
        self,
        env: simpy.Environment,
        minutes_per_bag: float,
        minutes_per_search: float,
        random_search_chance: float,
        worker_count: int,
    ):
        self.env = env
        self.minutes_per_bag = minutes_per_bag
        self.minutes_per_search = minutes_per_search
        self.random_search_chance = random_search_chance

        self.workers = simpy.Resource(env, worker_count)

    def _should_randomly_search(self):
        """
        Decide whether or not the passenger should be chosen for a
        random search.
        """
        perc = self.random_search_chance / 100
        return perc <= random.random()

    def check_bags(self, passenger: Passenger):
        arrival_time = self.env.now
        with self.workers.request() as request:
            # Wait for a security agent
            yield request

            # Add time for checking each of the passengers bags
            for _ in range(passenger.hand_luggage):
                yield self.env.timeout(self.minutes_per_bag)

            # If passenger has metal in bag, or is chosen for random search,
            # simulate time it takes to pat them down.
            if passenger._triggers_metal_detector():
                yield self.env.timeout(self.minutes_per_search)
                passenger.triggered_metal_detector = True

            if not passenger.triggered_metal_detector and self._should_randomly_search():
                yield self.env.timeout(self.minutes_per_search)
                passenger.was_randomly_searched = True
        passenger.total_wait_time += self.env.now - arrival_time


class BoardingGate:
    def __init__(
        self, env: simpy.Environment, minutes_per_ticket: float, worker_count: int
    ):
        self.env = env
        self.minutes_per_ticket = minutes_per_ticket
        self.name = faker.city()
        self.workers = simpy.Resource(env, worker_count)
        self.passengers = np.array([p for p in self._generate_passengers()])

    def check_ticket(self, passenger: Passenger):
        arrival_time = self.env.now
        with self.workers.request() as request:
            # Wait for a boarding gate worker
            yield request

            # Simulate time taken to check passengers ticket
            yield self.env.timeout(self.minutes_per_ticket)

        passenger.total_wait_time += self.env.now - arrival_time

    def _generate_passengers(self):
        """
        Return a generator of a random amount of passengers, with
        randomly chosen properties.
        """
        count = 0
        passenger_amount = random.randint(*passenger_range)

        while count < passenger_amount:
            passenger = Passenger(
                suitcases=random.randint(*suitcase_range),
                hand_luggage=random.randint(*hand_luggage_range),
                age=random.randint(*age_range),
                metal_chance=random.randint(*triggers_metal_range),
                overweight_chance=random.randint(*overweight_suitcase_range),
            )
            count += 1
            yield passenger

    def _generate_passengers_with_params(self, passenger_amount: int):
        count = 0

        while count < passenger_amount:
            passenger = Passenger(
                suitcases=random.randint(*suitcase_range),
                hand_luggage=random.randint(*hand_luggage_range),
                age=random.randint(*age_range),
                metal_chance=random.randint(*triggers_metal_range),
                overweight_chance=random.randint(*overweight_suitcase_range),
            )
            count += 1
            yield passenger


class AirportModel:
    def __init__(
        self,
        env: simpy.Environment,
    ):
        self.env = env
        (self.check_in, self.security) = self._generate_facilities()
        self.boarding_gates = np.array([gate for gate in self._generate_gates()])
        self.wait_times = {}

    def _generate_gates(self, gate_amount: Union[int, None] = None):
        if not gate_amount:
            gate_amount = random.randint(*gate_amount_range)

        for _ in range(gate_amount):
            yield BoardingGate(
                env=self.env,
                minutes_per_ticket=random.randint(*minutes_per_search_range),
                worker_count=random.randint(*boarding_worker_range),
            )

    def _generate_facilities(
        self,
        check_in_params: Union[dict, None] = None,
        security_params: Union[dict, None] = None,
    ):
        if not check_in_params:
            check_in_params = dict(
                minutes_per_suitcase=random.randint(*minutes_per_suitcase_range),
                overweight_delay=random.randint(*overweight_delay_range),
                worker_count=random.randint(*check_in_worker_range),
            )

        if not security_params:
            security_params = dict(
                minutes_per_bag=random.randint(*minutes_per_bag_range),
                minutes_per_search=random.randint(*minutes_per_search_range),
                random_search_chance=random.randint(*random_search_range),
                worker_count=random.randint(*security_worker_range),
            )
        return (
            CheckIn(env=self.env, **check_in_params),
            Security(env=self.env, **security_params),
        )

    def run_airport(self, passenger_delay: Union[int, None] = None):
        if not passenger_delay:
            passenger_delay = random.randint(*passenger_arrival_range)

        for gate in self.boarding_gates:
            for passenger in gate.passengers:
                yield self.env.timeout(passenger_delay)

                self.env.process(self.check_in.check_in(passenger))
                self.env.process(self.security.check_bags(passenger))
                self.env.process(gate.check_ticket(passenger))

    def _get_max_wait(self, df: pd.DataFrame):
        column = df["Total Wait Time (Minutes)"]
        max_value = column.max()

        row = df.loc[column == max_value]

        return max_value, row

    def survey_passengers(self):
        for gate in self.boarding_gates:
            for passenger in gate.passengers:
                self.wait_times[passenger.name] = [
                    gate.name,
                    passenger.total_wait_time,
                    passenger.age,
                    passenger.suitcases,
                    passenger.hand_luggage,
                    passenger.was_randomly_searched,
                    passenger.triggered_metal_detector,
                    passenger.had_overweight_luggage,
                ]

        df = pd.DataFrame.from_dict(
            self.wait_times,
            orient="index",
            columns=[
                "Gate Destination",
                "Total Wait Time (Minutes)",
                "Age",
                "# Of Suitcases",
                "# Of Hand Luggages",
                "Was Randomly Searched",
                "Triggered Metal Detector",
                "Had Overweight Luggage",
            ],
        )

        print(df)
        print()
        print(self._get_max_wait(df))
        pd.DataFrame.to_csv(df, "output.csv")


if __name__ == "__main__":
    env = simpy.Environment()
    airport = AirportModel(env=env)
    env.process(airport.run_airport())
    env.run()
    airport.survey_passengers()
