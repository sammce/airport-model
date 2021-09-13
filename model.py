# Used for simulating the randomness of aiport passengers
import random

# Used for getting CLI arguments
import sys

# Mainly for numpy.array in order to improve performance
import numpy as np

# Used for analysing the output of the model, and recording
# the results in CSV files
import pandas as pd

# A module for simulating scenarios using an event loop system.
# Documentation can be found at: https://simpy.readthedocs.io/en/latest/
import simpy

# My own module for printing coloured text to the terminal
from samutil.formatting import ColorCodes, Formatter

# Default ranges in the form (min, max)
# to be use in random generation.

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
overweight_suitcase_range = (2, 8)
# The amount of passengers per gate
passenger_range = (20, 150)


################
# * Check In * #
################

# The time it takes to check in each suitcase
minutes_per_suitcase_range = (2, 6)
# The delay an overweight bag causes
overweight_delay_range = (5, 15)
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
minutes_per_ticket_range = (2.5, 4)
# The amount of workers working on the gate
boarding_worker_range = (1, 2)


###############
# * Airport * #
###############

# The delay between the arrival of passengers
passenger_arrival_range = (5, 10)
# The amount of gates (flights) open at the airport
gate_amount_range = 5
# The maximum acceptable wait time
wait_time_threshold_range = 5

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

# Instantiate formatter class
formatter = Formatter()

# Set pandas output options
pd.set_option("display.show_dimensions", False)
pd.set_option("display.max_columns", 7)
pd.set_option("display.width", 130)


def hyphenate_iterable(*args):
    return " - ".join(args)


def convert_all(*args: tuple, data_type: type = str, iter_type: type = tuple):
    return wrap_all(*args, callback=data_type, iter_type=iter_type)


def wrap_all(*args: tuple, callback: function, iter_type: type = tuple):
    return iter_type(map(lambda x: callback(x), args))


def call_method_on_all(*args, method: str, iter_type: type = tuple):
    for val in args:
        assert method in dir(
            val
        ), f"Method {method} doesn't exist on value '{val}' of type: {type(val)}"

    return iter_type(map(lambda x: eval(f"{x}.{method}")))


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

    def _triggers_metal_detector(self) -> bool:
        """
        Decide whether or not the passenger has metal in their hand luggage.
        """
        perc = self.metal_chance
        return random.uniform(1, 100) <= perc

    def _overweight_suitcase(self):
        """
        Decide whether the passenger has at least 1 overweight suitcase.
        """
        perc = (self.overweight_chance) * self.suitcases
        return random.uniform(1, 100) <= perc


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
        perc = self.random_search_chance
        return random.uniform(1, 100) <= perc

    def check_bags(self, passenger: Passenger):
        """
        Simulate the process of checking a passenger's carry on luggage at security.
        """
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
        self,
        env: simpy.Environment,
        minutes_per_ticket: float,
        worker_count: int,
        passenger_amount: int,
    ):
        self.env = env
        self.minutes_per_ticket = minutes_per_ticket
        self.workers = simpy.Resource(env, worker_count)
        self.passengers = np.array(
            [p for p in self._generate_passengers(passenger_amount)]
        )

    def check_ticket(self, passenger: Passenger):
        """
        Simulate the process of checking a passenger's ticket at the boarding gate.
        """
        arrival_time = self.env.now
        with self.workers.request() as request:
            # Wait for a boarding gate worker
            yield request

            # Simulate time taken to check passengers ticket
            yield self.env.timeout(self.minutes_per_ticket)

        passenger.total_wait_time += self.env.now - arrival_time

    def _generate_passengers(self, passenger_amount: int):
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

    def __len__(self):
        """
        Method called by the `len()` function
        """
        return len(self.passengers)

    def __str__(self):
        return f"Boarding Gate with {len(self)} passengers and {len(self.workers)} staff."


class AirportModel:
    def __init__(self, ask_politely: bool = False):
        self.env = simpy.Environment()
        self.ask_politely = ask_politely
        self.params = self._ask_politely_for_inputs()
        self.wait_time_threshold = self.params.get("wait_time_threshold")
        self.check_in, self.security = self._generate_facilities(
            self.params.get("check_in_params"), self.params.get("security_params")
        )

        self.boarding_gates = np.array(
            [gate for gate in self._generate_gates(self.params.get("gate_params"))]
        )

        self.column_map = {
            "Total Wait Time": "minutes",
            "Age": "years",
            "Suitcases": "suitcases",
            "Hand Luggage": "items",
            "Randomly Searched": "%",
            "Metal Detector": "%",
            "Overweight Luggage": "%",
        }
        self.columns = tuple(self.column_map.keys())
        self.env.process(self._run_airport(self.params.get("passenger_delay")))
        self.env.run()
        self._survey_passengers()
        self._output_nicely()

    def _generate_gates(self, gate_params: dict):
        for _ in range(gate_params.pop("gate_amount")):
            yield BoardingGate(env=self.env, **gate_params)

    def _generate_facilities(
        self,
        check_in_params: dict = {},
        security_params: dict = {},
    ):
        return (
            CheckIn(env=self.env, **check_in_params),
            Security(env=self.env, **security_params),
        )

    def _run_airport(self, passenger_delay: int):
        for gate in self.boarding_gates:
            for passenger in gate.passengers:
                yield self.env.timeout(passenger_delay)

                self.env.process(self.check_in.check_in(passenger))
                self.env.process(self.security.check_bags(passenger))
                self.env.process(gate.check_ticket(passenger))

    def _get_wait_minmax(
        self,
        df: pd.DataFrame,
    ):
        column = df[self.columns[0]]
        max_value = column.max()
        min_value = column.min()

        return (
            df.loc[column == min_value].iloc[0:1],
            df.loc[column == max_value].iloc[0:1],
        )

    def _survey_passengers(self):

        for gate in self.boarding_gates:
            gate.passenger_data = np.zeros((len(gate), 7), dtype="object")

            for index, passenger in enumerate(gate.passengers):

                gate.passenger_data[index] = np.array(
                    [
                        passenger.total_wait_time,
                        passenger.age,
                        passenger.suitcases,
                        passenger.hand_luggage,
                        passenger.was_randomly_searched,
                        passenger.triggered_metal_detector,
                        passenger.had_overweight_luggage,
                    ]
                )

            gate.passenger_data = pd.DataFrame(
                data=gate.passenger_data, columns=self.columns
            )

    def _ask(
        self,
        input_message: str = "",
        convert_to_type: type = str,
        instance_of: tuple = [],
        default: tuple = None,
    ):
        if not instance_of:
            instance_of = [convert_to_type]

        valid = False

        while not valid:
            try:
                if self.ask_politely:
                    user_input = input("   • " + input_message + ColorCodes.MAGENTA)
                else:
                    user_input = ""

                print(ColorCodes.END, end="")

                # If user doesn't enter a value, or ask_politely is False
                # use the default ranges set at the top of the file.
                if user_input.strip() == "":
                    # If `default` is a literal, convert it to a range tuple.
                    if not isinstance(default, (list, tuple)):
                        user_input = (default, default)
                    else:
                        user_input = default
                    self.ask_politely and print(
                        "     Using random value in range: "
                        + hyphenate_iterable(user_input)
                    )

                    return user_input

                # Check if user input is a range
                if not user_input.isdigit() and "-" in user_input:

                    user_input = convert_all(
                        call_method_on_all(*user_input.split("-"), method="split"),
                        data_type=convert_to_type,
                    )

                    if len(user_input) >= 2 and not (
                        user_input[0] <= user_input[1] and len(user_input) == 2
                    ):
                        raise TypeError

                else:
                    user_input = convert_to_type(user_input)

                return user_input
            except ValueError:
                print(
                    formatter.bold(formatter.error("\n  Invalid Parameter:")),
                    f"Value '{user_input}' couldn't be called by function: {convert_to_type.__name__}",
                )
                continue
            except TypeError:
                print(
                    formatter.bold(formatter.error("\n  Invalid Parameter Range:")),
                    f"Range {user_input} must contain 2 ascending numerical values (EG 1 - 4)",
                )
                continue

    def _ask_politely_for_inputs(self):
        """
        Repeatedly ask for values to use as inputs into the model.

         Ask for:
           Passenger arrival delay

           Boarding Gate:
               The amount of boarding gates open
               The time it takes a worker to check 1 ticket
               The amount of workers at the gate
               The amount of passengers per gate

           Check in params:
               Time taken to check in 1 suitcase
               Delay when a bag is overweight
               The amount of workers at the check in desk

           Security params:
               Time taken for 1 worker to check each bag
               Time taken for 1 worker each random search
               Chance of a random search taking place
               Amount of workers at the security area
        """
        print("\t+------------------------+")
        print("\t|     ", formatter.bold("AirportModel"), "     |")
        print(
            "\t|   " + formatter.success("By:", formatter.bold("Sam McElligott")) + "  |"
        )
        print("\t+------------------------+\n\n")

        if self.ask_politely:
            print(formatter.bold(formatter.info("Please enter: (value, range or blank)")))
            print("\n  " + formatter.underline(formatter.bold("Passengers")))

        wait_time_threshold = self._ask(
            "Maximum acceptable wait time: ",
            float,
            default=wait_time_threshold_range,
        )
        passenger_delay = self._ask(
            "Delay between the arrival of passengers: ",
            float,
            default=passenger_arrival_range,
        )

        self.ask_politely and print(
            "\n  " + formatter.underline(formatter.bold("Boarding Gate"))
        )
        gate_params = dict(
            gate_amount=self._ask(
                "Amount of gates open: ", int, default=gate_amount_range
            ),
            minutes_per_ticket=self._ask(
                "Time it takes to examine a ticket: ",
                float,
                default=minutes_per_ticket_range,
            ),
            worker_count=self._ask(
                "Amount of workers on each gate: ", int, default=boarding_worker_range
            ),
            passenger_amount=self._ask(
                "Amount of passengers for each gate: ", int, default=passenger_range
            ),
        )

        self.ask_politely and print(
            "\n  " + formatter.underline(formatter.bold("Check In"))
        )

        check_in_params = dict(
            minutes_per_suitcase=self._ask(
                "Time it takes to check in each suitcase: ",
                float,
                default=minutes_per_suitcase_range,
            ),
            overweight_delay=self._ask(
                "Delay caused by an overweight suitcase: ",
                float,
                default=overweight_delay_range,
            ),
            worker_count=self._ask(
                "Amount of workers at the check in desk: ",
                int,
                default=check_in_worker_range,
            ),
        )

        self.ask_politely and print(
            "\n  " + formatter.underline(formatter.bold("Security"))
        )

        security_params = dict(
            minutes_per_bag=self._ask(
                "Time it takes to screen each carry on bag: ",
                float,
                default=minutes_per_bag_range,
            ),
            minutes_per_search=self._ask(
                "Time it takes for each pat-down: ",
                float,
                default=minutes_per_search_range,
            ),
            random_search_chance=self._ask(
                "The chance a passenger will be randomly searched (%) : ",
                float,
                default=random_search_range,
            ),
            worker_count=self._ask(
                "Amount of workers at security: ", int, default=security_worker_range
            ),
        )

        return dict(
            passenger_delay=passenger_delay,
            wait_time_threshold=wait_time_threshold,
            gate_params=gate_params,
            check_in_params=check_in_params,
            security_params=security_params,
        )

    def _output_nicely(self):
        """
        Output the passenger data for each gate in a visually pleasing format.
        """
        for index, gate in enumerate(self.boarding_gates):
            print(
                formatter.underline(formatter.bold(formatter.info("\nGate", index + 1)))
            )

            min_row, max_row = self._get_wait_minmax(gate.passenger_data)

            print(formatter.warning("Longest wait time:\n"), max_row)
            print(formatter.success("Shortest wait time:\n"), min_row)

            print(formatter.info("Averages:"))
            averages = str(gate.passenger_data.mean())

            # Print each average prefixed with a tab and 4 backspaces,
            # suffixed with the value's data unit,
            # except for the last value, which is info about the dtype.
            print(
                *tuple(
                    map(
                        lambda enum: f"\t\b\b\b\b{enum[1]} {self.column_map[self.columns[enum[0]]]}\n",
                        enumerate(averages.split("\n")[0:-1]),
                    )
                )
            )
            median_wait = gate.passenger_data[self.columns[0]].median()
            if median_wait < self.wait_time_threshold:
                color = formatter.success
            else:
                color = formatter.error
            print("    Median Wait Time: " + (" " * 5) + color(round(median_wait, 2)))
            print("    Target: " + (" " * 15) + formatter.bold(self.wait_time_threshold))


if __name__ == "__main__":
    should_ask_politely = "--params" in sys.argv
    AirportModel(should_ask_politely)
