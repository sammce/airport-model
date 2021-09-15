def play_tennis(outlook, humidity, wind):
    """
    Use a decision tree to decide whether or not you should play
    tennis given various parameters regarding conditions.
    """
    # Convert all values to lowercase
    outlook = outlook.lower()
    humidity = humidity.lower()
    wind = wind.lower()

    # Decide if tennis should be played
    if outlook == "sunny":
        if humidity == "high":
            return False
        elif humidity == "normal":
            return True
        else:
            raise TypeError("Parameter 'humidity' must be 'high' or 'normal'")
    elif outlook == "overcast":
        return True
    elif outlook == "rain":
        if wind == "strong":
            return False
        elif wind == "weak":
            return True
        else:
            raise TypeError("Parameter 'wind' must be 'strong' or 'weak'")

    else:
        raise TypeError("Parameter 'outlook' must be 'sunny', 'overcast' or 'rain'")


def test_tennis_data(path_to_csv):
    """
    Read a CSV file of tennis data and determine for each line
    whether or not the tennis player should play.
    """
    import pandas as pd

    # Read in data from CSV file
    df = pd.read_csv(path_to_csv)

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        print(f"\nTest {index + 1}:")

        # Get data from dataframe.
        outlook, humidity, wind = (row["Outlook"], row["Humidity"], row["Wind"])

        # The try-except block may be a bit advanced, but I think it
        # is useful in this context. This code is also useless if the
        # play_tennis function doesn't raise a TypeError
        try:
            # Assumes a function called 'play_tennis' exists and takes 3 parameters
            # outlook, humidity and wind in that order.
            result = play_tennis(outlook, humidity, wind)
        except TypeError as e:
            print(f"\nInvalid parameter on line {index + 1} of '{path_to_csv}'")
            print(e)
            return

        # Show feedback to the user
        print(f"\toutlook =", outlook)
        print("\thumidity =", humidity)
        print("\twind =", wind)
        print(f"\n\tShould play tennis: {result}")


def simple_test_tennis_data(path_to_csv):
    """
    Read a CSV file of tennis data and determine for each line
    whether or not the tennis player should play.
    """
    import pandas as pd

    # Read in data from CSV file
    df = pd.read_csv(path_to_csv)

    for index, row in df.iterrows():
        # Get data from dataframe.
        outlook, humidity, wind = (row["Outlook"], row["Humidity"], row["Wind"])

        # Assumes a function called 'play_tennis' exists and takes 3 parameters
        # outlook, humidity and wind, in that order.
        result = play_tennis(outlook, humidity, wind)

        # Show feedback to the user
        print(f"\toutlook =", outlook)
        print("\thumidity =", humidity)
        print("\twind =", wind)
        print(f"\n\tShould play tennis: {result}")


if __name__ == "__main__":
    test_tennis_data("tennis_data.csv")
