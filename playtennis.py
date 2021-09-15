def playTennis(outlook: str, humidity: str, wind: str) -> bool:
    outlook = outlook.lower()
    humidity = humidity.lower()
    wind = wind.lower()
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


def testTennis(csv_file: str):
    import pandas as pd

    df = pd.read_csv(csv_file)

    for row in df:
        outlook, humidity, wind = row.loc["Outlook"], row.loc["Humidity"], row.loc["Wind"]
        result = playTennis(outlook, humidity, wind)

        print(f"Running test with outlook={outlook}, humidity={humidity} and wind={wind}")

        print("Should play tennis: ", result)


if __name__ == "__main__":
    print(testTennis("./test.csv"))
