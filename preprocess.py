import pandas as pd
import numpy as np
import datetime
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

def rearrange_date(df):
    """
    Merges YEAR, MONTH, DAY and puts it into a datetime object.
    Creates WEEK_DAY based on date.
    """
    # Create a Date Time Variable - YEAR-MONTH-DAY all in the same field - train
    df = df.astype({'YEAR': 'str', 'MONTH': 'str', 'DAY': 'str'})
    df["DATE"] = df["DAY"].str.cat(df[['MONTH', 'YEAR']], sep='-')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['WEEK_DAY'] = df['DATE'].dt.day_name()
    df.drop(columns = ['YEAR', 'MONTH', 'DAY', "DAY_OF_WEEK"], inplace=True)
    return df


def departure_delay(day, scheduled_departure, departure, arrival_delay=None):
    '''
    Calculate the departure delay.
    Assumption: the delay is always positive unless the negative delay is less than 90 minutes.

    Input:
    date - date of the schedule departure. - datetime
    scheduled_departure - time (hhmm) for the scheduled departure. - int64
    departure - actual time (hhmm) for the departure. - float64
    '''
    # Transform to datetime objects
    scheduled_departure = "{0:04d}".format(int(scheduled_departure))
    scheduled_departure = day + datetime.timedelta(hours=int(scheduled_departure[0:2]),
                                                   minutes=int(scheduled_departure[2:4]))
    departure = "{0:04d}".format(int(departure))
    departure = datetime.timedelta(hours=int(departure[0:2]), minutes=int(departure[2:4]))

    # Create array with possible intervals
    previous_day = day - datetime.timedelta(days=1)
    previous_day_departure = previous_day + departure
    day_departure = day + departure
    next_day = day + datetime.timedelta(days=1)
    next_day_departure = next_day + departure
    time_differences = np.array([(previous_day_departure - scheduled_departure).total_seconds(),
                                 (day_departure - scheduled_departure).total_seconds(),
                                 (next_day_departure - scheduled_departure).total_seconds()]) / 60
    time_differences = time_differences[time_differences > -90]
    min_index = np.argmin(np.abs(time_differences))
    if arrival_delay:
        arrival_departure_difference = np.abs(arrival_delay - time_differences)
        min_index = np.argmin(arrival_departure_difference)
    # Get minimum interval
    return time_differences[min_index]


def one_hot_encode(df):
    df = pd.get_dummies(df, columns=['WEEK_DAY'], prefix='', prefix_sep='')
    df = pd.get_dummies(df, columns=['AIRLINE'], prefix='AIRLINE', prefix_sep='_')
    #df = pd.get_dummies(df, columns=['ORIGIN_AIRPORT'], prefix='ORIGIN', prefix_sep='_')
    #df = pd.get_dummies(df, columns=['DESTINATION_AIRPORT'], prefix='DESTINATION', prefix_sep='_')
    return df


def create_output(predictions):
    """
    Function that creates output csv submission file. Returns the submission dataframe.
    Input:
    - predictions: an array with shape (n_predictions, 1) with the ARRIVAL_DELAY outputs (ordered).
    Return:
    - submission: dataframe with id and ARRIVAL_DELAY fields.
    """
    submission = pd.DataFrame(data={"id": np.arange(len(predictions)),
                                    "ARRIVAL_DELAY": np.reshape(predictions,(len(predictions)))})
    submission.to_csv("data/submission.csv", index=False)
    return submission


if __name__ == "__main__":
    # Read Data
    # Original
    airlines_raw = pd.read_csv("data/airlines.csv")
    airports_raw = pd.read_csv("data/airports.csv")
    flights_train_raw = pd.read_csv("data/flights_train.csv")
    flights_test_raw = pd.read_csv("data/flights_test.csv")
    # Copies (work on the copies)
    airlines = airlines_raw.copy()
    airports = airports_raw.copy()
    flights_train = flights_train_raw.copy()
    flights_test = flights_test_raw.copy()

    # Drop columns that don't add information
    flights_train.drop(columns=["id", "WHEELS_OFF"])

    # Create Date with Datetime obj
    flights_train = rearrange_date(flights_train)
    flights_test = rearrange_date(flights_test)

    # Calculate Departure Delay
    flights_train["DEPARTURE_DELAY"] = flights_train.apply(lambda row: departure_delay(row["DATE"],
                                                                                        row["SCHEDULED_DEPARTURE"],
                                                                                        row["DEPARTURE_TIME"],
                                                                                        row["ARRIVAL_DELAY"]), axis=1)
    flights_test["DEPARTURE_DELAY"] = flights_test.apply(lambda row: departure_delay(row["DATE"],
                                                                                        row["SCHEDULED_DEPARTURE"],
                                                                                        row["DEPARTURE_TIME"]), axis=1)

    # OneHot Encode Labels
    flights_train = one_hot_encode(flights_train)
    flights_test = one_hot_encode(flights_test)

    # Save pre-processed data
    flights_train.to_csv("data/pp_flights_train.csv", index=False)
    flights_test.to_csv("data/pp_flights_test.csv", index=False)
    airports.to_csv("data/pp_airports.csv", index=False)
    airlines.to_csv("data/pp_airlines.csv", index=False)