import pandas as pd
import numpy as np
import category_encoders as ce
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

def rearrange_date(df):
    """
    Merges YEAR, MONTH, DAY and puts it into a datetime object.
    Creates DAY_OF_WEEK based on date.
    """
    # Create a Date Time Variable - YEAR-MONTH-DAY all in the same field - train
    df = df.astype({'YEAR': 'str', 'MONTH': 'str', 'DAY': 'str'})
    df["DATE"] = df["DAY"].str.cat(df[['MONTH', 'YEAR']], sep='-')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['DAY_OF_WEEK'] = df['DATE'].dt.day_name()
    df['MONTH'] = df['DATE'].dt.month_name()
    df.drop(columns=['YEAR', 'DAY'], inplace=True)
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


def one_hot_encode_main(df_train, df_test):
    """
    Receives train and test dataset. One Hot encodes variables MONTH,DAY_OF_WEEK,AIRLINE.
    :param df_train:
    :param df_test:
    :return:
    """
    encoder = ce.OneHotEncoder(cols=['MONTH', 'DAY_OF_WEEK', 'AIRLINE'], return_df=True, use_cat_names=True)
    encoder_fit = encoder.fit(df_train)
    df_train = encoder_fit.transform(df_train)
    df_test = encoder_fit.transform(df_test)
    return df_train, df_test


def one_hot_encode_airports(df_train, df_test):
    # Create object for binary encoding
    #encoder = ce.BinaryEncoder(cols=['AIRPORTS'], return_df=True)
    # Create target encoding object
    encoder_target = ce.TargetEncoder(cols=['AIRPORTS'], return_df=True)
    # Encode Train Set
    df_train['AIRPORTS'] = df_train['ORIGIN_AIRPORT'].str.cat(df_train['DESTINATION_AIRPORT'], sep="-")
    df_train['AIRPORT_TargetEncoding'] = encoder_target.fit_transform(df_train['AIRPORTS'], df_train['ARRIVAL_DELAY'])
    #df_train = encoder.fit_transform(df_train)
    # Encode Test Set
    df_test['AIRPORTS'] = df_test['ORIGIN_AIRPORT'].str.cat(df_test['DESTINATION_AIRPORT'], sep="-")
    df_test['AIRPORT_TargetEncoding'] = encoder_target.fit_transform(df_test['AIRPORTS'], df_test['ARRIVAL_DELAY'])
    #df_test = encoder.fit_transform(df_test)
    return df_train, df_test


def feature_encode_airports(df_train, df_test):
    """

    :param df_train:
    :param df_test:
    :return:
    """
    df_train['AIRPORTS'] = df_train['ORIGIN_AIRPORT'].str.cat(df_train['DESTINATION_AIRPORT'], sep="-")
    df_test['AIRPORTS'] = df_test['ORIGIN_AIRPORT'].str.cat(df_test['DESTINATION_AIRPORT'], sep="-")
    encoder = ce.TargetEncoder(cols=['AIRPORTS'], return_df=True)
    encoder_fit = encoder.fit(df_train['AIRPORTS'], df_train['ARRIVAL_DELAY'])
    df_train['AIRPORT_TargetEncoding'] = encoder_fit.transform(df_train['AIRPORTS'])
    df_test['AIRPORT_TargetEncoding'] = encoder_fit.transform(df_test['AIRPORTS'])
    return df_train, df_test


def cat_encode(df_train, df_test):
    df_train['AIRPORTS'] = df_train['ORIGIN_AIRPORT'].str.cat(df_train['DESTINATION_AIRPORT'], sep="-")
    df_test['AIRPORTS'] = df_test['ORIGIN_AIRPORT'].str.cat(df_test['DESTINATION_AIRPORT'], sep="-")
    cat_features = ['MONTH', "DAY_OF_WEEK", "AIRLINE", "FLIGHT_NUMBER", "TAIL_NUMBER", "ORIGIN_AIRPORT",
                    'DESTINATION_AIRPORT', "AIRPORTS"]
    cat_encoder = ce.LeaveOneOutEncoder()
    cat_encoder.fit(df_train[cat_features], df_train["ARRIVAL_DELAY"])
    df_train[cat_features] = cat_encoder.transform(df_train[cat_features])
    df_test[cat_features] = cat_encoder.transform(df_test[cat_features])
    return df_train, df_test


if __name__ == "__main__":
    # Read Data
    # Original
    flights_train_raw = pd.read_csv("data/flights_train.csv")
    flights_test_raw = pd.read_csv("data/flights_test.csv")
    # Copies (work on the copies)
    flights_train = flights_train_raw.copy()
    flights_test = flights_test_raw.copy()
    flights_test["ARRIVAL_DELAY"] = np.nan

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
    #flights_train, flights_test = one_hot_encode_main(flights_train, flights_test)
    # Origins and Destinations
    #flights_train, flights_test = feature_encode_airports(flights_train, flights_test)
    # Encode everything
    flights_train, flights_test = cat_encode(flights_train, flights_test)

    # Fit Linear Model
    X, Y = flights_train['DEPARTURE_DELAY'].to_numpy().reshape(-1, 1), flights_train['ARRIVAL_DELAY']
    lm = LinearRegression().fit(X, Y)
    # Build prediction for Test Set
    flights_train["LM_PREDICTIONS"] = lm.predict(X)
    flights_train["RESIDUALS"] = flights_train["ARRIVAL_DELAY"] - flights_train["LM_PREDICTIONS"]

    # Drop unnecessary columns
    #flights_train.drop(columns=["WHEELS_OFF", 'id', 'DATE'], inplace=True)
    #flights_test.drop(columns=["WHEELS_OFF", 'id', 'DATE'], inplace=True)


    print("Done")
    # Save pre-processed data
    flights_train.to_csv("data/pp_flights3_train.csv", index=False)
    flights_test.to_csv("data/pp_flights3_test.csv", index=False)