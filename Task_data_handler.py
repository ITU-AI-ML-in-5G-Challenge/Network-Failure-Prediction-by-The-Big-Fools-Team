import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Read csv files
training = pd.read_csv('ML5G-PS-005_train.csv')
test = pd.read_csv('ML5G-PS-005_test.csv')


# Format time column 
def get_time_obj(t):
    return datetime.datetime(t['date'][0], t['date'][1], t['date'][2], t['time'][0], t['time'][1], t['time'][2])

def separate_time(t):
    return {'date': (int(t[:4]), int(t[4:6]), int(t[6:8])), 'time': (int(t[8:10]), int(t[10:12]), int(t[12:]))}

# calculate detection time from time column
def calculate_time_difference(time_string):
    start = separate_time(time_string.split('-')[0])
    end = separate_time(time_string.split('-')[1].split('_')[1])
    
    difference = get_time_obj(start) - get_time_obj(end)
    # The preparation phase is donoted by negative time, so to prevent errors make it positive also
    if int(difference.days) < 0:
        difference = get_time_obj(end) - get_time_obj(start)
    return difference.seconds

# Get number from the time column
def get_number(time_string):
    num = time_string.split('-')[1].split('_')[0]
    return int(num)

# Get all indices from single test cycle (get consective 70 rows which forms a test scenario)
def get_test_cycle(df, time_string, iteration):
    results = []
    original_num = get_number(time_string)
    i = iteration
    while i >= 0:
        current_num = get_number(df.time.iloc[i]) 
        if original_num==current_num:
            results.append(i)
        else:
            break
        i -= 1
    i = iteration
    while i < len(df)-1:
        i += 1
        current_num = get_number(df.time.iloc[i]) 
        if original_num==current_num:
            results.append(i)
        else:
            break
        
    results.sort()
    return results

# prevent preparation phase and select row from registration and deregistration phase only
def allowed_index(df, one_cycle):
    for i in one_cycle:
        # Following if statement may produce out of bounds error (Take care when reusing it)
        if (calculate_time_difference(df.iloc[i].time) - calculate_time_difference(df.iloc[i+1].time)) < 0:
            return i+1
 
# Select rows and combine them to form new dataset     
def get_min_cycles(df, time_limit=600):
    all_test_cycles = []
    result = pd.DataFrame(columns=df.columns)
    i = 0
    # split the dataset into different test scenarios
    while i < len(df):
        one_cycle = get_test_cycle(df, df.time.iloc[i], i)
        all_test_cycles.append(one_cycle)
        i += len(one_cycle)
    
    current_index=0
    # select and combine the rows from each test scenario
    for i, one_cycle in enumerate(all_test_cycles):
        current_index = allowed_index(df, one_cycle)
        while current_index < len(one_cycle)*i+len(one_cycle):
            if calculate_time_difference(df.time.iloc[current_index]) >= time_limit:
                result = pd.concat([result, pd.DataFrame(df.iloc[current_index].to_numpy().reshape(1, -1), columns=df.columns)], ignore_index=True)
                break
            current_index += 1
    return result

# Separate training and testing dataset
def split(df, target, label, train_size=600):
    X_train = df.iloc[0:train_size, :]
    X_test = df.iloc[train_size:, :]
    train_label = label.iloc[0:train_size]
    test_label = label.iloc[train_size:]
    y_train = target.iloc[0:train_size]
    y_test = target.iloc[train_size:]
    return X_train, y_train, X_test, y_test, train_label, test_label

# Smote the training set (oversample the training set)
def manage_imbalance(df, label, fail):
    label = label.replace('normal', 0)
    label = label.replace('br-cp_bridge-loss-congestion-with-time-start', 1)
    
    sm = SMOTE(random_state=42)
    x_sm, y_sm = sm.fit_resample(fail.to_frame().join(df), label)
    
    df = x_sm.drop(['amf.amf.app.five-g.RM.RegInitFail'], axis=1)
    fail = x_sm['amf.amf.app.five-g.RM.RegInitFail']
    
    label = pd.DataFrame(y_sm, columns=['label'])
    label.replace(0, 'normal', inplace=True)
    label.replace(1, 'br-cp_bridge-loss-congestion-with-time-start', inplace=True)
    
    return df, label, fail

# return selected dataset for testing
def get_test_df(time_limit):
    return get_min_cycles(test, time_limit)

# return selected dataset for training
def get_train_df(time_limit):
    return get_min_cycles(training, time_limit)

# combine training and testing dataset
def get_df(time_limit):
    train_df = get_train_df(time_limit)
    return pd.concat([train_df, get_test_df(time_limit)], ignore_index=True), train_df.shape[0]

# return all selected datasets and perform simple data preprocessing 
def get_df_preprocessed(time_limit):
    df, train_size = get_df(time_limit)
    label = df.label
    target = df['amf.amf.app.five-g.RM.RegInitFail']
    df.drop(['time', 'label', 'amf.amf.app.five-g.RM.RegInitFail'], axis=1, inplace=True)
    df = df.astype('float32')
    target = target.astype('float32')
    
    df = pd.DataFrame(df, columns = df.columns)
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    
    
    return df, target, label, train_size
    