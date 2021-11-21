import os
import pandas as pd
import numpy as np

# dataset column headings
columns = ['ALG', 'TESTSUITE_ID', 'TEST_ID', 'TIME', 'IS_VALID', 'IS_PASSED', 'MAX_OOB_PCT']

# specify a name to data file
data_file_given_name = 'beamng_tests'

# set a data file type such as csv, tsv
file_type = 'csv'

# set save directory, currently set to relative location
main_save_dir = '../data/beamng_tests'

# set is passed threshold as float, currently set as 0.95 (95 percent) this will be entered as "N" in "IS_PASSED" column
# if the value in "MAX_OOB_PCT" is more than the given threshold otherwise this will be entered as "Y"
is_passed_threshold = 0.95


# by calling this function with a session info "%Y-%m-%d_%H%M%S" as an argument
# will return a dictionary with session details
def get_session_file_info(sess, alg_id):

    curr_save_dir = os.path.join(main_save_dir, alg_id)
    if not os.path.isdir(curr_save_dir):
        os.makedirs(curr_save_dir)

    curr_sess_data_file = ''.join([i for i in os.listdir(curr_save_dir) if sess in i])
    new_session = True if not curr_sess_data_file else False

    if not new_session:
        curr_data_file_path = os.path.join(curr_save_dir, curr_sess_data_file)
        testsuite = int(curr_sess_data_file.rsplit('testsuite-', )[1].rsplit('.', 1)[0])

        file_info = {'path': curr_data_file_path,
                     'testsuite_id': testsuite,
                     'new_session': new_session}

        return file_info

    else:
        test_sessions = [i for i in os.listdir(curr_save_dir)]
        testsuite = max([int(i.rsplit('testsuite-', )[1].rsplit('.', 1)[0])
                         if 'testsuite-' in i else 0 for i in test_sessions], default=0) + 1

        data_file_name = '{}_{}_{}_testsuite-{}.{}'.format(data_file_given_name, alg_id, sess, testsuite, file_type)
        curr_data_file_path = os.path.join(curr_save_dir, data_file_name)

        file_info = {'path': curr_data_file_path,
                     'testsuite_id': testsuite,
                     'new_session': new_session}

        return file_info


# by calling the following function with a dictionary of session's test data as an argument as follows
'''
test_info = {
             'ALG': 'wgan-1',
             'TIME': 0.33,
             'IS_VALID': 'Y',
             'MAX_OOB_PCT': np.array([[0.2323], [0.5599], [0.9601], [0.8119], [0.02566], '...n']),
             'session': '2021-11-1_123481',
             'initial_tests': 50
          }
'''
# this will create a new hard copy of a data file or update the current sessions data file
def update_session_dataset(data):

    new_test_data = data
    curr_sess_data_file = get_session_file_info(new_test_data['session'], new_test_data['ALG'])
    curr_sess_is_new = curr_sess_data_file['new_session']

    del new_test_data['session']

    for header in columns:
        if header not in new_test_data.keys():
            new_test_data.update({header: []})

    # if the session exists the code will update the hard dataset with a new entry on current data
    if not curr_sess_is_new:
        new_test_data.update({'TESTSUITE_ID': curr_sess_data_file['testsuite_id']})
        curr_data_df = pd.read_csv(curr_sess_data_file['path'])
        new_test_data['TEST_ID'] = max(curr_data_df['TEST_ID']) + 1
        new_test_data['MAX_OOB_PCT'] = [i[0] for i in new_test_data['MAX_OOB_PCT'].tolist()]\
                                       [new_test_data['initial_tests']:][new_test_data['TEST_ID'] - 1]

        if new_test_data['MAX_OOB_PCT'] < is_passed_threshold:
            new_test_data['IS_PASSED'] = 'Y'

        else:
            new_test_data['IS_PASSED'] = 'N'

        del new_test_data['initial_tests']

        for col in new_test_data:
            col_elements = list(curr_data_df[col])
            col_elements.append(new_test_data[col])
            new_test_data[col] = col_elements

        df = pd.DataFrame(new_test_data).reindex(columns, axis=1)
        df.to_csv(curr_sess_data_file['path'], index=False)

    # if the session does not exist the code will create a new hard dataset with a new entry on current data
    else:
        new_test_data.update({'TESTSUITE_ID': curr_sess_data_file['testsuite_id']})
        new_test_data['TEST_ID'] = 1
        new_test_data['MAX_OOB_PCT'] = [i[0] for i in new_test_data['MAX_OOB_PCT'].tolist()]\
                                       [new_test_data['initial_tests']:][new_test_data['TEST_ID'] - 1]

        if new_test_data['MAX_OOB_PCT'] < is_passed_threshold:
            new_test_data['IS_PASSED'] = 'Y'

        else:
            new_test_data['IS_PASSED'] = 'N'

        del new_test_data['initial_tests']

        for col in new_test_data:
            new_test_data[col] = [new_test_data[col]]

        df = pd.DataFrame(new_test_data).reindex(columns, axis=1)
        df.to_csv(curr_sess_data_file['path'], index=False)

'''
test_info = {
             'ALG': 'wgan_random',
             'TIME': 0.33,
             'IS_VALID': 'Y',
             'MAX_OOB_PCT': np.array([[0.2323], [0.5599], [0.9601], [0.8119], [0.02566], '...n']),
             'session': '2021-11-1_123481',
             'initial_tests': 0
          }
'''

# update_session_dataset(test_info)
