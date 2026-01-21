# ---------------------------------------------- ML 20/04/2020 -----------------------------------------------------#
#
# Generate a sample of EV sessions data.
#               This file can be used to generate the sample of a data using the saved SDG model.
#                   - User can choose between a default trained SDG model, or a latest model trained from a new dataset
#                   - Each SDG model is a list of 3 models - AM,MMc and MMe
#                       Arrival model, mixture model for connected time and mixture model for energy required
#                   - Use can specify which AM to use.
#                           - IAT: mean/poly/loses
#                           - AC: poisson_fit/neg_bio_reg
#                   - Sample is generated for the given horizon. horizon is defined by a starting and ending date
#  dependencies: modeling.generate_sample.py
# -------------------------------------------------------------------------------------------------------------------- #

import glob
import os
import pickle
import datetime
import argparse
import random
import time


MODELS = {"ACpoisson_fit": "SDG Model (AC,poisson_fit)",
          "AVneg_bio_reg": "SDG Model (AC,neg_bio_reg)",
          "IATloess": "SDG Model (IAT,poly)",
          "IATmean": "SDG Model (IAT,mean)"}

def main(args):
    horizon_start = datetime.datetime.strptime('01/01/2015', '%d/%m/%Y') + datetime.timedelta(days=random.randint(1, 364))
    horizon_end = horizon_start + datetime.timedelta(days=1)
    # print(horizon_start, horizon_end)

    if str(args['use']) == 'default':
        model_loc = '.'
        model_name = MODELS[str(args['model']+args['lambdamod'])] #config['models'][str(args['model']+args['lambdamod'])]
        file = os.path.join(model_loc, model_name)
    if str(args['use']) == 'latest':
        list_of_files = glob.glob('SDG Model (' + str(args['model']) + ',' + str(args['lambdamod']) + ')*')  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        file = os.path.join(latest_file)

    with open(file, 'rb') as f:
        x = pickle.load(f)
    
    # these are the three saved models.
    AM,MMc,MMe = x[0], x[1], x[2]

    # we load the latest model that was saved
    save_loc = '.'
    save_name = 'Generated sample ('+str(args['model'])+','+str(args['lambdamod'])+') Horizon =' + str(horizon_start.date()) + "-to-" + str(horizon_end.date()) + '.csv'

    # this function will generate the data using models AM,MMc and MMe in the given horizon.
    from generate_sample import generate_sample
    
    restart = True
    while restart:
        try:
            gen_sample = generate_sample(AM=AM, MMc=MMc, MMe=MMe,
                                         horizon_start=horizon_start, horizon_end=horizon_end, n=args['N'])
            restart = False
        except:
            # traceback.print_exc()
            restart = True
            horizon_start = datetime.datetime.strptime('01/01/2015', '%d/%m/%Y') + datetime.timedelta(days=random.randint(1, 364))
            horizon_end = horizon_start + datetime.timedelta(days=1)
            time.sleep(500)
    
    gen_sample.to_csv(os.path.join(save_loc, save_name), index=False)

    print(" EV sessions data saved at : ", save_loc)
    print(" EV sessions data filename : ", save_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments to SDG sample generate: \n '
                                                 'Generates a sample of EV sessions data.')
    parser.add_argument('-start_date',
                        help='First date of the horizon for data generation. \n'
                             'Format: dd/mm/YYYY')
    parser.add_argument('-end_date',
                        help='Last date of the horizon for data generation. \n'
                             'Format: dd/mm/YYYY')
    parser.add_argument('-use', default='default',
                        help='Which kind of models to use.  \n'
                             '\t\t "default" for using the default models \n'
                             '\t\t "latest" for using the lastest trained models')
    parser.add_argument('-model', default='AC',
                        help='Modeling method to be used for modeling arrival times: \n'
                             '\t\t AC for arrival count models \n'
                             '\t\t IAT for inter-arrival time models')
    parser.add_argument('-lambdamod', default='poisson_fit',
                        help='Method to be used for modeling lambda:\n'
                             '\t\t AC: has two options, poisson_fit/neg_bio_reg \n'
                             '\t\t IAT: has three options, mean/loess/poly')
    parser.add_argument('-N', default=100, type=int,
                        help='Number of charging requests to be generated (integer).')
    parser.add_argument('-verbose', default=0,
                        help='0 to print nothing; >0 values for printing more information. Possible values: 0, 1, 2, 3. (integer)')

    args = parser.parse_args()
    main(vars(args))
