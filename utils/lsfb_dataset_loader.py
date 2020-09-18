import pandas as pd
import glob
from os.path import sep

def load_lsfb_dataset(path:str, verbose: bool = True):
  '''
  Read the LSFB corpus in a dataframe containing the label of each signs, a number associated to each label,
  the path of the sign and which subset the signs come from (test or train). 

  PARAMETERS:
    path : The path to lsfb corpus folder

  OUTPUT:
    dataset : A dataframre containing 4 columns (label, label_nbr, path, subset) 
  '''

  signs_folders = glob.glob(f"{path}{sep}*")
  map_label = {value.split(sep)[-1]: idx for (idx, value) in enumerate(signs_folders)}

  dataset = pd.DataFrame(columns=['label', 'label_nbr', 'path', 'subset'])

  for (idx, folder) in enumerate(signs_folders):
    train_path = f"{folder}{sep}train{sep}*"
    test_path = f"{folder}{sep}test{sep}*"
    label = folder.split(sep)[-1]

    if(verbose):
      printProgressBar(idx+1, len(signs_folders), prefix=f'Loading dataset')

    for sign in glob.glob(train_path):
      row = {}
      row['label'] = label 
      row['label_nbr'] = map_label[label]
      row['path'] = sign
      row['subset'] = 'train'

      dataset = dataset.append(row, ignore_index=True)

    for sign in glob.glob(test_path):
      row = {}
      row['label'] = label 
      row['label_nbr'] = map_label[label]
      row['path'] = sign
      row['subset'] = 'test'

      dataset = dataset.append(row, ignore_index=True)

  return dataset



# Print iterations progress
def printProgressBar (iteration:int, total:int, prefix:str = '', suffix:str = '', decimals:int = 1, length:int = 100, fill:str = '█', printEnd:str = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

    




