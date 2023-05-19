"""
Convert dat files to csv files (gbk encoding)
"""
import os
 
path ="data/"   # converts directory
file = "2023-05-17-15-30-06_after_preprocess_dataset_clean_english_only_new.txt"
 
dir_path = os.path.join(path, file)
# Separate file name and file type
file_name = os.path.splitext(file)[0]
file_type = os.path.splitext(file)[1]

file_old = open(dir_path,'rb')  # Read the original file

encoding = 'utf-8'
more_str = "text_clean\n" # Add the csv table header

# Convert a.dat file to a.csv file
if file_type=='.dat':   # switch to '.csv' from '.dat'
    new_dir = os.path.join(path,str(file_name)+'.csv')
    #print(new_dir)
    file_new = open(new_dir,'wb')  # Create/modify a new file
    for lines in file_old.readlines():
        lines=lines.decode(encoding)
        # str_data = ",".join(lines.split(' '))
        str_data = lines
        file_new.write(str_data.encode(encoding))
    file_old.close()
    file_new.close()
elif file_type=='.txt': # switch to '.csv' from '.txt'
    new_dir = os.path.join(path,str(file_name)+'.csv')
    #print(new_dir)
    file_new = open(new_dir,'wb')  # Create/modify a new file
    file_new.write(more_str.encode(encoding)) # New adding

    for lines in file_old.readlines():
        lines=lines.decode(encoding)
        # str_data = ",".join(lines.split(' '))
        str_data = lines
        file_new.write(str_data.encode(encoding))
    file_old.close()
    file_new.close()