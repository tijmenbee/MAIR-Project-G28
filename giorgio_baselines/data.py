from sklearn.model_selection import train_test_split

with open('C:\\Users\yanni\Documents\GitHub\MAIR-Project-G28\dialog_acts.dat', 'r') as f:
    dialog_acts = f.readlines()


data = [line.lower().strip().split(" ", maxsplit=1) for line in dialog_acts]
print(data[5][1])

data_nodupes = []
for i in range(0,len(data)-1):
    duplicate = False
    for j in range(i+1,len(data)):
        if data[i][1] == data[j][1]:
            duplicate = True
    if duplicate == False:            
        data_nodupes.append(data[i])

train_data, test_data = train_test_split(data, test_size=0.15, random_state=42)

print(len(data))
print(len(data_nodupes))
del data  # To avoid misusing original data.
