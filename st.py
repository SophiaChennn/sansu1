with open('result.txt','r') as f:
    str1 = f.readlines()

predict = 0
real = 0
all = 0
precision = 0
recall_predict = 0
recall_real = 0
thred=0.6
for str11 in str1:
    item = str11.split('\t')
    all = all + 1
    if item[1] == item[2]:
        precision = precision + 1
    if float(item[3]) >= thred:
        predict = predict + 1
        if item[1] == '0':
            real = real + 1
    if item[1] == '0':
        recall_predict = recall_predict + 1
        if float(item[3]) >=thred:
            recall_real = recall_real + 1

print('sansu recall',recall_real/recall_predict)
print('accuracy',precision/all)
print('sansu precision',real/predict)
