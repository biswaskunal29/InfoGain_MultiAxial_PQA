import csv


img_ques_ans_write_csv = "Dataset_uniqid_ques_ans.csv"

with open(img_ques_ans_write_csv, mode = 'r') as f:
    reader = csv.reader(f)
    data = list(reader)





for i in range(2):
#    print(data[i][0],data[i][1])
#    print(i)

    sample_label = data[i][0]
    quesno = data[i][1]
    ans = data[i][2]
    print(sample_label)
    print(quesno)
    print(ans)
    
#    parts = [x.strip() for x in sample_label.split("_")]
#    (uid,imgid) = parts
#    print(uid)
#    print(imgid)

    











#    print(data[i])














