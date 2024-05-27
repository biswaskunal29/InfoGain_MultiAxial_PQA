import csv

FOLDER = r'F:/PhD\Datasets/twitter-collection-master/twitter-collection-master/Final Dataset v9/'

"""
image labels (done)
image HW recog (done)
image desc (done)

profile labels (done)
profile HW recog (done)
profile desc (done)

banner labels (done)
banner HW recog (done)
"""

def get_image_labels(uid,imgid):        #returns a string of all the labels separeated by a space
    filename = FOLDER + uid + "/" + imgid + "_labels.csv"  
    try:
        with open(filename, newline='\n', encoding='utf-8') as csv_file:
            cf = csv.reader(csv_file, delimiter=',',quotechar='"')
            line_count = 0
            label_list = []
            for row in cf:
                if line_count <= 1:
                    line_count+=1
                else:
                    k, v = row
                    label_list.append(k)
                    line_count += 1
        
        labels = " ".join(t for t in label_list)
        return labels
    except:
        return ""
    
def get_profile_labels(uid,imgid):      #returns a string of all the labels separeated by a space
    filename = FOLDER + uid + "/" + uid + "_profile_labels.csv" 
    try:
        with open(filename, newline='\n', encoding='utf-8') as csv_file:
            cf = csv.reader(csv_file, delimiter=',',quotechar='"')
            line_count = 0
            label_list = []
            for row in cf:
                if line_count <= 1:
                    line_count+=1
                else:
                    k, v = row
                    label_list.append(k)
                    line_count += 1
        
        labels = " ".join(t for t in label_list)
        return labels
    except:
        return ""
    
def get_banner_labels(uid,imgid):       #returns a string of all the labels separeated by a space
    filename = FOLDER + uid + "/" + uid + "_banner_labels.csv" 
    try:
        with open(filename, newline='\n', encoding='utf-8') as csv_file:
            cf = csv.reader(csv_file, delimiter=',',quotechar='"')
            line_count = 0
            label_list = []
            for row in cf:
                if line_count <= 1:
                    line_count+=1
                else:
                    k, v = row
                    label_list.append(k)
                    line_count += 1
        
        labels = " ".join(t for t in label_list)
        return labels
    except:
        return ""
 
def get_image_HW(uid,imgid):        #returns a string of all the recognised handwriting(HW) separeated by a space
    filename = FOLDER + uid + "/" + imgid + "_paragraph.csv"   
    try:
        with open(filename, newline='\n', encoding='utf-8') as csv_file:
            cf = csv.reader(csv_file, delimiter=',',quotechar='"')
            line_count = 0
            HW_list = []
            for row in cf:
                if line_count <= 0:
                    line_count+=1
                else:
                    k = row
                    HW_list.append("".join(k))
                    line_count += 1
    #    print(f"{type(HW_list)}\n{HW_list}")
        HW = " ".join(t for t in HW_list)
        return HW
    except:
        return ""

def get_profile_HW(uid,imgid):      #returns a string of all the recognised handwriting(HW) separeated by a space
    filename = FOLDER + uid + "/" + uid + "_profile_paragraph.csv"   
    try:
        with open(filename, newline='\n', encoding='utf-8') as csv_file:
            cf = csv.reader(csv_file, delimiter=',',quotechar='"')
            line_count = 0
            HW_list = []
            for row in cf:
                if line_count <= 0:
                    line_count+=1
                else:
                    k = row
                    HW_list.append("".join(k))
                    line_count += 1
    #    print(f"{type(HW_list)}\n{HW_list}")
        HW = " ".join(t for t in HW_list)
        return HW
    except:
        return ""
    
def get_banner_HW(uid,imgid):      #returns a string of all the recognised handwriting(HW) separeated by a space
    filename = FOLDER + uid + "/" + uid + "_banner_paragraph.csv"   
    try:
        with open(filename, newline='\n', encoding='utf-8') as csv_file:
            cf = csv.reader(csv_file, delimiter=',',quotechar='"')
            line_count = 0
            HW_list = []
            for row in cf:
                if line_count <= 0:
                    line_count+=1
                else:
                    k = row
                    HW_list.append("".join(k))
                    line_count += 1
    #    print(f"{type(HW_list)}\n{HW_list}")
        HW = " ".join(t for t in HW_list)
        return HW
    except:
        return ""
    
def get_image_desc(uid,imgid):      #returns the captions/image description written by the user about the image
    filename = FOLDER + uid + "/" + uid + ".csv"
    try:
        with open(filename, newline='\n', encoding='utf-8') as csv_file:
            cf = csv.reader(csv_file, delimiter=',',quotechar='"')
            d = {}
            for row in cf:
                k, v = row
                d[k] = v
        desc = d[imgid]
        return desc
    except:
        return ""
    
def get_profile_desc(uid,imgid):      #returns the description written by the user on his profile
    filename = FOLDER + uid + "/" + uid + "_profiledata.csv"
    try:
        with open(filename, newline='\n', encoding='utf-8') as csv_file:
            cf = csv.reader(csv_file, delimiter=',',quotechar='"')
            d = {}
            for row in cf:
                k, v = row
                d[k] = v
        desc = d["description"]
        return desc
    except:
        return ""

# =============================================================================
#     reader = csv.reader(open(filename, 'r'))
#     d = {}
#     for row in reader:
#         k, v = row
#         d[k] = v
# =============================================================================
# =============================================================================
# with open('employee_file.csv', newline='\n', encoding='utf-8') as csv_file:
#     cf = csv.reader(csv_file, delimiter=',',quotechar='"')
#     line_count = 0
#     for row in cf:
#         if line_count == 0:
#             line_count += 1
#             print(f'Column names are \n{", ".join(row)}\n')
#         else:   
#             print(', '.join(row))
#             line_count += 1
# =============================================================================
# =============================================================================
#     else:
#         return 0
# =============================================================================

# =============================================================================
# uid = "749003"
# imgid = "46"
# 
# result = get_profile_desc(uid,imgid)
# print(f"{type(result)}\nNo. of characters :{len(result)}\n\n{result}")
# =============================================================================

