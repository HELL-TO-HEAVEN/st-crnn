import config
import io
LA_TWEETS = config.LA_TWEETS

def process_data_la():
    tsf = io.open(LA_TWEETS, encoding='utf-8', mode='r')
    merge = io.open(LA_TWEETS+".merge", encoding='utf-8', mode='w')
    tsfls = tsf.readlines()
    print(tsfls[0].split('')[0])
    x = []
    y = []
    pre_part = ""
    for i in range(len(tsfls)):
        attrs = tsfls[i].split('')
        # if pre_part == "":
        #     pre_part = tsfls[i]
        if len(attrs) == 13:
            merge.write(tsfls[i])
            pre_part = ""
        else:
            pre_part = pre_part + tsfls[i].strip()
            print("pre_part=%s--i=%s" % (pre_part, i))
            attrs = pre_part.split('')
            print ("len:%d" % (len(attrs)))
            if len(attrs) == 13:
                merge.write(pre_part)
                pre_part = ""

if __name__ == '__main__':
    process_data_la()