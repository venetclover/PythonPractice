def main():
    with open('spambase_1/spambase.data', 'r') as fr, \
         open("spambase_1/spambase.data.new.txt",'w') as newf:
        for line in fr:  # the idiom of reading lines from a file
            ts = line.strip().split(',')

            print >> newf, ts.pop(),

            for i, t in enumerate(ts):
                if t != '0':
                    print >> newf, '%d:%s' % (i, t),

            print >> newf


if __name__ == '__main__':
    main()