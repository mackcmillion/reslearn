import re
import csv

with open('/home/max/Studium/Kurse/BA2/data/yelp/sample_submission.csv', 'r') as f:
    # final = ''
    # line = f.read()
    # line = line.replace('\n', '')
    # split = line.split(',')
    # for i, word in enumerate(split):
    #     if i >= 2 and i % 2 == 0 and ']' in word:
    #         bracket = word.index(']')
    #         inserted = word[:bracket + 1] + '\n' + word[bracket + 1:]
    #         final += inserted + ','
    #     else:
    #         final += word + ','

    # final = ''
    # for line in f:
    #     line_comma = re.sub(r'(\ )+', ',', line)
    #     line_replaced = line_comma.replace(',]', ']')
    #     line_replaced2 = line_replaced.replace('[,', '[')
    #     line_replaced3 = line_replaced2.replace('.,', '.0,')
    #     line_replaced4 = line_replaced3.replace('.]', '.0]')
    #     final += line_replaced4
    # print final
    with open('/home/max/Studium/Kurse/BA2/results/submission.csv') as f2:
        csvreader1 = csv.DictReader(f)
        csvreader2 = csv.DictReader(f2)
        set1 = set(row['business_id'] for row in csvreader1)
        set2 = set(row['business_id'] for row in csvreader2)
        print set1.difference(set2)

