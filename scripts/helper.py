import re

with open('/home/max/Studium/Kurse/BA2/results/prediction_map_val', 'r') as f:
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
    #
    final = ''
    for line in f:
        line_comma = re.sub(r'(\ )+', ',', line)
        line_replaced = line_comma.replace(',]', ']')
        line_replaced2 = line_replaced.replace('[,', '[')
        line_replaced3 = line_replaced2.replace('.,', '.0,')
        line_replaced4 = line_replaced3.replace('.]', '.0]')
        final += line_replaced4
    print final
    with open('/home/max/Studium/Kurse/BA2/results/prediction_map_val2', 'w') as f2:
        f2.write(final)
