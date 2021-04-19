input_filename = 'dev.txt'
output_filename = 'dev_notag.txt'

with open(output_filename, 'w', encoding='utf-8') as f_w:
    with open(input_filename, 'r', encoding='utf-8') as f_r:
        line = f_r.readline()
        while line:
            words_w_tags = line.split()
            words = [word_w_tag.split('/')[0] for word_w_tag in words_w_tags]

            f_w.write(' '.join(words) + '\n')
            line = f_r.readline()