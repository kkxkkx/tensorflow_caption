from nltk.translate.bleu_score import corpus_bleu

from utils_tf import bleu


def load_data(param):
    data = []
    # with open('D:/Learn/four/graduate/baseline/nlp/jddc2020_baseline/mhred/tensorflow_caption/data/' + param, 'r',
    with open('./data/' + param, 'r',
              encoding='utf-8') as f:
        lines = f.readlines()
        data_size = len(lines)
        for i in range(0, data_size - 1):
            lines[i] = lines[i].strip('\n')
            data.append(lines[i])
    return data, data_size


if __name__ == '__main__':
    max_order = 1
    smooth = False
    per_segment_references, data_size = load_data('answer.txt')
    translations, _ = load_data('result.txt')
    sum = 0
    for i in range(0, data_size - 1):
        bleu_score, precisions, bp, ratio, translation_length, reference_length = bleu.compute_bleu(
            per_segment_references[i],
            translations[i],
            max_order,
            smooth)
        sum += bleu_score
    print('bleu:{}'.format(sum/data_size))
