from strsimpy.levenshtein import Levenshtein


def sber_metric(pred, true):
    levenshtein = Levenshtein()
    len_pred = len(pred)
    len_true = len(true)
    return 1 - (levenshtein.distance(pred, true) / max(len_pred, len_true))


if __name__ == '__main__':
    print(sber_metric('sber', 'sver'))