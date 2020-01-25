AT_PARAMS = {
    'pgd-linf': {
        'nb_its': 10,
        'eps': {
            'cifar10':    [1, 2, 4, 8, 16, 32],
            'imagenet100':[1, 2, 4, 8, 16, 32],
        },
    },
    'pgd-l2': {
        'nb_its': 10,
        'eps': {
            'cifar10':    [40,  80,  160, 320,  640,  2560],
            'imagenet100':[150, 300, 600, 1200, 2400, 4800],
        },
    },
    'fw-l1': {
        'nb_its': 10,
        'eps': {
            'cifar10':    [195,    390,   780,   1560,   6240,   24960],
            'imagenet100':[9562.5, 19125, 76500, 153000, 306000, 612000],
        },
    },
    'elastic-linf': {
        'nb_its': 30,
        'eps': {
            'cifar10':    [0.125, 0.25, 0.5, 1, 2, 8],
            'imagenet100':[0.25,  0.5,  2,   4, 8, 16],
        },
    },
}