import sys
import time
import random
from binascii import hexlify
from blackbox import BlackBox

filter_length = 69997
hash_num = 30

def generate_hash_list(hash_num):
    # f(x) = (ax + b) % m
    # p is any prime number, m is the number of bins
    random.seed(0)
    a = random.sample(range(1, 10000), hash_num)
    b = random.sample(range(1, 10000), hash_num)
    # p = random.sample(prime, hash_num)
    return (a, b)

hash_list = generate_hash_list(hash_num)


def myhashs(user):
    calc_hash = lambda a, x, b: (a * x + b) % 163119

    result = []
    user_id = int(hexlify(user.encode('utf-8')), 16)

    for i in range(hash_num):
        result.append(calc_hash(hash_list[0][i], user_id, hash_list[1][i]) % filter_length)
    # print(result)
    
    return result


if __name__ == '__main__':

    start = time.time()

    input_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_path = sys.argv[4]

    bx = BlackBox()
    bloom_filter = [0 for i in range(filter_length)]
    FPR = []
    user_set = set([])

    for _ in range(num_of_asks):
        stream_users = bx.ask(input_path, stream_size)
        FP = 0
        TN = 0
        for user in stream_users:
            user_hash = myhashs(user)
            is_positive = True
            for h in user_hash:
                if bloom_filter[h] == 0:
                    TN += 1
                    is_positive = False
                    break
            if is_positive and user not in user_set:
                FP += 1

            # insert this user in filter
            for h in user_hash:
                bloom_filter[h] = 1
            user_set.add(user)

        if FP == 0:
            FPR.append(0.0)
        else:
            FPR.append(FP*1.0/(TN+FP))
        print(TN, FP)
        # print(FPR)

    with open(output_path, 'w') as f:
        f.write('Time,FPR\n')
        for i in range(num_of_asks):
            f.write('{},{}\n'.format(i, FPR[i]))

    # print(FPR)
    # print(bloom_filter)
    print('Duration: {}'.format(time.time()-start))




